# fmt: off
import numpy as np
import pandas as pd
import h5py
import scipy.signal
import skimage as sk
import os
import pickle
import sys
# import h5py_cache  # no longer using this
import copy
import shutil
import pickle as pkl
from parse import compile
from time import sleep
from distributed.client import futures_of
from dask.distributed import wait

import dask.dataframe as dd
import dask.delayed as delayed

from skimage import filters
from .trcluster import hdf5lock
from .utils import multifov,pandas_hdf5_handler,writedir
from .daskutils import add_list_to_column
from tifffile import imread

### Hacky memory trim
import ctypes

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

# def get_focus_score(img_arr):
#     # computes focus score from single image

#     Sx = sk.filters.sobel_h(img_arr)
#     Sy = sk.filters.sobel_v(img_arr)
#     Ften = np.sum(Sx**2 + Sy**2)
#     return Ften

def get_focus_score(img_arr):
    ## computes focus score from single image
    img_min = np.min(img_arr)
    img_max = np.max(img_arr)
    I = (img_arr-img_min)/(img_max-img_min)

    total_I = np.sum(I)

    Sx = sk.filters.sobel_h(I)
    Sy = sk.filters.sobel_v(I)
    edge_magnitude = np.sqrt(Sx**2 + Sy**2)
    norm_edge_magnitude = edge_magnitude / total_I
    Ften = np.sum(norm_edge_magnitude)
    return Ften

def get_grid_lookups(global_df, delta = 10):
    first_tpt = global_df.loc[pd.IndexSlice[:, slice(0, 0)], :]

    x_vals = first_tpt["x"].values
    y_vals = first_tpt["y"].values

    x_dist = np.abs(np.subtract.outer(x_vals,x_vals))
    y_dist = np.abs(np.subtract.outer(y_vals,y_vals))

    close_x = x_dist<delta
    close_y = y_dist<delta

    x_groups = []
    for x_idx in range(close_x.shape[0]):
        x_groups.append(np.where(close_x[x_idx])[0])
    x_groups = [np.array(item) for item in set(list(tuple(arr) for arr in x_groups))]
    x_groups = sorted(x_groups, key=lambda x: x[0])
    x_lookup = {item:k for k,group in enumerate(x_groups) for item in group}

    y_groups = []
    for y_idx in range(close_y.shape[0]):
        y_groups.append(np.where(close_y[y_idx])[0])
    y_groups = [np.array(item) for item in set(list(tuple(arr) for arr in y_groups))]
    y_groups = sorted(y_groups, key=lambda x: x[0])
    y_lookup = {item:k for k,group in enumerate(y_groups) for item in group}

    return x_lookup, y_lookup

def get_grid_indices(global_df, delta = 10):
    x_lookup, y_lookup = get_grid_lookups(global_df, delta = 10)

    column_indices = [x_lookup[fov_idx] for fov_idx in global_df.index.get_level_values("fov").tolist()]
    row_indices = [y_lookup[fov_idx] for fov_idx in global_df.index.get_level_values("fov").tolist()]

    return column_indices,row_indices

def get_global_row_lookups(fov_timepoint_df, delta = 20):
    first_tpt = fov_timepoint_df.loc[pd.IndexSlice[:, slice(0, 0)], :]

    y_vals = first_tpt["y (global)"].values
    y_dist = np.abs(np.subtract.outer(y_vals,y_vals))
    close_y = y_dist<delta

    y_groups = []
    for y_idx in range(close_y.shape[0]):
        y_groups.append(np.where(close_y[y_idx])[0])
    y_groups = [np.array(item) for item in set(list(tuple(arr) for arr in y_groups))]
    y_groups = sorted(y_groups, key=lambda x: x[0])
    y_lookup = {item:k for k,group in enumerate(y_groups) for item in group}

    row_indices = [y_lookup[idx] for idx in range(len(first_tpt))]

    first_tpt["Initial Global Row"] = row_indices
    first_row_df = first_tpt.groupby("Initial Global Row").first().reset_index().sort_values("y (global)")
    first_row_df["Global Row"] = range(len(first_row_df))

    init_global_row_to_global_row = {list(single_dict.keys())[0]:single_dict[list(single_dict.keys())[0]] \
    for single_dict in first_row_df.apply(lambda x: {x["Initial Global Row"]:x["Global Row"]}, axis=1).tolist()}

    first_tpt["Global Row"] = first_tpt.apply(lambda x: init_global_row_to_global_row[x["Initial Global Row"]], axis=1)
    first_tpt = first_tpt.drop("Initial Global Row", axis=1)

    return first_tpt

# def match_midpoints(midpoints, consensus_midpoints, midpoint_dist_tolerence):
#     diff_mat = np.subtract.outer(midpoints,consensus_midpoints)
#     min_cons_idx = np.argmin(abs(diff_mat),axis=0)
#     min_midpoint_idx = np.argmin(abs(diff_mat),axis=1)
#     count_arr = np.add.accumulate(np.ones(len(min_cons_idx),dtype=int))-1
#     match_mask = min_midpoint_idx[min_cons_idx] == count_arr
#     under_dist_tol = np.min(abs(diff_mat),axis=0) < midpoint_dist_tolerence
#     match_mask = match_mask*under_dist_tol
#     matched_min_cons_idx = min_cons_idx[match_mask]
#     return matched_min_cons_idx, match_mask

def match_midpoints(midpoints, consensus_midpoints, midpoint_dist_tolerence):
    assignment_dist_arr = []
    translation_arr = np.linspace(-midpoint_dist_tolerence,midpoint_dist_tolerence,num=500)

    for translation in translation_arr:
        translated_midpoints = midpoints+translation
        diff_mat = np.abs(np.subtract.outer(translated_midpoints,consensus_midpoints))
        min_dist_assignments = np.min(diff_mat,axis=1)
        mean_dist_of_assignment = np.mean(min_dist_assignments)
        assignment_dist_arr.append(mean_dist_of_assignment)

    assignment_dist_arr = np.array(assignment_dist_arr)
    optimal_translation = translation_arr[np.argmin(assignment_dist_arr)]
    translated_midpoints = midpoints+optimal_translation

    diff_mat = np.subtract.outer(translated_midpoints,consensus_midpoints)
    min_cons_idx = np.argmin(abs(diff_mat),axis=0)
    min_midpoint_idx = np.argmin(abs(diff_mat),axis=1)
    count_arr = np.add.accumulate(np.ones(len(min_cons_idx),dtype=int))-1
    match_mask = min_midpoint_idx[min_cons_idx] == count_arr
    under_dist_tol = np.min(abs(diff_mat),axis=0) < midpoint_dist_tolerence
    match_mask = match_mask*under_dist_tol
    matched_min_cons_idx = min_cons_idx[match_mask]

    return matched_min_cons_idx, match_mask

class kymograph_cluster:
    def __init__(self,headpath="",paramfile=False,all_channels=[""],filter_channel=None,trench_len_y=270,padding_y=20,trench_width_x=30,use_median_drift=False,\
                 invert=False,y_percentile=85,y_foreground_percentile=80,y_min_edge_dist=50,midpoint_dist_tolerence=50,smoothing_kernel_y=(1,9),y_percentile_threshold=0.2,\
                 top_orientation=0,expected_num_rows=None,alternate_orientation=True,alternate_over_rows=False,consensus_orientations=None,\
                 consensus_midpoints=None,x_percentile=85,background_kernel_x=(1,21),smoothing_kernel_x=(1,9),otsu_scaling=1.,min_threshold=0,trench_present_thr=0.,\
                o2_file_rename_latency=10.):

        if paramfile:
            parampath = headpath + "/kymograph.par"
            with open(parampath, 'rb') as infile:
                param_dict = pickle.load(infile)

            all_channels = param_dict["All Channels"]
            filter_channel = param_dict["Filter Channel"]
            trench_len_y = param_dict["Trench Length"]
            padding_y = param_dict["Y Padding"]
            trench_width_x = param_dict["Trench Width"]
            use_median_drift = param_dict['Use Median Drift?']
#             t_range = param_dict["Time Range"]
            invert = param_dict["Invert"]
            y_percentile = param_dict["Y Percentile"]
            y_foreground_percentile = param_dict["Y Foreground Percentile"]
            y_min_edge_dist = param_dict["Minimum Trench Length"]
            midpoint_dist_tolerence = param_dict["Midpoint Distance Tolerance"]
            smoothing_kernel_y = (1,param_dict["Y Smoothing Kernel"])
            y_percentile_threshold = param_dict['Y Percentile Threshold']
            top_orientation = param_dict["Orientation Detection Method"]
            expected_num_rows = param_dict["Expected Number of Rows (Manual Orientation Detection)"]
            alternate_orientation = param_dict['Alternate Orientation']
            alternate_over_rows = param_dict['Alternate Orientation Over Rows?']
            consensus_orientations = param_dict['Consensus Orientations']
            consensus_midpoints = param_dict['Consensus Midpoints']
            x_percentile = param_dict["X Percentile"]
            background_kernel_x = (1,param_dict["X Background Kernel"])
            smoothing_kernel_x = (1,param_dict["X Smoothing Kernel"])
            otsu_scaling = param_dict["Otsu Threshold Scaling"]
            min_threshold= param_dict['Minimum X Threshold']
            trench_present_thr =  param_dict["Trench Presence Threshold"]

        self.headpath = headpath
        self.kymographpath = self.headpath + "/kymograph"
        self.hdf5path = self.headpath + "/hdf5"
        self.all_channels = all_channels
        self.filter_channel = filter_channel
        self.seg_channel = self.all_channels[0]
        self.metapath = self.headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)

#         self.t_range = t_range
        self.invert = invert

        #### important paramaters to set
        self.trench_len_y = trench_len_y
        self.padding_y = padding_y
        ttl_len_y = trench_len_y+padding_y
        self.ttl_len_y = ttl_len_y
        self.trench_width_x = trench_width_x
        self.use_median_drift = use_median_drift

        #### params for y
        ## parameter for reducing signal to one dim
        self.y_percentile = y_percentile
        self.y_foreground_percentile = y_foreground_percentile
        self.y_min_edge_dist = y_min_edge_dist
        ## parameters for threshold finding
        self.smoothing_kernel_y = smoothing_kernel_y
        self.y_percentile_threshold = y_percentile_threshold
        ###
        self.top_orientation = top_orientation
        self.expected_num_rows = expected_num_rows
        self.alternate_orientation = alternate_orientation
        self.alternate_over_rows = alternate_over_rows
        ### Additions from Y consensus
        self.midpoint_dist_tolerence = midpoint_dist_tolerence
        self.consensus_orientations = consensus_orientations
        self.consensus_midpoints = consensus_midpoints
        #### params for x
        ## parameter for reducing signal to one dim
        self.x_percentile = x_percentile
        ## parameters for midpoint finding
        self.background_kernel_x = background_kernel_x
        self.smoothing_kernel_x = smoothing_kernel_x
        ## parameters for threshold finding
        self.otsu_scaling = otsu_scaling
        self.min_threshold = min_threshold
        ## New
        self.trench_present_thr = trench_present_thr

        self.output_chunk_shape = (1,1,self.ttl_len_y,(self.trench_width_x//2)*2)
        self.output_chunk_bytes = (2*np.multiply.accumulate(np.array(self.output_chunk_shape))[-1])
        self.output_chunk_cache_mem_size = 2*self.output_chunk_bytes

        #hardcoded wait time for file name changes, helps when o2 updates paths slowly
        self.o2_file_rename_latency = o2_file_rename_latency

        self.kymograph_params = {"trench_len_y":trench_len_y,"padding_y":padding_y,"ttl_len_y":ttl_len_y,\
                                 "trench_width_x":trench_width_x,"y_percentile":y_percentile,"y_foreground_percentile":y_foreground_percentile,"invert":invert,\
                             "y_min_edge_dist":y_min_edge_dist,"midpoint_dist_tolerence":midpoint_dist_tolerence,"smoothing_kernel_y":smoothing_kernel_y,\
                                 "y_percentile_threshold":y_percentile_threshold,\
                                 "top_orientation":top_orientation,"expected_num_rows":expected_num_rows,"alternate_orientation":alternate_orientation,\
                                 "alternate_over_rows":alternate_over_rows,"consensus_orientations":consensus_orientations,"consensus_midpoints":consensus_midpoints,\
                                 "x_percentile":x_percentile,"background_kernel_x":background_kernel_x,"smoothing_kernel_x":smoothing_kernel_x,\
                                "otsu_scaling":otsu_scaling,"min_x_threshold":min_threshold,"trench_present_thr":trench_present_thr}


    def median_filter_2d(self,array,smoothing_kernel,edge_width=3):
        """Two-dimensional median filter, with average smoothing at the signal
        edges in the second dimension (the non-time dimension).

        Args:
            array_list (list): List containing a single array of 2 dimensional signal to be smoothed.
            smoothing_kernel (tuple): A tuple of ints specifying the kernel under which
            the median will be taken.

        Returns:
            array: Median-filtered 2 dimensional signal.
        """
        kernel = np.array(smoothing_kernel) #1,9
        kernel_pad = kernel//2 + 1 #1,5
        med_filter = scipy.signal.medfilt(array,kernel_size=kernel)
#         start_edge = np.mean(med_filter[:,kernel_pad[1]:kernel[1]])
#         end_edge = np.mean(med_filter[:,-kernel[1]:-kernel_pad[1]])
        start_edge = np.mean(med_filter[:,kernel_pad[1]:kernel_pad[1]+edge_width],axis=1,keepdims=True)
        end_edge = np.mean(med_filter[:,-kernel_pad[1]-edge_width:-kernel_pad[1]],axis=1,keepdims=True)
        med_filter[:,:kernel_pad[1]] = start_edge
        med_filter[:,-kernel_pad[1]:] = end_edge
        return med_filter

    def get_smoothed_y_percentiles(self,file_idx,y_percentile,y_foreground_percentile,smoothing_kernel_y):
        """For each imported array, computes the percentile along the x-axis of
        the segmentation channel, generating a (y,t) array. Then performs
        median filtering of this array for smoothing.

        Args:
            imported_hdf5_handle (h5py.File): Hdf5 file handle corresponding to the input hdf5 dataset
            "data" of shape (channel,y,x,t).
            y_percentile (int): Percentile to apply along the x-axis.
            smoothing_kernel_y (tuple): Kernel to use for median filtering.

        Returns:
            h5py.File: Hdf5 file handle corresponding to the output hdf5 dataset "data", a smoothed
            percentile array of shape (y,t).
        """
        with h5py.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",rdcc_nbytes=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
        # with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            img_arr = imported_hdf5_handle[self.seg_channel][:] #t x y
            if self.invert:
                img_arr = sk.util.invert(img_arr)
            perc_arr = np.percentile(img_arr,y_percentile,axis=2,interpolation='lower')
            y_percentiles_smoothed = self.median_filter_2d(perc_arr,smoothing_kernel_y)

            min_qth_percentile = y_percentiles_smoothed.min(axis=1)[:, np.newaxis]
##             max_qth_percentile = y_percentiles_smoothed.max(axis=1)[:, np.newaxis]
            max_qth_percentile = np.percentile(y_percentiles_smoothed,y_foreground_percentile,axis=1)[:, np.newaxis]
            y_percentiles_smoothed = (y_percentiles_smoothed - min_qth_percentile)/(max_qth_percentile - min_qth_percentile)
            y_percentiles_smoothed = np.minimum(y_percentiles_smoothed,1.)

        return y_percentiles_smoothed

    def get_edges_from_mask(self,mask):
        """Finds edges from a boolean mask of shape (t,y). Filters out rows of
        length smaller than y_min_edge_dist.

        Args:
            mask (array): Boolean of shape (y,t) resulting from triangle thresholding.
            y_min_edge_dist (int): Minimum row length necessary for detection.

        Returns:
            list: List containing arrays of edges for each timepoint, filtered for rows that are too small.
        """
        edges_list = []
        start_above_list = []
        end_above_list = []
        for t in range(mask.shape[0]):
            edge_mask = (mask[t,1:] != mask[t,:-1])
            start_above,end_above = (mask[t,0]==True,mask[t,-1]==True)
            edges = np.where(edge_mask)[0]
            edges_list.append(edges)
            start_above_list.append(start_above)
            end_above_list.append(end_above)
        return edges_list,start_above_list,end_above_list

    def get_trench_edges_y(self,y_percentiles_smoothed_array,y_percentile_threshold,y_min_edge_dist):
        """Detects edges in the shape (t,y) smoothed percentile arrays for each
        input array.

        Args:
            y_percentiles_smoothed_array (array): A shape (y,t) smoothed percentile array.
            triangle_nbins (int): Number of bins to be used to construct the thresholding histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
            y_min_edge_dist (int): Minimum row length necessary for detection.

        Returns:
            list: List containing arrays of edges for each timepoint, filtered for rows that are too small.
        """

        trench_mask_y = y_percentiles_smoothed_array>y_percentile_threshold
        edges_list,start_above_list,end_above_list = self.get_edges_from_mask(trench_mask_y)
        return edges_list,start_above_list,end_above_list

    def repair_out_of_frame(self,trench_edges_y,start_above,end_above):
        if start_above:
            trench_edges_y =  np.array([0] + trench_edges_y.tolist())
        if end_above:
            trench_edges_y = np.array(trench_edges_y.tolist() + [int(self.metadata['height'])])
        return trench_edges_y

    def remove_small_rows(self,edges,min_edge_dist):
        """Filters out small rows when performing automated row detection.

        Args:
            edges (array): Array of edges along y-axis.
            min_edge_dist (int): Minimum row length necessary for detection.

        Returns:
            array: Array of edges, filtered for rows that are too small.
        """
        grouped_edges = edges.reshape(-1,2)
        row_lens = np.diff(grouped_edges,axis=1)
        row_mask = (row_lens>min_edge_dist).flatten()
        filtered_edges = grouped_edges[row_mask]
        return filtered_edges.flatten()

    def remove_out_of_frame(self,orientations,repaired_trench_edges_y,start_above,end_above):
        """Takes an array of trench row edges and removes the first/last edge,
        if that edge does not have a proper partner (i.e. trench row mask takes
        value True at boundaries of image).

        Args:
            edges (array): Array of edges along y-axis.
            start_above (bool): True if the trench row mask takes value True at the
            starting edge of the mask.
            end_above (bool): True if the trench row mask takes value True at the
            ending edge of the mask.

        Returns:
            array: Array of edges along y-axis, corrected for edge pairs that
            are out of frame.
        """
        drop_first_row,drop_last_row = (False,False)
        if start_above and orientations[0] == 0: #if the top is facing down and is cut
            drop_first_row = True
            orientations = orientations[1:]
            repaired_trench_edges_y = repaired_trench_edges_y[2:]
        if end_above and orientations[-1] == 1: #if the bottom is facing up and is cut
            drop_last_row = True
            orientations = orientations[:-1]
            repaired_trench_edges_y = repaired_trench_edges_y[:-2]
        return orientations,drop_first_row,drop_last_row,repaired_trench_edges_y

    def get_manual_orientations(self,trench_edges_y_list,start_above_list,end_above_list,expected_num_rows,alternate_orientation,top_orientation,current_row,y_min_edge_dist):
        trench_edges_y = trench_edges_y_list[0]
        start_above = start_above_list[0]
        end_above = end_above_list[0]
        orientations = []

        repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
        repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

        if repaired_trench_edges_y.shape[0]//2 == expected_num_rows:
            orientation = top_orientation
            ### Correction for alternating top on different FOV rows
            orientation = (orientation+current_row)%2
            for row in range(repaired_trench_edges_y.shape[0]//2):
                orientations.append(orientation)
                if alternate_orientation:
                    orientation = (orientation+1)%2
            orientations,drop_first_row,drop_last_row,repaired_trench_edges_y = self.remove_out_of_frame(orientations,repaired_trench_edges_y,start_above,end_above)

        else:
            print("Start frame does not have expected number of rows!")

        return orientations,drop_first_row,drop_last_row

    def get_trench_ends(self,trench_edges_y_list,start_above_list,end_above_list,orientations,drop_first_row,drop_last_row,y_min_edge_dist):
        top_orientation = orientations[0]

        y_ends_list = []

        for t,trench_edges_y in enumerate(trench_edges_y_list):
            start_above = start_above_list[t]
            end_above = end_above_list[t]

            repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
            repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

            if (repaired_trench_edges_y.shape[0]//2 > len(orientations)) and drop_first_row:
                repaired_trench_edges_y = repaired_trench_edges_y[2:]
            if (repaired_trench_edges_y.shape[0]//2 > len(orientations)) and drop_last_row:
                repaired_trench_edges_y = repaired_trench_edges_y[:-2]
            grouped_edges = repaired_trench_edges_y.reshape(-1,2) # or,2
            y_ends = []
            for edges,orientation in enumerate(orientations):
                y_ends.append(grouped_edges[edges,orientation])
            y_ends = np.array(y_ends)
            y_ends_list.append(y_ends)
        return y_ends_list

    ### NEW CONSENSUS SECTION

    def get_manual_orientations_withcons(self,trench_edges_y_list,start_above_list,end_above_list,consensus_orientations,consensus_midpoints,\
                                         y_min_edge_dist,midpoint_dist_tolerence):

        trench_edges_y = trench_edges_y_list[0]
        start_above = start_above_list[0]
        end_above = end_above_list[0]

        repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
        repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

        midpoints = (repaired_trench_edges_y[1:]+repaired_trench_edges_y[:-1])/2

        matched_min_cons_idx, match_mask = match_midpoints(midpoints, consensus_midpoints, midpoint_dist_tolerence)
        repaired_trench_edges_y = [repaired_trench_edges_y[idx:idx+2] for idx in matched_min_cons_idx]
        repaired_trench_edges_y = np.array([part for item in repaired_trench_edges_y for part in item])
        orientations = np.array(consensus_orientations)[match_mask].tolist()
        filtered_consensus_midpoints = np.array(consensus_midpoints)[match_mask].tolist()

        orientations,drop_first_row,drop_last_row,_ = self.remove_out_of_frame(orientations,repaired_trench_edges_y,start_above,end_above)
        if drop_first_row:
            filtered_consensus_midpoints = filtered_consensus_midpoints[1:]
        if drop_last_row:
            filtered_consensus_midpoints = filtered_consensus_midpoints[:-1]

        return orientations,filtered_consensus_midpoints

    def get_trench_ends_withcons(self,trench_edges_y_list,start_above_list,end_above_list,orientations,filtered_consensus_midpoints,\
                                y_min_edge_dist,midpoint_dist_tolerence):

        y_ends_list = []

        for t,trench_edges_y in enumerate(trench_edges_y_list):
            start_above = start_above_list[t]
            end_above = end_above_list[t]

            repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
            repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

            midpoints = (repaired_trench_edges_y[1:]+repaired_trench_edges_y[:-1])/2

            matched_min_cons_idx, match_mask = match_midpoints(midpoints, filtered_consensus_midpoints, midpoint_dist_tolerence)
            repaired_trench_edges_y = [repaired_trench_edges_y[idx:idx+2] for idx in matched_min_cons_idx]
            repaired_trench_edges_y = np.array([part for item in repaired_trench_edges_y for part in item])
            matched_orientations = np.array(orientations)[match_mask].tolist()

            grouped_edges = repaired_trench_edges_y.reshape(-1,2) # or,2
            y_ends = []
            for edges,orientation in enumerate(matched_orientations):
                y_ends.append(grouped_edges[edges,orientation])
            y_ends = np.array(y_ends)
            y_ends_list.append(y_ends)

        return y_ends_list

    ### END

    def get_y_drift(self,y_ends_list): ## IF THIS WORKS PUSH TO INTER Y DRIFT
        """Given a list of midpoints, computes the average drift in y for every
        timepoint.

        Args:
            y_midpoints_list (list): A list containing, for each fov, a list of the form [time_list,[midpoint_array]]
            containing the trench row midpoints.

        Returns:
            list: A nested list of the form [time_list,[y_drift_int]] for fov i.
        """
        y_drift = []
        for t in range(len(y_ends_list)-1):
            diff_mat = np.subtract.outer(y_ends_list[t+1],y_ends_list[t])
            if diff_mat.shape[0] > 0 and diff_mat.shape[1] > 0:
                min_dist_idx = np.argmin(abs(diff_mat),axis=0)
                min_dist_inv_idx = np.argmin(abs(diff_mat),axis=1)
                count_arr = np.add.accumulate(np.ones(len(min_dist_idx),dtype=int))-1
                match_mask = min_dist_inv_idx[min_dist_idx] == count_arr
                min_dists = diff_mat[min_dist_idx[match_mask],min_dist_inv_idx[min_dist_idx[match_mask]]]
                median_translation = np.median(min_dists)
            else:
                median_translation = 0
            y_drift.append(median_translation)
        net_y_drift = np.append(np.array([0]),np.add.accumulate(y_drift)).astype(int)
        return net_y_drift

    def keep_in_frame_kernels(self,y_ends_list,y_drift,orientations,padding_y,trench_len_y):
        """Removes those kernels which drift out of the image during any timepoint.
        Args:
            trench_edges_y_lists (list): A list containing, for each fov, a time-ordered list of trench edge arrays.
            y_drift_list (list): A list containing, for each fov, a nested list of the form [time_list,[y_drift_int]].
            imported_array_list (int): A numpy array containing the hdf5 file image data.
            padding_y (int): Y-dimensional padding for cropping.

        Returns:
            list: Time-ordered list of trench edge arrays, filtered for images which
            stay in frame for all timepoints, for fov i.
        """

        init_y_ends = y_ends_list[0]
        max_y_dim = self.metadata['height']
        max_drift,min_drift = np.max(y_drift),np.min(y_drift)

        valid_init_y_ends = []
        valid_orientations = []
        for j,orientation in enumerate(orientations):  ## this is  a static list across all time
            y_end = init_y_ends[j]
            if orientation == 0:
                bottom_edge = y_end+trench_len_y+max_drift
                top_edge = y_end-padding_y+min_drift
                edge_under_max = bottom_edge<max_y_dim
                edge_over_min = top_edge >= 0
            else:
                bottom_edge = y_end+padding_y+max_drift
                top_edge = y_end-trench_len_y+min_drift
                edge_under_max = bottom_edge<max_y_dim
                edge_over_min = top_edge >= 0

            edge_in_bounds = edge_under_max*edge_over_min

            if edge_in_bounds:
                valid_init_y_ends.append(init_y_ends[j])
                valid_orientations.append(orientation)

        valid_init_y_ends = np.array(valid_init_y_ends)

        return valid_init_y_ends,valid_orientations

    def get_ends_and_orientations(self,fov_idx,edges_futures,expected_num_rows,alternate_orientation,top_orientation,alternate_over_rows,\
                                  consensus_orientations,consensus_midpoints,y_min_edge_dist,midpoint_dist_tolerence,padding_y,trench_len_y):

        fovdf = self.meta_handle.read_df("global",read_metadata=False)
        if alternate_over_rows:
            _, y_lookup = get_grid_lookups(fovdf, delta = 10)
            current_row = y_lookup[fov_idx]
        else:
            current_row = 0
#         fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        working_fovdf = fovdf.loc[fov_idx]

        trench_edges_y_list = []
        start_above_list = []
        end_above_list = []

        for j,file_idx in enumerate(working_fovdf["File Index"].unique().tolist()):
            working_filedf = working_fovdf[working_fovdf["File Index"]==file_idx]
            img_indices = working_filedf["Image Index"].unique()
            first_idx,last_idx = (img_indices[0],img_indices[-1])
            trench_edges_y_list += edges_futures[j][0][first_idx:last_idx+1]
            start_above_list += edges_futures[j][1][first_idx:last_idx+1]
            end_above_list += edges_futures[j][2][first_idx:last_idx+1]

        if consensus_orientations == None:

            orientations,drop_first_row,drop_last_row = self.get_manual_orientations(trench_edges_y_list,start_above_list,end_above_list,expected_num_rows,\
                                                                                     alternate_orientation,top_orientation,current_row,y_min_edge_dist)

            y_ends_list = self.get_trench_ends(trench_edges_y_list,start_above_list,end_above_list,orientations,drop_first_row,drop_last_row,y_min_edge_dist)

        else:
            orientations,filtered_consensus_midpoints = self.get_manual_orientations_withcons(trench_edges_y_list,start_above_list,end_above_list,\
                                                                consensus_orientations,consensus_midpoints,y_min_edge_dist,midpoint_dist_tolerence)

            y_ends_list = self.get_trench_ends_withcons(trench_edges_y_list,start_above_list,end_above_list,orientations,filtered_consensus_midpoints,\
                                y_min_edge_dist,midpoint_dist_tolerence)

        y_drift = self.get_y_drift(y_ends_list)
        valid_init_y_ends,valid_orientations = self.keep_in_frame_kernels(y_ends_list,y_drift,orientations,padding_y,trench_len_y)

        return y_drift,valid_orientations,valid_init_y_ends

    def get_median_y_drift(self,drift_orientation_and_initend_futures):
        y_drift_list = [item[0] for item in drift_orientation_and_initend_futures]
        median_drift = np.round(np.median(np.array(y_drift_list),axis=0)).astype(int)
        return median_drift

    def update_y_drift_futures(self,new_y_drift,drift_orientation_and_initend_future):
        drift_orientation_and_initend_future = tuple((new_y_drift,drift_orientation_and_initend_future[1],drift_orientation_and_initend_future[2]))
        return drift_orientation_and_initend_future

    def crop_y(self,file_idx,drift_orientation_and_initend_future,padding_y,trench_len_y):
        """Performs cropping of the images in the y-dimension.

        Args:
            i (int): Specifies the current fov index.
            trench_edges_y_list (list): List containing, for each fov entry, a list of time-sorted edge arrays.
            row_num_list (list): List containing The number of trench rows detected in each fov.
            imported_array_list (list): A list containing numpy arrays containing the hdf5 file image
            data of shape (channel,y,x,t).
            padding_y (int): Padding to be used when cropping in the y-dimension.
            trench_len_y (int): Length from the end of the tenches to be used when cropping in the
            y-dimension.
            top_orientation (int, optional): The orientation of the top-most row where 0 corresponds to a trench with
            a downward-oriented trench opening and 1 corresponds to a trench with an upward-oriented trench opening.
        Returns:
            array: A y-cropped array of shape (rows,channels,x,y,t).
        """
        fovdf = self.meta_handle.read_df("global",read_metadata=False)
#         fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]

        filedf = fovdf.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()
        working_filedf = filedf.loc[file_idx]

        timepoint_indices = working_filedf["timepoints"].unique().tolist()
        image_indices = working_filedf.index.get_level_values("Image Index").unique().tolist()
#         first_idx,last_idx = (timepoint_indices[0]-self.t_range[0],timepoint_indices[-1]-self.t_range[0])
        first_idx,last_idx = (timepoint_indices[0],timepoint_indices[-1])


        y_drift = drift_orientation_and_initend_future[0][first_idx:last_idx+1]
        valid_orientations,valid_init_y_ends = drift_orientation_and_initend_future[1:]

        drift_corrected_edges = np.add.outer(y_drift,valid_init_y_ends)


        channel_arr_list = []
        for c,channel in enumerate(self.all_channels):
            with h5py.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",rdcc_nbytes=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            # with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
                img_arr = imported_hdf5_handle[channel][image_indices[0]:image_indices[-1]+1]
            time_list = []
            lane_y_coords_list = []
            for t in range(img_arr.shape[0]):
                trench_ends_y = drift_corrected_edges[t]
                row_list = []
                lane_y_coords = []
                for r,orientation in enumerate(valid_orientations):
                    trench_end = trench_ends_y[r]
                    if orientation == 0:
                        upper = max(trench_end-padding_y,0)
                        lower = min(trench_end+trench_len_y,img_arr.shape[1])
                    else:
                        upper = max(trench_end-trench_len_y,0)
                        lower = min(trench_end+padding_y,img_arr.shape[1])
                    lane_y_coords.append(upper)
                    output_array = img_arr[t,upper:lower,:]
                    row_list.append(output_array)
                time_list.append(row_list)
                lane_y_coords_list.append(lane_y_coords)
            cropped_in_y = np.array(time_list) # t x row x y x x
            if len(cropped_in_y.shape) != 4:
                print("Error in crop_y")
                raise
            else:
                channel_arr_list.append(cropped_in_y)
        return channel_arr_list,lane_y_coords_list

    def get_smoothed_x_percentiles(self,file_idx,drift_orientation_and_initend_future,padding_y,trench_len_y,x_percentile,background_kernel_x,smoothing_kernel_x):
        """Summary.

        Args:
            array_tuple (tuple): A singleton tuple containing the y-cropped hdf5 array of shape (rows,x,y,t).
            background_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing background subtraction
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.
            smoothing_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing smoothing
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.

        Returns:
            array: A smoothed and background subtracted percentile array of shape (rows,x,t)
        """
        channel_arr_list,_ = self.crop_y(file_idx,drift_orientation_and_initend_future,padding_y,trench_len_y)
        cropped_in_y = channel_arr_list[0]
        if self.invert:
            cropped_in_y = sk.util.invert(cropped_in_y)
#         cropped_in_y = y_crop_future[0][0] # t x row x y x x     # (24, 1, 330, 2048)

        x_percentiles_smoothed = []
        for row_num in range(cropped_in_y.shape[1]):
            cropped_in_y_seg = cropped_in_y[:,row_num] # t x y x x
            x_percentiles = np.percentile(cropped_in_y_seg,x_percentile,axis=1) # t x x
            x_background_filtered = x_percentiles - self.median_filter_2d(x_percentiles,background_kernel_x)
            x_smooth_filtered = self.median_filter_2d(x_background_filtered,smoothing_kernel_x)
            x_smooth_filtered[x_smooth_filtered<0.] = 0.
            x_percentiles_smoothed.append(x_smooth_filtered)
        x_percentiles_smoothed=np.array(x_percentiles_smoothed) # row x t x x
        return x_percentiles_smoothed

    def get_midpoints_from_mask(self,mask):
        """Using a boolean x mask, computes the positions of trench midpoints.

        Args:
            mask (array): x boolean array, specifying where trenches are present.

        Returns:
            array: array of trench midpoint x positions.
        """
        transitions = mask[:-1].astype(int) - mask[1:].astype(int)

        trans_up = np.where((transitions==-1))[0]
        trans_dn = np.where((transitions==1))[0]

        if len(np.where(trans_dn>trans_up[0])[0])>0:
            first_dn = np.where(trans_dn>trans_up[0])[0][0]
            trans_dn = trans_dn[first_dn:]
            trans_up = trans_up[:len(trans_dn)]
            midpoints = (trans_dn + trans_up)//2
        else:
            midpoints = []
        return midpoints

    def get_x_row_midpoints(self,x_percentiles_t,otsu_scaling,min_threshold):
        """Given an array of signal in x, determines the position of trench
        midpoints.

        Args:
            x_percentiles_t (array): array of trench intensities in x, at time t.
            otsu_nbins (int): Number of bins to use when applying Otsu's method to x-dimension signal.
            otsu_scaling (float): Threshold scaling factor for Otsu's method thresholding.

        Returns:
            array: array of trench midpoint x positions.
        """

        otsu_threshold = sk.filters.threshold_otsu(x_percentiles_t[:,np.newaxis],nbins=50)*otsu_scaling
        modified_otsu_threshold = max(otsu_threshold,min_threshold)

        x_mask = x_percentiles_t>modified_otsu_threshold
        midpoints = self.get_midpoints_from_mask(x_mask)
        return midpoints

    def get_x_midpoints(self,x_percentiles_smoothed,otsu_scaling,min_threshold):
        """Given an x percentile array of shape (rows,t,x), determines the
        trench midpoints of each row array at each time t.

        Args:
            x_percentiles_smoothed_array (array): A smoothed and background subtracted percentile array of shape (rows,x,t)
            otsu_nbins (int): Number of bins to use when applying Otsu's method to x-dimension signal.
            otsu_scaling (float): Threshold scaling factor for Otsu's method thresholding.

        Returns:
            list: A nested list of the form [row_list,[time_list,[midpoint_array]]].
        """
        all_midpoints_list = []
        for row in range(x_percentiles_smoothed.shape[0]):
            row_x_percentiles = x_percentiles_smoothed[row]
            all_midpoints = []
            midpoints = self.get_x_row_midpoints(row_x_percentiles[0],otsu_scaling,min_threshold)
            if len(midpoints) == 0:
                return None
            all_midpoints.append(midpoints)

            for t in range(1,row_x_percentiles.shape[0]):
                midpoints = self.get_x_row_midpoints(row_x_percentiles[t],otsu_scaling,min_threshold)
                if len(midpoints)/(len(all_midpoints[-1])+1) < 0.5:
                    all_midpoints.append(all_midpoints[-1])
                else:
                    all_midpoints.append(midpoints)
            all_midpoints_list.append(all_midpoints)
        return all_midpoints_list

    def compile_midpoint_futures(self,midpoint_futures):
        num_rows = len(midpoint_futures[0])
        all_midpoints_list = []
        for row in range(num_rows):
            row_midpoints_list = []
            for midpoint_future in midpoint_futures:
                row_midpoints_list += midpoint_future[row]
            all_midpoints_list.append(row_midpoints_list)
        return all_midpoints_list

    def get_x_drift(self,midpoint_futures):
        """Given a list of midpoints, computes the average drift in x for every
        timepoint.

        Args:
            all_midpoints_list (list): A nested list of the form [row_list,[time_list,[midpoint_array]]] containing
            the trench midpoints.

        Returns:
            list: A nested list of the form [row_list,[time_list,[x_drift_int]]].
        """
        all_midpoints_list = self.compile_midpoint_futures(midpoint_futures)

        x_drift_list = []
        for all_midpoints in all_midpoints_list:
            x_drift = []
            for t in range(len(all_midpoints)-1):
                diff_mat = np.subtract.outer(all_midpoints[t+1],all_midpoints[t])
                min_dist_idx = np.argmin(abs(diff_mat),axis=0)
                min_dists = diff_mat[min_dist_idx]
                median_translation = int(np.median(min_dists))
                x_drift.append(median_translation)
            net_x_drift = np.append(np.array([0]),np.add.accumulate(x_drift))
            x_drift_list.append(net_x_drift)
        return x_drift_list

    def get_median_x_drift(self,x_drift_futures):
        uppacked_x_drift_futures = [row for fov in x_drift_futures for row in fov]
        median_drift = np.round(np.median(np.array(uppacked_x_drift_futures),axis=0)).astype(int)
        return median_drift

    def update_x_drift_futures(self,new_x_drift,x_drift_future):
        x_drift_future = [copy.copy(new_x_drift) for row in x_drift_future]
        return x_drift_future

    def filter_midpoints(self,all_midpoints,x_drift,trench_width_x,trench_present_thr):

        drift_corrected_midpoints = []
        for t in range(len(x_drift)):
            drift_corrected_t = all_midpoints[t]-x_drift[t]
            drift_corrected_midpoints.append(drift_corrected_t)
        midpoints_up,midpoints_dn = (all_midpoints[0]-trench_width_x//2,\
                                     all_midpoints[0]+trench_width_x//2+1)

        trench_present_t = []
        for t in range(len(drift_corrected_midpoints)):
            above_mask = np.greater.outer(drift_corrected_midpoints[t],midpoints_up)
            below_mask = np.less.outer(drift_corrected_midpoints[t],midpoints_dn)
            in_bound_mask = (above_mask*below_mask)
            trench_present = np.any(in_bound_mask,axis=0)
            trench_present_t.append(trench_present)
        trench_present_t = np.array(trench_present_t)
        trench_present_perc = np.sum(trench_present_t,axis=0)/trench_present_t.shape[0]

        presence_filter_mask = trench_present_perc>=trench_present_thr

        midpoint_seeds = all_midpoints[0][presence_filter_mask]
        return midpoint_seeds

    def get_in_bounds(self,all_midpoints,x_drift,trench_width_x,trench_present_thr):
        """Produces and writes a trench mask of shape (y_dim,t_dim,x_dim). This
        will be used to mask out trenches from the reshaped "cropped_in_y"
        array at a later step.

        Args:
            cropped_in_y (array): A y-cropped hdf5 array of shape (rows,y,x,t) containing y-cropped image data.
            all_midpoints (list): A list containing, for each time t, an array of trench midpoints.
            x_drift (list): A list containing, for each time t, an int corresponding to the drift of the midpoints in x.
            trench_width_x (int): Width to be used when cropping in the x-dimension.

        Returns:
            h5py.File: Hdf5 file handle corresponding to the trench mask hdf5 dataset
            "data" of shape (y_dim,t_dim,x_dim).
            int: Total number of trenches detected in the image.
        """

        midpoint_seeds = self.filter_midpoints(all_midpoints,x_drift,trench_width_x,trench_present_thr)
        corrected_midpoints = x_drift[:,np.newaxis].astype(int)+midpoint_seeds[np.newaxis,:].astype(int) ### DIRTY FIX

        midpoints_up,midpoints_dn = (corrected_midpoints-trench_width_x//2,\
                                     corrected_midpoints+trench_width_x//2+1)
        stays_in_frame = np.all(midpoints_up>=0,axis=0)*np.all(midpoints_dn<=self.metadata["width"],axis=0) #filters out midpoints that stay in the frame for the whole time...
#         no_overlap = np.append(np.array([True]),(corrected_midpoints[0,1:]-corrected_midpoints[0,:-1])>=(trench_width_x+1)) #corrects for overlap
#         if np.sum(no_overlap)/len(no_overlap)<0.9:
#             print("Trench overlap issue!!!")

#         valid_mask = stays_in_frame*no_overlap
        in_bounds = np.array([midpoints_up[:,stays_in_frame],\
                            midpoints_dn[:,stays_in_frame]])
        k_tot = in_bounds.shape[2]

        x_coords = np.mean(in_bounds,axis=0).T #CHANGED TO BE MIDDLE OF KYMO XCOORD
#         x_coords = in_bounds[0].T
        return in_bounds,x_coords,k_tot

    def get_all_in_bounds(self,midpoint_futures,x_drift_future,trench_width_x,trench_present_thr):
        """Generates complete kymograph arrays for all trenches in the fov in
        every channel listed in 'self.all_channels'. Writes hdf5 files
        containing datasets of shape (trench_num,y_dim,x_dim,t_dim) for each
        row,channel combination. Dataset keys follow the convention.

        ["[row_number]/[channel_name]"].

        Args:
            cropped_in_y_handle (h5py.File): Hdf5 file handle corresponding to the y-cropped hdf5 dataset
            "data" of shape (rows,channels,x,y,t).
            all_midpoints_list (list): A nested list of the form [row_list,[time_list,[midpoint_array]]] containing
            the trench midpoints.
            x_drift_list (list): A nested list of the form [row_list,[time_list,[x_drift_int]]] containing the computed
            drift in the x dimension.
            trench_width_x (int): Width to be used when cropping in the x-dimension.
        """
        all_midpoints_list = self.compile_midpoint_futures(midpoint_futures)

        in_bounds_list = []
        x_coords_list = []
        k_tot_list = []

        for row_num,all_midpoints in enumerate(all_midpoints_list):
            x_drift = x_drift_future[row_num]
            in_bounds,x_coords,k_tot = self.get_in_bounds(all_midpoints,x_drift,trench_width_x,trench_present_thr)

            in_bounds_list.append(in_bounds)
            x_coords_list.append(x_coords)
            k_tot_list.append(k_tot)

        return in_bounds_list,x_coords_list,k_tot_list

    def crop_with_bounds(self,output_kymograph,cropped_in_y_list,working_in_bounds,k_tot,row_num):
        """Generates and writes kymographs of a single row from the already
        y-cropped image data, using a pregenerated kymograph mask of shape
        (y_dim,t_dim,x_dim).

        Args:
            cropped_in_y_handle (h5py.File): Hdf5 file handle corresponding to the y-cropped hdf5 dataset
            "data" of shape (rows,channels,x,y,t).
            k_mask_handle (h5py.File): Hdf5 file handle corresponding to the trench mask hdf5 dataset
            "data" of shape (y_dim,t_dim,x_dim).
            row_num (int): The row number to crop kymographs from.
            k_tot (int): Int specifying the total number of detected trenches in the fov.
        """

        for c,channel in enumerate(self.all_channels):
            dataset_name = str(row_num) + "/" + str(channel)
            cropped_in_y = cropped_in_y_list[c][:,row_num] # t,y,x
            k_len,t_len,y_len,x_len = (working_in_bounds.shape[2],working_in_bounds.shape[1],cropped_in_y.shape[1],working_in_bounds[1,0,0]-working_in_bounds[0,0,0])

            kymo_out = np.zeros((k_len,t_len,y_len,x_len),dtype="uint16")

            for t in range(working_in_bounds.shape[1]):
                for k in range(working_in_bounds.shape[2]):
                    bounds = working_in_bounds[:,t,k]
                    kymo_out[k,t] = cropped_in_y[t,:,bounds[0]:bounds[1]]

#             kymo_out = self.apply_kymo_mask(kymo_mask,cropped_in_y,k_tot) # k x t x y x x

            hdf5_dataset = output_kymograph.create_dataset(dataset_name,data=kymo_out,chunks=self.output_chunk_shape, dtype='uint16')

    def crop_x(self,file_idx,drift_orientation_and_initend_future,in_bounds_future,padding_y,trench_len_y):
        """Generates complete kymograph arrays for all trenches in the fov in
        every channel listed in 'self.all_channels'. Writes hdf5 files
        containing datasets of shape (trench_num,y_dim,x_dim,t_dim) for each
        row,channel combination. Dataset keys follow the convention.

        ["[row_number]/[channel_name]"].

        Args:
            cropped_in_y_handle (h5py.File): Hdf5 file handle corresponding to the y-cropped hdf5 dataset
            "data" of shape (rows,channels,x,y,t).
            all_midpoints_list (list): A nested list of the form [row_list,[time_list,[midpoint_array]]] containing
            the trench midpoints.
            x_drift_list (list): A nested list of the form [row_list,[time_list,[x_drift_int]]] containing the computed
            drift in the x dimension.
            trench_width_x (int): Width to be used when cropping in the x-dimension.
        """
        fovdf = self.meta_handle.read_df("global",read_metadata=False)
#         fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        filedf = fovdf.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()
        working_filedf = filedf.loc[file_idx]

        timepoint_indices = working_filedf["timepoints"].unique().tolist()
        image_indices = working_filedf.index.get_level_values("Image Index").unique().tolist()
#         first_idx,last_idx = (timepoint_indices[0]-self.t_range[0],timepoint_indices[-1]-self.t_range[0])  #CHANGED
        first_idx,last_idx = (timepoint_indices[0],timepoint_indices[-1])  #CHANGED


        channel_arr_list,lane_y_coords_list = self.crop_y(file_idx,drift_orientation_and_initend_future,padding_y,trench_len_y)
        num_rows = channel_arr_list[0].shape[1]

        in_bounds_list,x_coords_list,k_tot_list = in_bounds_future
#         counting_arr = self.init_counting_arr(self.metadata["width"])
        with h5py.File(self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5","w",rdcc_nbytes=self.output_chunk_cache_mem_size) as output_kymograph:
        # with h5py_cache.File(self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5","w",chunk_cache_mem_size=self.output_chunk_cache_mem_size) as output_kymograph:
            for row_num in range(num_rows):
                in_bounds,k_tot = (in_bounds_list[row_num],k_tot_list[row_num])
                working_in_bounds = in_bounds[:,first_idx:last_idx+1]
#                 kymo_mask = self.get_trench_mask(in_bounds[:,first_idx:last_idx+1],counting_arr)
                self.crop_with_bounds(output_kymograph,channel_arr_list,working_in_bounds,k_tot,row_num)

        return lane_y_coords_list

    def save_coords(self,fov_idx,x_crop_futures,in_bounds_future,drift_orientation_and_initend_future):
        fovdf = self.meta_handle.read_df("global",read_metadata=False)
#         fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        fovdf = fovdf.loc[fov_idx]

        x_coords_list = in_bounds_future[1]
        orientations = drift_orientation_and_initend_future[1]

        y_coords_list = []
        for j,file_idx in enumerate(fovdf["File Index"].unique().tolist()):
            working_filedf = fovdf[fovdf["File Index"]==file_idx]
            img_indices = working_filedf["Image Index"].unique()
            y_coords_list += x_crop_futures[j] # t x row list

        pixel_microns = self.metadata['pixel_microns']
        y_coords = np.array(y_coords_list) # t x row array
        scaled_y_coords = y_coords*pixel_microns
        t_len = scaled_y_coords.shape[0]
        fs = np.repeat([fov_idx],t_len)
        orit_dict = {0:"top",1:"bottom"}
        tpts = np.array(range(t_len))

        missing_metadata = ('x' not in fovdf.columns)

        if not missing_metadata:
            global_x,global_y,ts,file_indices,img_indices = (fovdf["x"].values,fovdf["y"].values,fovdf["t"].values,fovdf["File Index"].values,fovdf["Image Index"].values)
        else:
            file_indices,img_indices = (fovdf["File Index"].values,fovdf["Image Index"].values)

        pd_output = []

        for l,x_coord in enumerate(x_coords_list):
            scaled_x_coord = x_coord*pixel_microns
            yt = scaled_y_coords[:,l]
            orit = np.repeat([orit_dict[orientations[l]]],t_len)
            if not missing_metadata:
                global_yt = yt+global_y
            ls = np.repeat([l],t_len)
            for k in range(scaled_x_coord.shape[0]):
                xt = scaled_x_coord[k]
                if not missing_metadata:
                    global_xt = xt+global_x
                ks = np.repeat([k],t_len)
                if not missing_metadata:
                    pd_output.append(np.array([fs,ls,ks,tpts,file_indices,img_indices,ts,orit,yt,xt,global_yt,global_xt]).T)
                else:
                    pd_output.append(np.array([fs,ls,ks,tpts,file_indices,img_indices,orit,yt,xt]).T)

        pd_output = np.concatenate(pd_output,axis=0)
        if not missing_metadata:
            df = pd.DataFrame(pd_output,columns=["fov","row","trench","timepoints","File Index","Image Index","time (s)","lane orientation","y (local)","x (local)","y (global)","x (global)"])
            df = df.astype({"fov":int,"row":int,"trench":int,"timepoints":int,"File Index":int,"Image Index":int,"time (s)":float,"lane orientation":str,"y (local)":float,"x (local)":float,\
                            "y (global)":float,"x (global)":float})
        else:
            df = pd.DataFrame(pd_output,columns=["fov","row","trench","timepoints","File Index","Image Index","lane orientation","y (local)","x (local)"])
            df = df.astype({"fov":int,"row":int,"trench":int,"timepoints":int,"File Index":int,"Image Index":int,"lane orientation":str,"y (local)":float,"x (local)":float,})

        fov_indices = df.apply(lambda x: int(f'{x["fov"]:04n}{x["row"]:04n}{x["trench"]:04n}{x["timepoints"]:04n}'), axis=1)
        temp_file_idx = df.apply(lambda x: int(f'{x["File Index"]:08n}{x["Image Index"]:04n}{x["timepoints"]:04n}'), axis=1)
        df["FOV Parquet Index"] = [item for item in fov_indices]
        df["Temp File Parquet Index"] = [item for item in temp_file_idx]
        df = df.set_index("FOV Parquet Index").sort_index()
        df = df.dropna()

        df.to_hdf(self.kymographpath + "/temp_output/temp_output_" + str(fov_idx) + ".hdf", "data",mode="w",format="table")

        return fov_idx

    def get_filter_scores(self,channel,file_idx):
        df = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)

        working_rowdfs = []

        proc_file_path = self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5"
        with h5py.File(proc_file_path,"r") as infile:
            start_index = int(f'{file_idx:08n}' + '00000000')
            end_index = int(f'{file_idx:08n}' + '99999999')
            working_filedf = df.loc[start_index:end_index].compute()
            del df
            working_filedf = working_filedf.set_index("FOV Parquet Index",drop=True).sort_index()
#             df[df["File Index"]==file_idx]
            row_list = working_filedf["row"].unique().tolist()
            for row in row_list:
                working_rowdf = working_filedf[working_filedf["row"]==row]
                kymo_arr = infile[str(row) + "/" + channel][:]
                original_shape = kymo_arr.shape
                kymo_arr = kymo_arr.reshape(-1,original_shape[2],original_shape[3])
                focus_scores = [get_focus_score(kymo_arr[i]) for i in range(kymo_arr.shape[0])]
                intensity_scores = [np.mean(kymo_arr[i]) for i in range(kymo_arr.shape[0])]

                working_rowdf[channel + " Focus Score"] = focus_scores
                working_rowdf[channel + " Mean Intensity"] = intensity_scores

                working_rowdfs.append(working_rowdf)

        out_df = pd.concat(working_rowdfs)
        return out_df

    def get_all_filter_scores(self,channel,dask_controller):
        df = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)
        file_list = df["File Index"].unique().compute().tolist()
        del df

        delayed_list = []

        for file_idx in file_list:
            df_delayed = delayed(self.get_filter_scores)(channel,file_idx)
            delayed_list.append(df_delayed.persist())

        ## filtering out non-failed dataframes ##
        all_delayed_futures = []
        for item in delayed_list:
            all_delayed_futures+=futures_of(item)
        while any(future.status == 'pending' for future in all_delayed_futures):
            sleep(0.1)

        good_delayed = []
        for item in delayed_list:
            if all([future.status == 'finished' for future in futures_of(item)]):
                good_delayed.append(item)

        ## compiling output dataframe ##
        df_out = dd.from_delayed(good_delayed).persist()
        df_out = df_out.repartition(partition_size="25MB").persist()
        dd.to_parquet(df_out, self.kymographpath + "/metadata_2",engine='fastparquet',compression='gzip',write_metadata_file=True)
        dask_controller.daskclient.cancel(df_out)
        shutil.rmtree(self.kymographpath + "/metadata")

    def generate_kymographs(self,dask_controller,first_fov_only=False,delta_global_rows=25):
        writedir(self.kymographpath,overwrite=True)
        writedir(self.kymographpath+ "/temp_output",overwrite=True)

        dask_controller.futures = {}

        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = fovdf.metadata
        if first_fov_only:
            first_fov_val = fovdf.index.get_level_values(0)[0]
            fovdf = fovdf.loc[first_fov_val:first_fov_val]
#         fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]

        filedf = fovdf.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()

        file_list = filedf.index.get_level_values("File Index").unique().values
        fov_list = fovdf.index.get_level_values("fov").unique().values
        num_file_jobs = len(file_list)
        num_fov_jobs = len(fov_list)

        ### smoothed y percentiles ###

        for k,file_idx in enumerate(file_list):
            future = dask_controller.daskclient.submit(self.get_smoothed_y_percentiles,file_idx,\
                                        self.y_percentile,self.y_foreground_percentile,self.smoothing_kernel_y,retries=1)
            dask_controller.futures["Smoothed Y Percentiles: " + str(file_idx)] = future

        ### get trench row edges, y midpoints ###

        for k,file_idx in enumerate(file_list):
            smoothed_y_future = dask_controller.futures["Smoothed Y Percentiles: " + str(file_idx)]
            future = dask_controller.daskclient.submit(self.get_trench_edges_y,smoothed_y_future,self.y_percentile_threshold,\
                                                       self.y_min_edge_dist,retries=1)

            dask_controller.futures["Y Trench Edges: " + str(file_idx)] = future

        ### get y drift, orientations, init edges ###

        for k,fov_idx in enumerate(fov_list):
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            edges_futures = [dask_controller.futures["Y Trench Edges: " + str(file_idx)] for file_idx in working_files]
            future = dask_controller.daskclient.submit(self.get_ends_and_orientations,fov_idx,edges_futures,self.expected_num_rows,self.alternate_orientation,\
                                                       self.top_orientation,self.alternate_over_rows,self.consensus_orientations,self.consensus_midpoints,\
                                                       self.y_min_edge_dist,self.midpoint_dist_tolerence,self.padding_y,self.trench_len_y,retries=1)

            dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)] = future

        ### optionally get median drift ###
        if self.use_median_drift:
            drift_orientation_and_initend_futures = [dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)] for fov_idx in fov_list]
            drift_orientation_and_initend_futures = dask_controller.daskclient.gather(drift_orientation_and_initend_futures,errors="skip")
            future = dask_controller.daskclient.submit(self.get_median_y_drift,drift_orientation_and_initend_futures,retries=1)
            dask_controller.futures["Y Median Drift"] = future
            for k,fov_idx in enumerate(fov_list):
                new_y_drift = dask_controller.futures["Y Median Drift"]
                drift_orientation_and_initend_future = dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)]
                future = dask_controller.daskclient.submit(self.update_y_drift_futures,new_y_drift,drift_orientation_and_initend_future,retries=1)
                dask_controller.futures["Median Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)] = future

        ### smoothed x percentiles ###

        for k,file_idx in enumerate(file_list):
            working_filedf = filedf.loc[file_idx]
            fov_idx = working_filedf["fov"].unique().tolist()[0]
            if self.use_median_drift:
                drift_orientation_and_initend_future = dask_controller.futures["Median Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)]
            else:
                drift_orientation_and_initend_future = dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)]
            future = dask_controller.daskclient.submit(self.get_smoothed_x_percentiles,file_idx,drift_orientation_and_initend_future,\
                                                       self.padding_y,self.trench_len_y,self.x_percentile,self.background_kernel_x,\
                                                       self.smoothing_kernel_x,retries=1)
            dask_controller.futures["Smoothed X Percentiles: " + str(file_idx)] = future

        ### get x midpoints ###

        for k,file_idx in enumerate(file_list):
            smoothed_x_future = dask_controller.futures["Smoothed X Percentiles: " + str(file_idx)]
            future = dask_controller.daskclient.submit(self.get_x_midpoints,smoothed_x_future,\
                                                       self.otsu_scaling,self.min_threshold,retries=1)
            dask_controller.futures["X Midpoints: " + str(file_idx)] = future

        ### get x drift ###

        for k,fov_idx in enumerate(fov_list):
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            midpoint_futures = [dask_controller.futures["X Midpoints: " + str(file_idx)] for file_idx in working_files]
            future = dask_controller.daskclient.submit(self.get_x_drift,midpoint_futures,retries=1)
            dask_controller.futures["X Drift: " + str(fov_idx)] = future

        ### optionally get median drift ###

        if self.use_median_drift:
            x_drift_futures = [dask_controller.futures["X Drift: " + str(fov_idx)] for fov_idx in fov_list]
            x_drift_futures = dask_controller.daskclient.gather(x_drift_futures,errors="skip")
            future = dask_controller.daskclient.submit(self.get_median_x_drift,x_drift_futures,retries=1)
            dask_controller.futures["X Median Drift"] = future
            for k,fov_idx in enumerate(fov_list):
                new_x_drift = dask_controller.futures["X Median Drift"]
                x_drift_future = dask_controller.futures["X Drift: " + str(fov_idx)]
                future = dask_controller.daskclient.submit(self.update_x_drift_futures,new_x_drift,x_drift_future,retries=1)
                dask_controller.futures["Median X Drift: " + str(fov_idx)] = future

        ### get kymograph masks ###

        for k,fov_idx in enumerate(fov_list):
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            midpoint_futures = [dask_controller.futures["X Midpoints: " + str(file_idx)] for file_idx in working_files]
            if self.use_median_drift:
                x_drift_future = dask_controller.futures["Median X Drift: " + str(fov_idx)]
            else:
                x_drift_future = dask_controller.futures["X Drift: " + str(fov_idx)]
            future = dask_controller.daskclient.submit(self.get_all_in_bounds,midpoint_futures,x_drift_future,\
                                                self.trench_width_x,self.trench_present_thr,retries=1)
            dask_controller.futures["X In Bounds: " + str(fov_idx)] = future

        ### crop in x ###


        for k,file_idx in enumerate(file_list):
            working_filedf = filedf.loc[file_idx]
            fov_idx = working_filedf["fov"].unique().tolist()[0]
            if self.use_median_drift:
                drift_orientation_and_initend_future = dask_controller.futures["Median Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)]
            else:
                drift_orientation_and_initend_future = dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)]
            in_bounds_future = dask_controller.futures["X In Bounds: " + str(fov_idx)]

            future = dask_controller.daskclient.submit(self.crop_x,file_idx,drift_orientation_and_initend_future,in_bounds_future,self.padding_y,self.trench_len_y,retries=0)
            dask_controller.futures["X Crop: " + str(file_idx)] = future

        ### get coords ###
        df_fov_idx_futures = []
        for k,fov_idx in enumerate(fov_list):
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            x_crop_futures = [dask_controller.futures["X Crop: " + str(file_idx)] for file_idx in working_files]
            in_bounds_future = dask_controller.futures["X In Bounds: " + str(fov_idx)]
            if self.use_median_drift:
                drift_orientation_and_initend_future = dask_controller.futures["Median Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)]
            else:
                drift_orientation_and_initend_future = dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Ends: " + str(fov_idx)]

            df_fov_idx_future = dask_controller.daskclient.submit(self.save_coords,fov_idx,x_crop_futures,in_bounds_future,drift_orientation_and_initend_future,retries=0)
            df_fov_idx_futures.append(df_fov_idx_future)

        while any(future.status == 'pending' for future in df_fov_idx_futures):
            sleep(0.1)

        good_futures = []
        for future in df_fov_idx_futures:
            if future.status == 'finished':
                good_futures.append(future.result())

        temp_output_file_list = [self.kymographpath + "/temp_output/temp_output_" + str(fov_idx) + ".hdf" for fov_idx in sorted(good_futures)]
        df_out = dd.read_hdf(temp_output_file_list,"data",mode="r",sorted_index=True)

        ## compiling output dataframe ##
        df_out = self.add_trenchids(df_out).persist()
        df_out = df_out.reset_index(drop=False).set_index("Temp File Parquet Index", drop=True, sorted=False).persist() #sorted=False) HERE

        successful_fovs = df_out["fov"].unique().compute().tolist()

        dd.to_parquet(df_out, self.kymographpath + "/metadata",engine='pyarrow',compression='gzip',write_metadata_file=True,overwrite=True)

        dask_controller.daskclient.cancel(df_out)
        dask_controller.daskclient.cancel(good_futures)

        if self.filter_channel != None:
            self.get_all_filter_scores(self.filter_channel,dask_controller)
        else:
            ### compiling output dataframe ##
            #### Dirty fix
            df_out = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)
            df_out = df_out.set_index("FOV Parquet Index",drop=True,sorted=False)
            df_out = df_out.repartition(partition_size="25MB")
            dd.to_parquet(df_out, self.kymographpath + "/metadata_2",engine='pyarrow',compression='gzip',write_metadata_file=True)
            dask_controller.daskclient.cancel(df_out)
            shutil.rmtree(self.kymographpath + "/metadata")

        # Adding global row column
        df_out = dd.read_parquet(self.kymographpath + "/metadata_2",calculate_divisions=True)
        fov_row_timepoint_df = df_out.groupby(["fov","row","timepoints"],sort=False).first().compute()
        fov_timepoint_df = fov_row_timepoint_df.reset_index(drop=False).set_index(["fov","timepoints"]).sort_index()

        first_timepoint_with_lane_rows = get_global_row_lookups(fov_timepoint_df, delta = delta_global_rows)
        fov_row_to_globalrow_dict = first_timepoint_with_lane_rows.reset_index()[["fov","row","Global Row"]].set_index(["fov","row"])["Global Row"].to_dict()
        df_out["Global Row"] = df_out.apply(lambda x: fov_row_to_globalrow_dict[(x["fov"],x["row"])], axis=1, meta=(None,int))
        dd.to_parquet(df_out, self.kymographpath + "/metadata_3",engine='pyarrow',compression='gzip',write_metadata_file=True)
        dask_controller.daskclient.cancel(df_out)
        shutil.rmtree(self.kymographpath + "/metadata_2")
        os.rename(self.kymographpath + "/metadata_3", self.kymographpath + "/metadata")
        sleep(self.o2_file_rename_latency)

        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        fov_list = fovdf.index.get_level_values("fov").unique().values
        failed_fovs = list(set(fov_list)-set(successful_fovs))

        kymograph_metadata = {"attempted_fov_list":fov_list,"successful_fov_list":successful_fovs,"failed_fov_list":failed_fovs,"kymograph_params":self.kymograph_params}

        with open(self.kymographpath + "/metadata.pkl", 'wb') as handle:
            pickle.dump(kymograph_metadata, handle)

        dask_controller.daskclient.cancel([val for key,val in dask_controller.futures.items()])
        dask_controller.daskclient.run(trim_memory)

    def add_trenchids(self,df):

        trench_preindex = df.apply(lambda x: int(f'{x["fov"]:04n}{x["row"]:04n}{x["trench"]:04n}'), axis=1)
        df["key"] = trench_preindex.persist()

        trenchids = df.set_index("key",sorted=True).groupby("key").size().reset_index().drop(0,axis=1).reset_index().compute()
        trenchids.columns = ["trenchid","key"]
        df = df.join(trenchids.set_index('key'),how="left",on="key",rsuffix="moo")
        del df["key"]

        return df

#     def add_list_to_column(self,df,list_to_add,column_name):
#         df = df.repartition(partition_size="25MB").persist()
#         df["index"] = 1
#         idx = df["index"].cumsum()
#         df["index"] = idx

#         list_to_add = pd.DataFrame(list_to_add)
#         list_to_add["index"] = idx
#         df = df.join(list_to_add.set_index('index'),how="left",on="index")

#         df = df.drop(["index"],axis=1)

#         df.columns = df.columns.tolist()[:-1] + [column_name]

#         return df

    def filter_trenchids(self,dask_controller,filter_channel,df,focus_threshold = 0.,intensity_threshold=0.,perc_above = 0.):

        def filter_single_trenchid(df: pd.DataFrame,
                               channel: str,
                               focus_threshold = 0.,
                               intensity_threshold = 0.,
                               num_timepoints_above_threshold_to_keep_trench = 0.):
            """
            Filter trenches based on focus and intensity scores.
            Initially intended to be used as part of a groupby-apply operation to filter trenches
            Args:
                df (pd.DataFrame): DataFrame containing trench data.
                channel (str): The channel to filter on.
                focus_threshold (float): Threshold for focus score.
                intensity_threshold (float): Threshold for mean intensity.
                num_timepoints_above_threshold_to_keep_trench (int): Minimum number of timepoints above threshold to keep trench.
            Returns:
                pd.DataFrame: Filtered DataFrame or None if trench is discarded.
            """
            mask_pass_focus_filter = df[channel + " Focus Score"] > focus_threshold
            mask_pass_intensity_filter = df[channel + " Mean Intensity"] > intensity_threshold
            mask_pass_combined = mask_pass_focus_filter & mask_pass_intensity_filter

            num_timepoints_passing_mask = np.sum(mask_pass_combined)
            if num_timepoints_passing_mask >= num_timepoints_above_threshold_to_keep_trench:
                return df
            else:
                return None

        num_above = np.round(len(df["timepoints"].unique().compute())*perc_above).astype(int)
        meta_df = df.head(0).reset_index()

        out_df = \
        (df
        .reset_index()
        .groupby("trenchid")
        .apply(filter_single_trenchid,
                meta=meta_df,
                channel=filter_channel,
                focus_threshold=focus_threshold,
                intensity_threshold=intensity_threshold,
                num_timepoints_above_threshold_to_keep_trench=num_above)
        .reset_index(drop=True)
        .set_index('FOV Parquet Index', sort=True) # This is very inefficient but I did not find another easy way to deal with the indices
        )
        
        ## compiling output dataframe ##
        dd.to_parquet(out_df, self.kymographpath + "/metadata_2",engine='fastparquet',compression='gzip',write_metadata_file=True)
        dask_controller.daskclient.cancel(out_df)
        shutil.rmtree(self.kymographpath + "/metadata")
        os.rename(self.kymographpath + "/metadata_2", self.kymographpath + "/metadata")
    
    def reindex_trenches(self,df):

        num_timepoints = len(df["timepoints"].unique())
        fov_row_idx = df.apply(lambda x: int(f'{x["fov"]:04n}{x["row"]:04n}'), axis=1)
        df["fov-row Index"] = fov_row_idx.persist()
        wait(df);

        ### NEW ##

        new_trenches = df.set_index("fov-row Index",sorted=True)["trench"].persist()
        wait(new_trenches);
        new_trenches = new_trenches.groupby("fov-row Index").apply(lambda x: list(np.repeat(list(range(0,len(x.unique()))),repeats=num_timepoints))).persist()
        wait(new_trenches);

#         new_trenches = df.set_index("fov-row Index",sorted=True).groupby("fov-row Index").apply(lambda x: np.repeat(list(range(0,len(x["trench"].unique()))),repeats=num_timepoints))
#         wait(new_trenches);
        new_trenches = new_trenches.compute().sort_index()
        new_trenches = new_trenches.apply(eval)
        new_trenches = [element for list_ in new_trenches for element in list_]
        new_trenches = pd.DataFrame(new_trenches)
        df = df.drop(["trenchid","trench","fov-row Index"],axis=1)
        new_trenches.index = df.index

        new_trenches_dask_df = dd.from_pandas(new_trenches,npartitions=df.npartitions).persist()
        wait(new_trenches_dask_df);
        new_trenches_dask_df = new_trenches_dask_df.repartition(divisions=df.divisions)
        # new_trenches_dask_df = new_trenches_dask_df.repartition(divisions=df.divisions,force=True)
        new_trenches_dask_df = new_trenches_dask_df.persist()
        wait(new_trenches_dask_df);
        df["trench"] = new_trenches_dask_df[0]

        cols = df.columns.tolist()
        reordered_columns = cols[:2] + cols[-1:] + cols[2:-1]
        df = df[reordered_columns]

        ## NEW CODE ##
        fov_idx = df.apply(lambda x: int(f'{x["fov"]:04n}{x["row"]:04n}{x["trench"]:04n}{x["timepoints"]:04n}'), axis=1)
        df["FOV Parquet Index"] = fov_idx.persist()
        wait(df);

        df = df.set_index("FOV Parquet Index")
        return df

    def reorg_kymograph(self,k,trenchids):
#         df = dd.read_parquet(self.kymographpath + "/metadata/")
#         trenchid_list = df["trenchid"].unique().compute().tolist()
#         trenchiddf = df.set_index("trenchid")
        working_trenchdf = dd.read_parquet(self.kymographpath + "/trenchiddf",calculate_divisions=True).loc[trenchids].compute(scheduler='threads')
        working_trenchdf = working_trenchdf.sort_values(["fov","File Index","row"])
        output_file_path = self.kymographpath+"/kymograph_"+str(k)+".hdf5"

        with h5py.File(output_file_path,"w") as outfile:
            for channel in self.all_channels:
                fov_list = working_trenchdf["fov"].unique().tolist()
                trench_arr_fovs = []
                for fov in fov_list:
                    working_fovdf = working_trenchdf[working_trenchdf["fov"]==fov]
                    file_list = working_fovdf["File Index"].unique().tolist()

                    trench_arr_files = []
                    for file_idx in file_list:
                        proc_file_path = self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5"
                        with h5py.File(proc_file_path,"r") as infile:
                            working_filedf = working_fovdf[working_fovdf["File Index"]==file_idx]
                            row_list = working_filedf["row"].unique().tolist()

                            trench_arr_rows = []
                            for row in row_list:
                                working_rowdf = working_filedf[working_filedf["row"]==row]
#                                 trenches = working_rowdf["trench"].unique().tolist()
#                                 first_trench_idx,last_trench_idx = (trenches[0],trenches[-1])
#                                 kymo_arr = infile[str(row) + "/" + channel][first_trench_idx:(last_trench_idx+1)]
                                trenches = working_rowdf["trench"].unique().tolist()
                                kymo_arr = infile[str(row) + "/" + channel][trenches]
                                trench_arr_rows.append(kymo_arr)
                        trench_arr_rows = np.concatenate(trench_arr_rows,axis=0) # k x t x y x x
                        trench_arr_files.append(trench_arr_rows)
                    trench_arr_files = np.concatenate(trench_arr_files,axis=1) # k x t x y x x
                    trench_arr_fovs.append(trench_arr_files)
                trench_arr_fovs = np.concatenate(trench_arr_fovs,axis=0) # k x t x y x x
                hdf5_dataset = outfile.create_dataset(str(channel), data=trench_arr_fovs, dtype="uint16")

    def cleanup_kymographs(self,reorg_futures,file_list):
        for file_idx in file_list:
            proc_file_path = self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5"
            os.remove(proc_file_path)
        return 1

    def post_process(self,dask_controller,trench_timepoints_per_file=25000,focus_thr=0,intensity_thr=0,perc_above_thr=0,filter_channel=None):

        dask_controller.futures = {}

        outputdf = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)

        ##hack until I can fix this
        outputdf = outputdf.repartition(npartitions=len(outputdf.divisions),force=True)
        outputdf = outputdf.reset_index().set_index("FOV Parquet Index",sorted=False)
        dd.to_parquet(outputdf, self.kymographpath + "/metadata_2",engine='pyarrow',compression='gzip',write_metadata_file=True,schema="infer")
        dask_controller.daskclient.cancel(outputdf)
        shutil.rmtree(self.kymographpath + "/metadata")
        os.rename(self.kymographpath + "/metadata_2", self.kymographpath + "/metadata")
        sleep(self.o2_file_rename_latency)
        outputdf = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)
        
        if os.path.exists(self.headpath + "/focus_filter.par"):
            print("Applying Focus Filter...")
            with open(self.headpath + "/focus_filter.par", 'rb') as infile:
                param_dict = pickle.load(infile)

            filter_channel = param_dict["Filter Channel"]
            focus_thr = param_dict["Focus Threshold"]
            intensity_thr = param_dict["Intensity Threshold"]
            perc_above_thr = param_dict["Percent Of Kymograph"]

            if filter_channel != None:
                self.filter_trenchids(dask_controller,filter_channel,outputdf,focus_threshold=focus_thr,\
                                      intensity_threshold=intensity_thr,perc_above=perc_above_thr)
                outputdf = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)
        
        if os.path.exists(self.kymographpath + "/global_rows.pkl"):
            print("Eliminating selected rows...")
            with open(self.kymographpath + "/global_rows.pkl", 'rb') as handle:
                rows_to_keep = pkl.load(handle)
            outputdf = outputdf[outputdf["Global Row"].apply(lambda x: x in rows_to_keep, meta=('Global Row',bool))]

            fov_to_rows_series = outputdf.groupby("fov")["row"].apply(lambda x: sorted(list(set(x)))).compute().sort_index()
            fov_to_rows_series = fov_to_rows_series.apply(eval)
            
            fov_to_rows_map = fov_to_rows_series.apply(lambda x: {old_row:new_row for new_row,old_row in enumerate(sorted(x))}).to_frame().reset_index()
            fov_to_rows_map = fov_to_rows_map.apply(lambda x: {(x["fov"],key):val for key,val in x["row"].items()}, axis=1).tolist()
            fov_to_rows_map = {k: v for d in fov_to_rows_map for k, v in d.items()}

            outputdf["new row"] = outputdf.apply(lambda x: int(fov_to_rows_map[(x["fov"],x["row"])]), axis=1, meta=("new row", int))
            outputdf = outputdf.reset_index(drop=True)
            outputdf["FOV Parquet Index"] = outputdf.apply(lambda x: int(f'{x["fov"]:04n}{x["new row"]:04n}{x["trench"]:04n}{x["timepoints"]:04n}'), axis=1, meta=("FOV Parquet Index", int))
            outputdf = outputdf.set_index("FOV Parquet Index",sorted=True)

            dd.to_parquet(outputdf, self.kymographpath + "/metadata_2",engine='pyarrow',compression='gzip',write_metadata_file=True,schema="infer")
            dask_controller.daskclient.cancel(outputdf)
            shutil.rmtree(self.kymographpath + "/metadata")
            os.rename(self.kymographpath + "/metadata_2", self.kymographpath + "/metadata")
            sleep(self.o2_file_rename_latency)

            outputdf = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)

        trenchid_list = outputdf["trenchid"].unique().compute().tolist()
        file_list = outputdf["File Index"].unique().compute().tolist()

        trenchiddf = outputdf.set_index("trenchid",sorted=False).persist()
        wait(trenchiddf);
        outputdf = outputdf.drop(columns = ["File Index","Image Index"]).persist()
        wait(outputdf);

#         writedir(self.kymographpath + "/metadata",overwrite=True)

        num_tpts = len(trenchiddf["timepoints"].unique().compute().tolist())
        trenches_per_file = trench_timepoints_per_file//num_tpts
        chunk_size = trenches_per_file*num_tpts

        print("Number of timepoints per trench: " + str(num_tpts))
        print("Number of trenches per file: " + str(trenches_per_file))

        if len(trenchid_list)%trenches_per_file == 0:
            num_files = (len(trenchid_list)//trenches_per_file)
        else:
            num_files = (len(trenchid_list)//trenches_per_file) + 1

        file_indices = np.repeat(np.array(range(num_files)),chunk_size)[:len(outputdf)]
        file_trenchid = np.repeat(np.array(range(trenches_per_file)),num_tpts)
        file_trenchid = np.repeat(file_trenchid[:,np.newaxis],num_files,axis=1).T.flatten()[:len(outputdf)]
        file_indices = pd.DataFrame(file_indices)
        file_trenchid = pd.DataFrame(file_trenchid)
        file_indices.index = outputdf.index
        file_trenchid.index = outputdf.index

        #test code
        file_indices_dask_df = dd.from_pandas(file_indices,npartitions=outputdf.npartitions).persist()
        wait(file_indices_dask_df);
        # file_indices_dask_df = file_indices_dask_df.repartition(divisions=outputdf.divisions)
        file_indices_dask_df = file_indices_dask_df.repartition(divisions=outputdf.divisions,force=True)
        file_indices_dask_df = file_indices_dask_df.persist()
        wait(file_indices_dask_df);
        outputdf["File Index"] = file_indices_dask_df[0]
        #test code
        file_trenchid_dask_df = dd.from_pandas(file_trenchid,npartitions=outputdf.npartitions).persist()
        wait(file_trenchid_dask_df);
        # file_trenchid_dask_df = file_trenchid_dask_df.repartition(divisions=outputdf.divisions)
        file_trenchid_dask_df = file_trenchid_dask_df.repartition(divisions=outputdf.divisions,force=True)
        file_trenchid_dask_df = file_trenchid_dask_df.persist()
        wait(file_trenchid_dask_df);
        outputdf["File Trench Index"] = file_trenchid_dask_df[0]
        outputdf = outputdf.persist()
        wait(outputdf);

##         outputdf = add_list_to_column(outputdf,file_indices[0].tolist(),"File Index").persist()
##         wait(outputdf);
#         outputdf = add_list_to_column(outputdf,file_trenchid[0].tolist(),"File Trench Index").persist()
#         wait(outputdf);

        parq_file_idx = outputdf.apply(lambda x: int(f'{int(x["File Index"]):08n}{int(x["File Trench Index"]):04n}{int(x["timepoints"]):04n}'), axis=1, meta=int)
        outputdf["File Parquet Index"] = parq_file_idx.persist()
        outputdf = outputdf.astype({"File Index":int,"File Trench Index":int,"File Parquet Index":int}).persist()
        wait(outputdf);

        dd.to_parquet(trenchiddf, self.kymographpath + "/trenchiddf",engine='fastparquet',compression='gzip',write_metadata_file=True)
        del trenchiddf

        random_priorities = np.random.uniform(size=(num_files,))
        for k in range(0,num_files):
            priority = random_priorities[k]

            trenchids = trenchid_list[k*trenches_per_file:(k+1)*trenches_per_file]

            future = dask_controller.daskclient.submit(self.reorg_kymograph,k,trenchids,retries=1,priority=priority)
            dask_controller.futures["Kymograph Reorganized: " + str(k)] = future

        reorg_futures = [dask_controller.futures["Kymograph Reorganized: " + str(k)] for k in range(num_files)]
        future = dask_controller.daskclient.submit(self.cleanup_kymographs,reorg_futures,file_list,retries=1,priority=priority)
        dask_controller.futures["Kymographs Cleaned Up"] = future
        dask_controller.daskclient.gather([future])

        if os.path.exists(self.kymographpath + "/global_rows.pkl"):
            outputdf = outputdf.drop("row",axis=1)
            outputdf = outputdf.rename(columns={"new row":"row"})

        #checkpoint
        dd.to_parquet(outputdf, self.kymographpath + "/metadata_2",engine='pyarrow',compression='gzip',write_metadata_file=True,schema="infer")
        dask_controller.daskclient.cancel(outputdf)
        shutil.rmtree(self.kymographpath + "/metadata")
        os.rename(self.kymographpath + "/metadata_2", self.kymographpath + "/metadata")
        sleep(self.o2_file_rename_latency)
        outputdf = dd.read_parquet(self.kymographpath + "/metadata",calculate_divisions=True)

        ## everything after this is slow

        outputdf = self.reindex_trenches(outputdf).persist()
        wait(outputdf);
        outputdf = self.add_trenchids(outputdf).persist()
        wait(outputdf);

        ### NEW INDEX
        outputdf["Trenchid Timepoint Index"] = outputdf.apply(lambda x: int(f'{x["trenchid"]:08n}{x["timepoints"]:04n}'), axis=1, meta=int)
        outputdf["Trenchid Timepoint Index"] = outputdf["Trenchid Timepoint Index"].astype(int)
        ### END
        outputdf = outputdf.drop(labels=["FOV Parquet Index"],axis=1)
        dd.to_parquet(outputdf, self.kymographpath + "/metadata_2",engine='pyarrow',compression='gzip',write_metadata_file=True,schema="infer")
        dask_controller.daskclient.cancel(outputdf)
        shutil.rmtree(self.kymographpath + "/metadata")
        os.rename(self.kymographpath + "/metadata_2", self.kymographpath + "/metadata")
        shutil.rmtree(self.kymographpath + "/trenchiddf")

    def kymo_report(self):
        df = dd.read_parquet(self.kymographpath + "/metadata/",calculate_divisions=True).persist()
        with open(self.kymographpath + "/metadata.pkl", 'rb') as handle:
            metadata = pickle.load(handle)

        fov_list = metadata["attempted_fov_list"]
        failed_fovs = metadata["failed_fov_list"]

        fovs_proc = len(df.groupby(["fov"]).size().compute())
        rows_proc = len(df.groupby(["fov","row"]).size().compute())
        trenches_proc = len(df.groupby(["fov","row","trench"]).size().compute())

        print("fovs processed: " + str(fovs_proc) + "/" + str(len(fov_list)))
        print("rows processed: " + str(rows_proc))
        print("trenches processed: " + str(trenches_proc))
        print("row/fov: " + str(rows_proc/fovs_proc))
        print("trenches/fov: " + str(trenches_proc/fovs_proc))

        print("failed fovs: " + str(failed_fovs))

class kymograph_multifov(multifov):
    def __init__(self,headpath):
        """The kymograph class is used to generate and visualize kymographs.
        The central function of this class is the method 'generate_kymograph',
        which takes an hdf5 file of images from a single fov and outputs an
        hdf5 file containing kymographs from all detected trenches.

        NOTE: I need to revisit the row detection, must ensure there can be no overlap...

        Args:
            input_file_prefix (string): File prefix for all input hdf5 files of the form
            [input_file_prefix][number].hdf5
            all_channels (list): list of strings corresponding to the different image channels
            available in the input hdf5 file, with the channel used for segmenting trenches in
            the first position. NOTE: these names must match those of the input hdf5 file datasets.
            trench_len_y (int): Length from the end of the tenches to be used when cropping in the
            y-dimension.
        """

        self.headpath = headpath
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        self.metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = self.metadf.metadata

    def import_hdf5(self,i):
        """Performs initial import of the hdf5 file to be processed. Converts
        the input hdf5 file's "channel" datasets into the first dimension of
        the array, ordered as specified by 'self.all_channels'. Outputs a numpy
        array.

        Args:
            i (int): Specifies the current fov index.

        Returns:
            array: A numpy array containing the hdf5 file image data.
        """
        fov = self.selected_fov_list[i]
        fovdf = self.metadf.loc[fov]
        last_idx = fovdf.index.get_level_values(0).unique().tolist()[-1]
        fovdf = fovdf.loc[slice(0,last_idx,self.t_subsample_step),:]
        file_indices = fovdf["File Index"].unique().tolist()

        channel_list = []
        for channel in self.all_channels:
            file_list = []
            for j,file_idx in enumerate(file_indices):
                filedf = fovdf[fovdf["File Index"]==file_idx]
                img_indices = filedf["Image Index"].unique().tolist()
                with h5py.File(self.headpath + "/hdf5/hdf5_" + str(file_idx) + ".hdf5", "r") as infile:
                    file_list += [infile[channel][idx][:,:,np.newaxis] for idx in img_indices]
            channel_list.append(np.concatenate(file_list,axis=2))
        channel_array = np.array(channel_list)
        if self.invert:
            channel_array = sk.util.invert(channel_array)
        return channel_array

    def import_hdf5_files(self,all_channels,seg_channel,filter_channel,invert,selected_fov_list,t_subsample_step):
        seg_channel_idx = all_channels.index(seg_channel)
        all_channels.insert(0, all_channels.pop(seg_channel_idx))
        self.all_channels = all_channels
        self.seg_channel = all_channels[0]
        self.filter_channel = filter_channel
        self.selected_fov_list = selected_fov_list
        self.t_subsample_step = t_subsample_step
        self.invert = invert

        super(kymograph_multifov, self).__init__(selected_fov_list)

        imported_array_list = self.map_to_fovs(self.import_hdf5)

        self.imported_array_list = imported_array_list

        return imported_array_list

    def median_filter_2d(self,array,smoothing_kernel,edge_width=3):
        """Two-dimensional median filter, with average smoothing at the signal
        edges in the first dimension.

        Args:
            array_list (list): List containing a single array of yt signal to be smoothed.

        Returns:
            array: Median-filtered yt signal.
        """
        kernel = np.array(smoothing_kernel)
        kernel_pad = kernel//2 + 1
        med_filter = scipy.signal.medfilt(array,kernel_size=kernel)
#         start_edge = np.mean(med_filter[kernel_pad[0]:kernel[0]])
#         end_edge = np.mean(med_filter[-kernel[0]:-kernel_pad[0]])
        start_edge = np.mean(med_filter[kernel_pad[0]:kernel_pad[0]+edge_width],axis=0)
        end_edge = np.mean(med_filter[-kernel_pad[0]-edge_width:-kernel_pad[0]],axis=0)
        med_filter[:kernel_pad[0]] = start_edge
        med_filter[-kernel_pad[0]:] = end_edge
        return med_filter

    def get_smoothed_y_percentiles(self,i,imported_array_list,y_percentile,y_foreground_percentile,smoothing_kernel_y):
        """For each imported array, computes the percentile along the x-axis of
        the segmentation channel, generating a (y,t) array. Then performs
        median filtering of this array for smoothing.

        Args:
            i (int): Specifies the current fov index.
            imported_array_list (list): A 3list containing numpy arrays containing the hdf5 file image
            data of shape (channel,y,x,t).
            y_percentile (int): Percentile to apply along the x-axis.
            smoothing_kernel_y (tuple): Kernel to use for median filtering.

        Returns:
            array: A smoothed percentile array of shape (y,t)
        """
        imported_array = imported_array_list[i]
        y_percentiles = np.percentile(imported_array[0],y_percentile,axis=1,interpolation='lower')
        y_percentiles_smoothed = self.median_filter_2d(y_percentiles,smoothing_kernel_y)
        # Normalize (scale by range and subtract minimum) to make scaling of thresholds make more sense
        min_qth_percentile = y_percentiles_smoothed.min(axis=0)
#         max_qth_percentile = y_percentiles_smoothed.max(axis=0)
        max_qth_percentile = np.percentile(y_percentiles_smoothed,y_foreground_percentile,axis=0)
        y_percentiles_smoothed = (y_percentiles_smoothed - min_qth_percentile)/(max_qth_percentile - min_qth_percentile)
        y_percentiles_smoothed = np.minimum(y_percentiles_smoothed,1.)
        return y_percentiles_smoothed

    def get_edges_from_mask(self,mask):
        """Finds edges from a boolean mask of shape (y,t). Filters out rows of
        length smaller than y_min_edge_dist.

        Args:
            mask (array): Boolean of shape (y,t) resulting from triangle thresholding.
            y_min_edge_dist (int): Minimum row length necessary for detection.

        Returns:
            list: List containing arrays of edges for each timepoint, filtered for rows that are too small.
        """
        edges_list = []
        start_above_list = []
        end_above_list = []
        for t in range(mask.shape[1]):
            edge_mask = (mask[1:,t] != mask[:-1,t])
            start_above,end_above = (mask[0,t]==True,mask[-1,t]==True)
            edges = np.where(edge_mask)[0]
            #edges = self.remove_out_of_frame(edges,start_above,end_above)
            #edges = self.remove_small_rows(edges,y_min_edge_dist)
            edges_list.append(edges)
            start_above_list.append(start_above)
            end_above_list.append(end_above)
        return edges_list,start_above_list,end_above_list

    def get_trench_edges_y(self,i,y_percentiles_smoothed_list,y_percentile_threshold):
        """Detects edges in the shape (y,t) smoothed percentile arrays for each
        input array.

        Args:
            i (int): Specifies the current fov index.
            y_percentiles_smoothed_list (list): List containing a smoothed percentile array for each input array.
            triangle_nbins (int): Number of bins to be used to construct the thresholding histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
            y_min_edge_dist (int): Minimum row length necessary for detection.

        Returns:
            list: List containing arrays of edges for each timepoint, filtered for rows that are too small.
        """
        y_percentiles_smoothed = y_percentiles_smoothed_list[i]
        trench_mask_y = y_percentiles_smoothed>y_percentile_threshold
        trench_edges_y_list,start_above_list,end_above_list = self.get_edges_from_mask(trench_mask_y)
        return trench_edges_y_list,start_above_list,end_above_list

    def repair_out_of_frame(self,trench_edges_y,start_above,end_above):
        if start_above:
            trench_edges_y =  np.array([0] + trench_edges_y.tolist())
        if end_above:
            trench_edges_y = np.array(trench_edges_y.tolist() + [int(self.metadata['height'])])
        return trench_edges_y

    def remove_small_rows(self,edges,min_edge_dist):
        """Filters out small rows when performing automated row detection.

        Args:
            edges (array): Array of edges along y-axis.
            min_edge_dist (int): Minimum row length necessary for detection.

        Returns:
            array: Array of edges, filtered for rows that are too small.
        """
        grouped_edges = edges.reshape(-1,2)
        row_lens = np.diff(grouped_edges,axis=1)
        row_mask = (row_lens>min_edge_dist).flatten()
        filtered_edges = grouped_edges[row_mask]
        return filtered_edges.flatten()

    def remove_out_of_frame(self,orientations,repaired_trench_edges_y,start_above,end_above):
        """Takes an array of trench row edges and removes the first/last edge,
        if that edge does not have a proper partner (i.e. trench row mask takes
        value True at boundaries of image).

        Args:
            edges (array): Array of edges along y-axis.
            start_above (bool): True if the trench row mask takes value True at the
            starting edge of the mask.
            end_above (bool): True if the trench row mask takes value True at the
            ending edge of the mask.

        Returns:
            array: Array of edges along y-axis, corrected for edge pairs that
            are out of frame.
        """
        drop_first_row,drop_last_row = (False,False)
        if start_above and orientations[0] == 0: #if the top is facing down and is cut
            drop_first_row = True
            orientations = orientations[1:]
            repaired_trench_edges_y = repaired_trench_edges_y[2:]
        if end_above and orientations[-1] == 1: #if the bottom is facing up and is cut
            drop_last_row = True
            orientations = orientations[:-1]
            repaired_trench_edges_y = repaired_trench_edges_y[:-2]
        return orientations,drop_first_row,drop_last_row,repaired_trench_edges_y



    def get_manual_orientations(self,i,trench_edges_y_lists,start_above_lists,end_above_lists,alternate_orientation,\
                                expected_num_rows,top_orientation,alternate_over_rows,y_min_edge_dist):
        trench_edges_y = trench_edges_y_lists[i][0]
        start_above = start_above_lists[i][0]
        end_above = end_above_lists[i][0]
        orientations = []

        repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)

        repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

        if repaired_trench_edges_y.shape[0]//2 == expected_num_rows:
            orientation = top_orientation
            if alternate_over_rows:
                fov = self.selected_fov_list[i]
                _, y_lookup = get_grid_lookups(self.metadf, delta = 10)
                row = y_lookup[fov]
                orientation = (orientation+row)%2

            for row in range(repaired_trench_edges_y.shape[0]//2):
                orientations.append(orientation)
                if alternate_orientation:
                    orientation = (orientation+1)%2
            orientations,drop_first_row,drop_last_row,repaired_trench_edges_y = self.remove_out_of_frame(orientations,repaired_trench_edges_y,start_above,end_above)

        else:
            print("Start frame does not have expected number of rows!")

        return orientations,drop_first_row,drop_last_row

    def get_trench_ends(self,i,trench_edges_y_lists,start_above_lists,end_above_lists,orientations_list,drop_first_row_list,drop_last_row_list,y_min_edge_dist):
        trench_edges_y_list = trench_edges_y_lists[i]
        start_above_list = start_above_lists[i]
        end_above_list = end_above_lists[i]
        orientations = orientations_list[i]
        drop_first_row,drop_last_row = (drop_first_row_list[i],drop_last_row_list[i])

        top_orientation = orientations[0]

        y_ends_list = []

        for t,trench_edges_y in enumerate(trench_edges_y_list):
            start_above = start_above_list[t]
            end_above = end_above_list[t]

            repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
            repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

            if (repaired_trench_edges_y.shape[0]//2 > len(orientations)) and drop_first_row:
                repaired_trench_edges_y = repaired_trench_edges_y[2:]
            if (repaired_trench_edges_y.shape[0]//2 > len(orientations)) and drop_last_row:
                repaired_trench_edges_y = repaired_trench_edges_y[:-2]
            grouped_edges = repaired_trench_edges_y.reshape(-1,2) # or,2
            y_ends = []
            for edges,orientation in enumerate(orientations):
                y_ends.append(grouped_edges[edges,orientation])
            y_ends = np.array(y_ends)
            y_ends_list.append(y_ends)
        return y_ends_list

### NEW STUFF HERE ###
    def get_y_consensus_med(self,y_percentile,y_foreground_percentile,smoothing_kernel_y,n_fovs = 50):
        rand_fovs = np.sort(np.random.choice(self.fov_list,size=(n_fovs,),replace=False))

        img_arr = []
        for fov_idx in rand_fovs:
            fovdf = self.metadf.loc[fov_idx]
            last_idx = fovdf.index.get_level_values(0).unique().tolist()[-1]
            init_timepoint_fovdf = fovdf.loc[0:0,:].iloc[0]
            file_idx = int(init_timepoint_fovdf["File Index"])
            img_idx = int(init_timepoint_fovdf["Image Index"])
            with h5py.File(self.headpath + "/hdf5/hdf5_" + str(file_idx) + ".hdf5", "r") as infile:
                img_arr.append(infile[self.seg_channel][img_idx][:,:,np.newaxis])

        img_arr = np.concatenate(img_arr,axis=2)
        y_percentiles = np.percentile(img_arr,y_percentile,axis=1,interpolation='lower')
        y_percentiles_smoothed = self.median_filter_2d(y_percentiles,smoothing_kernel_y)
        # Normalize (scale by range and subtract minimum) to make scaling of thresholds make more sense
        min_qth_percentile = y_percentiles_smoothed.min(axis=0)
        max_qth_percentile = np.percentile(y_percentiles_smoothed,y_foreground_percentile,axis=0)
        y_percentiles_smoothed = np.minimum(((y_percentiles_smoothed - min_qth_percentile)/(max_qth_percentile - min_qth_percentile)),1.)
        consensus_med = np.median(y_percentiles_smoothed,axis=1,keepdims=True)

        return consensus_med

    def get_consensus_orientations(self,y_percentile_threshold,consensus_med,alternate_orientation,expected_num_rows,top_orientation,y_min_edge_dist): ##currently disables alternate_over_rows
        trench_mask_y = consensus_med>y_percentile_threshold
        trench_edges_y_list,start_above_list,end_above_list = self.get_edges_from_mask(trench_mask_y)
        consensus_orientations,drop_first_row,drop_last_row = self.get_manual_orientations(0,[trench_edges_y_list],[start_above_list],[end_above_list],alternate_orientation,\
                                                                                                  expected_num_rows,top_orientation,False,y_min_edge_dist)
        y_ends_list = self.get_trench_ends(0,[trench_edges_y_list],[start_above_list],[end_above_list],[consensus_orientations],[drop_first_row],[drop_last_row],y_min_edge_dist)
        repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y_list[0],start_above_list[0],end_above_list[0])
        repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)
        consensus_midpoints = [np.mean(repaired_trench_edges_y[i*2:(i+1)*2]) for i in range(len(consensus_orientations))]

        return consensus_orientations,consensus_midpoints

    def get_manual_orientations_withcons(self,i,trench_edges_y_lists,start_above_lists,end_above_lists,\
                                         consensus_orientations,consensus_midpoints,y_min_edge_dist,midpoint_dist_tolerence):
        trench_edges_y = trench_edges_y_lists[i][0]
        start_above = start_above_lists[i][0]
        end_above = end_above_lists[i][0]

        repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
        repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

        midpoints = (repaired_trench_edges_y[1:]+repaired_trench_edges_y[:-1])/2

        matched_min_cons_idx, match_mask = match_midpoints(midpoints, consensus_midpoints, midpoint_dist_tolerence)
        repaired_trench_edges_y = [repaired_trench_edges_y[idx:idx+2] for idx in matched_min_cons_idx]
        repaired_trench_edges_y = np.array([part for item in repaired_trench_edges_y for part in item])
        orientations = np.array(consensus_orientations)[match_mask].tolist()
        filtered_consensus_midpoints = np.array(consensus_midpoints)[match_mask].tolist()

        orientations,drop_first_row,drop_last_row,_ = self.remove_out_of_frame(orientations,repaired_trench_edges_y,start_above,end_above)
        if drop_first_row:
            filtered_consensus_midpoints = filtered_consensus_midpoints[1:]
        if drop_last_row:
            filtered_consensus_midpoints = filtered_consensus_midpoints[:-1]

        return orientations,filtered_consensus_midpoints

    def get_trench_ends_withcons(self,i,trench_edges_y_lists,start_above_lists,end_above_lists,\
                                 orientations_list,filtered_consensus_midpoints_list,y_min_edge_dist,midpoint_dist_tolerence):
        trench_edges_y_list = trench_edges_y_lists[i]
        start_above_list = start_above_lists[i]
        end_above_list = end_above_lists[i]
        orientations = orientations_list[i]
        filtered_consensus_midpoints = filtered_consensus_midpoints_list[i]

        y_ends_list = []

        for t,trench_edges_y in enumerate(trench_edges_y_list):
            start_above = start_above_list[t]
            end_above = end_above_list[t]

            repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
            repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)

            midpoints = (repaired_trench_edges_y[1:]+repaired_trench_edges_y[:-1])/2

            matched_min_cons_idx, match_mask = match_midpoints(midpoints, filtered_consensus_midpoints, midpoint_dist_tolerence)
            repaired_trench_edges_y = [repaired_trench_edges_y[idx:idx+2] for idx in matched_min_cons_idx]
            repaired_trench_edges_y = np.array([part for item in repaired_trench_edges_y for part in item])
            matched_orientations = np.array(orientations)[match_mask].tolist()

            grouped_edges = repaired_trench_edges_y.reshape(-1,2) # or,2
            y_ends = []
            for edges,orientation in enumerate(matched_orientations):
                y_ends.append(grouped_edges[edges,orientation])
            y_ends = np.array(y_ends)
            y_ends_list.append(y_ends)

        return y_ends_list

    ### END

    def get_y_drift(self,i,y_ends_lists):
        """Given a list of midpoints, computes the average drift in y for every
        timepoint.

        Args:
            y_midpoints_list (list): A list containing, for each fov, a list of the form [time_list,[midpoint_array]]
            containing the trench row midpoints.

        Returns:
            list: A nested list of the form [time_list,[y_drift_int]] for fov i.
        """
        y_ends_list = y_ends_lists[i]
        y_drift = []
        for t in range(len(y_ends_list)-1):
            diff_mat = np.subtract.outer(y_ends_list[t+1],y_ends_list[t])
            if diff_mat.shape[0] > 0 and diff_mat.shape[1] > 0:
                min_dist_idx = np.argmin(abs(diff_mat),axis=0)
                min_dist_inv_idx = np.argmin(abs(diff_mat),axis=1)
                count_arr = np.add.accumulate(np.ones(len(min_dist_idx),dtype=int))-1
                match_mask = min_dist_inv_idx[min_dist_idx] == count_arr
                min_dists = diff_mat[min_dist_idx[match_mask],min_dist_inv_idx[min_dist_idx[match_mask]]]
                median_translation = np.median(min_dists)
            else:
                median_translation = 0
            y_drift.append(median_translation)
        net_y_drift = np.append(np.array([0]),np.add.accumulate(y_drift)).astype(int)
        return net_y_drift

    def keep_in_frame_kernels(self,i,y_ends_lists,y_drift_list,imported_array_list,orientations_list,padding_y,trench_len_y):
        """Removes those kernels which drift out of the image during any timepoint.
        Args:
            trench_edges_y_lists (list): A list containing, for each fov, a time-ordered list of trench edge arrays.
            y_drift_list (list): A list containing, for each fov, a nested list of the form [time_list,[y_drift_int]].
            imported_array_list (int): A numpy array containing the hdf5 file image data.
            padding_y (int): Y-dimensional padding for cropping.

        Returns:
            list: Time-ordered list of trench edge arrays, filtered for images which
            stay in frame for all timepoints, for fov i.
        """

        y_ends_list = y_ends_lists[i]
        init_y_ends = y_ends_list[0]
        y_drift = y_drift_list[i]
        max_y_dim = imported_array_list[i].shape[1]
        orientations = orientations_list[i]

        max_drift,min_drift = np.max(y_drift),np.min(y_drift)

        valid_init_y_ends = []
        valid_orientations = []
        for j,orientation in enumerate(orientations):
            y_end = init_y_ends[j]
            if orientation == 0:
                bottom_edge = y_end+trench_len_y+max_drift
                top_edge = y_end-padding_y+min_drift
                edge_under_max = bottom_edge<max_y_dim
                edge_over_min = top_edge >= 0
            else:
                bottom_edge = y_end+padding_y+max_drift
                top_edge = y_end-trench_len_y+min_drift
                edge_under_max = bottom_edge<max_y_dim
                edge_over_min = top_edge >= 0

            edge_in_bounds = edge_under_max*edge_over_min

            if edge_in_bounds:
                valid_init_y_ends.append(init_y_ends[j])
                valid_orientations.append(orientation)

        valid_init_y_ends = np.array(valid_init_y_ends)

        return valid_init_y_ends,valid_orientations

    def crop_y(self,i,imported_array_list,y_drift_list,valid_init_y_ends_list,valid_orientations_list,padding_y,trench_len_y):
        """Performs cropping of the images in the y-dimension.

        Args:
            i (int): Specifies the current fov index.
            trench_edges_y_list (list): List containing, for each fov entry, a list of time-sorted edge arrays.
            row_num_list (list): List containing The number of trench rows detected in each fov.
            imported_array_list (list): A list containing numpy arrays containing the hdf5 file image
            data of shape (channel,y,x,t).
            padding_y (int): Padding to be used when cropping in the y-dimension.
            trench_len_y (int): Length from the end of the tenches to be used when cropping in the
            y-dimension.
            top_orientation (int, optional): The orientation of the top-most row where 0 corresponds to a trench with
            a downward-oriented trench opening and 1 corresponds to a trench with an upward-oriented trench opening.
        Returns:
            array: A y-cropped array of shape (rows,channels,x,y,t).
        """
        imported_array = imported_array_list[i]
        y_drift = y_drift_list[i]
        valid_init_y_ends = valid_init_y_ends_list[i]
        valid_orientations = valid_orientations_list[i]

        drift_corrected_edges = np.add.outer(y_drift,valid_init_y_ends)
        time_list = []

        for t in range(imported_array.shape[3]):
            trench_ends_y = drift_corrected_edges[t]
            row_list = []

            for r,orientation in enumerate(valid_orientations):
                trench_end = trench_ends_y[r]
                if orientation == 0:
                    upper = max(trench_end-padding_y,0)
                    lower = min(trench_end+trench_len_y,imported_array.shape[1])
                else:
                    upper = max(trench_end-trench_len_y,0)
                    lower = min(trench_end+padding_y,imported_array.shape[1])

                channel_list = []
                for c in range(imported_array.shape[0]):

                    output_array = imported_array[c,upper:lower,:,t]
                    channel_list.append(output_array)
                row_list.append(channel_list)
            time_list.append(row_list)

        cropped_in_y = np.array(time_list)
        if len(cropped_in_y.shape) != 5:
            print("Error in crop_y")
            return None
        else:
            cropped_in_y = np.moveaxis(cropped_in_y,(0,1,2,3,4),(4,0,1,2,3))
            return cropped_in_y

    def get_smoothed_x_percentiles(self,i,cropped_in_y_list,x_percentile,background_kernel_x,smoothing_kernel_x):
        """Summary.

        Args:
            i (int): Specifies the current fov index.
            cropped_in_y_list (list): List containing, for each fov entry, a y-cropped numpy array of shape (rows,channels,x,y,t).
            x_percentile (int): Used for reducing signal in xyt to only the xt dimension when cropping
            in the x-dimension.
            background_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing background subtraction
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.
            smoothing_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing smoothing
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.

        Returns:
            array: A smoothed and background subtracted percentile array of shape (rows,x,t)
        """
        cropped_in_y = cropped_in_y_list[i]
        x_percentiles_smoothed_rows = []
        for row_num in range(cropped_in_y.shape[0]):
            cropped_in_y_arr = cropped_in_y[row_num,0]
            x_percentiles = np.percentile(cropped_in_y_arr,x_percentile,axis=0)
            x_background_filtered = x_percentiles - self.median_filter_2d(x_percentiles,background_kernel_x)
            x_smooth_filtered = self.median_filter_2d(x_background_filtered,smoothing_kernel_x)
            x_smooth_filtered[x_smooth_filtered<0.] = 0.
            x_percentiles_smoothed_rows.append(x_smooth_filtered)
        x_percentiles_smoothed_rows=np.array(x_percentiles_smoothed_rows)
        return x_percentiles_smoothed_rows

    def get_midpoints_from_mask(self,mask):
        """Using a boolean x mask, computes the positions of trench midpoints.

        Args:
            mask (array): x boolean array, specifying where trenches are present.

        Returns:
            array: array of trench midpoint x positions.
        """
        transitions = mask[:-1].astype(int) - mask[1:].astype(int)

        trans_up = np.where((transitions==-1))[0]
        trans_dn = np.where((transitions==1))[0]

        if len(np.where(trans_dn>trans_up[0])[0])>0:
            first_dn = np.where(trans_dn>trans_up[0])[0][0]
            trans_dn = trans_dn[first_dn:]
            trans_up = trans_up[:len(trans_dn)]
            midpoints = (trans_dn + trans_up)//2
        else:
            midpoints = []
        return midpoints

    def get_midpoints(self,x_percentiles_t,otsu_scaling,min_threshold):
        """Given an array of signal in x, determines the position of trench
        midpoints.

        Args:
            x_percentiles_t (array): array of trench intensities in x, at time t.
            otsu_nbins (int): Number of bins to use when applying Otsu's method to x-dimension signal.
            otsu_scaling (float): Threshold scaling factor for Otsu's method thresholding.

        Returns:
            array: array of trench midpoint x positions.
        """
        otsu_threshold = sk.filters.threshold_otsu(x_percentiles_t[:,np.newaxis],nbins=50)*otsu_scaling
        modified_otsu_threshold = max(otsu_threshold,min_threshold)

        x_mask = x_percentiles_t>modified_otsu_threshold
        midpoints = self.get_midpoints_from_mask(x_mask)
        return midpoints,modified_otsu_threshold

    def get_all_midpoints(self,i,x_percentiles_smoothed_list,otsu_scaling,min_threshold):
        """Given an x percentile array of shape (rows,x,t), determines the
        trench midpoints of each row array at each time t.

        Args:
            i (int): Specifies the current fov index.
            x_percentiles_smoothed (array): A smoothed and background subtracted percentile array of shape (rows,x,t)
            otsu_nbins (TYPE): Description
            otsu_scaling (TYPE): Description

        Returns:
            list: A nested list of the form [row_list,[time_list,[midpoint_array]]].
        """
        x_percentiles_smoothed_row = x_percentiles_smoothed_list[i]
        midpoints_row_list = []
        for j in range(x_percentiles_smoothed_row.shape[0]):
            x_percentiles_smoothed = x_percentiles_smoothed_row[j]
            all_midpoints = []
            midpoints,_ = self.get_midpoints(x_percentiles_smoothed[:,0],otsu_scaling,min_threshold)
            if len(midpoints) == 0:
                return None
            all_midpoints.append(midpoints)
            for t in range(1,x_percentiles_smoothed.shape[1]):
                midpoints,_ = self.get_midpoints(x_percentiles_smoothed[:,t],otsu_scaling,min_threshold)
                if len(midpoints)/(len(all_midpoints[-1])+1) < 0.5:
                    all_midpoints.append(all_midpoints[-1])
                else:
                    all_midpoints.append(midpoints)
            midpoints_row_list.append(all_midpoints)
        return midpoints_row_list

    def get_x_drift(self,i,all_midpoints_list):
        """Given an t by x array of midpoints, computed the average drift in x
        for every timepoint.

        Args:
            i (int): Specifies the current fov index.
            all_midpoints_list (list): A nested list of the form [selected_fov_list,[row_list,[time_list,[midpoint_array]]]] containing
            the trench midpoints.

        Returns:
            list: A nested list of the form [row_list,[time_list,[x_drift_int]]].
        """
        midpoints_row_list = all_midpoints_list[i]
        x_drift_row_list = []
        for all_midpoints in midpoints_row_list:
            x_drift = []
            for t in range(len(all_midpoints)-1):
                diff_mat = np.subtract.outer(all_midpoints[t+1],all_midpoints[t])
                min_dist_idx = np.argmin(abs(diff_mat),axis=0)
                min_dists = diff_mat[min_dist_idx]
                median_translation = int(np.median(min_dists))
                x_drift.append(median_translation)
            net_x_drift = np.append(np.array([0]),np.add.accumulate(x_drift))
            x_drift_row_list.append(net_x_drift)

        return x_drift_row_list

    def init_counting_arr(self,x_dim,t_dim):
        """Initializes a counting array of shape (x_dim,t_dim) which counts
        from 0 to x_dim on axis 0 for all positions in axis 1.

        Args:
            x_dim (int): Size of x axis to use.
            t_dim (int): Size of t axis to use.

        Returns:
            array: Counting array to be used for masking out trenches in x.
        """
        ones_arr = np.ones(x_dim)
        counting_arr = np.add.accumulate(np.ones(x_dim)).astype(int) - 1
        counting_arr_repeated = np.repeat(counting_arr[:,np.newaxis],t_dim,axis=1)
        return counting_arr_repeated

    def get_k_mask(self,in_bounds,counting_arr,k):
        """Generates a boolean trench mask of shape (x_dim,t_dim) for a given
        trench k, using the trench boundary values in in_bounds_list.

        Args:
            in_bounds (array): A shape (2,t_dim,k_dim) array specifying the start and end bounds in x of a
            given trench k over time.
            counting_arr (array): Counting array to be used for masking out trenches in x.
            k (int): Int specifying the trench to generate a mask for.

        Returns:
            array: Boolean trench mask of shape (x_dim,t_dim) for a given trench k.
        """
        working_t_dim = in_bounds.shape[1]
        cropped_counting_arr = counting_arr[:,:working_t_dim]
        k_mask = np.logical_and(cropped_counting_arr>in_bounds[0,:,k],cropped_counting_arr<in_bounds[1,:,k]).T
        return k_mask

    def apply_kymo_mask(self,img_arr,mask_arr,row_num,channel):
        """Given a y-cropped image and a boolean trench mask of shape
        (x_dim,t_dim), masks that image in xt to generate an output kymograph
        of shape (y_dim,x_dim,t_dim).

        Args:
            img_arr (array): A numpy array of a y-cropped image
            mask_arr (array): A boolean trench mask of shape (x_dim,t_dim) for a given trench k
            row_num (int): Int specifying the current row.
            channel (int): Int specifying which channel we are getting midpoints from (order specified by
            self.all_channels).

        Returns:
            array: Kymograph array of shape (y_dim,x_dim,t_dim).
        """
        working_img_arr = img_arr[row_num,channel]
        reshaped_arr = np.swapaxes(working_img_arr,1,2)
        masked_arr = reshaped_arr[:,mask_arr.astype(bool)]
        reshaped_masked_arr = masked_arr.reshape(reshaped_arr.shape[0],reshaped_arr.shape[1],-1)
        swapped_masked_arr = np.swapaxes(reshaped_masked_arr,1,2)
        return swapped_masked_arr

    def get_corrected_midpoints(self,i,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr):
        midpoints_row_list = all_midpoints_list[i]
        x_drift_row_list = x_drift_list[i]
        corrected_midpoints = []
        for row_num,all_midpoints in enumerate(midpoints_row_list):
            x_drift = x_drift_row_list[row_num]
            midpoint_seeds = self.filter_midpoints(all_midpoints,x_drift,trench_width_x,trench_present_thr)
            corrected_midpoints_row = x_drift[:,np.newaxis]+midpoint_seeds[np.newaxis,:]
            corrected_midpoints.append(corrected_midpoints_row)
        return corrected_midpoints

    def filter_midpoints(self,all_midpoints,x_drift,trench_width_x,trench_present_thr):

        drift_corrected_midpoints = []
        for t in range(len(x_drift)):
            drift_corrected_t = all_midpoints[t]-x_drift[t]
            drift_corrected_midpoints.append(drift_corrected_t)
        midpoints_up,midpoints_dn = (all_midpoints[0]-trench_width_x//2,\
                                     all_midpoints[0]+trench_width_x//2+1)

        trench_present_t = []
        for t in range(len(drift_corrected_midpoints)):
            above_mask = np.greater.outer(drift_corrected_midpoints[t],midpoints_up)
            below_mask = np.less.outer(drift_corrected_midpoints[t],midpoints_dn)
            in_bound_mask = (above_mask*below_mask)
            trench_present = np.any(in_bound_mask,axis=0)
            trench_present_t.append(trench_present)
        trench_present_t = np.array(trench_present_t)
        trench_present_perc = np.sum(trench_present_t,axis=0)/trench_present_t.shape[0]

        presence_filter_mask = trench_present_perc>=trench_present_thr

        midpoint_seeds = all_midpoints[0][presence_filter_mask]
        return midpoint_seeds

    def get_k_masks(self,cropped_in_y,all_midpoints,x_drift,trench_width_x,trench_present_thr):
        """Generates a boolean trench mask of shape (x_dim,t_dim) for each
        trench k. This will be used to mask out each trench at a later step.

        Args:
            cropped_in_y (array): A y-cropped numpy array of shape (rows,channels,x,y,t) containing y-cropped image data.
            all_midpoints (list): A list containing, for each time t, an array of trench midpoints.
            x_drift (list): A list containing, for each time t, an int corresponding to the drift of the midpoints in x.
            trench_width_x (int): Width to be used when cropping in the x-dimension.

        Returns:
            list:  A list containing, for each trench k, a boolean trench mask of shape (x_dim,t_dim).
        """
        midpoint_seeds = self.filter_midpoints(all_midpoints,x_drift,trench_width_x,trench_present_thr)
        corrected_midpoints = x_drift[:,np.newaxis]+midpoint_seeds[np.newaxis,:]
        midpoints_up,midpoints_dn = (corrected_midpoints-trench_width_x//2,\
                                     corrected_midpoints+trench_width_x//2+1)
        valid_mask = np.all(midpoints_up>=0,axis=0)*np.all(midpoints_dn<=cropped_in_y.shape[3],axis=0)
        in_bounds = np.array([midpoints_up[:,valid_mask],\
                            midpoints_dn[:,valid_mask]])
        counting_arr = self.init_counting_arr(cropped_in_y.shape[3],cropped_in_y.shape[4])

        k_masks = []
        for k in range(in_bounds.shape[2]):
            k_mask = self.get_k_mask(in_bounds,counting_arr,k)
            k_masks.append(k_mask)
        return k_masks

    def crop_with_k_masks(self,cropped_in_y,row_num,k_masks):
        """Performs cropping of the aleady y-cropped image data, using
        pregenerated kymograph masks of shape (x_dim,t_dim).

        Args:
            cropped_in_y (array): A y-cropped array of shape (rows,channels,x,y,t).
            row_num (int): The row number to crop kymographs from.
            k_masks (list): A list containing, for each trench k, a boolean trench mask of shape (x_dim,t_dim).

        Returns:
            list: A kymograph array of shape (channels,trenches,y_dim,x_dim,t_dim)
        """
        x_cropped = []
        for channel in range(len(self.all_channels)):
            kymographs = []
            for k in range(len(k_masks)):
                k_mask = k_masks[k]
                kymograph = self.apply_kymo_mask(cropped_in_y,k_mask,row_num,channel)
                kymographs.append(kymograph)
            x_cropped.append(np.array(kymographs))
        x_cropped = np.array(x_cropped)
        return x_cropped

    def get_crop_in_x(self,i,cropped_in_y_list,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr):
        """Generates complete kymograph arrays for all trenches in the fov in
        every channel listed in 'self.all_channels'. Outputs a list of these
        kymograph arrays, with entries corresponding to each row in the fov
        with index i.

        Args:
            i (int): Specifies the current fov index.
            cropped_in_y_list (list): List containing, for each fov entry, a y-cropped numpy array of shape (rows,channels,x,y,t).
            all_midpoints_list (list): A nested list of the form [selected_fov_list,[row_list,[time_list,[midpoint_array]]]] containing
            the trench midpoints.
            x_drift_list (list): A nested list of the form [selected_fov_list,[row_list,[time_list,[x_drift_int]]]] containing the computed
            drift in the x dimension.
            trench_width_x (int): Width to be used when cropping in the x-dimension.

        Returns:
            list: A list containing, for each row, a kymograph array of shape (channels,trenches,y_dim,x_dim,t_dim).
        """
        cropped_in_y = cropped_in_y_list[i]
        midpoints_row_list = all_midpoints_list[i]
        x_drift_row_list = x_drift_list[i]
        crop_in_x_row_list = []
        for row_num,all_midpoints in enumerate(midpoints_row_list):
            x_drift = x_drift_row_list[row_num]
            k_masks = self.get_k_masks(cropped_in_y,all_midpoints,x_drift,trench_width_x,trench_present_thr)
            x_cropped = self.crop_with_k_masks(cropped_in_y,row_num,k_masks)
            crop_in_x_row_list.append(x_cropped)
        return crop_in_x_row_list

    def crop_trenches_in_x(self,cropped_in_y_list):
        """Performs cropping of the images in the x-dimension.

        Args:
            cropped_in_y_list (list): List containing, for each fov entry, a y-cropped numpy array of shape (rows,channels,x,y,t).

        Returns:
            list: A nested list of the form [selected_fov_list,[row_list,[kymograph_array]]], containing kymograph arrays of
            shape (channels,trenches,y_dim,x_dim,t_dim).
        """
        smoothed_x_percentiles_list = self.map_to_fovs(self.get_smoothed_x_percentiles,cropped_in_y_list,self.x_percentile,\
                                                                 self.background_kernel_x,self.smoothing_kernel_x)
        all_midpoints_list = self.map_to_fovs(self.get_all_midpoints,smoothed_x_percentiles_list,self.otsu_scaling)
        x_drift_list = self.map_to_fovs(self.get_x_drift,all_midpoints_list)
        cropped_in_x_list = self.map_to_fovs(self.get_crop_in_x,cropped_in_y_list,all_midpoints_list,x_drift_list,self.trench_width_x,self.trench_present_thr)
        return cropped_in_x_list

    def generate_kymograph(self):
        """Master function for generating kymographs for the set of fovs
        specified on initialization.

        Returns:
            list: A nested list of the form [selected_fov_list,[row_list,[kymograph_array]]], containing kymograph arrays of
            shape (channels,trenches,y_dim,x_dim,t_dim).
        """
        array_list = self.map_to_fovs(self.import_hdf5)
        cropped_in_y_list = self.crop_trenches_in_y(array_list)
        cropped_in_x_list = self.crop_trenches_in_x(cropped_in_y_list)

        return cropped_in_x_list

class tiff_sequence_kymograph():
    """Class for getting kymographs from tiff stack (see classes in
    ndextract.py for details on how this works)"""
    def __init__(self, headpath, tiffpath, all_channels, filename_format_string, trenches_per_file=5, upside_down=False, time_interval=60, manual_metadata_params={}):
        self.headpath = headpath
        self.kymographpath = self.headpath + "/kymograph"
        self.hdf5path = self.headpath + "/hdf5"
        self.all_channels = all_channels
        self.tiffpath = tiffpath
        self.metapath = self.headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        self.trenches_per_file = trenches_per_file
        self.filename_format_string = filename_format_string
        self.upside_down = upside_down
        self.manual_metadata_params = manual_metadata_params
        self.time_interval = time_interval

    def assignidx(self, metadf):
        outdf = copy.deepcopy(metadf)
        numchannels = len(pd.unique(metadf["channel"]))
        num_total_files = (outdf.shape[0]//(self.trenches_per_file*numchannels)) + 1
        remainder = (outdf.shape[0]//numchannels)%(self.trenches_per_file)

        trench_file_idx = np.repeat(list(range(num_total_files)), self.trenches_per_file*numchannels)[:-(self.trenches_per_file-remainder)*numchannels]

        trench_file_trench_idx = np.repeat(np.repeat(np.array(list(range(self.trenches_per_file))), numchannels)[np.newaxis,:],num_total_files,axis=0)
        trench_file_trench_idx = trench_file_trench_idx.flatten()[:-(self.trenches_per_file-remainder)*numchannels]

        outdf["File Index"] = trench_file_idx
        outdf["File Trench Index"] = trench_file_trench_idx
        return outdf

    def writemetadata(self, parser, tiff_files):
        kymograph_metadata = {}
        exp_metadata = {}
        first_img = imread(tiff_files[0])
        exp_metadata["num_frames"] = first_img.shape[0]
        exp_metadata["height"] = first_img.shape[1]
        exp_metadata["width"] = first_img.shape[2]

        self.output_chunk_shape = (1,1,first_img.shape[1],first_img.shape[2])
        self.output_chunk_bytes = (2*np.multiply.accumulate(np.array(self.output_chunk_shape))[-1])
        self.chunk_cache_mem_size = 2*self.output_chunk_bytes

        kymograph_metadata = dict([(key, [value]) for key, value in parser.search(tiff_files[0]).named.items()])
        kymograph_metadata["Image Path"] = [tiff_files[0]]
        kymograph_metadata["Image Path"] = [tiff_files[0]]
        for f in tiff_files[1:]:
            fov_frame_dict = parser.search(f).named
            for key, value in fov_frame_dict.items():
                kymograph_metadata[key].append(value)
            kymograph_metadata["Image Path"].append(f)
        if "lane" not in kymograph_metadata:
            kymograph_metadata["lane"] = [1]*len(tiff_files)
        if "row" not in kymograph_metadata:
            kymograph_metadata["row"] = [0]*len(tiff_files)


        kymograph_metadata = pd.DataFrame(kymograph_metadata)

        old_labels_fov = [list(frozen_array) for frozen_array in kymograph_metadata.set_index(["lane", "fov"]).index.unique().labels]
        old_labels_trench = [list(frozen_array) for frozen_array in kymograph_metadata.set_index(["lane", "fov", "trench"]).index.unique().labels]
        old_labels_fov = list(zip(old_labels_fov[0], old_labels_fov[1]))
        old_labels_trench = list(zip(old_labels_trench[0], old_labels_trench[1], old_labels_trench[2]))

        fov_label_mapping = {}
        trench_label_mapping = {}

        for i in range(len(old_labels_fov)):
            fov_label_mapping[old_labels_fov[i]] = i
        for i in range(len(old_labels_trench)):
            trench_label_mapping[old_labels_trench[i]] = i

        old_labels_fov = np.array(kymograph_metadata.set_index(["lane", "fov"]).index.labels).T
        old_labels_trench = np.array(kymograph_metadata.set_index(["lane", "fov", "trench"]).index.labels).T

        new_labels_fov = np.empty(old_labels_fov.shape[0])
        new_labels_trench = np.empty(old_labels_fov.shape[0])

        for i in range(old_labels_fov.shape[0]):
            old_label = (old_labels_fov[i, 0], old_labels_fov[i, 1])
            new_labels_fov[i] = fov_label_mapping[old_label]

            old_label = (old_labels_trench[i, 0], old_labels_trench[i, 1], old_labels_trench[i, 2])
            new_labels_trench[i] = trench_label_mapping[old_label]

        kymograph_metadata = kymograph_metadata.reset_index()
        del kymograph_metadata["index"]
        kymograph_metadata["fov"] = new_labels_fov
        kymograph_metadata["trenchid"] = new_labels_trench

        exp_metadata["fields_of_view"] = sorted(list(pd.unique(kymograph_metadata["fov"])))
        exp_metadata["num_fovs"] = len(exp_metadata["fields_of_view"])
        exp_metadata["channels"] = list(pd.unique(kymograph_metadata["channel"]))

        self.meta_handle = pandas_hdf5_handler(self.metapath)

        assignment_metadata = self.assignidx(kymograph_metadata.set_index(["trenchid"]).sort_values("trenchid"))

        channel_tidx_paths_by_file_index = assignment_metadata.reset_index()[["File Index", "row", "channel", "File Trench Index", "Image Path"]].set_index(["File Index", "row"])
        indices = [list(frozenlist) for frozenlist in channel_tidx_paths_by_file_index.index.unique().labels]
        indices = list(zip(indices[0], indices[1]))
        channel_tidx_paths_by_file_index = [(file_index, row, list(channel_tidx_paths_by_file_index.loc[file_index, row]["channel"]), list(channel_tidx_paths_by_file_index.loc[file_index, row]["File Trench Index"]), list(channel_tidx_paths_by_file_index.loc[file_index, row]["Image Path"])) for file_index, row, in indices]

        assignment_metadata = assignment_metadata.drop_duplicates(subset=["File Index", "File Trench Index"])
        assignment_metadata = assignment_metadata[["fov", "row", "trench", "File Index", "File Trench Index"]]

        timepoints = np.repeat(np.array(list(range(exp_metadata["num_frames"])))[np.newaxis,:], assignment_metadata.shape[0], axis=0).flatten()

        assignment_metadata = assignment_metadata.reset_index()
        assignment_metadata = pd.DataFrame(np.repeat(assignment_metadata.values, exp_metadata["num_frames"], axis=0), columns=assignment_metadata.columns)
        assignment_metadata["timepoints"] = timepoints
        assignment_metadata["time (s)"] = assignment_metadata["timepoints"]*self.time_interval
        assignment_metadata = assignment_metadata.set_index(["trenchid", "timepoints"])

        for param, val in self.manual_metadata_params.items():
            exp_metadata[param] = val

        self.meta_handle.write_df("kymograph",assignment_metadata,metadata=exp_metadata)
        self.meta_handle.write_df("global", pd.DataFrame(), metadata=exp_metadata)
        return channel_tidx_paths_by_file_index

    def extract(self, dask_controller):
        writedir(self.kymographpath ,overwrite=True)
        parser = compile(self.filename_format_string)
        tiff_files = []
        for root, _, files in os.walk(self.tiffpath):
            tiff_files.extend([os.path.join(root, f) for f in files if ".tif" in os.path.splitext(f)[1]])

        channel_tidx_paths_by_file_index = self.writemetadata(parser, tiff_files)
        metadf = self.meta_handle.read_df("kymograph",read_metadata=True)
        self.metadata = metadf.metadata

        dask_controller.futures = {}

        def writehdf5(fidx_channels_paths):
            y_dim = self.metadata['height']
            x_dim = self.metadata['width']
            time = self.metadata['num_frames']
            num_channels = len(self.all_channels)

            file_idx, row, channels, trench_indices, filepaths = fidx_channels_paths
            datasets = {}
            with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5","w",rdcc_nbytes=self.chunk_cache_mem_size) as h5pyfile:
            # with h5py_cache.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5","w",chunk_cache_mem_size=self.chunk_cache_mem_size) as h5pyfile:

                for i,channel in enumerate(self.all_channels):
                    hdf5_dataset = h5pyfile.create_dataset(str(channel),\
                    (len(filepaths)/num_channels,time,y_dim,x_dim), chunks=self.output_chunk_shape, dtype='uint16')
                    datasets[str(row) + "/" + channel] = hdf5_dataset
                for i in range(len(filepaths)):
                    curr_channel = channels[i]
                    curr_file = filepaths[i]
                    curr_trench = trench_indices[i]
                    data = imread(curr_file)
                    if self.upside_down:
                        data = np.flip(data, axis=1)
                    datasets[str(row) + "/" + curr_channel][curr_trench,:,:,:] = data
            return "Done."
        dask_controller.futures["extract file"] = dask_controller.daskclient.map(writehdf5, channel_tidx_paths_by_file_index)
