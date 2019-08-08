import numpy as np
import pandas as pd
import h5py
import scipy.signal
import shutil
import skimage as sk
import os
import pickle
import sys
import h5py_cache

from skimage import filters
from .cluster import hdf5lock
from .utils import multifov,pandas_hdf5_handler

class kymograph_cluster:
    def __init__(self,headpath="",trenches_per_file=20,paramfile=False,all_channels=[""],trench_len_y=270,padding_y=20,trench_width_x=30,\
                 t_range=(0,None),y_percentile=85,y_min_edge_dist=50,smoothing_kernel_y=(1,9),triangle_nbins=50,triangle_scaling=1.,\
                 top_orientation=0,expected_num_rows=None,orientation_on_fail=None,x_percentile=85,background_kernel_x=(1,21),smoothing_kernel_x=(1,9),\
                 otsu_nbins=50,otsu_scaling=1.,trench_present_thr=0.):
        
        if paramfile:
            parampath = headpath + "/kymograph.par"
            with open(parampath, 'rb') as infile:
                param_dict = pickle.load(infile)
                
            all_channels = param_dict["All Channels"]
            trench_len_y = param_dict["Trench Length"]
            padding_y = param_dict["Y Padding"]
            trench_width_x = param_dict["Trench Width"]
            t_range = param_dict["Time Range"]
            y_percentile = param_dict["Y Percentile"]
            y_min_edge_dist = param_dict["Minimum Trench Length"]
            smoothing_kernel_y = (1,param_dict["Y Smoothing Kernel"])
            triangle_nbins = param_dict["Triangle Threshold Bins"]
            triangle_scaling = param_dict["Triangle Threshold Scaling"]
            top_orientation = param_dict["Orientation Detection Method"]
            expected_num_rows = param_dict["Expected Number of Rows (Manual Orientation Detection)"]
            orientation_on_fail = param_dict["Top Orientation when Row Drifts Out (Manual Orientation Detection)"]
            x_percentile = param_dict["X Percentile"]
            background_kernel_x = (1,param_dict["X Background Kernel"])
            smoothing_kernel_x = (1,param_dict["X Smoothing Kernel"])
            otsu_nbins = param_dict["Otsu Threshold Bins"]
            otsu_scaling = param_dict["Otsu Threshold Scaling"]
            trench_present_thr =  param_dict["Trench Presence Threshold"]
                
        self.headpath = headpath
        self.kymographpath = self.headpath + "/kymograph"
        self.hdf5path = self.headpath + "/hdf5"
        self.all_channels = all_channels
        self.seg_channel = self.all_channels[0]
        self.metapath = self.headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        self.trenches_per_file = trenches_per_file

        self.t_range = t_range

        #### important paramaters to set
        self.trench_len_y = trench_len_y
        self.padding_y = padding_y
        ttl_len_y = trench_len_y+padding_y
        self.ttl_len_y = ttl_len_y
        self.trench_width_x = trench_width_x
                
        #### params for y
        ## parameter for reducing signal to one dim
        self.y_percentile = y_percentile
        self.y_min_edge_dist = y_min_edge_dist
        ## parameters for threshold finding
        self.smoothing_kernel_y = smoothing_kernel_y
        self.triangle_nbins = triangle_nbins
        self.triangle_scaling = triangle_scaling
        ### 
        self.top_orientation = top_orientation
        self.expected_num_rows = expected_num_rows
        self.orientation_on_fail = orientation_on_fail
        #### params for x
        ## parameter for reducing signal to one dim
        self.x_percentile = x_percentile
        ## parameters for midpoint finding
        self.background_kernel_x = background_kernel_x
        self.smoothing_kernel_x = smoothing_kernel_x
        ## parameters for threshold finding
        self.otsu_nbins = otsu_nbins
        self.otsu_scaling = otsu_scaling
        ## New
        self.trench_present_thr = trench_present_thr
        
        self.output_chunk_shape = (1,1,self.ttl_len_y,(self.trench_width_x//2)*2)
        self.output_chunk_bytes = (2*np.multiply.accumulate(np.array(self.output_chunk_shape))[-1])
        self.output_chunk_cache_mem_size = 2*self.output_chunk_bytes
        
        self.kymograph_params = {"trench_len_y":trench_len_y,"padding_y":padding_y,"ttl_len_y":ttl_len_y,\
                                 "trench_width_x":trench_width_x,"y_percentile":y_percentile,\
                             "y_min_edge_dist":y_min_edge_dist,"smoothing_kernel_y":smoothing_kernel_y,\
                                 "triangle_nbins":triangle_nbins,"triangle_scaling":triangle_scaling,\
                                 "top_orientation":top_orientation,"expected_num_rows":expected_num_rows,\
                                 "orientation_on_fail":orientation_on_fail,"x_percentile":x_percentile,\
                                 "background_kernel_x":background_kernel_x,"smoothing_kernel_x":smoothing_kernel_x,\
                                "otsu_nbins":otsu_nbins,"otsu_scaling":otsu_scaling,"trench_present_thr":trench_present_thr}
        
    def writedir(self,directory,overwrite=False):
        """Creates an empty directory at the specified location. If a directory is
        already at this location, it will be overwritten if 'overwrite' is true,
        otherwise it will be left alone.
        
        Args:
            directory (str): Path to directory to be overwritten/created.
            overwrite (bool, optional): Whether to overwrite a directory that
            already exists in this location.
        """
        if overwrite:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
    def median_filter_2d(self,array,smoothing_kernel):
        """Two-dimensional median filter, with average smoothing at the signal edges in
        the second dimension (the non-time dimension).
        
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
        start_edge = np.mean(med_filter[:,kernel_pad[1]:kernel[1]])
        end_edge = np.mean(med_filter[:,-kernel[1]:-kernel_pad[1]])
        med_filter[:,:kernel_pad[1]] = start_edge
        med_filter[:,-kernel_pad[1]:] = end_edge
        return med_filter

    def get_smoothed_y_percentiles(self,file_idx,y_percentile,smoothing_kernel_y):
        """For each imported array, computes the percentile along the x-axis of the segmentation
        channel, generating a (y,t) array. Then performs median filtering of this array for smoothing.
        
        Args:
            imported_hdf5_handle (h5py.File): Hdf5 file handle corresponding to the input hdf5 dataset
            "data" of shape (channel,y,x,t).
            y_percentile (int): Percentile to apply along the x-axis.
            smoothing_kernel_y (tuple): Kernel to use for median filtering.
        
        Returns:
            h5py.File: Hdf5 file handle corresponding to the output hdf5 dataset "data", a smoothed
            percentile array of shape (y,t).
        """
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            img_arr = imported_hdf5_handle[self.seg_channel][:] #t x y
            perc_arr = np.percentile(img_arr,y_percentile,axis=2,interpolation='lower')
            y_percentiles_smoothed = self.median_filter_2d(perc_arr,smoothing_kernel_y)
        return y_percentiles_smoothed
    
    def triangle_threshold(self,img_arr,triangle_nbins,triangle_scaling):
        """Applies a triangle threshold to each timepoint in a (t,y) input array, returning a boolean mask.
        
        Args:
            img_arr (array): ndarray to be thresholded.
            triangle_nbins (int): Number of bins to be used to construct the thresholding
            histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
        
        Returns:
            array: Boolean mask produced by the threshold.
        """
        all_thresholds = np.apply_along_axis(sk.filters.threshold_triangle,1,img_arr,nbins=triangle_nbins)*triangle_scaling
        triangle_mask = img_arr>all_thresholds[:,np.newaxis]
        return triangle_mask
    
    def remove_out_of_frame(self,edges,start_above,end_above):
        """Takes an array of trench row edges and removes the first/last
        edge, if that edge does not have a proper partner (i.e. trench row mask
        takes value True at boundaries of image).
        
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
        if start_above:
            edges = edges[1:]
        if end_above:
            edges = edges[:-1]
        return edges

    def remove_small_rows(self,edges,y_min_edge_dist):
        """Filters out small rows when performing automated row detection.
        
        Args:
            edges (array): Array of edges along y-axis.
            y_min_edge_dist (int): Minimum row length necessary for detection.
        
        Returns:
            array: Array of edges, filtered for rows that are too small.
        """
        grouped_edges = edges.reshape(-1,2)
        row_lens = np.diff(grouped_edges,axis=1)
        row_mask = (row_lens>y_min_edge_dist).flatten()
        filtered_edges = grouped_edges[row_mask]
        return filtered_edges.flatten()
    
    def get_edges_from_mask(self,mask,y_min_edge_dist):
        """Finds edges from a boolean mask of shape (t,y). Filters out rows of length
        smaller than y_min_edge_dist.
        
        Args:
            mask (array): Boolean of shape (y,t) resulting from triangle thresholding.
            y_min_edge_dist (int): Minimum row length necessary for detection.
        
        Returns:
            list: List containing arrays of edges for each timepoint, filtered for rows that are too small.
        """
        edges_list = []
        for t in range(mask.shape[0]):
            edge_mask = (mask[t,1:] != mask[t,:-1])
            start_above,end_above = (mask[t,0]==True,mask[t,-1]==True)
            edges = np.where(edge_mask)[0]
            edges = self.remove_out_of_frame(edges,start_above,end_above)
            edges = self.remove_small_rows(edges,y_min_edge_dist)
            edges_list.append(edges)
        return edges_list

    def get_y_midpoints(self,trench_edges_y_list):
        """Outputs trench row midpoints for each time point.
        
        Args:
            trench_edges_y_list (list): Time-ordered list of trench edge arrays.
        
        Returns:
            list: Time-ordered list of trench midpoint arrays.
        """
        midpoints = []
        for t in range(len(trench_edges_y_list)):
            midpoints_t = []
            for r in range(0,trench_edges_y_list[t].shape[0],2):
                midpoints_t.append(int(np.mean(trench_edges_y_list[t][r:r+2])))
            midpoints.append(np.array(midpoints_t))
        return midpoints
    
    def get_y_edges_and_midpoints(self,y_percentiles_smoothed_array,triangle_nbins,triangle_scaling,y_min_edge_dist):
        """Detects edges in the shape (t,y) smoothed percentile arrays for each input array.
        
        Args:
            y_percentiles_smoothed_array (array): A shape (y,t) smoothed percentile array.
            triangle_nbins (int): Number of bins to be used to construct the thresholding histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
            y_min_edge_dist (int): Minimum row length necessary for detection.
        
        Returns:
            list: List containing arrays of edges for each timepoint, filtered for rows that are too small.
        """
        
        trench_mask_y = self.triangle_threshold(y_percentiles_smoothed_array,triangle_nbins,triangle_scaling)
        trench_edges_y_list = self.get_edges_from_mask(trench_mask_y,y_min_edge_dist)
        midpoints = self.get_y_midpoints(trench_edges_y_list)

        return trench_edges_y_list,midpoints
    
    def get_y_drift(self,y_midpoints):
        """Given a list of midpoints, computes the average drift in y for every timepoint.

        Args:
            y_midpoints (list): A  list of the form [time_list,[midpoint_array]] containing
            the trench row midpoints.

        Returns:
            list: A nested list of the form [time_list,[y_drift_int]].
        """         
        y_drift = []
        for t in range(len(y_midpoints)-1):
            diff_mat = np.subtract.outer(y_midpoints[t+1],y_midpoints[t])
            if len(diff_mat) > 0:
                min_dist_idx = np.argmin(abs(diff_mat),axis=0)
                min_dists = []
                for row in range(diff_mat.shape[0]):
                    min_dists.append(diff_mat[row,min_dist_idx[row]])
                min_dists = np.array(min_dists)
                median_translation = np.median(min_dists)
            else:
                median_translation = 0
            y_drift.append(median_translation)
        net_y_drift = np.append(np.array([0]),np.add.accumulate(y_drift)).astype(int)
        return net_y_drift
    
    def keep_in_frame_kernels(self,trench_edges_y_list,y_drift,padding_y):
        """Removes those kernels which drift out of the image during any timepoint.
        
        Args:
            trench_edges_y_list (list): Time-ordered list of trench edge arrays.
            y_drift (list): A nested list of the form [time_list,[y_drift_int]].
            max_y_dim (int): Size of the y-dimension.
            padding_y (int): Y-dimensional padding for cropping.
        
        Returns:
            list: Time-ordered list of trench edge arrays, filtered for images which
            stay in frame for all timepoints.
        """
        max_y_dim = self.metadata['height']
        
        max_drift,min_drift = np.max(y_drift),np.min(y_drift)
        edge_under_max = np.all((trench_edges_y_list+max_drift+padding_y)<max_y_dim,axis=0) 
        edge_over_min = np.all((trench_edges_y_list+min_drift-padding_y)>=0,axis=0)
        edge_in_bounds = edge_under_max*edge_over_min

        valid_edge_mask = []
        valid_orientation_mask = []
        for i in range(0,len(edge_in_bounds),2):
            if np.all(edge_in_bounds[i:i+2]):
                valid_edge_mask+=[True,True]
                valid_orientation_mask+=[True]
            else:
                valid_edge_mask+=[False,False]
                valid_orientation_mask+=[False]

        valid_edges_y_list = [trench_edges_y[valid_edge_mask] for trench_edges_y in trench_edges_y_list]
        return valid_edges_y_list,valid_orientation_mask
    
    def get_orientations_and_init_edges(self,trench_edges_y_list,y_drift,expected_num_rows,top_orientation,orientation_on_fail,padding_y):
        
        if top_orientation not in [0,1]:
            raise ValueError
            
        orientations = []
        if trench_edges_y_list[0].shape[0]//2 == expected_num_rows:
            orientation = top_orientation
            for row in range(trench_edges_y_list[0].shape[0]//2):
                orientations.append(orientation)
                orientation = (orientation+1)%2
        elif (trench_edges_y_list[0].shape[0]//2 < expected_num_rows) and orientation_on_fail is not None:
            orientation = orientation_on_fail
            for row in range(trench_edges_y_list[0].shape[0]//2):
                orientations.append(orientation)
                orientation = (orientation+1)%2
        else:
            print("Start frame does not have expected number of rows!")
            
        valid_edges_y_list,valid_orientation_mask = self.keep_in_frame_kernels(trench_edges_y_list,y_drift,padding_y)
        orientations = np.array(orientations)[valid_orientation_mask].tolist()
        init_trench_edges = valid_edges_y_list[0]
        
        return orientations,init_trench_edges
    
    def get_y_drift_and_orientations(self,fov_idx,edges_and_midpoint_futures,expected_num_rows,top_orientation,orientation_on_fail,padding_y):
        """Given a list of midpoints, computes the average drift in y for every timepoint.

        Args:
            y_midpoints (list): A  list of the form [time_list,[midpoint_array]] containing
            the trench row midpoints.

        Returns:
            list: A nested list of the form [time_list,[y_drift_int]].
        """
        
        fovdf = self.meta_handle.read_df("global",read_metadata=False)
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        working_fovdf = fovdf.loc[fov_idx]
                
        trench_edges_y_list = []
        y_midpoints = []
        
        for j,file_idx in enumerate(working_fovdf["File Index"].unique().tolist()):
            working_filedf = working_fovdf[working_fovdf["File Index"]==file_idx]
            img_indices = working_filedf["Image Index"].unique()
            first_idx,last_idx = (img_indices[0],img_indices[-1])
            trench_edges_y_list += edges_and_midpoint_futures[j][0][first_idx:last_idx+1]
            y_midpoints += edges_and_midpoint_futures[j][1][first_idx:last_idx+1]
            
        y_drift = self.get_y_drift(y_midpoints)
        
        orientations,init_trench_edges = self.get_orientations_and_init_edges(trench_edges_y_list,y_drift,expected_num_rows,top_orientation,orientation_on_fail,padding_y)
        
        return y_drift,orientations,init_trench_edges
        
    def crop_y(self,file_idx,drift_orientation_and_initedge_future,padding_y,trench_len_y):
        """Performs cropping of the images in the y-dimension.
        
        Args:
            array_tuple (tuple): Tuple containing the imported hdf5 array (y,x,t), 
            and the time-ordered list of edge arrays.
            padding_y (int): Padding to be used when cropping in the y-dimension.
            trench_len_y (int): Length from the end of the tenches to be used when cropping in the 
            y-dimension.
            
        Returns:
            array: A y-cropped array of shape (rows,y,x,t).
        """        
        fovdf = self.meta_handle.read_df("global",read_metadata=False)
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        filedf = fovdf.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()
        working_filedf = filedf.loc[file_idx]
        
        timepoint_indices = working_filedf["timepoints"].unique().tolist()
        image_indices = working_filedf.index.get_level_values("Image Index").unique().tolist()
        first_idx,last_idx = (timepoint_indices[0],timepoint_indices[-1])
        y_drift = drift_orientation_and_initedge_future[0][first_idx:last_idx+1]
        orientations,init_trench_edges = drift_orientation_and_initedge_future[1:]

        drift_corrected_edges = np.add.outer(y_drift,init_trench_edges)

        channel_arr_list = []
        for c,channel in enumerate(self.all_channels):

            with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
                img_arr = imported_hdf5_handle[channel][image_indices[0]:image_indices[-1]+1]

            time_list = []
            lane_y_coords_list = []
            for t in range(img_arr.shape[0]):
                trench_edges_y = drift_corrected_edges[t]          
                row_list = []
                lane_y_coords = []
                for r,orientation in enumerate(orientations):
                    if orientation == 0:
                        trench_edge_y = trench_edges_y[2*r]
                        upper = max(trench_edge_y-padding_y,0)
                        lower = min(trench_edge_y+trench_len_y,img_arr.shape[1])
                        pad = upper+trench_len_y+padding_y-lower
                        output_array = np.pad(img_arr[t,upper:lower,:],((pad, 0),(0,0)),'constant')
                        lane_y_coords.append(upper)
                    else:
                        trench_edge_y = trench_edges_y[(2*r)+1]
                        upper = max(trench_edge_y-trench_len_y,0)
                        lower = min(trench_edge_y+padding_y,img_arr.shape[1])
                        pad = upper+trench_len_y+padding_y-lower
                        output_array = np.pad(img_arr[t,upper:lower,:],((0, pad),(0,0)),'constant')
                        lane_y_coords.append(upper)
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
    
    def get_smoothed_x_percentiles(self,y_crop_future,x_percentile,background_kernel_x,smoothing_kernel_x):
                
        """Summary
        
        Args:
            array_tuple (tuple): A singleton tuple containing the y-cropped hdf5 array of shape (rows,x,y,t).
            background_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing background subtraction
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.
            smoothing_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing smoothing
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.
        
        Returns:
            array: A smoothed and background subtracted percentile array of shape (rows,x,t)
        """
        
        cropped_in_y = y_crop_future[0][0] # t x row x y x x     # (24, 1, 330, 2048)   
        
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
    
    def get_x_row_midpoints(self,x_percentiles_t,otsu_nbins,otsu_scaling):
        """Given an array of signal in x, determines the position of trench midpoints.
        
        Args:
            x_percentiles_t (array): array of trench intensities in x, at time t.
            otsu_nbins (int): Number of bins to use when applying Otsu's method to x-dimension signal.
            otsu_scaling (float): Threshold scaling factor for Otsu's method thresholding.
        
        Returns:
            array: array of trench midpoint x positions.
        """

        otsu_threshold = sk.filters.threshold_otsu(x_percentiles_t[:,np.newaxis],nbins=otsu_nbins)*otsu_scaling
        x_mask = x_percentiles_t>otsu_threshold
        midpoints = self.get_midpoints_from_mask(x_mask)
        return midpoints
    
    def get_x_midpoints(self,x_percentiles_smoothed,otsu_nbins,otsu_scaling):
        """Given an x percentile array of shape (rows,t,x), determines the trench midpoints of each row array
        at each time t.
        
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
            midpoints = self.get_x_row_midpoints(row_x_percentiles[0],otsu_nbins,otsu_scaling)
            if len(midpoints) == 0:
                return None
            all_midpoints.append(midpoints)
            
            for t in range(1,row_x_percentiles.shape[0]):
                midpoints = self.get_x_row_midpoints(row_x_percentiles[t],otsu_nbins,otsu_scaling)
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
        """Given a list of midpoints, computes the average drift in x for every timepoint.
        
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
    
    def init_counting_arr(self,x_dim):
        """Initializes a counting array of shape (x_dim,) which counts from 0 to
        x_dim on axis 0.
        
        Args:
            x_dim (int): Size of x axis to use.
        
        Returns:
            array: Counting array to be used for masking out trenches in x.
        """
        ones_arr = np.ones(x_dim)
        counting_arr = np.add.accumulate(np.ones(x_dim)).astype(int) - 1
        return counting_arr
    
    def get_trench_mask(self,in_bounds,counting_arr):
        """Produce a trench mask of shape (y_dim,t_dim,x_dim) which will correspond
        to the reshaped "cropped_in_y" array that will be made later.
        
        Args:
            array_tuple (tuple): Singleton tuple containing the trench boundary array of shape
            (2,t_dim,num_trenches)
            cropped_in_y (array): A y-cropped hdf5 array of shape (rows,y,x,t) containing y-cropped image data.
            counting_arr (array): Counting array to be used for masking out trenches in x, of shape (x_dim,).
        
        Returns:
            array: A trench mask of shape (y_dim,t_dim,x_dim).
        """
        counting_arr_repeated = np.repeat(counting_arr[:,np.newaxis],in_bounds.shape[1],axis=1)
        masks = []
        for k in range(in_bounds.shape[2]):
            mask = np.logical_and(counting_arr_repeated>in_bounds[0,:,k],counting_arr_repeated<in_bounds[1,:,k]).T
            masks.append(mask)
        all_mask = np.any(np.array(masks),axis=0)
        k_mask = np.repeat(all_mask[np.newaxis,:,:],self.ttl_len_y,axis=0)
        return k_mask
    
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

    def get_kymo_masks(self,all_midpoints,x_drift,trench_width_x,trench_present_thr):
        """Produces and writes a trench mask of shape (y_dim,t_dim,x_dim). This will be used to mask out
        trenches from the reshaped "cropped_in_y" array at a later step.
        
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
        corrected_midpoints = x_drift[:,np.newaxis]+midpoint_seeds[np.newaxis,:]
      
        midpoints_up,midpoints_dn = (corrected_midpoints-trench_width_x//2,\
                                     corrected_midpoints+trench_width_x//2+1)
        stays_in_frame = np.all(midpoints_up>=0,axis=0)*np.all(midpoints_dn<=self.metadata["width"],axis=0) #filters out midpoints that stay in the frame for the whole time...
        no_overlap = np.append(np.array([True]),(corrected_midpoints[0,1:]-corrected_midpoints[0,:-1])>=(trench_width_x+1)) #corrects for overlap 
        if np.sum(no_overlap)/len(no_overlap)<0.9:
            print("Trench overlap issue!!!")
        
        valid_mask = stays_in_frame*no_overlap
        in_bounds = np.array([midpoints_up[:,valid_mask],\
                            midpoints_dn[:,valid_mask]])
        k_tot = in_bounds.shape[2]
        counting_arr = self.init_counting_arr(self.metadata["width"])
                
        kymo_mask = self.get_trench_mask(in_bounds,counting_arr)
        x_coords = in_bounds[0].T
        return kymo_mask,x_coords,k_tot
    
    def get_all_kymo_masks(self,midpoint_futures,x_drift_future,trench_width_x,trench_present_thr):
        """Generates complete kymograph arrays for all trenches in the fov in every channel listed in 'self.all_channels'.
        Writes hdf5 files containing datasets of shape (trench_num,y_dim,x_dim,t_dim) for each row,channel combination. 
        Dataset keys follow the convention ["[row_number]/[channel_name]"].
        
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
        kymo_mask_list = []
        x_coords_list = []
        k_tot_list = []
        
        for row_num,all_midpoints in enumerate(all_midpoints_list):
            x_drift = x_drift_future[row_num]
            kymo_mask,x_coords,k_tot = self.get_kymo_masks(all_midpoints,x_drift,trench_width_x,trench_present_thr)
            kymo_mask_list.append(kymo_mask)
            x_coords_list.append(x_coords)
            k_tot_list.append(k_tot)
        
        return kymo_mask_list,x_coords_list,k_tot_list
    
    def apply_kymo_mask(self,k_mask,img_arr,k_tot):
        """Given a y-cropped image and a boolean trench mask of shape (y_dim,t_dim,x_dim), masks that image to 
        generate an output kymograph of shape (trench_num,y_dim,x_dim,t_dim). Masked trenches must be a fized size,
        so this only detects trenches that are totally in frame for the whole timelapse. 

        Args:
            array_tuple (tuple): Tuple containing the y-cropped hdf5 array of shape (t,y,x), and
            the boolean trench mask of shape (y_dim,t_dim,x_dim).
            row_num (int): Int specifying the current row.
            k_tot (int): Int specifying the total number of detected trenches in the fov.

        Returns:
            array: Kymograph array of shape (trench_num,y_dim,x_dim,t_dim).
        """
        
        img_arr_swap = np.moveaxis(img_arr,(0,1,2),(1,0,2))
        cropped_img_arr = img_arr_swap[k_mask]
        cropped_img_arr = cropped_img_arr.reshape(img_arr_swap.shape[0],img_arr_swap.shape[1],-1)
        cropped_img_arr = np.moveaxis(cropped_img_arr,(0,1,2),(1,0,2)) # t x y x x 
        kymo_out = np.stack(np.split(cropped_img_arr,k_tot,axis=2),axis=0) # k x t x y x x 
        return kymo_out

    def crop_with_k_masks(self,output_kymograph,cropped_in_y_list,kymo_mask,k_tot,row_num):
        """Generates and writes kymographs of a single row from the already y-cropped image data, using a pregenerated kymograph mask
        of shape (y_dim,t_dim,x_dim).
        
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
            cropped_in_y = cropped_in_y_list[c][:,row_num]
            kymo_out = self.apply_kymo_mask(kymo_mask,cropped_in_y,k_tot) # k x t x y x x 
            
            hdf5_dataset = output_kymograph.create_dataset(dataset_name,data=kymo_out,chunks=self.output_chunk_shape, dtype='uint16')
            
    def crop_x(self,file_idx,y_crop_future,kymo_mask_future):
        """Generates complete kymograph arrays for all trenches in the fov in every channel listed in 'self.all_channels'.
        Writes hdf5 files containing datasets of shape (trench_num,y_dim,x_dim,t_dim) for each row,channel combination. 
        Dataset keys follow the convention ["[row_number]/[channel_name]"].
        
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
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        filedf = fovdf.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()
        working_filedf = filedf.loc[file_idx]
        
        timepoint_indices = working_filedf["timepoints"].unique().tolist()
        image_indices = working_filedf.index.get_level_values("Image Index").unique().tolist()
        first_idx,last_idx = (timepoint_indices[0],timepoint_indices[-1])
        cropped_in_y_list = y_crop_future[0]
        num_rows = cropped_in_y_list[0].shape[1]
        
        kymo_mask_list,_,k_tot_list = kymo_mask_future
        
        with h5py_cache.File(self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5","w",chunk_cache_mem_size=self.output_chunk_cache_mem_size) as output_kymograph:        
            for row_num in range(num_rows):
                kymo_mask,k_tot = (kymo_mask_list[row_num],k_tot_list[row_num])
                kymo_mask = kymo_mask[:,first_idx:last_idx+1]
                self.crop_with_k_masks(output_kymograph,cropped_in_y_list,kymo_mask,k_tot,row_num)
                
    def save_coords(self,fov_idx,y_crop_futures,kymo_mask_future,drift_orientation_and_initedge_future):
        fovdf = self.meta_handle.read_df("global",read_metadata=False)
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        fovdf = fovdf.loc[fov_idx]

        x_coords_list = kymo_mask_future[1]
        orientations = drift_orientation_and_initedge_future[1]

        y_coords_list = []        
        for j,file_idx in enumerate(fovdf["File Index"].unique().tolist()):
            working_filedf = fovdf[fovdf["File Index"]==file_idx]
            img_indices = working_filedf["Image Index"].unique()
            first_idx,last_idx = (img_indices[0],img_indices[-1])
            y_coords_list += y_crop_futures[j][1][first_idx:last_idx+1] # t x row list

        pixel_microns = self.metadata['pixel_microns']
        y_coords = np.array(y_coords_list) # t x row array
        scaled_y_coords = y_coords*pixel_microns
        t_len = scaled_y_coords.shape[0] 
        fs = np.repeat([fov_idx],t_len)
        global_x,global_y,ts,file_indices,img_indices = (fovdf["x"].values,fovdf["y"].values,fovdf["t"].values,fovdf["File Index"].values,fovdf["Image Index"].values)
        tpts = np.array(range(ts.shape[0]))
        orit_dict = {0:"top",1:"bottom"}

        pd_output = []

        for l,x_coord in enumerate(x_coords_list):
            scaled_x_coord = x_coord*pixel_microns
            yt = scaled_y_coords[:,l]
            orit = np.repeat([orit_dict[orientations[l]]],t_len)
            global_yt = yt+global_y
            ls = np.repeat([l],t_len)
            for k in range(scaled_x_coord.shape[0]):
                xt = scaled_x_coord[k]
                global_xt = xt+global_x
                ks = np.repeat([k],t_len)
                pd_output.append(np.array([fs,ls,ks,tpts,file_indices,img_indices,ts,orit,yt,xt,global_yt,global_xt]).T)
        pd_output = np.concatenate(pd_output,axis=0)
        df = pd.DataFrame(pd_output,columns=["fov","row","trench","timepoints","File Index","Image Index","time (s)","lane orientation","y (local)","x (local)","y (global)","x (global)"])
        df = df.astype({"fov":int,"row":int,"trench":int,"timepoints":int,"File Index":int,"Image Index":int,"time (s)":float,"lane orientation":str,"y (local)":float,"x (local)":float,\
                        "y (global)":float,"x (global)":float})
        return df

    def generate_kymographs(self,dask_controller):
        self.writedir(self.kymographpath,overwrite=True)
        
        dask_controller.futures = {}
        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = fovdf.metadata
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        
        filedf = fovdf.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()
        
        file_list = filedf.index.get_level_values("File Index").unique().values
        fov_list = fovdf.index.get_level_values("fov").unique().values
        num_file_jobs = len(file_list)
        num_fov_jobs = len(fov_list)
        
        ### smoothed y percentiles ###
        
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            future = dask_controller.daskclient.submit(self.get_smoothed_y_percentiles,file_idx,\
                                        self.y_percentile,self.smoothing_kernel_y,retries=1,priority=priority)
            dask_controller.futures["Smoothed Y Percentiles: " + str(file_idx)] = future
            
        ### get trench row edges, y midpoints ###
        
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            smoothed_y_future = dask_controller.futures["Smoothed Y Percentiles: " + str(file_idx)]            
            future = dask_controller.daskclient.submit(self.get_y_edges_and_midpoints,smoothed_y_future,self.triangle_nbins,\
                                                       self.triangle_scaling,self.y_min_edge_dist,retries=1,priority=priority)
            dask_controller.futures["Y Trench Edges and Midpoints: " + str(file_idx)] = future
            
        ### get y drift, orientations, init edges ###
        
        random_priorities = np.random.uniform(size=(num_fov_jobs,))
        for k,fov_idx in enumerate(fov_list):
            priority = random_priorities[k]
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            edges_and_midpoint_futures = [dask_controller.futures["Y Trench Edges and Midpoints: " + str(file_idx)] for file_idx in working_files]
            future = dask_controller.daskclient.submit(self.get_y_drift_and_orientations,fov_idx,edges_and_midpoint_futures,self.expected_num_rows,\
                                                       self.top_orientation,self.orientation_on_fail,self.padding_y,retries=1,priority=priority)                
            dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Edges: " + str(fov_idx)] = future
            
        ### crop y dim ###
        
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            working_filedf = filedf.loc[file_idx]
            fov_idx = working_filedf["fov"].unique().tolist()[0]
            drift_orientation_and_initedge_future = dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Edges: " + str(fov_idx)]                        
            future = dask_controller.daskclient.submit(self.crop_y,file_idx,drift_orientation_and_initedge_future,self.padding_y,\
                                                       self.trench_len_y,retries=1,priority=priority)
            dask_controller.futures["Y Crop: " + str(file_idx)] = future
            
        ### smoothed x percentiles ###
        
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            y_crop_future = dask_controller.futures["Y Crop: " + str(file_idx)]
            future = dask_controller.daskclient.submit(self.get_smoothed_x_percentiles,y_crop_future,\
                                        self.x_percentile,self.background_kernel_x,self.smoothing_kernel_x,retries=1,priority=priority)
            dask_controller.futures["Smoothed X Percentiles: " + str(file_idx)] = future
        
        ### get x midpoints ###

        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            smoothed_x_future = dask_controller.futures["Smoothed X Percentiles: " + str(file_idx)]            
            future = dask_controller.daskclient.submit(self.get_x_midpoints,smoothed_x_future,\
                                                       self.otsu_nbins,self.otsu_scaling,retries=1,priority=priority)
            dask_controller.futures["X Midpoints: " + str(file_idx)] = future
            
        ### get x drift ###
        
        random_priorities = np.random.uniform(size=(num_fov_jobs,))
        for k,fov_idx in enumerate(fov_list):
            priority = random_priorities[k]
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            midpoint_futures = [dask_controller.futures["X Midpoints: " + str(file_idx)] for file_idx in working_files]
            future = dask_controller.daskclient.submit(self.get_x_drift,midpoint_futures,retries=1,priority=priority)
            dask_controller.futures["X Drift: " + str(fov_idx)] = future
        
        ### get kymograph masks ###
        
        random_priorities = np.random.uniform(size=(num_fov_jobs,))
        for k,fov_idx in enumerate(fov_list):
            priority = random_priorities[k]
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            midpoint_futures = [dask_controller.futures["X Midpoints: " + str(file_idx)] for file_idx in working_files]
            x_drift_future = dask_controller.futures["X Drift: " + str(fov_idx)]
            future = dask_controller.daskclient.submit(self.get_all_kymo_masks,midpoint_futures,x_drift_future,\
                                                self.trench_width_x,self.trench_present_thr,retries=1,priority=priority)
            dask_controller.futures["X Mask: " + str(fov_idx)] = future
            
        ### crop in x ###
            
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            working_filedf = filedf.loc[file_idx]
            fov_idx = working_filedf["fov"].unique().tolist()[0]
            
            y_crop_future = dask_controller.futures["Y Crop: " + str(file_idx)]
            kymo_mask_future = dask_controller.futures["X Mask: " + str(fov_idx)]
            
            future = dask_controller.daskclient.submit(self.crop_x,file_idx,y_crop_future,kymo_mask_future,retries=1,priority=priority)
            dask_controller.futures["X Crop: " + str(file_idx)] = future
            
        ### get coords ###
        
        random_priorities = np.random.uniform(size=(num_fov_jobs,))
        for k,fov_idx in enumerate(fov_list):
            priority = random_priorities[k]
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            y_crop_futures = [dask_controller.futures["Y Crop: " + str(file_idx)] for file_idx in working_files]
            kymo_mask_future = dask_controller.futures["X Mask: " + str(fov_idx)]
            drift_orientation_and_initedge_future = dask_controller.futures["Y Trench Drift, Orientations and Initial Trench Edges: " + str(fov_idx)]
            
            future = dask_controller.daskclient.submit(self.save_coords,fov_idx,y_crop_futures,kymo_mask_future,drift_orientation_and_initedge_future,retries=1,priority=priority)
            dask_controller.futures["Coords: " + str(fov_idx)] = future
            
    def collect_metadata(self,dask_controller):
        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        fov_list = fovdf.index.get_level_values("fov").unique().values        
        
        coord_futures = [dask_controller.futures["Coords: " + str(fov_idx)] for fov_idx in fov_list]
        coord_list = dask_controller.daskclient.gather(coord_futures, errors='skip')
        df_out = pd.concat(coord_list)
        df_out = df_out.set_index(["fov","row","trench","timepoints"], drop=True, append=False, inplace=False)
        
        idx_df = df_out.groupby(["fov","row","trench"]).size().reset_index().drop(0,axis=1).reset_index()
        idx_df = idx_df.set_index(["fov","row","trench"], drop=True, append=False, inplace=False)
        idx_df = idx_df.reindex(labels=df_out.index)
        df_out["trenchid"] = idx_df["index"]
        
        successful_fovs = df_out.index.get_level_values("fov").unique().tolist()
        failed_fovs = list(set(fov_list)-set(successful_fovs))
                
        meta_out_handle = pandas_hdf5_handler(self.metapath)
        kymograph_metadata = {"attempted_fov_list":fov_list,"successful_fov_list":successful_fovs,"failed_fov_list":failed_fovs,"kymograph_params":self.kymograph_params}
        meta_out_handle.write_df("temp_kymograph",df_out,metadata=kymograph_metadata)

    def reorg_kymograph(self,k):
        fovdf = self.meta_handle.read_df("temp_kymograph",read_metadata=True)
        metadata = fovdf.metadata
        trenchiddf = fovdf.reset_index(inplace=False)
        trenchiddf = trenchiddf.set_index(["trenchid","timepoints"], drop=True, append=False, inplace=False)
        trenchiddf = trenchiddf.sort_index()
        trenchid_list = trenchiddf.index.get_level_values("trenchid").unique().tolist()

        output_file_path = self.kymographpath+"/kymograph_"+str(k)+".hdf5"
        with h5py.File(output_file_path,"w") as outfile:
            for channel in self.all_channels:
                print(k)
                print(k*trenches_per_file)
                trenchids = trenchid_list[k*self.trenches_per_file:(k+1)*self.trenches_per_file]
                working_trenchdf = trenchiddf.loc[trenchids]
                fov_list = working_trenchdf["fov"].unique().tolist()
                print(trenchids)
                print(fov_list)

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
                                trenches = working_rowdf["trench"].unique().tolist()
                                first_trench_idx,last_trench_idx = (trenches[0],trenches[-1])
                                kymo_arr = infile[str(row) + "/" + channel][first_trench_idx:(last_trench_idx+1)]
                                trench_arr_rows.append(kymo_arr)
                        trench_arr_rows = np.concatenate(trench_arr_rows,axis=0) # k x t x y x x
                        trench_arr_files.append(trench_arr_rows)
                    trench_arr_files = np.concatenate(trench_arr_files,axis=1) # k x t x y x x
                    trench_arr_fovs.append(trench_arr_files)
                trench_arr_fovs = np.concatenate(trench_arr_fovs,axis=0) # k x t x y x x
                hdf5_dataset = outfile.create_dataset(str(channel), data=trench_arr_fovs, dtype="uint16")

    def reorg_all_kymographs(self,dask_controller):
        fovdf = self.meta_handle.read_df("temp_kymograph",read_metadata=True)
        metadata = fovdf.metadata
        trenchiddf = fovdf.reset_index(inplace=False)
        trenchiddf = trenchiddf.set_index(["trenchid","timepoints"], drop=True, append=False, inplace=False)
        trenchiddf = trenchiddf.sort_index()
        trenchid_list = trenchiddf.index.get_level_values("trenchid").unique().tolist()

        outputdf = trenchiddf.drop(columns = ["File Index","Image Index"])
        num_tpts = len(outputdf.index.get_level_values("timepoints").unique().tolist())
        chunk_size = self.trenches_per_file*num_tpts
        num_files = (len(trenchid_list)//self.trenches_per_file) + 1

        file_indices = np.repeat(np.array(range(num_files)),chunk_size)[:len(outputdf)]
        file_trenchid = np.repeat(np.array(range(self.trenches_per_file)),num_tpts)
        file_trenchid = np.repeat(file_trenchid[:,np.newaxis],num_files,axis=1).T.flatten()[:len(outputdf)]
        outputdf["File Index"] = file_indices
        outputdf["File Trench Index"] = file_trenchid
        fovdf = self.meta_handle.write_df("kymograph",outputdf,metadata=metadata)

        random_priorities = np.random.uniform(size=(num_files,))
        for k in range(0,num_files):
            priority = random_priorities[k]
            future = dask_controller.daskclient.submit(self.reorg_kymograph,k,retries=1,priority=priority)
            dask_controller.futures["Kymograph Reorganized: " + str(k)] = future

    def post_process(self,dask_controller):
        self.collect_metadata(dask_controller)
        self.reorg_all_kymographs(dask_controller)
        
    def kymo_report(self):
        df_in = self.meta_handle.read_df("kymograph",read_metadata=True)
        
        fov_list = df_in.metadata["attempted_fov_list"]
        failed_fovs = df_in.metadata["failed_fov_list"]

        fovs_proc = len(df_in.groupby(["fov"]).size())
        rows_proc = len(df_in.groupby(["fov","row"]).size())
        trenches_proc = len(df_in.groupby(["fov","row","trench"]).size())

        print("fovs processed: " + str(fovs_proc) + "/" + str(len(fov_list)))
        print("lanes processed: " + str(rows_proc))
        print("trenches processed: " + str(trenches_proc))
        print("row/fov: " + str(rows_proc/fovs_proc))
        print("trenches/fov: " + str(trenches_proc/fovs_proc))

        print("failed fovs: " + str(failed_fovs))

class kymograph_multifov(multifov):
    def __init__(self,headpath):
        """The kymograph class is used to generate and visualize kymographs. The central function of this
        class is the method 'generate_kymograph', which takes an hdf5 file of images from a single fov and
        outputs an hdf5 file containing kymographs from all detected trenches.

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
        """Performs initial import of the hdf5 file to be processed. Converts the input hdf5 file's "channel"
        datasets into the first dimension of the array, ordered as specified by 'self.all_channels'. Outputs
        a numpy array.
        
        Args:
            i (int): Specifies the current fov index.
        
        Returns:
            array: A numpy array containing the hdf5 file image data.
        """
        fov = self.fov_list[i]
        fovdf = self.metadf.loc[fov]
        fovdf = fovdf.loc[(slice(self.t_range[0],self.t_range[1],self.t_subsample_step)),:]
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
        return channel_array
    
    def import_hdf5_files(self,all_channels,seg_channel,fov_list,t_range,t_subsample_step):
        seg_channel_idx = all_channels.index(seg_channel)
        all_channels.insert(0, all_channels.pop(seg_channel_idx))
        self.all_channels = all_channels
        self.seg_channel = all_channels[0]
        self.fov_list = fov_list
        self.t_range = (t_range[0],t_range[1]+1)
        self.t_subsample_step = t_subsample_step
        
        super(kymograph_multifov, self).__init__(fov_list)
        
        imported_array_list = self.map_to_fovs(self.import_hdf5)
        
        return imported_array_list
        
    def median_filter_2d(self,array,smoothing_kernel):
        """Two-dimensional median filter, with average smoothing at the signal edges in
        the first dimension.
        
        Args:
            array_list (list): List containing a single array of yt signal to be smoothed.
        
        Returns:
            array: Median-filtered yt signal.
        """
        kernel = np.array(smoothing_kernel)
        kernel_pad = kernel//2 + 1
        med_filter = scipy.signal.medfilt(array,kernel_size=kernel)
        start_edge = np.mean(med_filter[kernel_pad[0]:kernel[0]])
        end_edge = np.mean(med_filter[-kernel[0]:-kernel_pad[0]])
        med_filter[:kernel_pad[0]] = start_edge
        med_filter[-kernel_pad[0]:] = end_edge
        return med_filter
    
    def get_smoothed_y_percentiles(self,i,imported_array_list,y_percentile,smoothing_kernel_y):
        """For each imported array, computes the percentile along the x-axis of the segmentation
        channel, generating a (y,t) array. Then performs median filtering of this array for smoothing.
        
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
        return y_percentiles_smoothed
    
    def triangle_threshold(self,img_arr,triangle_nbins,triangle_scaling):
        """Applys a triangle threshold to each timepoint in a (y,t) input array, returning a boolean mask.
        
        Args:
            img_arr (array): Image array to be thresholded.
            triangle_nbins (int): Number of bins to be used to construct the thresholding
            histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
        
        Returns:
            array: Boolean mask produced by the threshold.
        """
        all_thresholds = np.apply_along_axis(sk.filters.threshold_triangle,0,img_arr,nbins=triangle_nbins)*triangle_scaling
        triangle_mask = img_arr>all_thresholds
        return triangle_mask,all_thresholds

    def remove_out_of_frame(self,edges,start_above,end_above):
        """Takes an array of trench row edges and removes the first/last
        edge, if that edge does not have a proper partner (i.e. trench row mask
        takes value True at boundaries of image).
        
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
        if start_above:
            edges = edges[1:]
        if end_above:
            edges = edges[:-1]
        return edges
    
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
    
    def get_edges_from_mask(self,mask,y_min_edge_dist):
        """Finds edges from a boolean mask of shape (y,t). Filters out rows of length
        smaller than y_min_edge_dist.
        
        Args:
            mask (array): Boolean of shape (y,t) resulting from triangle thresholding.
            y_min_edge_dist (int): Minimum row length necessary for detection.
        
        Returns:
            list: List containing arrays of edges for each timepoint, filtered for rows that are too small.
        """
        edges_list = []
        for t in range(mask.shape[1]):
            edge_mask = (mask[1:,t] != mask[:-1,t])
            start_above,end_above = (mask[0,t]==True,mask[-1,t]==True)
            edges = np.where(edge_mask)[0]
            edges = self.remove_out_of_frame(edges,start_above,end_above)
            edges = self.remove_small_rows(edges,y_min_edge_dist)
            edges_list.append(edges)
        return edges_list
    
    def get_trench_edges_y(self,i,y_percentiles_smoothed_list,triangle_nbins,triangle_scaling,y_min_edge_dist):
        """Detects edges in the shape (y,t) smoothed percentile arrays for each input array.
        
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
        trench_mask_y,_ = self.triangle_threshold(y_percentiles_smoothed,triangle_nbins,triangle_scaling)
        trench_edges_y_list = self.get_edges_from_mask(trench_mask_y,y_min_edge_dist)
        return trench_edges_y_list
    
    def get_manual_orientations(self,i,trench_edges_y_lists,expected_num_rows,top_orientation,orientation_on_fail):
        trench_edges_y_list = trench_edges_y_lists[i]
        orientations = []
        if trench_edges_y_list[0].shape[0]//2 == expected_num_rows:
            orientation = top_orientation
            for row in range(trench_edges_y_list[0].shape[0]//2):
                orientations.append(orientation)
                orientation = (orientation+1)%2
        elif (trench_edges_y_list[0].shape[0]//2 < expected_num_rows) and orientation_on_fail is not None:
            orientation = orientation_on_fail
            for row in range(trench_edges_y_list[0].shape[0]//2):
                orientations.append(orientation)
                orientation = (orientation+1)%2
        else:
            print("Start frame does not have expected number of rows!")
        return orientations

    def get_phase_orientations(self,i,y_percentiles_smoothed_list,valid_edges_y_lists,pad=50,percentile=90):
        """Automatically determines the orientations of trench rows when segmenting with phase. Only
        considers the first timepoint. Currently the only mechanism to do this, until a manual version
        is implemented.
        
        Args:
            y_percentiles_smoothed_handle (h5py.File): Hdf5 file handle corresponding to smoothed y percentiles data. 
            valid_edges_y_list (list): Time-ordered list of trench edge arrays.
            pad (int, optional): Padding to be used to bin "start" and "end" values from trench row peaks.
            percentile (int, optional): Percentile to be used when scoring the "start" and "end" values
            from trench row peaks.
        
        Returns:
            list: List of ints representing the oreintation of each trench row, starting with the top row.
        """

        y_percentiles_smoothed = y_percentiles_smoothed_list[i]
        valid_edges_y_list = valid_edges_y_lists[i]
        orientations = []
        for row in range(valid_edges_y_list[0].shape[0]//2):
            edge_1,edge_2 = valid_edges_y_list[0][2*row],valid_edges_y_list[0][2*row+1]
            edge_1_val = np.percentile(y_percentiles_smoothed[:,0][edge_1:edge_1+pad],percentile)
            edge_2_val = np.percentile(y_percentiles_smoothed[:,0][edge_2-pad:edge_2],percentile)
            if edge_2_val>edge_1_val:
                orientations.append(0)
            else:
                orientations.append(1)
        return orientations

    def get_y_midpoints(self,i,trench_edges_y_lists):
        """Outputs trench row midpoints for each time point.
        
        Args:
            i (int): Specifies the current fov index.
            trench_edges_y_lists (list): List containing, for each fov entry, a time-ordered list
            of trench edge arrays.
        
        Returns:
            list: Time-ordered list of trench midpoint arrays for the fov of index i.
        """
        trench_edges_y_list = trench_edges_y_lists[i]
        midpoints = []
        for t in range(len(trench_edges_y_list)):
            midpoints_t = []
            for r in range(0,trench_edges_y_list[t].shape[0],2):
                midpoints_t.append(int(np.mean(trench_edges_y_list[t][r:r+2])))
            midpoints.append(np.array(midpoints_t))
        return midpoints

    def get_y_drift(self,i,y_midpoints_list):
        """Given a list of midpoints, computes the average drift in y for every timepoint.

        Args:
            y_midpoints_list (list): A list containing, for each fov, a list of the form [time_list,[midpoint_array]]
            containing the trench row midpoints.

        Returns:
            list: A nested list of the form [time_list,[y_drift_int]] for fov i.
        """
        y_midpoints = y_midpoints_list[i]
        y_drift = []
        for t in range(len(y_midpoints)-1):
            diff_mat = np.subtract.outer(y_midpoints[t+1],y_midpoints[t])
            if len(diff_mat) > 0:
                min_dist_idx = np.argmin(abs(diff_mat),axis=0)
                min_dists = []
                for row in range(diff_mat.shape[0]):
                    min_dists.append(diff_mat[row,min_dist_idx[row]])
                min_dists = np.array(min_dists)
                median_translation = np.median(min_dists)
            else:
                median_translation = 0
            y_drift.append(median_translation)
        net_y_drift = np.append(np.array([0]),np.add.accumulate(y_drift)).astype(int)
        return net_y_drift

    def keep_in_frame_kernels(self,i,trench_edges_y_lists,y_drift_list,imported_array_list,padding_y):
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
        trench_edges_y_list = trench_edges_y_lists[i]
        y_drift = y_drift_list[i]
        max_y_dim = imported_array_list[i].shape[1]

        max_drift,min_drift = np.max(y_drift),np.min(y_drift)
        edge_under_max = np.all((trench_edges_y_list+max_drift+padding_y)<max_y_dim,axis=0) 
        edge_over_min = np.all((trench_edges_y_list+min_drift-padding_y)>=0,axis=0)
        edge_in_bounds = edge_under_max*edge_over_min

        valid_edge_mask = []
        valid_orientation_mask = []
        for i in range(0,len(edge_in_bounds),2):
            if np.all(edge_in_bounds[i:i+2]):
                valid_edge_mask+=[True,True]
                valid_orientation_mask+=[True]
            else:
                valid_edge_mask+=[False,False]
                valid_orientation_mask+=[False]

        valid_edges_y_list = [trench_edges_y[valid_edge_mask] for trench_edges_y in trench_edges_y_list]
        return valid_edges_y_list,valid_orientation_mask
    
    def get_row_numbers(self,i,trench_edges_y_list):
        """Computes the number of trench rows in the fov, from the detected edges.
        
        Args:
            i (int): Specifies the current fov index.
            trench_edges_y_list (list): List containing, for each fov entry, a list of time-sorted edge arrays.
        
        Returns:
            int: The number of trench rows detected in the fov of index i.
        """
        trench_edges_y = trench_edges_y_list[i]
        edge_num_list = [len(item) for item in trench_edges_y]
        trench_row_num = (np.median(edge_num_list).astype(int))//2
        return trench_row_num

    def crop_y(self,i,imported_array_list,y_drift_list,valid_edges_y_lists,trench_orientations_list,padding_y,trench_len_y):
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
        valid_edges_y_list = valid_edges_y_lists[i]
        trench_orientations = trench_orientations_list[i]

        drift_corrected_edges = np.add.outer(y_drift,valid_edges_y_list[0])
        time_list = []

        for t in range(imported_array.shape[3]):
            trench_edges_y = drift_corrected_edges[t]
            row_list = []

            for r,orientation in enumerate(trench_orientations):
                if orientation == 0:
                    trench_edge_y = trench_edges_y[2*r]
                    upper = max(trench_edge_y-padding_y,0)
                    lower = min(trench_edge_y+trench_len_y,imported_array.shape[1])
                    pad = upper+trench_len_y+padding_y-lower

                else:
                    trench_edge_y = trench_edges_y[(2*r)+1]
                    upper = max(trench_edge_y-trench_len_y,0)
                    lower = min(trench_edge_y+padding_y,imported_array.shape[1])
                    pad = upper+trench_len_y+padding_y-lower

                channel_list = []
                for c in range(imported_array.shape[0]):
                    output_array = np.pad(imported_array[c,upper:lower,:,t],((pad, 0),(0,0)),'constant')
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
        
    def crop_trenches_in_y(self,imported_array_list):
        """Master function for cropping the input hdf5 file in the y-dimension.
        
        Args:
            imported_array_list (list): List containing, for each fov entry, a numpy array containing
            the corresponding hdf5 file image data.
        
        Returns:
            list: List containing, for each fov entry, a y-cropped numpy array of shape (rows,channels,x,y,t).
        """        
        y_percentiles_smoothed_list = self.map_to_fovs(self.get_smoothed_y_percentiles,imported_array_list,\
                                                       self.y_percentile,self.smoothing_kernel_y)
        
        trench_edges_y_lists = self.map_to_fovs(self.get_trench_edges_y,y_percentiles_smoothed_list,self.triangle_nbins,\
                                               self.triangle_scaling,self.y_min_edge_dist)
        
        y_midpoints_list = self.map_to_fovs(self.get_y_midpoints,trench_edges_y_lists)
        y_drift_list = self.map_to_fovs(self.get_y_drift,y_midpoints_list)

        if self.orientation_detection == 'phase':
            valid_edges_y_output = self.map_to_fovs(self.keep_in_frame_kernels,trench_edges_y_lists,y_drift_list,imported_array_list,self.padding_y)
            valid_edges_y_lists = [item[0] for item in valid_edges_y_output]
            trench_orientations_list = self.map_to_fovs(self.get_phase_orientations,y_percentiles_smoothed_list,valid_edges_y_lists)
        
        elif self.orientation_detection == 0 or self.orientation_detection == 1:
            trench_orientations_list = self.map_to_fovs(self.get_manual_orientations,trench_edges_y_lists,self.expected_num_rows,self.orientation_detection,self.orientation_on_fail)
            valid_edges_y_output = self.map_to_fovs(self.keep_in_frame_kernels,trench_edges_y_lists,y_drift_list,imported_array_list,self.padding_y)
            valid_edges_y_lists = [item[0] for item in valid_edges_y_output]
            valid_orientation_lists = [item[1] for item in valid_edges_y_output]
            trench_orientations_list = [np.array(item)[valid_orientation_lists[i]].tolist() for i,item in enumerate(trench_orientations_list)]
            
        else:
            print("Orientation detection value invalid!")

        cropped_in_y_list = self.map_to_fovs(self.crop_y,imported_array_list,y_drift_list,valid_edges_y_lists,trench_orientations_list,self.padding_y,\
                                             self.trench_len_y)
        return cropped_in_y_list
    
    def get_smoothed_x_percentiles(self,i,cropped_in_y_list,x_percentile,background_kernel_x,smoothing_kernel_x):
        """Summary
        
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
    
    def get_midpoints(self,x_percentiles_t,otsu_nbins,otsu_scaling):
        """Given an array of signal in x, determines the position of trench midpoints.
        
        Args:
            x_percentiles_t (array): array of trench intensities in x, at time t.
            otsu_nbins (int): Number of bins to use when applying Otsu's method to x-dimension signal.
            otsu_scaling (float): Threshold scaling factor for Otsu's method thresholding.
        
        Returns:
            array: array of trench midpoint x positions.
        """
        otsu_threshold = sk.filters.threshold_otsu(x_percentiles_t[:,np.newaxis],nbins=otsu_nbins)*otsu_scaling
        x_mask = x_percentiles_t>otsu_threshold
        midpoints = self.get_midpoints_from_mask(x_mask)
        return midpoints,otsu_threshold
    
    def get_all_midpoints(self,i,x_percentiles_smoothed_list,otsu_nbins,otsu_scaling):
        """Given an x percentile array of shape (rows,x,t), determines the trench midpoints of each row array
        at each time t.
        
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
            midpoints,_ = self.get_midpoints(x_percentiles_smoothed[:,0],otsu_nbins,otsu_scaling)
            if len(midpoints) == 0:
                return None
            all_midpoints.append(midpoints)
            for t in range(1,x_percentiles_smoothed.shape[1]):
                midpoints,_ = self.get_midpoints(x_percentiles_smoothed[:,t],otsu_nbins,otsu_scaling)
                if len(midpoints)/(len(all_midpoints[-1])+1) < 0.5:
                    all_midpoints.append(all_midpoints[-1])
                else:
                    all_midpoints.append(midpoints)
            midpoints_row_list.append(all_midpoints)
        return midpoints_row_list

    def get_x_drift(self,i,all_midpoints_list):
        """Given an t by x array of midpoints, computed the average drift in x for every timepoint.
        
        Args:
            i (int): Specifies the current fov index.
            all_midpoints_list (list): A nested list of the form [fov_list,[row_list,[time_list,[midpoint_array]]]] containing
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
        """Initializes a counting array of shape (x_dim,t_dim) which counts from 0 to
        x_dim on axis 0 for all positions in axis 1.
        
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
        """Generates a boolean trench mask of shape (x_dim,t_dim) for a given trench k, using
        the trench boundary values in in_bounds_list.
        
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
        """Given a y-cropped image and a boolean trench mask of shape (x_dim,t_dim), masks that image in
        xt to generate an output kymograph of shape (y_dim,x_dim,t_dim). 
        
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
        """Generates a boolean trench mask of shape (x_dim,t_dim) for each trench k. This will be used to mask
        out each trench at a later step.
        
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
        """Performs cropping of the aleady y-cropped image data, using pregenerated kymograph masks
        of shape (x_dim,t_dim).
        
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
        """Generates complete kymograph arrays for all trenches in the fov in every channel listed in 'self.all_channels'.
        Outputs a list of these kymograph arrays, with entries corresponding to each row in the fov with index i.
        
        Args:
            i (int): Specifies the current fov index.
            cropped_in_y_list (list): List containing, for each fov entry, a y-cropped numpy array of shape (rows,channels,x,y,t).
            all_midpoints_list (list): A nested list of the form [fov_list,[row_list,[time_list,[midpoint_array]]]] containing
            the trench midpoints.
            x_drift_list (list): A nested list of the form [fov_list,[row_list,[time_list,[x_drift_int]]]] containing the computed
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
            list: A nested list of the form [fov_list,[row_list,[kymograph_array]]], containing kymograph arrays of
            shape (channels,trenches,y_dim,x_dim,t_dim).
        """
        smoothed_x_percentiles_list = self.map_to_fovs(self.get_smoothed_x_percentiles,cropped_in_y_list,self.x_percentile,\
                                                                 self.background_kernel_x,self.smoothing_kernel_x)
        all_midpoints_list = self.map_to_fovs(self.get_all_midpoints,smoothed_x_percentiles_list,self.otsu_nbins,self.otsu_scaling)
        x_drift_list = self.map_to_fovs(self.get_x_drift,all_midpoints_list)
        cropped_in_x_list = self.map_to_fovs(self.get_crop_in_x,cropped_in_y_list,all_midpoints_list,x_drift_list,self.trench_width_x,self.trench_present_thr)
        return cropped_in_x_list
        
    def generate_kymograph(self):
        """Master function for generating kymographs for the set of fovs specified on initialization.
        
        Returns:
            list: A nested list of the form [fov_list,[row_list,[kymograph_array]]], containing kymograph arrays of
            shape (channels,trenches,y_dim,x_dim,t_dim).
        """
        array_list = self.map_to_fovs(self.import_hdf5)
        cropped_in_y_list = self.crop_trenches_in_y(array_list)
        cropped_in_x_list = self.crop_trenches_in_x(cropped_in_y_list)
        
        return cropped_in_x_list