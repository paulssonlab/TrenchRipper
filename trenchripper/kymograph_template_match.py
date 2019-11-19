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
from .utils import multifov,pandas_hdf5_handler,writedir
from .drift_detection import find_seed_image, get_orb_pois, find_drift_poi, find_template, find_drift_template

class kymograph_cluster:
    def __init__(self,headpath="",trenches_per_file=20,paramfile=False,all_channels=[""],trench_len_y=270,padding_y=20,trench_width_x=30,\
                 t_range=(0,None),y_percentile=85,y_min_edge_dist=50,smoothing_kernel_y=(1,9),triangle_nbins=50,triangle_scaling=1.,\
                 triangle_max_threshold=0,triangle_min_threshold=65535,top_orientation=0,expected_num_rows=None,orientation_on_fail=None,\
                 x_percentile=85,background_kernel_x=(1,21),smoothing_kernel_x=(1,9),otsu_nbins=50,otsu_scaling=1.,trench_present_thr=0.):
        
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
            smoothing_kernel_y = param_dict["Y Smoothing Kernel"]
            triangle_nbins = param_dict["Triangle Threshold Bins"]
            triangle_scaling = param_dict["Triangle Threshold Scaling"]
            triangle_max_threshold = param_dict['Triangle Max Threshold']
            triangle_min_threshold = param_dict['Triangle Min Threshold']
            top_orientation = param_dict["Orientation Detection Method"]
            expected_num_rows = param_dict["Expected Number of Rows (Manual Orientation Detection)"]
            orientation_on_fail = param_dict["Top Orientation when Row Drifts Out (Manual Orientation Detection)"]
            x_percentile = param_dict["X Percentile"]
            background_kernel_x = param_dict["X Background Kernel"]
            smoothing_kernel_x = param_dict["X Smoothing Kernel"]
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
        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = fovdf.metadata
        
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
        self.triangle_max_threshold = triangle_max_threshold
        self.triangle_min_threshold = triangle_min_threshold
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
                                 "triangle_max_threshold":triangle_max_threshold,"triangle_min_threshold":triangle_min_threshold,\
                                 "top_orientation":top_orientation,"expected_num_rows":expected_num_rows,\
                                 "orientation_on_fail":orientation_on_fail,"x_percentile":x_percentile,\
                                 "background_kernel_x":background_kernel_x,"smoothing_kernel_x":smoothing_kernel_x,\
                                "otsu_nbins":otsu_nbins,"otsu_scaling":otsu_scaling,"trench_present_thr":trench_present_thr}

    def find_seed_image(self, file_idx):
        """ Get an image for each file to act as reference for drift measurements. This is to account for
        the possibility that fiduciary markers are not available in the first timepoint or that it is out
        of focus, even though the majority of images may still be fine. Bad timepoints are easier to filter
        at the kymograph stage.

        Inputs:
            file_idx(int): index of hdf5 archive
        Outputs:
            seed_index(int): index of reference image within stack contained in the hdf5 archive
        """
        # Load hdf5 archive
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            # Open time stack of images
            img_arr = imported_hdf5_handle[self.seg_channel][:] #t x y x x
            # Find the seed index
            seed_index = find_seed_image(img_arr)
        return seed_index


    def find_median_outliers(self, points, thresh = 3.5):
        """ Detect outliers using median absolute deviation method

        Inputs:
            points (numpy.ndarray): set of x, y points (i.e. drifts of shape t x 2)
        Outputs:
            array (numpy.ndarray): boolean array (of length t) indicating which points are outliers
        
        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        if len(points.shape) == 1:
            points = points[:,None]
        # Calculate median x and y drift
        median = np.nanmedian(points, axis=0)
        # Calclulate sum of squared difference of x and y to median
        diff = np.sqrt(np.sum((points - median)**2, axis=-1))
        # Take the median of these sequared deviations
        med_abs_deviation = np.nanmedian(diff)
        # Z score - if it's above a threshold it's an outlier. You can think of this as
        # how much a particular point deviates from the median
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thresh

    def get_drifts(self, file_idx, seed_idx, max_poi_std=5):
        """ Find drifts within each hdf5 archive using interest point matching, starting from the seed index

        Inputs: 
            file_idx (int): index of hdf5 archive
            seed_idx (int): index of reference image within stack contained in the hdf5 archive
            max_poi_std (float): maximum standard deviation between selected interest points used to register images.
                        If the measured std is higher, interpolate between timepoints.
        Outputs:
            drifts (numpy.ndarray): t x 2 numpy.ndarray of x and y drift for the time stack
        """
        # Load hdf5 archive
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            img_arr = imported_hdf5_handle[self.seg_channel][:] #t x y
        drifts = np.zeros((img_arr.shape[0], 2))
        
        for i in range(0, seed_idx):
            # Get points of interest using ORB algorithm
            points1, points2, std_distance = get_orb_pois(img_arr[seed_idx], img_arr[i])
            
            # If the standard deviation of distance between the interest point matches across timepoints
            # is too high, instead label as nan for now and interpolate (the matches can't be trusted)
            if std_distance > max_poi_std:
                drifts[i,:] = [np.nan, np.nan]
            else:
                # Find the translation according to the interest point matches using RANSAC to account
                # for outliers
                drifts[i,:] = find_drift_poi(points1, points2)
        for i in range(seed_idx+1, img_arr.shape[0]):
            points1, points2, std_distance = get_orb_pois(img_arr[seed_idx], img_arr[i])
            if std_distance > max_poi_std:
                drifts[i,:] = [np.nan, np.nan]
            else:
                drifts[i,:] = find_drift_poi(points1, points2)
        # Find outliers where drift is too large and also set to nan (similar to median filter but
        # leave inliers alone)
        outliers = self.find_median_outliers(drifts)
        drifts[outliers,:] = [np.nan, np.nan]
        # Interpolate for timepoints where drift could not be calculated
        for i in range(img_arr.shape[0]):
            if np.isnan(drifts[i,0]):
                # Get correct drift from the right
                if i==0:
                    right_idx = 1
                    while np.isnan(drifts[right_idx, 0]):
                        right_idx += 1
                    for j in range(i, right_idx):
                        drifts[j,:] = drifts[right_idx,:]
                # Get correct drift from the left
                elif i == img_arr.shape[0] - 1:
                    left_idx = i - 1
                    while np.isnan(drifts[left_idx, 0]):
                        left_idx -= 1
                    for j in range(left_idx+1, i+1):
                        drifts[j,:] = drifts[left_idx,:]
                # Interpolate between left and right points
                else:
                    left_idx = i - 1
                    right_idx = i + 1
                    while np.isnan(drifts[right_idx, 0]) and right_idx < img_arr.shape[0]-1:
                        right_idx += 1
                    while np.isnan(drifts[left_idx, 0]) and left_idx > 0:
                        left_idx -= 1
                    if np.isnan(drifts[right_idx, 0]):
                        for j in range(left_idx+1, right_idx+1):
                            drifts[j,:] = drifts[left_idx,:]
                    elif np.isnan(drifts[left_idx, 0]):
                        for j in range(left_idx, right_idx):
                            drifts[j,:] = drifts[right_idx,:]
                    else:
                        for j in range(left_idx+1, right_idx):
                            drifts[j,:] = drifts[left_idx,:] + (drifts[right_idx,:]-drifts[left_idx,:]) * (j-left_idx)/(right_idx-left_idx)
        return drifts

    
    def find_seed_image_and_template(self, file_idx):
        """ Get an image for each file to act as reference for drift measurements. This is to account for
        the possibility that fiduciary markers are not available in the first timepoint or that it is out
        of focus, even though the majority of images may still be fine. Bad timepoints are easier to filter
        at the kymograph stage. Use points of interest to determine a patch of the image to use for template
        matching.

        Inputs:
            file_idx (int): index of hdf5 archive
        Outputs:
            seed_index (int): index of reference image within stack contained in the hdf5 archive
            template (numpy.ndarray): 2D image patch to use as template
            top_left (tuple): x, y coordinates (int, int) of top-left corner of template, used to calculate drift
                    later
        """
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            img_arr = imported_hdf5_handle[self.seg_channel][:] #t x y
            seed_index = find_seed_image(img_arr)
            template, top_left = find_template(img_arr[seed_index], img_arr[seed_index+1])
        return seed_index, template, top_left
    
    def get_drifts_template(self, file_idx, seed_image_and_template_future, max_sqdiff=0.1):
        """ Detect drifts using MSD template matching

        Inputs:
            file_idx (int): index of hdf5 archive
            seed_image_and_template_future (tuple): tuple of (seed_index, template, top_left) (to be
            compatible with map-reduce framework in Dask)
            seed_index (int): index of reference image within stack contained in the hdf5 archive
            template (numpy.ndarray): 2D image patch to use as template
            top_left (tuple): x, y coordinates (int, int) of top-left corner of template
            max_sqdiff (double, <=1): normalized square difference maximum above which to throw out the
                        template match and instead interpolate
        Outputs:
            seed_index - index of reference image within stack contained in the hdf5 archive
        """
        # Unpack tuple
        seed_idx, template, top_left = seed_image_and_template_future
        # Open HDF5 archive
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            img_arr = imported_hdf5_handle[self.seg_channel][:] #t x y
        drifts = np.zeros((img_arr.shape[0], 2))
        # Template match
        for i in range(0, seed_idx):
            drifts[i,:], min_sqdiff = find_drift_template(template, top_left, img_arr[i])
            # Throw out bad matches
            if min_sqdiff > max_sqdiff:
                drifts[i,:] = [np.nan, np.nan]
        for i in range(seed_idx+1, img_arr.shape[0]):
            drifts[i,:], min_sqdiff = find_drift_template(template, top_left, img_arr[i])
            if min_sqdiff > max_sqdiff:
                drifts[i,:] = [np.nan, np.nan]
        # Find outliers where drift is too large and also set to nan (similar to median filter but
        # leave inliers alone)        
        outliers = self.find_median_outliers(drifts)
        drifts[outliers,:] = [np.nan, np.nan]
        # Interpolate for timepoints where drift could not be calculated
        for i in range(img_arr.shape[0]):
            if np.isnan(drifts[i,0]):
                # Get correct drift from left
                if i==0:
                    right_idx = 1
                    while np.isnan(drifts[right_idx, 0]):
                        right_idx += 1
                    for j in range(i, right_idx):
                        drifts[j,:] = drifts[right_idx,:]
                # Get correct drift from right
                elif i == img_arr.shape[0] - 1:
                    left_idx = i - 1
                    while np.isnan(drifts[left_idx, 0]):
                        left_idx -= 1
                    for j in range(left_idx+1, i+1):
                        drifts[j,:] = drifts[left_idx,:]
                # Interpolate between left and right
                else:
                    left_idx = i - 1
                    right_idx = i + 1
                    while np.isnan(drifts[right_idx, 0]) and right_idx < img_arr.shape[0]-1:
                        right_idx += 1
                    while np.isnan(drifts[left_idx, 0]) and left_idx > 0:
                        left_idx -= 1
                    if np.isnan(drifts[right_idx, 0]):
                        for j in range(left_idx+1, right_idx+1):
                            drifts[j,:] = drifts[left_idx,:]
                    elif np.isnan(drifts[left_idx, 0]):
                        for j in range(left_idx, right_idx):
                            drifts[j,:] = drifts[right_idx,:]
                    else:
                        for j in range(left_idx+1, right_idx):
                            drifts[j,:] = drifts[left_idx,:] + (drifts[right_idx,:]-drifts[left_idx,:]) * (j-left_idx)/(right_idx-left_idx)
        return drifts

    def link_drifts(self, file_indices, seed_images, within_file_drifts):
        """ Put all drifts of a field of view with respect to first file by linking seed drifts for each hdf5 archive
        
        Inputs:
            file_indices (list): hdf5 archive indices (int) that belong to a field of view
            seed_images (list): seed indices (int) for each hdf5 archive
            within_file_difts (list): t x 2 numpy arrays containing the drifts with respect to
                                each seed image 
        """
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_indices[0])+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            first_seed_img = imported_hdf5_handle[self.seg_channel][seed_images[0],:,:] #t x y
        # Start with first file's drift w.r.t its seed image
        drifts = [within_file_drifts[0]]
        for k, file_idx in enumerate(file_indices[1:]):
            with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
                # Find the seed image in the other file
                comp_seed_img = imported_hdf5_handle[self.seg_channel][seed_images[k],:,:] #t x y
                # Interest point match the seed images (assume well behaved from checking previously so no need
                # for error correction)
                points1, points2, _ = get_orb_pois(first_seed_img, comp_seed_img)
                file_to_file_drift = find_drift_poi(points1, points2).reshape(1, 2)
                drifts.append(within_file_drifts[k+1] + file_to_file_drift)
        drifts = np.concatenate(drifts, axis=0)
        return drifts

    def median_filter(self,array,smoothing_kernel):
        """1D median filter, with average smoothing at the signal edges.
        
        Args:
            array_list (list): List containing a single array to be smoothed.
            smoothing_kernel (int) : The size of the  kernel under which
            the median will be taken.
        
        Returns:
            med_filter (numpy.ndarray): Median-filtered 1D dimensional signal (y,).
        """
        kernel = smoothing_kernel #1,9
        kernel_pad = kernel//2 + 1 #1,5
        med_filter = scipy.signal.medfilt(array,kernel_size=kernel)
        start_edge = np.mean(med_filter[kernel_pad:kernel])
        end_edge = np.mean(med_filter[-kernel:-kernel_pad])
        med_filter[:kernel_pad] = start_edge
        med_filter[-kernel_pad:] = end_edge
        return med_filter

    def get_smoothed_y_percentiles(self,file_idx,seed_image_idx,y_percentile,smoothing_kernel_y):
        """For each imported file, computes the percentile along the x-axis of the segmentation
        channel for the previously identified seed image, generating a (y,) array.
        Then performs median filtering of this array for smoothing.
        
        Args:
            imported_hdf5_handle (h5py.File): Hdf5 file handle corresponding to the input hdf5 dataset
            "data" of shape (channel,y,x,t).
            seed_image_idx (int): index of seed image on which to perform segmentation
            y_percentile (int): Percentile to apply along the x-axis.
            smoothing_kernel_y (int): Kernel to use for median filtering.
        
        Returns:
            y_percentiles_smoothed (numpy.ndarray): a smoothed percentile array of shape (y,).
        """
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            seed_img = imported_hdf5_handle[self.seg_channel][seed_image_idx]
            perc_arr = np.percentile(seed_img,y_percentile,axis=1,interpolation='lower')
            y_percentiles_smoothed = self.median_filter(perc_arr,smoothing_kernel_y)
            
            min_qth_percentile = y_percentiles_smoothed.min()
            max_qth_percentile = y_percentiles_smoothed.max()
            y_percentiles_smoothed = (y_percentiles_smoothed - min_qth_percentile)/(max_qth_percentile - min_qth_percentile)
            
        return y_percentiles_smoothed
    
    def triangle_threshold(self,y_percentiles_smoothed,triangle_nbins,triangle_scaling,triangle_max_threshold,triangle_min_threshold):
        """Applies a triangle threshold to a smoothed y percentile array, returning a boolean mask.
        
        Args:
            y_percentiles_smoothed (array): 1-D array to be thresholded.
            triangle_nbins (int): Number of bins to be used to construct the thresholding
            histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
        
        Returns:
            triangle_mask (numpy.ndarray): Boolean mask produced by the threshold.
        """
        threshold = sk.filters.threshold_triangle(y_percentiles_smoothed,nbins=triangle_nbins)*triangle_scaling
        threshold = min(threshold, triangle_max_threshold)
        threshold = max(threshold, triangle_min_threshold)
        
        triangle_mask = y_percentiles_smoothed>threshold
        return triangle_mask
    
    def get_edges_from_mask(self,mask):
        """Finds edges from a boolean mask of shape (y,).
        
        Args:
            mask (array): Boolean of shape (y,) resulting from triangle thresholding.
            y_min_edge_dist (int): Minimum row length necessary for detection.
        
        Returns:
            edges (numpy.ndarray): list of y-edges of trenches (each pair = 1 trench)
            start_above (bool): Whether the top edge of the image is inside a trench
            end_above (bool): Whether the bottom edge of the image is inside a trench
        """

        edge_mask = (mask[1:] != mask[:-1])
        start_above,end_above = (mask[0]==True,mask[-1]==True)
        edges = np.where(edge_mask)[0]

        return edges,start_above,end_above
    
    def get_trench_edges_y(self,y_percentiles_smoothed,triangle_nbins,triangle_scaling,triangle_max_threshold,triangle_min_threshold,y_min_edge_dist):
        """Detects edges in the shape (y,) smoothed percentile array for each seed image.
        
        Args:
            y_percentiles_smoothed (array): A shape (y,) smoothed percentile array for the seed image.
            triangle_nbins (int): Number of bins to be used to construct the thresholding histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
            y_min_edge_dist (int): Minimum row length necessary for detection.
        
        Returns:
            edges (nu,[y/mdarray]): list of y-edges of the trench (each pair = 1 trench)
            start_above (bool): Whether the top edge of the image is inside a trench
            end_above (bool): Whether the bottom edge of the image is inside a trench
        """
        
        trench_mask_y = self.triangle_threshold(y_percentiles_smoothed,triangle_nbins,triangle_scaling,triangle_max_threshold,triangle_min_threshold)
        edges,start_above,end_above = self.get_edges_from_mask(trench_mask_y)
        return edges,start_above,end_above
    
    def repair_out_of_frame(self,trench_edges_y,start_above,end_above):
        """ Replace missing trench edges with the edges of the image
        
        Args:
            trench_edges_y (numpy.ndarray): list of y-edges of the trench (each pair = 1 trench)
            start_above (bool): Whether the top edge of the image is inside a trench
            end_above (bool): Whether the bottom edge of the image is inside a trench
        Returns:
            trench_edges_y (numpy.ndarray): list of repaired y-edges of the trench (each pair = 1 trench)

        """
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
    
    def get_manual_orientations(self,trench_edges_y,start_above,end_above,expected_num_rows,top_orientation,orientation_on_fail,y_min_edge_dist):
        repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
        repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)
        
        orientations = []
        if repaired_trench_edges_y.shape[0]//2 == expected_num_rows:
            orientation = top_orientation
            for row in range(repaired_trench_edges_y.shape[0]//2):
                orientations.append(orientation)
                orientation = (orientation+1)%2
            orientations,drop_first_row,drop_last_row,repaired_trench_edges_y = self.remove_out_of_frame(orientations,repaired_trench_edges_y,start_above,end_above)
                
        elif (repaired_trench_edges_y.shape[0]//2 < expected_num_rows) and orientation_on_fail is not None:
            orientation = orientation_on_fail
            for row in range(repaired_trench_edges_y.shape[0]//2):
                orientations.append(orientation)
                orientation = (orientation+1)%2
            orientations,drop_first_row,drop_last_row,repaired_trench_edges_y = self.remove_out_of_frame(orientations,repaired_trench_edges_y,start_above,end_above)
        else:
            print("Start frame does not have expected number of rows!")
            
        return repaired_trench_edges_y, orientations,drop_first_row,drop_last_row

    def get_trench_ends(self, seed_image_idx, seed_trench_edges_y, y_drifts, orientations,drop_first_row,drop_last_row, padding_y, trench_len_y):        
        """ Get trench ends (i.e. y-boundaries) for all times in a field of view based on seed image segmentation and calculated drift
        
        Args:
            seed_image_idx (int): Index of seed image in the full image stack for the field of view
            seed_trench_edges_y (numpy.ndarrray, int): List of segmented edges in seed mage
            orientations (numpy.ndarray, int): List of orientations for each trench row in a field of view
            drop_first_row (bool): Whether to ignore the first row of trenches
            drop_last_row (bool): Whether to ignore the second row of trenches
            padding_y (int): Extra space to add beyond the segmentation
            trench_len_y (int): Length to cut from the dead end of the trench

        Returns:
            valid_y_ends (numpy.ndarray): List of valid dead ends of the trenches for each row
            valid_orientations (numpy.ndarray): Corresponding orientations
        """
        # Get trench edges at each timepoint according to the trench edges of the seed index and the drifts
        trench_edges_y_list = np.tile(seed_trench_edges_y, (y_drifts.shape[0], 1)) + y_drifts[:,None]
        # Cut edges that are outside the image
        trench_edges_y_list[:, 0] = np.maximum(trench_edges_y_list[:,0], 0)
        trench_edges_y_list[:, -1] = np.minimum(trench_edges_y_list[:,-1], int(self.metadata['height']))

        # Remove any trenches that should be dropped
        if trench_edges_y_list.shape[1]//2 > len(orientations) and drop_first_row:
            trench_edges_y_list = trench_edges_y_list[:,2:]
        if trench_edges_y_list.shape[1]//2 > len(orientations) and drop_last_row:
            trench_edges_y_list = trench_edges_y_list[:,:-2]

        y_ends_list = []

        # Pair orientations and y ends of trenches for each timepoint        
        for t in range(trench_edges_y_list.shape[0]):
            grouped_edges = trench_edges_y_list[t,:].reshape(-1,2) # or,2
            y_ends = []
            for edges,orientation in enumerate(orientations):
                y_ends.append(grouped_edges[edges,orientation])
            y_ends = np.array(y_ends)
            y_ends_list.append(y_ends)
        
        valid_y_ends_list = []
        valid_orientations = []

        seed_y_ends = y_ends_list[seed_image_idx]
        max_y_dim = self.metadata['height']
        max_drift,min_drift = np.max(y_drifts),np.min(y_drifts)

        # Check that the trench does not drift outside the image at any timepoint         
        for j,orientation in enumerate(orientations):
            y_end = seed_y_ends[j]
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
                valid_y_ends_list.append([y_end[j] for y_end in y_ends_list])
                valid_orientations.append(orientation)
        
        valid_y_ends = np.round(np.array(valid_y_ends_list).T).astype(int) # t,edge
        if len(valid_y_ends) == 0:
            raise Exception("No valid y-ends with seed ends %s and drift bounds %d, %d" %(str(seed_y_ends), max_drift, min_drift))
        return valid_y_ends,valid_orientations
    
    def get_ends_and_orientations(self,seed_image_idx,drift_future,edges_future,expected_num_rows,top_orientation,orientation_on_fail,y_min_edge_dist,padding_y,trench_len_y):
        """ Get orientations and dead ends of trenches that have not drifted out of the image at any timepoint

        Args:
            seed_image_idx (int): index of seed image
            drift_future (numpy.ndarray): t x 2 array of x and y drifts
            edges_future (tuple): tuple of (seed_trench_edges, seed_start_above, seed_end_above)
            seed_trench_edges (numpy.ndarray): list of detected trench edges for the seed image
            seed_start_above (bool): Whether the top edge of the image is within a trench
            seed_end_above (bool): Whether the bottom edge of the image is within a trench
            top_orientation (int): Whether the first trench in an image is pointing up or down
            orientation_on_fail (int): ???
            y_min_edge_dist (int): The minimum distance between edges to consider a signal as a possible trench
            trench_len_y (int): Length of trench to cut
        Returns:

        """
        # Unpack edge future
        seed_trench_edges, seed_start_above, seed_end_above = edges_future
        # Unpack y from drift future
        y_drift = drift_future[:,1]
        
        repaired_trench_edges_y, orientations,drop_first_row,drop_last_row = self.get_manual_orientations(seed_trench_edges,seed_start_above,seed_end_above,expected_num_rows,top_orientation,orientation_on_fail,y_min_edge_dist)
        valid_y_ends, valid_orientations = self.get_trench_ends(seed_image_idx, repaired_trench_edges_y, y_drift, orientations,drop_first_row,drop_last_row,padding_y, trench_len_y)
        
        return valid_orientations,valid_y_ends
    
    def crop_y(self,file_idx,orientation_and_initend_future,padding_y,trench_len_y):
        """Performs cropping of the images in the y-dimension.
        
        Args:
            file_idx (int):
            orientation_and_initend_future (tuple): tuple of valid_orientations, valid_y_ends
        Returns:
            array: A y-cropped array of shape (rows,channels,x,y,t).
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

        valid_orientations,valid_y_ends = orientation_and_initend_future
        drift_corrected_edges = valid_y_ends
        
        channel_arr_list = []
        for c,channel in enumerate(self.all_channels):
            with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
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
    
    def crop_y_single(self, file_idx, seed_image_idx, orientation_and_initend_future, padding_y, trench_len_y):        
        with h5py_cache.File(self.hdf5path+"/hdf5_"+str(file_idx)+".hdf5","r",chunk_cache_mem_size=self.metadata["chunk_cache_mem_size"]) as imported_hdf5_handle:
            seed_image = imported_hdf5_handle[self.seg_channel][seed_image_idx]
        
        time_list = []
        lane_y_coords_list = []
        valid_orientations,valid_y_ends = orientation_and_initend_future
        
        trench_ends_y = valid_y_ends[seed_image_idx]
        row_list = []
        lane_y_coords = []
        for r,orientation in enumerate(valid_orientations):
            trench_end = trench_ends_y[r]
            if orientation == 0:
                upper = max(trench_end-padding_y,0)
                lower = min(trench_end+trench_len_y,seed_image.shape[0])
            else:
                upper = max(trench_end-trench_len_y,0)
                lower = min(trench_end+padding_y,seed_image.shape[0])
            row_crop = seed_image[upper:lower,:]
            row_list.append(row_crop)

        cropped_in_y = np.array(row_list) # row x y x x
        return cropped_in_y

    def get_smoothed_x_percentiles(self,file_idx,seed_image_idx,orientation_and_initend_future,padding_y,trench_len_y,x_percentile,background_kernel_x,smoothing_kernel_x):
                
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
        cropped_in_y = self.crop_y_single(file_idx,seed_image_idx,orientation_and_initend_future,padding_y,trench_len_y)
    
        x_percentiles_smoothed = []
        for row_num in range(cropped_in_y.shape[0]):
            cropped_in_y_seg = cropped_in_y[row_num] # t x y x x   
            x_percentiles = np.percentile(cropped_in_y_seg,x_percentile,axis=0) # t x x  
            x_background_filtered = x_percentiles - self.median_filter(x_percentiles,background_kernel_x)
            x_smooth_filtered = self.median_filter(x_background_filtered,smoothing_kernel_x)
            x_smooth_filtered[x_smooth_filtered<0.] = 0.
            x_percentiles_smoothed.append(x_smooth_filtered)
        x_percentiles_smoothed=np.array(x_percentiles_smoothed) # row x x
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
        """Given an x percentile array of shape (rows,x), determines the trench midpoints of each row array
        at each time.
        
        Args:
            x_percentiles_smoothed_array (array): A smoothed and background subtracted percentile array of shape (rows,x)
            otsu_nbins (int): Number of bins to use when applying Otsu's method to x-dimension signal.
            otsu_scaling (float): Threshold scaling factor for Otsu's method thresholding.
        
        Returns:
            list: A nested list of the form [row_list,[midpoint_array]].
        """
        all_midpoints_list = []
        for row in range(x_percentiles_smoothed.shape[0]):
            row_x_percentiles = x_percentiles_smoothed[row]
            midpoints = self.get_x_row_midpoints(row_x_percentiles,otsu_nbins,otsu_scaling)
            if len(midpoints) == 0:
                return None
            all_midpoints_list.append(midpoints)
        return all_midpoints_list

    def get_in_bounds(self,row_midpoints,x_drift,trench_width_x,trench_present_thr):
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
        midpoints_time = np.tile(row_midpoints, (x_drift.shape[0], 1))
        corrected_midpoints = np.round(midpoints_time + x_drift[:, None]).astype(int)
      
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
#         counting_arr = self.init_counting_arr(self.metadata["width"])
                
#         kymo_mask = self.get_trench_mask(in_bounds,counting_arr)
        x_coords = in_bounds[0].T
        return in_bounds,x_coords,k_tot
    
    def get_all_in_bounds(self,seed_image_idx,midpoint_futures,drift_future,trench_width_x,trench_present_thr):
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

        x_drift = drift_future[:,0]
        in_bounds_list = []
        x_coords_list = []
        k_tot_list = []
        
        for row_midpoints in midpoint_futures:
            in_bounds,x_coords,k_tot = self.get_in_bounds(row_midpoints,x_drift,trench_width_x,trench_present_thr)
            in_bounds_list.append(in_bounds)
            x_coords_list.append(x_coords)
            k_tot_list.append(k_tot)
        
        return in_bounds_list,x_coords_list,k_tot_list
    
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
            
    def crop_x(self,file_idx,orientation_and_initend_future,in_bounds_future,padding_y,trench_len_y):
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
        
        channel_arr_list,lane_y_coords_list = self.crop_y(file_idx,orientation_and_initend_future,padding_y,trench_len_y)
        num_rows = channel_arr_list[0].shape[1]
        
        in_bounds_list,x_coords_list,k_tot_list = in_bounds_future
        counting_arr = self.init_counting_arr(self.metadata["width"])
        
        with h5py_cache.File(self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5","w",chunk_cache_mem_size=self.output_chunk_cache_mem_size) as output_kymograph:        
            for row_num in range(num_rows):
                in_bounds,k_tot = (in_bounds_list[row_num],k_tot_list[row_num])                
                kymo_mask = self.get_trench_mask(in_bounds[:,first_idx:last_idx+1],counting_arr)
#                 kymo_mask = kymo_mask[:,first_idx:last_idx+1]
                self.crop_with_k_masks(output_kymograph,channel_arr_list,kymo_mask,k_tot,row_num)
                
        return lane_y_coords_list
                
    def save_coords(self,fov_idx,x_crop_futures,in_bounds_future,orientation_and_initend_future):
        fovdf = self.meta_handle.read_df("global",read_metadata=False)
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        fovdf = fovdf.loc[fov_idx]

        x_coords_list = in_bounds_future[1]
        orientations = orientation_and_initend_future[0]

        y_coords_list = []        
        for j,file_idx in enumerate(fovdf["File Index"].unique().tolist()):
            working_filedf = fovdf[fovdf["File Index"]==file_idx]
            img_indices = working_filedf["Image Index"].unique()
            first_idx,last_idx = (img_indices[0],img_indices[-1])
            y_coords_list += x_crop_futures[j][first_idx:last_idx+1] # t x row list

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
        temp_meta_handle = pandas_hdf5_handler(self.kymographpath + "/temp_metadata_" + str(fov_idx) + ".hdf5")
        temp_meta_handle.write_df("temp",df)

    def generate_kymographs(self,dask_controller):
        writedir(self.kymographpath,overwrite=True)
        
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
        fov_first_file_index = [fovdf.loc[fov_idx]["File Index"].unique().tolist()[0] for fov_idx in fov_list]
        
        ### Find seed image for each file index (i.e. an image that we know is good)
        for file_idx in file_list:
            future = dask_controller.daskclient.submit(self.find_seed_image, file_idx)
            dask_controller.futures["Seed Image Index: " + str(file_idx)] = future
        
        ### Find drift over time
        for file_idx in file_list:
            seed_idx_future = dask_controller.futures["Seed Image Index: " + str(file_idx)]
            future = dask_controller.daskclient.submit(self.get_drifts, file_idx, seed_idx_future)
            dask_controller.futures["Drift: " + str(file_idx)] = future

        ### Link drifts across files
        for fov_idx in fov_list:
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()        
            within_file_drift_futures = [dask_controller.futures["Drift: " + str(file_idx)] for file_idx in working_files]
            seed_image_index_futures = [dask_controller.futures["Seed Image Index: " + str(file_idx)] for file_idx in working_files]
            future = dask_controller.daskclient.submit(self.link_drifts,working_files, seed_image_index_futures, within_file_drift_futures,retries=1)                
            dask_controller.futures["FoV Drifts: " + str(fov_idx)] = future

        ### smoothed y percentiles ###
        
        for k, fov_idx in enumerate(fov_list):
            first_file_idx = fov_first_file_index[k]
            seed_image_future = dask_controller.futures["Seed Image Index: " + str(first_file_idx)]
            future = dask_controller.daskclient.submit(self.get_smoothed_y_percentiles,first_file_idx,seed_idx_future,\
                                        self.y_percentile,self.smoothing_kernel_y,retries=1)
            dask_controller.futures["Smoothed Y Percentiles: " + str(fov_idx)] = future
            
        ### get trench row edges ###
        
        for fov_idx in fov_list:
            smoothed_y_future = dask_controller.futures["Smoothed Y Percentiles: " + str(fov_idx)]            
            future = dask_controller.daskclient.submit(self.get_trench_edges_y,smoothed_y_future,self.triangle_nbins,\
                                                       self.triangle_scaling,self.triangle_max_threshold,self.triangle_min_threshold,\
                                                       self.y_min_edge_dist,retries=1)#,priority=priority)
            
            dask_controller.futures["Y Trench Edges: " + str(fov_idx)] = future   
        
        ### get orientations, init edges ###

        for k, fov_idx in enumerate(fov_list):
            edges_future = dask_controller.futures["Y Trench Edges: " + str(fov_idx)]
            drift_future = dask_controller.futures["FoV Drifts: " + str(fov_idx)]
            seed_image_future = dask_controller.futures["Seed Image Index: " + str(first_file_idx)]
            future = dask_controller.daskclient.submit(self.get_ends_and_orientations, seed_image_future, drift_future, edges_future, self.expected_num_rows,\
                                                       self.top_orientation,self.orientation_on_fail,self.y_min_edge_dist,self.padding_y,self.trench_len_y,retries=1)#,priority=priority)                
            dask_controller.futures["Trench Orientations and Initial Trench Ends: " + str(fov_idx)] = future
                        
        ### smoothed x percentiles ###
        
        for k,fov_idx in enumerate(fov_list):
            first_file_idx = fov_first_file_index[k]
            seed_image_future = dask_controller.futures["Seed Image Index: " + str(first_file_idx)]
            orientation_and_initend_future = dask_controller.futures["Trench Orientations and Initial Trench Ends: " + str(fov_idx)]
            future = dask_controller.daskclient.submit(self.get_smoothed_x_percentiles, first_file_idx, seed_image_future, orientation_and_initend_future,\
                                                       self.padding_y,self.trench_len_y,self.x_percentile,self.background_kernel_x,\
                                                       self.smoothing_kernel_x,retries=1)#,priority=priority)
            dask_controller.futures["Smoothed X Percentiles: " + str(fov_idx)] = future
            
        ### get x midpoints ###

        for fov_idx in fov_list:
            smoothed_x_future = dask_controller.futures["Smoothed X Percentiles: " + str(fov_idx)]
            future = dask_controller.daskclient.submit(self.get_x_midpoints,smoothed_x_future,\
                                                       self.otsu_nbins,self.otsu_scaling,retries=1)#,priority=priority)
            dask_controller.futures["X Midpoints: " + str(fov_idx)] = future
        
        ### get kymograph masks ###

        for k,fov_idx in enumerate(fov_list):
            midpoint_futures = dask_controller.futures["X Midpoints: " + str(fov_idx)]
            drift_future = dask_controller.futures["FoV Drifts: " + str(fov_idx)]
            seed_image_future = dask_controller.futures["Seed Image Index: " + str(first_file_idx)]
            future = dask_controller.daskclient.submit(self.get_all_in_bounds,seed_image_future,midpoint_futures,drift_future,\
                                                self.trench_width_x,self.trench_present_thr,retries=1)#,priority=priority)
            dask_controller.futures["X In Bounds: " + str(fov_idx)] = future
            
        ### crop in x ###
            
        for k,file_idx in enumerate(file_list):
            working_filedf = filedf.loc[file_idx]
            fov_idx = working_filedf["fov"].unique().tolist()[0]
            drift_future = dask_controller.futures["FoV Drifts: " + str(fov_idx)]
            orientation_and_initend_future = dask_controller.futures["Trench Orientations and Initial Trench Ends: " + str(fov_idx)]
            in_bounds_future = dask_controller.futures["X In Bounds: " + str(fov_idx)]
            
            future = dask_controller.daskclient.submit(self.crop_x,file_idx,orientation_and_initend_future,in_bounds_future,self.padding_y,self.trench_len_y,retries=0)#,priority=priority)
            dask_controller.futures["X Crop: " + str(file_idx)] = future
            
        ### get coords ###
        
        for k,fov_idx in enumerate(fov_list):
            working_fovdf = fovdf.loc[fov_idx]
            working_files = working_fovdf["File Index"].unique().tolist()
            x_crop_futures = [dask_controller.futures["X Crop: " + str(file_idx)] for file_idx in working_files]
            in_bounds_future = dask_controller.futures["X In Bounds: " + str(fov_idx)]
            orientation_and_initend_future = dask_controller.futures["Trench Orientations and Initial Trench Ends: " + str(fov_idx)]

            future = dask_controller.daskclient.submit(self.save_coords,fov_idx,x_crop_futures,in_bounds_future,orientation_and_initend_future,retries=1)#,priority=priority)
            dask_controller.futures["Coords: " + str(fov_idx)] = future
            
    def collect_metadata(self,dask_controller):
        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        fovdf = fovdf.loc[(slice(None), slice(self.t_range[0],self.t_range[1])),:]
        fov_list = fovdf.index.get_level_values("fov").unique().values        
        
        completed_list = []
        for filename in os.listdir(self.kymographpath):
            if "temp_metadata" in filename:
                filename_list = filename.split("_")
                endstr = filename_list[-1]
                idx = int(endstr.split(".")[0])             
                completed_list.append(idx)
                
        df_list = []
        for fov_idx in completed_list:
            temp_meta_path = self.kymographpath + "/temp_metadata_" + str(fov_idx) + ".hdf5"
            temp_meta_handle = pandas_hdf5_handler(temp_meta_path)
            temp_df = temp_meta_handle.read_df("temp")
            df_list.append(temp_df)
            os.remove(temp_meta_path)
        
        df_out = pd.concat(df_list)
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
                trenchids = trenchid_list[k*self.trenches_per_file:(k+1)*self.trenches_per_file]
                working_trenchdf = trenchiddf.loc[trenchids]
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
                
    def cleanup_kymographs(self,reorg_futures,file_list):
        for file_idx in file_list:
            proc_file_path = self.kymographpath+"/kymograph_processed_"+str(file_idx)+".hdf5"
            os.remove(proc_file_path)

    def reorg_all_kymographs(self,dask_controller):
        fovdf = self.meta_handle.read_df("temp_kymograph",read_metadata=True)
        file_list = fovdf["File Index"].unique().tolist()
        metadata = fovdf.metadata
        trenchiddf = fovdf.reset_index(inplace=False)
        trenchiddf = trenchiddf.set_index(["trenchid","timepoints"], drop=True, append=False, inplace=False)
        trenchiddf = trenchiddf.sort_index()
        trenchid_list = trenchiddf.index.get_level_values("trenchid").unique().tolist()

        outputdf = trenchiddf.drop(columns = ["File Index","Image Index"])
        num_tpts = len(outputdf.index.get_level_values("timepoints").unique().tolist())
        chunk_size = self.trenches_per_file*num_tpts
        if len(trenchid_list)%self.trenches_per_file == 0:
            num_files = (len(trenchid_list)//self.trenches_per_file)
        else:
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
            
        reorg_futures = [dask_controller.futures["Kymograph Reorganized: " + str(k)] for k in range(num_files)]
        future = dask_controller.daskclient.submit(self.cleanup_kymographs,reorg_futures,file_list,retries=1,priority=priority)
        dask_controller.futures["Kymographs Cleaned Up"] = future

    def post_process(self,dask_controller):
        dask_controller.daskclient.restart()
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
        # Normalize (scale by range and subtract minimum) to make scaling of thresholds make more sense
        min_qth_percentile = y_percentiles_smoothed.min(axis=0)
        max_qth_percentile = y_percentiles_smoothed.max(axis=0)
        y_percentiles_smoothed = (y_percentiles_smoothed - min_qth_percentile)/(max_qth_percentile - min_qth_percentile)
        return y_percentiles_smoothed
    
    def triangle_threshold(self,img_arr,triangle_nbins,triangle_scaling,triangle_max_threshold,triangle_min_threshold):
        """Applys a triangle threshold to each timepoint in a (y,t) input array, returning a boolean mask.
        
        Args:
            img_arr (array): Image array to be thresholded.
            triangle_nbins (int): Number of bins to be used to construct the thresholding
            histogram.
            triangle_scaling (float): Factor by which to scale the threshold.
        
        Returns:
            array: Boolean mask produced by the threshold.
        """
        all_thresholds = np.apply_along_axis(sk.filters.threshold_triangle,0,img_arr,nbins=triangle_nbins)*triangle_scaling #(t,) array
        
        thresholds_above_min = all_thresholds > triangle_min_threshold
        thresholds_below_max = all_thresholds < triangle_max_threshold
        all_thresholds[~thresholds_above_min] = triangle_min_threshold
        all_thresholds[~thresholds_below_max] = triangle_max_threshold
        
        triangle_mask = img_arr>all_thresholds
        return triangle_mask,all_thresholds
    
    def get_edges_from_mask(self,mask):
        """Finds edges from a boolean mask of shape (y,t). Filters out rows of length
        smaller than y_min_edge_dist.
        
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
    
    def get_trench_edges_y(self,i,y_percentiles_smoothed_list,triangle_nbins,triangle_scaling,triangle_max_threshold,triangle_min_threshold):
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
        trench_mask_y,_ = self.triangle_threshold(y_percentiles_smoothed,triangle_nbins,triangle_scaling,triangle_max_threshold,triangle_min_threshold)
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
        
#         if start_above and orientations[0] == 0: #if the top is facing down and is cut
#             orientations = orientations[1:]
#             repaired_trench_edges_y = repaired_trench_edges_y[2:]
#         if end_above and orientations[-1] == 1: #if the bottom is facing up and is cut
#             orientations = orientations[:-1]
#             repaired_trench_edges_y = repaired_trench_edges_y[:-2]
#         return orientations,repaired_trench_edges_y
    
#     def assign_orientation(self,orientations,repaired_trench_edges_y,start_above,end_above):
#         """Takes an array of trench row edges and removes the first/last
#         edge, if that edge does not have a proper partner (i.e. trench row mask
#         takes value True at boundaries of image).
        
#         Args:
#             edges (array): Array of edges along y-axis.
#             start_above (bool): True if the trench row mask takes value True at the
#             starting edge of the mask.
#             end_above (bool): True if the trench row mask takes value True at the
#             ending edge of the mask.
        
#         Returns:
#             array: Array of edges along y-axis, corrected for edge pairs that
#             are out of frame.
#         """
#         drop_first_row,drop_last_row = (False,False)
#         if start_above and orientations[0] == 0: #if the top is facing down and is cut
#             drop_first_row = True
#             orientations = orientations[1:]
#             repaired_trench_edges_y = repaired_trench_edges_y[2:]
#         if end_above and orientations[-1] == 1: #if the bottom is facing up and is cut
#             drop_last_row = True
#             orientations = orientations[:-1]
#             repaired_trench_edges_y = repaired_trench_edges_y[:-2]
#         return orientations,drop_first_row,drop_last_row,repaired_trench_edges_y
    
    def get_manual_orientations(self,i,trench_edges_y_lists,start_above_lists,end_above_lists,\
                                expected_num_rows,top_orientation,orientation_on_fail,y_min_edge_dist):
        trench_edges_y = trench_edges_y_lists[i][0]
        start_above = start_above_lists[i][0]
        end_above = end_above_lists[i][0]
        orientations = []
        
        repaired_trench_edges_y = self.repair_out_of_frame(trench_edges_y,start_above,end_above)
        repaired_trench_edges_y = self.remove_small_rows(repaired_trench_edges_y,y_min_edge_dist)
        
#         trench_edges_y_no_drift = self.remove_out_of_frame(top_orientation,trench_edges_y,start_above,end_above)
#         trench_edges_y_no_drift = self.remove_small_rows(trench_edges_y_no_drift,y_min_edge_dist)
        
        if repaired_trench_edges_y.shape[0]//2 == expected_num_rows:
            orientation = top_orientation
            for row in range(repaired_trench_edges_y.shape[0]//2):
                orientations.append(orientation)
                orientation = (orientation+1)%2
            orientations,drop_first_row,drop_last_row,repaired_trench_edges_y = self.remove_out_of_frame(orientations,repaired_trench_edges_y,start_above,end_above)
                
        elif (repaired_trench_edges_y.shape[0]//2 < expected_num_rows) and orientation_on_fail is not None:
            orientation = orientation_on_fail
            for row in range(repaired_trench_edges_y.shape[0]//2):
                orientations.append(orientation)
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

    def get_y_drift(self,i,y_ends_lists):
        """Given a list of midpoints, computes the average drift in y for every timepoint.

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
        
        valid_y_ends_list = []
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
                valid_y_ends_list.append([y_end[j] for y_end in y_ends_list])
                valid_orientations.append(orientation)
        
        valid_y_ends = np.array(valid_y_ends_list).T # t,edge
     
        return valid_y_ends,valid_orientations

# 
#         valid_edge_mask = []
#         valid_orientation_mask = []
#         for i in range(0,len(edge_in_bounds),2):
#             if np.all(edge_in_bounds[i:i+2]):
#                 valid_edge_mask+=[True,True]
#                 valid_orientation_mask+=[True]
#             else:
#                 valid_edge_mask+=[False,False]
#                 valid_orientation_mask+=[False]

#         valid_edges_y_list = [trench_edges_y[valid_edge_mask] for trench_edges_y in trench_edges_y_list]
        
    
#     def get_row_numbers(self,i,trench_edges_y_list):
#         """Computes the number of trench rows in the fov, from the detected edges.
        
#         Args:
#             i (int): Specifies the current fov index.
#             trench_edges_y_list (list): List containing, for each fov entry, a list of time-sorted edge arrays.
        
#         Returns:
#             int: The number of trench rows detected in the fov of index i.
#         """
#         trench_edges_y = trench_edges_y_list[i]
#         edge_num_list = [len(item) for item in trench_edges_y]
#         trench_row_num = (np.median(edge_num_list).astype(int))//2
#         return trench_row_num

    def crop_y(self,i,imported_array_list,y_drift_list,valid_y_ends_list,valid_orientations_list,padding_y,trench_len_y):
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
        valid_y_ends = valid_y_ends_list[i]
        valid_orientations = valid_orientations_list[i]

        drift_corrected_edges = np.add.outer(y_drift,valid_y_ends[0])
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
#                     output_array = np.pad(imported_array[c,upper:lower,:,t],((pad, 0),(0,0)),'constant')
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
        
#     def crop_trenches_in_y(self,imported_array_list):
#         """Master function for cropping the input hdf5 file in the y-dimension.
        
#         Args:
#             imported_array_list (list): List containing, for each fov entry, a numpy array containing
#             the corresponding hdf5 file image data.
        
#         Returns:
#             list: List containing, for each fov entry, a y-cropped numpy array of shape (rows,channels,x,y,t).
#         """        
#         y_percentiles_smoothed_list = self.map_to_fovs(self.get_smoothed_y_percentiles,imported_array_list,\
#                                                        self.y_percentile,self.smoothing_kernel_y)
        
#         get_trench_edges_y_output = self.map_to_fovs(self.get_trench_edges_y,y_percentiles_smoothed_list,self.triangle_nbins,self.triangle_scaling,self.triangle_max_threshold,self.triangle_min_threshold)
#         trench_edges_y_lists = [item[0] for item in get_trench_edges_y_output]
#         start_above_lists = [item[1] for item in get_trench_edges_y_output]
#         end_above_lists = [item[2] for item in get_trench_edges_y_output]
                
#         orientations_list = self.map_to_fovs(self.get_manual_orientations,trench_edges_y_lists,start_above_lists,end_above_lists,self.expected_num_rows,\
#                                              self.orientation_detection,self.orientation_on_fail,self.y_min_edge_dist)
        
#         y_ends_lists = self.map_to_fovs(self.get_trench_ends,trench_edges_y_lists,start_above_lists,end_above_lists,orientations_list,self.y_min_edge_dist)

#         y_drift_list = self.map_to_fovs(self.get_y_drift,y_ends_lists)
        
#         keep_in_frame_kernels_output = self.map_to_fovs(self.keep_in_frame_kernels,y_ends_lists,y_drift_list,imported_array_list,orientations_list,self.padding_y,self.trench_len_y)
#         valid_y_ends_lists = [item[0] for item in keep_in_frame_kernels_output]
#         valid_orientations_list = [item[1] for item in keep_in_frame_kernels_output]
        
#         cropped_in_y_list = self.map_to_fovs(self.crop_y,imported_array_list,y_drift_list,valid_y_ends_lists,orientations_list,self.padding_y,self.trench_len_y)
        
#         return cropped_in_y_list
    
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
#         x_mask = x_percentiles_t<otsu_threshold
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