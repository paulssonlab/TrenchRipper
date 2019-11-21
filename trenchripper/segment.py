import numpy as np
import skimage as sk
import h5py
import os
import copy
import pickle
import shutil
import cv2
import pandas as pd

from skimage import measure,feature,segmentation,future,util,morphology,filters
from .utils import kymo_handle,pandas_hdf5_handler,writedir
from .cluster import hdf5lock
from time import sleep
import scipy.ndimage.morphology as morph
import mahotas as mh
from dask.distributed import worker_client
from pandas import HDFStore


from matplotlib import pyplot as plt

class phase_segmentation:
    """ Segmentation algorithm for high mag phase images

    Attributes:
        init_niblack_k (float): k parameter for niblack thresholding of cells
        maxima_niblack_k (float): k parameter for more rigorous thresholding to determine
                        watershed seeds
        init_smooth_sigma (float): smoothing value for cell segmentation
        maxima_smooth_sigma (float): smoothing value for seed segmentation
        maxima_niblack_window_size (int): window size for niblack thresholding of seeds
        init_niblack_window_size (int): window size for niblack thresholding of cells
        min_cell_size (int): throw out mask regions smaller than this size
        deviation_from_median (float): Relative deviation from median cell size for outlier
                                detection
        max_perc_contrast (float):
        wrap_pad (int):

    """
    def __init__(self, init_niblack_k=-0.5, maxima_niblack_k=-0.6, init_smooth_sigma=3, maxima_smooth_sigma=2, init_niblack_window_size=13, maxima_niblack_window_size=13, min_cell_size=100, deviation_from_median=0.3, max_perc_contrast=97, wrap_pad = 0):
        
        self.init_niblack_k = -init_niblack_k
        self.maxima_niblack_k = -maxima_niblack_k
        self.init_smooth_sigma = init_smooth_sigma
        self.maxima_smooth_sigma = maxima_smooth_sigma
        self.maxima_niblack_window_size = maxima_niblack_window_size
        self.init_niblack_window_size = init_niblack_window_size
        self.min_cell_size = min_cell_size
        self.deviation_from_median = deviation_from_median
        self.max_perc_contrast= max_perc_contrast
        self.wrap_pad = 0
        
    def remove_large_objects(self, conn_comp, max_size=4000):
        """ Remove segmented regions that are too large

        Args:
            conn_comp(numpy.ndarray, int): Segmented connected components where each
                                        component is numbered
            max_size(int): Throw out regions where area is greater than this
        """
        out = np.copy(conn_comp)
        component_sizes = np.bincount(conn_comp.ravel())
        too_big = component_sizes > max_size
        too_big_mask = too_big[conn_comp]
        out[too_big_mask] = 0
        return out
    
    def to_8bit(self,img_arr,bit_max=None):
        img_max = np.max(img_arr)+0.0001
        if bit_max is None:
            max_val = img_max
        else:
            max_val = max(img_max,bit_max)
        min_val = np.min(img_arr)
        norm_array = (img_arr-min_val)/(max_val-min_val)
        norm_byte_array = sk.img_as_ubyte(norm_array)
        return norm_byte_array
    
    def preprocess_img(self,img_arr,sigma=1.,bit_max=0):
        img_smooth = copy.copy(img_arr)
        for t in range(img_arr.shape[0]):
#             img_smooth[t] = sk.filters.gaussian(img_arr[t],sigma=sigma,preserve_range=True,mode='reflect')
            img_smooth[t] = self.to_8bit(sk.filters.gaussian(img_arr[t],sigma=sigma,preserve_range=True,mode='reflect'),bit_max=bit_max)
        return img_smooth
    
    def detect_rough_trenches(self, img, show_plots=False):
        def fill_holes(img):
            seed = np.copy(img)
            seed[1:-1, 1:-1] = img.max()
            strel = sk.morphology.square(3, dtype=bool)
            img_filled = sk.morphology.reconstruction(seed, img, selem=strel, method='erosion')
            img_filled = img_filled.astype(bool)
            return img_filled
    
        max_val = np.percentile(img,self.max_perc_contrast)
        img_contrast = sk.exposure.rescale_intensity(img, in_range=(0,max_val))
        
        img_median = mh.median_filter(img_contrast,Bc=np.ones((2,2)))
#         img_median = cv2.medianBlur(img_contrast, 2)
        img_edges = sk.filters.sobel(img_median)

        T_otsu = sk.filters.threshold_otsu(img_edges)
        img_otsu = (img_edges > T_otsu).astype(bool)
        img_otsu = img_otsu
#         img_otsu = cv2.threshold(img_edges,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        img_close = morph.binary_closing(img_otsu, structure = np.ones((3,3)),iterations=6)
#         img_close = img_otsu
#         strel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         for i in range(6):
#             img_close = cv2.morphologyEx(img_close, cv2.MORPH_CLOSE, strel)

        img_filled = fill_holes(img_close)
        img_open = morph.binary_opening(img_filled, structure = np.ones((3,3)),iterations=2)
#         img_open = img_filled
#         strel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         for i in range(6):
#             img_open = cv2.morphologyEx(img_open, cv.MORPH_OPEN, strel)
#         strel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,1))
#         trench_masks = cv2.dilate(img_open, strel)
        trench_masks = morph.binary_dilation(img_open,structure = np.ones((9,1)))
        
        
        if show_plots:        
            fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(8,10))
            ax1.imshow(img)
            ax2.imshow(img_edges)
            ax3.imshow(img_otsu)
            ax4.imshow(img_close)
            ax5.imshow(img_open)
            ax6.imshow(trench_masks)
        return trench_masks

    def detect_trenches(self, img, show_plots=False):
        trench_masks = self.detect_rough_trenches(img, show_plots)
#         strel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
#         for i in range(6):
#             img_open = cv2.morphologyEx(img_open, cv2.MORPH_OPEN, strel)
        trench_masks = morph.binary_closing(trench_masks,structure=np.ones((15,1)),iterations=6)
        trench_masks = morph.binary_erosion(trench_masks,structure=np.ones((3,3)),iterations=4)
        conn_comp = sk.measure.label(trench_masks,neighbors=4)
        rp_area = [r.area for r in sk.measure.regionprops(conn_comp, cache=False)]
        if len(rp_area) > 0:
            max_region = np.argmax(rp_area)+1
            trench_masks[conn_comp != max_region] = 0
        return trench_masks
    
    def medianFilter(self, img):
        img_median = mh.median_filter(img,Bc=np.ones((3,3)))
        return img_median
    
    def get_trench_mask_and_median_filter(self, img):
        return self.detect_trenches(img), self.medianFilter(img)
    
    def findCellsInTrenches(self, img, mask,sigma,window_size,niblack_k):
        img_smooth = img

        if sigma > 0:    
            img_smooth = sk.filters.gaussian(img,sigma=sigma,preserve_range=True,mode='reflect')

        thresh_niblack = sk.filters.threshold_niblack(img_smooth, window_size = window_size, k= niblack_k)
        threshed = (img <= thresh_niblack).astype(bool)
        if mask is not None:
            threshed = threshed*mask
        return threshed
    
    def findWatershedMaxima(self, img,mask):
        maxima = self.findCellsInTrenches(img,mask,self.maxima_smooth_sigma,self.maxima_niblack_window_size,self.maxima_niblack_k)
    #         maxima = mh.erode(maxima,Bc=np.ones((7,5))) #Initial
    #         maxima = mh.erode(maxima,Bc=np.ones((5,3)))
        reg_props = sk.measure.regionprops(sk.measure.label(maxima,neighbors=4), cache=False)
            # make sure that in case there is more than one region, we grab the largest region
        rp_area = [r.area for r in reg_props]
        med_size = np.median(rp_area)
        std_size = np.std(rp_area)
        cutoff_size = int(max(0,med_size/6))

        maxima = sk.morphology.remove_small_objects(maxima,min_size=cutoff_size)
        return maxima

    def findWatershedMask(self, img,mask):
        img_mask = self.findCellsInTrenches(img,mask,self.init_smooth_sigma,self.init_niblack_window_size,self.init_niblack_k)
        img_mask = mh.dilate(mh.dilate(img_mask),Bc=np.ones((1,3),dtype=np.bool))
        img_mask = sk.morphology.remove_small_objects(img_mask,min_size=4)
        return img_mask

    def extract_connected_components_phase(self, threshed, maxima, return_all = False, show_plots=False):

        """
        phase segmentation and connected components detection algorithm

        :param img: numpy array containing image
        :param trench_masks: you can supply your own trench_mask rather than computing it each time
        :param flip_trenches: if mothers are on bottom of image set to True
        :param cut_from_bottom: how far to crop the bottom of the detected trenches to avoid impacting segmentation
        :param above_trench_pad: how muching padding above the mother cell
        :param init_smooth_sigma: how much smoothing to apply to the image for the initial niblack segmentation
        :param init_niblack_window_size: size of niblack window for segmentation
        :param init_niblack_k: k-offset for initial niblack segmentation
        :param maxima_smooth_sigma: how much smoothing to use for image that determines maxima used to seed watershed
        :param maxima_niblack_window_size: size of niblack window for maxima determination
        :param maxima_niblack_k: k-offset for maxima determination using niblack
        :param min_cell_size: minimum size of cell in pixels
        :param max_perc_contrast: scale contrast before median filter application
        :param return_all: whether just the connected component or the connected component, 
        thresholded image pre-watershed and maxima used in watershed
        :return: 
            if return_all = False: a connected component matrix
            if return_all = True: connected component matrix, 
            thresholded image pre-watershed and maxima used in watershed

        """

        distances = mh.stretch(mh.distance(threshed))
        spots, n_spots = mh.label(maxima,Bc=np.ones((3,3)))
        surface = (distances.max() - distances)
        conn_comp = sk.morphology.watershed(surface, spots, mask=threshed)
        conn_comp = sk.measure.label(conn_comp,neighbors=4)
        conn_comp = sk.morphology.remove_small_objects(conn_comp,min_size=self.min_cell_size)
        
        if show_plots:        
            fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(5,10))
            ax1.imshow(distances)
            ax2.imshow(maxima)
            ax3.imshow(spots)
            ax4.imshow(conn_comp)
        return conn_comp
        
    def segment(self, img, return_all=False, show_plots=False):
        trench_masks = self.detect_trenches(img, show_plots)
#         loaded = (img < sk.filters.threshold_otsu(img)) * trench_masks
#         denom = np.sum(trench_masks)
#         loaded_fraction = loaded/denom
        img_median = mh.median_filter(img,Bc=np.ones((3,3)))
        img_mask = self.findWatershedMask(img_median,trench_masks)
        maxima = self.findWatershedMaxima(img_median,img_mask)
        conn_comp = self.extract_connected_components_phase(img_mask, maxima, show_plots=show_plots)
        
        if return_all:
            return conn_comp, trench_masks, img_mask, maxima
        else:
            return conn_comp, loaded_fraction
    
    def find_trench_masks_and_median_filtered_list(self, trench_array_list):
        trench_masks_list = np.empty_like(trench_array_list, dtype=bool)
        img_medians_list = np.empty_like(trench_array_list, dtype=np.uint8)
        for tr in range(trench_array_list.shape[0]):
            for t in range(trench_array_list.shape[1]):
                trench_masks_list[tr,t,:,:] = self.detect_trenches(trench_array_list[tr,t,:,:])
                img_medians_list[tr,t,:,:] = self.medianFilter(trench_array_list[tr,t,:,:])
        return trench_masks_list, img_medians_list
    
    def find_watershed_mask_list(self, trench_mask_and_median_filtered_list):
        trench_mask_array_list, img_array_list = trench_mask_and_median_filtered_list
        watershed_masks_list = np.empty_like(img_array_list, dtype=bool)
        for tr in range(img_array_list.shape[0]):
            for t in range(img_array_list.shape[1]):
                watershed_masks_list[tr,t,:,:] = self.findWatershedMask(img_array_list[tr,t,:,:], trench_mask_array_list[tr,t,:,:])
        return watershed_masks_list, img_array_list
        
    def find_watershed_maxima_list(self, watershed_mask_and_median_filtered_list):
        masks_list, img_array_list = watershed_mask_and_median_filtered_list
        watershed_maxima_list = np.empty_like(img_array_list, dtype=bool)
        for tr in range(img_array_list.shape[0]):
            for t in range(img_array_list.shape[1]):
                watershed_maxima_list[tr,t,:,:] = self.findWatershedMaxima(img_array_list[tr,t,:,:], masks_list[tr,t,:,:])
        return watershed_maxima_list, masks_list
    
    def find_conn_comp_list(self, watershed_maxima_and_masks_list): #masks_arr is tr,t,y,x
#         input_kymo = kymo_handle()
#         input_kymo.import_wrap(img_arr,scale=self.scale_timepoints,scale_perc=self.scaling_percentage)
        maxima_list, masks_list = watershed_maxima_and_masks_list
        final_masks_list = np.empty_like(masks_list, dtype=np.uint8)
        for tr in range(masks_list.shape[0]):
            for t in range(masks_list.shape[1]):
                final_masks_list[tr,t,:,:] = self.extract_connected_components_phase(masks_list[tr,t,:,:], maxima_list[tr,t,:,:])
        return final_masks_list
    
    def get_trench_loading_fraction(self, img, t_idx=-1, trench_idx=-1, file_idx=-1):
        try:
            trench_mask = self.detect_trenches(img)
        except ValueError as e:
            trench_mask = self.detect_trenches(img, show_plots=True)
            raise Exception("ValueError happened detecting trench at f%dtr%dt%d, %s" %(file_idx, trench_idx, t_idx, str(e)))
        try:
            loaded = (img < sk.filters.threshold_otsu(img)) * trench_mask
        except ValueError as e:
            raise Exception("ValueError happened applying otsu threshold at f%dtr%dt%d, %s" %(file_idx, trench_idx, t_idx, str(e)))
        denom = np.sum(trench_mask)
        if denom == 0:
            return 0
        else:
            return np.sum(loaded)/denom
    
    def measure_trench_loading(self, trench_array_list, file_idx=-1):
        trench_loadings = np.empty((trench_array_list.shape[0], trench_array_list.shape[1]))
        for tr in range(trench_array_list.shape[0]):
            for t in range(trench_array_list.shape[1]):
                trench_loadings[tr, t] = self.get_trench_loading_fraction(trench_array_list[tr, t, :, :], t_idx=t, trench_idx=tr, file_idx=file_idx)
        trench_loadings = trench_loadings.flatten()
        return trench_loadings

class fluo_segmentation:
    def __init__(self,scale_timepoints=False,scaling_percentage=0.9,smooth_sigma=0.75,bit_max=0,wrap_pad=3,hess_pad=6,min_obj_size=30,cell_mask_method='global',global_threshold=1000,\
                 cell_otsu_scaling=1.,local_otsu_r=15,edge_threshold_scaling=1.,threshold_step_perc=0.1,threshold_perc_num_steps=2,convex_threshold=0.8):
        
        self.scale_timepoints=scale_timepoints
        self.scaling_percentage=scaling_percentage
        self.smooth_sigma = smooth_sigma
        self.bit_max = bit_max
        self.wrap_pad = wrap_pad
        self.hess_pad = hess_pad
        self.min_obj_size = min_obj_size
        self.global_threshold = global_threshold
        self.cell_mask_method = cell_mask_method
        self.cell_otsu_scaling = cell_otsu_scaling
        self.local_otsu_r = local_otsu_r
        self.edge_threshold_scaling = edge_threshold_scaling
        self.threshold_step_perc = threshold_step_perc
        self.threshold_perc_num_steps = threshold_perc_num_steps
        self.convex_threshold = convex_threshold
    
    def scale_kymo(self,wrap_arr,percentile):
        perc_t = np.percentile(wrap_arr[:].reshape(wrap_arr.shape[0],-1),percentile,axis=1)
        norm_perc_t = perc_t/np.max(perc_t)
        scaled_arr = wrap_arr.astype(float)/norm_perc_t[:,np.newaxis,np.newaxis]
        scaled_arr[scaled_arr>255.] = 255.
        scaled_arr = scaled_arr.astype("uint8")
        return scaled_arr
    
    def to_8bit(self,img_arr,bit_max=None):
        img_max = np.max(img_arr)+0.0001
        if bit_max is None:
            max_val = img_max
        else:
            max_val = max(img_max,bit_max)
        min_val = np.min(img_arr)
#         min_val = np.min(img_arr)
        norm_array = (img_arr-min_val)/(max_val-min_val)
        norm_byte_array = sk.img_as_ubyte(norm_array)
        return norm_byte_array
    
    def preprocess_img(self,img_arr,sigma=1.,bit_max=0,scale_timepoints=False,scaling_percentage=None):
        img_smooth = copy.copy(img_arr)
        for t in range(img_arr.shape[0]):
            img_smooth[t] = self.to_8bit(sk.filters.gaussian(img_arr[t],sigma=sigma,preserve_range=True,mode='reflect'),bit_max=bit_max)
        if scale_timepoints:
            img_smooth = self.scale_kymo(img_smooth,scaling_percentage)
        
        return img_smooth

#     def cell_region_mask(self,img_arr,method='global',global_scaling=1.,cell_otsu_scaling=1.,local_otsu_r=15):
    def cell_region_mask(self,img_arr,method='global',global_threshold=1000,cell_otsu_scaling=1.,local_otsu_r=15):
        global_mask_kymo = []
        for t in range(img_arr.shape[0]):
            cell_mask = img_arr[t,:,:]>global_threshold
            global_mask_kymo.append(cell_mask)
        global_mask_kymo = np.array(global_mask_kymo)
                    
        if method == 'global':
            return global_mask_kymo
                    
        elif method == 'local':
            otsu_selem = sk.morphology.disk(local_otsu_r)
            local_mask_kymo = []
            for t in range(img_arr.shape[0]):
                above_threshold = np.any(global_mask_kymo[t,:,:]) # time saving
                if above_threshold:
                    local_thr_arr = sk.filters.rank.otsu(img_arr[t,:,:], otsu_selem)
                    local_mask = img_arr[t,:,:]>local_thr_arr
                else:
                    local_mask = np.zeros(img_arr[t,:,:].shape,dtype=bool)
                local_mask_kymo.append(local_mask)
            local_mask_kymo = np.array(local_mask_kymo)

            final_cell_mask = global_mask_kymo*local_mask_kymo
            del global_mask_kymo
            del local_mask_kymo
            return final_cell_mask
            
        else:
            print("no valid cell threshold method chosen!!!")
                
    def hessian_contrast_enc(self,img_arr,edge_padding=0):
        img_arr = np.pad(img_arr, edge_padding, 'reflect')
        hessian = sk.feature.hessian_matrix(img_arr,order="rc")
        eigvals = sk.feature.hessian_matrix_eigvals(hessian)
        min_eigvals = np.min(eigvals,axis=0)
        if edge_padding>0:
            min_eigvals = min_eigvals[edge_padding:-edge_padding,edge_padding:-edge_padding]
        return min_eigvals
    
    def find_mask(self,cell_local_mask,min_eigvals,edge_threshold,min_obj_size=30):
        edge_mask = min_eigvals>edge_threshold
        composite_mask = cell_local_mask*edge_mask
        composite_mask = sk.morphology.remove_small_objects(composite_mask,min_size=min_obj_size)
        composite_mask = sk.morphology.remove_small_holes(composite_mask)
        return composite_mask
    
    def compute_convexity(self,curr_obj):    
        area = np.sum(curr_obj)
        convex_hull = sk.morphology.convex_hull_image(curr_obj)
        convex_hull_area = np.sum(convex_hull)
        convexity = area/convex_hull_area
        return convexity
    
    def get_object_coords(self,obj_thresh,padding=1):    
        x_dim_max = np.max(obj_thresh,axis=0)
        x_indices = np.where(x_dim_max)[0]
        x_min = max(np.min(x_indices)-padding,0)
        x_max = min(np.max(x_indices)+padding+1,obj_thresh.shape[1])

        y_dim_max = np.max(obj_thresh,axis=1)
        y_indices = np.where(y_dim_max)[0]
        y_min = max(np.min(y_indices)-padding,0)
        y_max = min(np.max(y_indices)+padding+1,obj_thresh.shape[0])

        return x_min,x_max,y_min,y_max
    
    def crop_object(self,conn_comp,obj_idx,padding=1):
        obj_thresh = (conn_comp==obj_idx)
        x_min,x_max,y_min,y_max = self.get_object_coords(obj_thresh,padding=padding)
        curr_obj = obj_thresh[y_min:y_max,x_min:x_max]
        return curr_obj

    def get_image_weights(self,conn_comp,padding=1):
        objects = np.unique(conn_comp)[1:]
        conv_weights = []
        for obj_idx in objects:
            curr_obj = self.crop_object(conn_comp,obj_idx,padding=padding)
            conv_weight = self.compute_convexity(curr_obj)            
            conv_weights.append(conv_weight)
        return np.array(conv_weights)
    
    def make_weight_arr(self,conn_comp,weights):
        weight_arr = np.zeros(conn_comp.shape)
        for i,obj_weight in enumerate(weights):
            obj_idx = i+1
            obj_mask = (conn_comp==obj_idx)
            weight_arr[obj_mask] = obj_weight
        return weight_arr

    def get_mid_threshold_arr(self,wrap_eig,edge_threshold_scaling=1.,padding=3):
        edge_threshold_kymo = []
        for t in range(wrap_eig.shape[0]):
            edge_threshold = sk.filters.threshold_otsu(wrap_eig[t])
            edge_thr_arr = edge_threshold*np.ones(wrap_eig.shape[1:],dtype='uint8')
            edge_threshold_kymo.append(edge_thr_arr)
        edge_threshold_kymo = np.array(edge_threshold_kymo)*edge_threshold_scaling
#         edge_threshold_kymo = np.moveaxis(edge_threshold_kymo,(0,1,2),(2,0,1))

        edge_thr_kymo = kymo_handle()
        edge_thr_kymo.import_wrap(edge_threshold_kymo,scale=False)
        mid_threshold_arr = edge_thr_kymo.return_unwrap(padding=padding)
        return mid_threshold_arr

    def get_scores(self,cell_mask,min_eigvals,mid_threshold_arr,threshold_step_perc=0.05,threshold_perc_num_steps=2,min_obj_size=30):
        threshold_percs = [1. + threshold_step_perc*step for step in range(-threshold_perc_num_steps,threshold_perc_num_steps+1)]
        edge_thresholds = [mid_threshold_arr*threshold_perc for threshold_perc in threshold_percs]
        conv_weight_arrs = []
        for edge_threshold in edge_thresholds:
            composite_mask = self.find_mask(cell_mask,min_eigvals,edge_threshold,min_obj_size=min_obj_size)
            conn_comp = sk.measure.label(composite_mask,neighbors=4,connectivity=2)
            conv_weights = self.get_image_weights(conn_comp)
            conv_weight_arr = self.make_weight_arr(conn_comp,conv_weights)        
            conv_weight_arrs.append(conv_weight_arr)
        conv_weight_arrs = np.array(conv_weight_arrs)
        conv_max_merged = np.max(conv_weight_arrs,axis=0)
        return conv_max_merged
        
    def segment(self,img_arr): #img_arr is t,y,x
#         input_kymo = kymo_handle()
#         input_kymo.import_wrap(img_arr,scale=self.scale_timepoints,scale_perc=self.scaling_percentage)
        t_tot = img_arr.shape[0]
        working_img = self.preprocess_img(img_arr,sigma=self.smooth_sigma,bit_max=self.bit_max,\
                                         scale_timepoints=self.scale_timepoints,scaling_percentage=self.scaling_percentage) #8_bit 
                
        inverted = np.array([sk.util.invert(working_img[t]) for t in range(working_img.shape[0])])
        min_eigvals = np.array([self.to_8bit(self.hessian_contrast_enc(inverted[t],self.hess_pad)) for t in range(inverted.shape[0])])
        del inverted        
        cell_mask = self.cell_region_mask(working_img,method=self.cell_mask_method,global_threshold=self.global_threshold,cell_otsu_scaling=self.cell_otsu_scaling,local_otsu_r=self.local_otsu_r)
                
        mid_threshold_arr = self.get_mid_threshold_arr(min_eigvals,edge_threshold_scaling=self.edge_threshold_scaling,padding=self.wrap_pad)
        
        cell_mask_kymo = kymo_handle()
        cell_mask_kymo.import_wrap(cell_mask)
        cell_mask = cell_mask_kymo.return_unwrap(padding=self.wrap_pad)
        
        min_eigvals_kymo = kymo_handle()
        min_eigvals_kymo.import_wrap(min_eigvals)
        min_eigvals = min_eigvals_kymo.return_unwrap(padding=self.wrap_pad)
        
        convex_scores = self.get_scores(cell_mask,min_eigvals,mid_threshold_arr,\
                                          threshold_step_perc=self.threshold_step_perc,threshold_perc_num_steps=self.threshold_perc_num_steps,\
                                                  min_obj_size=self.min_obj_size)
        del cell_mask
        del min_eigvals
        del mid_threshold_arr

        final_mask = (convex_scores>self.convex_threshold)
        
        final_mask = sk.morphology.remove_small_objects(final_mask,min_size=self.min_obj_size)
        final_mask = sk.morphology.remove_small_holes(final_mask)

        output_kymo = kymo_handle()
        output_kymo.import_unwrap(final_mask,t_tot,padding=self.wrap_pad)
        segmented = output_kymo.return_wrap()
        return segmented
    

class fluo_segmentation_cluster(fluo_segmentation):
    def __init__(self,headpath,paramfile=True,seg_channel="",scale_timepoints=False,scaling_percentage=0.9,smooth_sigma=0.75,bit_max=0,wrap_pad=0,\
                 hess_pad=6,min_obj_size=30,cell_mask_method='local',global_threshold=255,cell_otsu_scaling=1.,local_otsu_r=15,\
                 edge_threshold_scaling=1.,threshold_step_perc=.1,threshold_perc_num_steps=2,convex_threshold=0.8):
        
        if paramfile:
            parampath = headpath + "/fluorescent_segmentation.par"
            with open(parampath, 'rb') as infile:
                param_dict = pickle.load(infile)
            
            scale_timepoints = param_dict["Scale Fluorescence?"]
            scaling_percentage = param_dict["Scaling Percentile:"]
            seg_channel = param_dict["Segmentation Channel:"]
            smooth_sigma = param_dict["Gaussian Kernel Sigma:"]
            bit_max = param_dict['8 Bit Maximum:']
            min_obj_size = param_dict["Minimum Object Size:"]
            cell_mask_method = param_dict["Cell Mask Thresholding Method:"]
            global_threshold = param_dict["Global Threshold:"]
            cell_otsu_scaling = param_dict["Cell Threshold Scaling:"]
            local_otsu_r = param_dict["Local Otsu Radius:"]
            edge_threshold_scaling = param_dict["Edge Threshold Scaling:"]
            threshold_step_perc = param_dict["Threshold Step Percent:"]
            threshold_perc_num_steps = param_dict["Number of Threshold Steps:"]
            convex_threshold = param_dict["Convexity Threshold:"]
        
        super(fluo_segmentation_cluster, self).__init__(scale_timepoints=scale_timepoints,scaling_percentage=scaling_percentage,smooth_sigma=smooth_sigma,bit_max=bit_max,\
                                                        wrap_pad=wrap_pad,hess_pad=hess_pad,min_obj_size=min_obj_size,cell_mask_method=cell_mask_method,\
                                                        global_threshold=global_threshold,cell_otsu_scaling=cell_otsu_scaling,local_otsu_r=local_otsu_r,\
                                                        edge_threshold_scaling=edge_threshold_scaling,threshold_step_perc=threshold_step_perc,\
                                                        threshold_perc_num_steps=threshold_perc_num_steps,convex_threshold=convex_threshold)

        self.headpath = headpath
        self.seg_channel = seg_channel
        self.kymographpath = headpath + "/kymograph"
        self.fluorsegmentationpath = headpath + "/fluorsegmentation"
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)

    def generate_segmentation(self,file_idx):
        with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5","r") as input_file:
            input_data = input_file[self.seg_channel]
            trench_output = []
            for trench_idx in range(input_data.shape[0]):
                trench_array = input_data[trench_idx]
                trench_array = self.segment(trench_array)
                trench_output.append(trench_array[np.newaxis])
                del trench_array
        trench_output = np.concatenate(trench_output,axis=0)
        with h5py.File(self.fluorsegmentationpath + "/segmentation_" + str(file_idx) + ".hdf5", "w") as h5pyfile:
            hdf5_dataset = h5pyfile.create_dataset("data", data=trench_output, dtype=bool)
        return "Done"

    def dask_segment(self,dask_controller):
        writedir(self.fluorsegmentationpath,overwrite=True)
        dask_controller.futures = {}
        
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        file_list = kymodf["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
                
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            
            future = dask_controller.daskclient.submit(self.generate_segmentation,file_idx,retries=1,priority=priority)
            dask_controller.futures["Segmentation: " + str(file_idx)] = future
            
    
class phase_segmentation_cluster(phase_segmentation):
    def __init__(self,headpath,paramfile=True,seg_channel="", init_niblack_k=-0.45, maxima_niblack_k=-0.8, init_smooth_sigma=4, maxima_smooth_sigma=3, init_niblack_window_size=13, maxima_niblack_window_size=13, min_cell_size=100, deviation_from_median=0.3, max_perc_contrast=97, wrap_pad=0):
        
        super(phase_segmentation_cluster, self).__init__(init_niblack_k=init_niblack_k, init_smooth_sigma=init_smooth_sigma, maxima_smooth_sigma=2, init_niblack_window_size=maxima_niblack_window_size, maxima_niblack_window_size=maxima_niblack_window_size, min_cell_size=min_cell_size, deviation_from_median=deviation_from_median, max_perc_contrast=max_perc_contrast, wrap_pad = wrap_pad)

        self.headpath = headpath
        self.seg_channel = seg_channel
        self.kymographpath = headpath + "/kymograph"
        self.phasesegmentationpath = headpath + "/phasesegmentation"
        self.phasedatapath = self.phasesegmentationpath + "/cell_data"
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        self.metadf = None
        self.bit_max = None
        self.persisted_futures = None
    
    def get_num_trenches_timepoints(self):
        metadf = self.meta_handle.read_df("kymograph",read_metadata=True)
        num_trenchid = len(metadf.index.unique("trenchid"))
        num_time = len(metadf.index.unique("timepoints"))
        return num_trenchid, num_time
    
    def view_kymograph(self, trench_idx, timepoint, channel):
        if self.metadf is None:
            self.metadf = self.meta_handle.read_df("kymograph",read_metadata=True)
        img_entry = self.metadf.loc[trench_idx,timepoint]
        file_idx = int(img_entry["File Index"])
        trench_idx = int(img_entry["File Trench Index"])
        with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5", "r") as infile:
            img_arr = infile[channel][trench_idx,timepoint,:,:]
        plt.imshow(img_arr)

    def load_trench_array_list(self, path_form, file_idx, key, to_8bit):
        with h5py.File(path_form + str(file_idx) + ".hdf5","r") as input_file:
            if to_8bit:
                trench_array_list = np.empty_like(input_file[key], dtype=np.uint8)
                for tr in range(trench_array_list.shape[0]):
                    for t in range(trench_array_list.shape[1]):
                        trench_array_list[tr,t,:,:] = self.to_8bit(input_file[key][tr,t,:,:])
            else:
                trench_array_list = input_file[key][:]
            return trench_array_list
    
    def save_masks_to_hdf(self, file_idx, final_masks_future, data_shape=None):
        with h5py.File(self.phasesegmentationpath + "/segmentation_" + str(file_idx) + ".hdf5", "w") as h5pyfile:
            hdf5_dataset = h5pyfile.create_dataset("data", data=final_masks_future, dtype=np.uint8)
        return "Done"
            
    def generate_trench_loading(self, file_idx):
        with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5","r") as input_file:
            input_data = input_file[self.seg_channel]
            trench_output = []
            for trench_idx in range(input_data.shape[0]):
                trench_array = input_data[trench_idx]
                trench_loading_array = self.measure_trench_loading(trench_array, file_idx=file_idx, trench_idx=trench_idx)
                trench_output.append(trench_loading_array[np.newaxis])
        trench_output = np.concatenate(trench_output,axis=0)
        return trench_output

    def dask_segment(self,dask_controller, file_list=None, overwrite=True):
        writedir(self.phasesegmentationpath,overwrite=overwrite)
        dask_controller.futures = {}
        if file_list is None:
            file_list = self.meta_handle.read_df("kymograph",read_metadata=True)["File Index"].unique().tolist()
        
        num_file_jobs = len(file_list)
        
        random_priorities = np.random.uniform(size=(num_file_jobs,6))
        for k,file_idx in enumerate(file_list):
            future = dask_controller.daskclient.submit(self.load_trench_array_list, self.kymographpath + "/kymograph_", file_idx, self.seg_channel, True, retries=1,priority=random_priorities[k,0]*0.1)
            future = dask_controller.daskclient.submit(self.find_trench_masks_and_median_filtered_list,future,retries=1,priority=random_priorities[k,1]*0.4)
            future = dask_controller.daskclient.submit(self.find_watershed_mask_list,future,retries=1,priority=random_priorities[k,2]*1.6)
            future = dask_controller.daskclient.submit(self.find_watershed_maxima_list,future,retries=1,priority=random_priorities[k,3]*6.4)
            future = dask_controller.daskclient.submit(self.find_conn_comp_list,future,retries=1,priority=random_priorities[k,4]*25.6)
            future = dask_controller.daskclient.submit(self.save_masks_to_hdf,file_idx,future,retries=1,priority=random_priorities[k,5]*51.2)
            dask_controller.futures["Segmentation: " + str(file_idx)] = future
    
    def dask_characterize_trench_loading(self, dask_controller, file_list=None):
        dask_controller.futures = {}
        dask_controller.futures["Trench Loading"] = []
        
        if file_list is None:
            file_list = self.meta_handle.read_df("kymograph",read_metadata=True)["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
        
        random_priorities = np.random.uniform(size=(num_file_jobs,2))
        for k,file_idx in enumerate(file_list):
            future = dask_controller.daskclient.submit(self.load_trench_array_list, self.kymographpath + "/kymograph_", file_idx, self.seg_channel, True, retries=1,priority=random_priorities[k,0]*0.1)
            future = dask_controller.daskclient.submit(self.measure_trench_loading,future,file_idx,retries=1,priority=random_priorities[k,0]*0.8)
            dask_controller.futures["Trench Loading"].append(future)
    
    def dask_postprocess_trench_loading(self, dask_controller):
        trench_loadings = np.concatenate(dask_controller.daskclient.gather(dask_controller.futures["Trench Loading"]), axis=0)
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        kymodf["Trench Loading"] = trench_loadings
        self.meta_handle.write_df("kymograph", kymodf, metadata=kymodf.metadata)

    def props_to_dict(self, regionprops, props_to_grab):
        props_dict = {}
        for prop in props_to_grab:
            props_dict[prop] = list(map(lambda x: x[prop], regionprops))
        del regionprops
        return props_dict
    
    def persist_futures(self, sleep_time=900):
        if self.persisted_futures is None:
            self.persisted_futures = set()
        for file_idx, (key, item) in enumerate(self.daskcontroller.futures.items()):
            if item.status == "finished":
                self.persisted_futures |= file_idx
        print("%d futures finished" % len(self.persisted_futures))
        sleep(sleep_time)
    
    def dask_extract_cell_data(self, dask_controller, props_to_grab, file_list=None, overwrite=True):
        dask_controller.futures = {}
        writedir(self.phasedatapath,overwrite=overwrite)
        
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        metadata = kymodf.metadata
        
        globaldf = self.meta_handle.read_df("global", read_metadata=True)
        pixel_scaling = globaldf.metadata["pixel_microns"]

        if file_list is None:
            file_list = kymodf["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
        kymodf = kymodf.reset_index()
        kymodf = kymodf.set_index(["File Index", "File Trench Index", "timepoints"])

        random_priorities = np.random.uniform(size=(num_file_jobs,2))
        for k,file_idx in enumerate(file_list):
            segmented_masks_file = "segmentation_" + str(file_idx) + ".hdf5"
            if segmented_masks_file in os.listdir(self.phasesegmentationpath):
                times = kymodf.loc[file_idx, "time (s)"]
                global_trench_indices = kymodf.loc[file_idx, "trenchid"]
                trench_loadings = kymodf.loc[file_idx, "Trench Loading"]
                fov_idx = kymodf.loc[file_idx, "fov"]
                dask_controller.futures["Cell Props %d: " % file_idx] = dask_controller.daskclient.submit(self.extract_cell_data, file_idx, fov_idx, times, global_trench_indices, trench_loadings, props_to_grab, pixel_scaling, metadata, priority=random_priorities[k, 1]*8)
    
    def extract_cell_data(self, file_idx, fov_idx, times, global_trench_indices, trench_loadings, props_to_grab, pixel_scaling, metadata=None):
        segmented_mask_array = self.load_trench_array_list(self.phasesegmentationpath + "/segmentation_", file_idx, "data", False)
        with HDFStore(os.path.join(self.phasedatapath, "data_%d.h5" % file_idx)) as store:
            if "/metrics" in store.keys():
                store.remove("/metrics")
            trench_time_dataframes = {}
            first_dict_flag = True
            for trench_idx in range(segmented_mask_array.shape[0]):
                for time_idx in range(segmented_mask_array.shape[1]):
                    seg_props = sk.measure.regionprops(segmented_mask_array[trench_idx, time_idx,:,:],cache=False)
                    if len(seg_props) > 0:
                        seg_props = self.props_to_dict(seg_props, props_to_grab)
                        seg_props["trenchid"] = [global_trench_indices.loc[trench_idx,time_idx]]*len(seg_props["label"])
                        seg_props["file_trench_index"] = [trench_idx]*len(seg_props["label"])
                        seg_props["time_s"] = [times.loc[trench_idx, time_idx]]*len(seg_props["label"])
                        seg_props["trench_loadings"] = [trench_loadings.loc[trench_idx,time_idx]]*len(seg_props["label"])
                        seg_props["fov"] = [fov_idx.loc[trench_idx,time_idx]]*len(seg_props["label"])
                        if first_dict_flag:
                            trench_time_dataframes = seg_props
                            first_dict_flag = False
                        else:
                            for prop in trench_time_dataframes.keys():
                                trench_time_dataframes[prop].extend(seg_props[prop])
                    del seg_props
            del segmented_mask_array
            seg_props = pd.DataFrame(trench_time_dataframes)
            seg_props["trench_cell_index"] = seg_props["label"]
            seg_props = seg_props.drop("label", axis=1)
            if "bbox" in props_to_grab:
                seg_props[['min_row', 'min_col', 'max_row', 'max_col']] = pd.DataFrame(seg_props['bbox'].tolist(), index=seg_props.index)
                seg_props = seg_props.drop("bbox", axis=1)
            if "centroid" in props_to_grab:
                seg_props[['centy', 'centx']] = pd.DataFrame(seg_props['centroid'].tolist(), index=seg_props.index)
                seg_props = seg_props.drop("centroid", axis=1)
            length_scale_measurements = set(["major_axis_length", "equivalent_diameter", "minor_axis_length", "perimeter"])
            for prop in props_to_grab:
                if prop in length_scale_measurements:
                    seg_props[prop] = seg_props[prop]*pixel_scaling
            if "area" in props_to_grab:
                seg_props["area"] = seg_props["area"]*(pixel_scaling)**2
            seg_props = seg_props.set_index(["file_trench_index", "time_s", "trench_cell_index"])
            store.put("metrics", seg_props, data_columns=True)
            if metadata is not None:
                store.get_storer("metrics").attrs.metadata = metadata
            del seg_props
            del trench_time_dataframes
        return "Done"
