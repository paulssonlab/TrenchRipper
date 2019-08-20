import numpy as np
import skimage as sk
import h5py
import os
import copy
import pickle
import shutil

from skimage import measure,feature,segmentation,future,util,morphology,filters
from .utils import kymo_handle,pandas_hdf5_handler
from .cluster import hdf5lock
from time import sleep

class fluo_segmentation:
    def __init__(self,scale_timepoints=False,scaling_percentage=0.9,smooth_sigma=0.75,wrap_pad=3,hess_pad=4,min_obj_size=30,cell_mask_method='global',global_otsu_scaling=1.,\
                 cell_otsu_scaling=1.,local_otsu_r=15,edge_threshold_scaling=1.,threshold_step_perc=0.1,threshold_perc_num_steps=2,convex_threshold=0.8):
        
        self.scale_timepoints=scale_timepoints
        self.scaling_percentage=scaling_percentage
        self.smooth_sigma = smooth_sigma
        self.wrap_pad = wrap_pad
        self.hess_pad = hess_pad
        self.min_obj_size = min_obj_size
        self.global_otsu_scaling = global_otsu_scaling
        self.cell_mask_method = cell_mask_method
        self.cell_otsu_scaling = cell_otsu_scaling
        self.local_otsu_r = local_otsu_r
        self.edge_threshold_scaling = edge_threshold_scaling
        self.threshold_step_perc = threshold_step_perc
        self.threshold_perc_num_steps = threshold_perc_num_steps
        self.convex_threshold = convex_threshold
    
    def to_8bit(self,img_arr):
        max_val = np.max(img_arr)
        min_val = np.min(img_arr)
        norm_array = (img_arr-min_val)/(max_val-min_val)
        norm_byte_array = sk.img_as_ubyte(norm_array)
        return norm_byte_array
    
    def preprocess_img(self,img_arr,sigma=1.):
        img_smooth = self.to_8bit(sk.filters.gaussian(img_arr,sigma=sigma,preserve_range=True,mode='reflect'))
        return img_smooth
    
    def cell_region_mask(self,img_arr,method='global',global_otsu_scaling=1.,cell_otsu_scaling=1.,t_tot=0,local_otsu_r=15):
        
        img_kymo = kymo_handle()
        img_kymo.import_unwrap(img_arr,t_tot)
        wrap_img = img_kymo.return_wrap()

        cell_threshold_kymo = []
        for t in range(t_tot):
            cell_threshold = sk.filters.threshold_otsu(wrap_img[t,:,:])*global_otsu_scaling
            cell_thr_arr = cell_threshold*np.ones(wrap_img[t,:,:].shape,dtype='uint8')
            cell_threshold_kymo.append(cell_thr_arr)
        cell_threshold_kymo = np.array(cell_threshold_kymo)

        thr_kymo = kymo_handle()
        thr_kymo.import_wrap(cell_threshold_kymo)
        global_cell_mask = img_arr>thr_kymo.return_unwrap()
            
        if method == 'global':
            
            return global_cell_mask
            
        elif method == 'local':
            otsu_selem = sk.morphology.disk(local_otsu_r)
            local_otsu = sk.filters.rank.otsu(img_arr, otsu_selem)
            cell_local_mask = img_arr>local_otsu
            del img_arr
            
            cell_region_threshold = sk.filters.threshold_otsu(local_otsu)*cell_otsu_scaling
            cell_region_mask = local_otsu>cell_region_threshold
            del local_otsu
            
            final_cell_mask=cell_local_mask*cell_region_mask*global_cell_mask
            del cell_region_mask
            
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
        for t in range(wrap_eig.shape[-1]):
            edge_threshold = sk.filters.threshold_otsu(wrap_eig[:,:,t])
            edge_thr_arr = edge_threshold*np.ones(wrap_eig[:,:,t].shape,dtype='uint8')
            edge_threshold_kymo.append(edge_thr_arr)
        edge_threshold_kymo = np.array(edge_threshold_kymo)*edge_threshold_scaling
        edge_threshold_kymo = np.moveaxis(edge_threshold_kymo,(0,1,2),(2,0,1))

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
    
    def segment(self,img_arr):
        input_kymo = kymo_handle()
        input_kymo.import_wrap(img_arr,scale=self.scale_timepoints,scale_perc=self.scaling_percentage)
        t_tot = input_kymo.kymo_arr.shape[0]

        working_img = self.preprocess_img(input_kymo.return_unwrap(padding=self.wrap_pad),sigma=self.smooth_sigma)
        del input_kymo

        inverted = sk.util.invert(working_img)
        min_eigvals = self.to_8bit(self.hessian_contrast_enc(inverted,self.hess_pad))
        del inverted
                
        cell_mask = self.cell_region_mask(working_img,method=self.cell_mask_method,global_otsu_scaling=self.global_otsu_scaling,cell_otsu_scaling=self.cell_otsu_scaling,t_tot=t_tot,local_otsu_r=self.local_otsu_r)

        eig_kymo = kymo_handle()
        eig_kymo.import_unwrap(min_eigvals,t_tot,padding=self.wrap_pad)
        wrap_eig = eig_kymo.return_wrap()
        mid_threshold_arr = self.get_mid_threshold_arr(wrap_eig,edge_threshold_scaling=self.edge_threshold_scaling,padding=self.wrap_pad)
        del wrap_eig
                
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
    def __init__(self,headpath,paramfile=True,seg_channel="",scale_timepoints=False,scaling_percentage=0.9,smooth_sigma=0.75,wrap_pad=0,\
                 hess_pad=4,min_obj_size=30,cell_mask_method='local',global_otsu_scaling=1.,cell_otsu_scaling=1.,local_otsu_r=15,\
                 edge_threshold_scaling=1.,threshold_step_perc=.1,threshold_perc_num_steps=2,convex_threshold=0.8):
        
        if paramfile:
            parampath = headpath + "/fluorescent_segmentation.par"
            with open(parampath, 'rb') as infile:
                param_dict = pickle.load(infile)
            
            scale_timepoints = param_dict["Scale Fluorescence?"]
            scaling_percentage = param_dict["Scaling Percentile:"]
            seg_channel = param_dict["Segmentation Channel:"]
            smooth_sigma = param_dict["Gaussian Kernel Sigma:"]
            min_obj_size = param_dict["Minimum Object Size:"]
            cell_mask_method = param_dict["Cell Mask Thresholding Method:"]
            global_otsu_scaling = param_dict["Global Threshold Scaling:"]
            cell_otsu_scaling = param_dict["Cell Threshold Scaling:"]
            local_otsu_r = param_dict["Local Otsu Radius:"]
            edge_threshold_scaling = param_dict["Edge Threshold Scaling:"]
            threshold_step_perc = param_dict["Threshold Step Percent:"]
            threshold_perc_num_steps = param_dict["Number of Threshold Steps:"]
            convex_threshold = param_dict["Convexity Threshold:"]
        
        super(fluo_segmentation_cluster, self).__init__(scale_timepoints=scale_timepoints,scaling_percentage=scaling_percentage,smooth_sigma=smooth_sigma,\
                                                        wrap_pad=wrap_pad,hess_pad=hess_pad,min_obj_size=min_obj_size,cell_mask_method=cell_mask_method,\
                                                        global_otsu_scaling=global_otsu_scaling,cell_otsu_scaling=cell_otsu_scaling,local_otsu_r=local_otsu_r,\
                                                        edge_threshold_scaling=edge_threshold_scaling,threshold_step_perc=threshold_step_perc,\
                                                        threshold_perc_num_steps=threshold_perc_num_steps,convex_threshold=convex_threshold)

        self.headpath = headpath
        self.seg_channel = seg_channel
        self.kymographpath = headpath + "/kymograph"
        self.fluorsegmentationpath = headpath + "/fluorsegmentation"
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)

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

    def generate_segmentation(self,file_idx):
        with h5py.File(self.fluorsegmentationpath + "/segmentation_" + str(file_idx) + ".hdf5", "w") as h5pyfile:
            with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5","r") as input_file:
                input_data = input_file[self.seg_channel]
                trench_output = []
                for trench_idx in range(input_data.shape[0]):
                    trench_array = input_data[trench_idx]
                    trench_array = self.segment(trench_array)
                    trench_output.append(trench_array[np.newaxis])
                    del trench_array
            trench_output = np.concatenate(trench_output,axis=0)
            hdf5_dataset = h5pyfile.create_dataset("data", data=trench_output, dtype=bool)
            return "Done"

    def dask_segment(self,dask_controller):
        self.writedir(self.fluorsegmentationpath,overwrite=True)
        dask_controller.futures = {}
        
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        file_list = kymodf["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
        
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            
            future = dask_controller.daskclient.submit(self.generate_segmentation,file_idx,retries=1,priority=priority)
            dask_controller.futures["Segmentation: " + str(file_idx)] = future

class phase_segmentation:
    def __init__(self,smooth_sigma=0.75,wrap_pad=0,hess_pad=4,min_obj_size=30,global_otsu_scaling=1.,\
                 cell_otsu_scaling=1.,local_otsu_r=15,edge_threshold_scaling=1.,threshold_step_perc=0.1,threshold_perc_num_steps=2,convex_threshold=0.8,trench_mask_scaling=1.):

        self.smooth_sigma = smooth_sigma
        self.wrap_pad = wrap_pad
        self.hess_pad = hess_pad
        self.min_obj_size = min_obj_size
        self.global_otsu_scaling = global_otsu_scaling
        self.cell_otsu_scaling = cell_otsu_scaling
        self.local_otsu_r = local_otsu_r
        self.edge_threshold_scaling = edge_threshold_scaling
        self.threshold_step_perc = threshold_step_perc
        self.threshold_perc_num_steps = threshold_perc_num_steps
        self.convex_threshold = convex_threshold
        self.trench_mask_scaling = trench_mask_scaling
    
    def to_8bit(self,img_arr):
        max_val = np.max(img_arr)
        min_val = np.min(img_arr)
        norm_array = (img_arr-min_val)/(max_val-min_val)
        norm_byte_array = sk.img_as_ubyte(norm_array)
        return norm_byte_array
    
    def preprocess_img(self,img_arr,sigma=1.):
        img_smooth = self.to_8bit(sk.filters.gaussian(img_arr,sigma=sigma,preserve_range=True,mode='reflect'))
        return img_smooth
    
    def cell_region_mask(self,img_arr,trench_mask,global_otsu_scaling=1.,cell_otsu_scaling=1.,t_tot=0,local_otsu_r=15):
        
        img_kymo = kymo_handle()
        img_kymo.import_unwrap(img_arr,t_tot)
        wrap_img = img_kymo.return_wrap()
        
        mask_kymo = kymo_handle()
        mask_kymo.import_unwrap(trench_mask,t_tot)
        wrap_mask = mask_kymo.return_wrap()

        cell_threshold_kymo = []
        for t in range(t_tot):
            trench_values = wrap_img[:,:,t][wrap_mask[:,:,t]]
            cell_threshold = sk.filters.threshold_otsu(trench_values)*global_otsu_scaling
            cell_thr_arr = cell_threshold*np.ones(wrap_img[:,:,t].shape,dtype='uint8')
            cell_threshold_kymo.append(cell_thr_arr)
        cell_threshold_kymo = np.array(cell_threshold_kymo)
        cell_threshold_kymo = np.moveaxis(cell_threshold_kymo,(0,1,2),(2,0,1))

        thr_kymo = kymo_handle()
        thr_kymo.import_wrap(cell_threshold_kymo)
        global_cell_mask = img_arr>thr_kymo.return_unwrap()
        
        
        otsu_selem = sk.morphology.disk(local_otsu_r)
        local_otsu = sk.filters.rank.otsu(img_arr, otsu_selem, mask=trench_mask)
        
        cell_local_mask = img_arr>local_otsu
        del img_arr
        del local_otsu

        final_cell_mask=cell_local_mask*global_cell_mask

        return final_cell_mask
                
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

    def get_mid_threshold_arr(self,wrap_eig,wrap_trench_mask,edge_threshold_scaling=1.,padding=3):
        edge_threshold_kymo = []
        for t in range(wrap_eig.shape[-1]):
            wrap_eig_masked = wrap_eig[:,:,t][wrap_trench_mask[:,:,t]]
            edge_threshold = sk.filters.threshold_otsu(wrap_eig_masked)
            edge_thr_arr = edge_threshold*np.ones(wrap_eig[:,:,t].shape,dtype='uint8')
            edge_threshold_kymo.append(edge_thr_arr)
        edge_threshold_kymo = np.array(edge_threshold_kymo)*edge_threshold_scaling
        edge_threshold_kymo = np.moveaxis(edge_threshold_kymo,(0,1,2),(2,0,1))

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
    
    def segment(self,img_arr):
        input_kymo = kymo_handle()
        input_kymo.import_wrap(img_arr,scale=True,scale_perc=95)
        t_tot = input_kymo.kymo_arr.shape[-1]

        working_img = self.preprocess_img(input_kymo.return_unwrap(padding=self.wrap_pad),sigma=self.smooth_sigma)
        del input_kymo
        trench_mask = working_img>(sk.filters.threshold_triangle(working_img)*self.trench_mask_scaling)
        
        trench_mask = sk.morphology.binary_closing(trench_mask)
        trench_mask = sk.morphology.binary_erosion(trench_mask)
        
        inverted = sk.util.invert(working_img)
        min_eigvals = self.to_8bit(self.hessian_contrast_enc(working_img,self.hess_pad))
        
        cell_mask = self.cell_region_mask(inverted,trench_mask,global_otsu_scaling=self.global_otsu_scaling,t_tot=t_tot,cell_otsu_scaling=self.cell_otsu_scaling,local_otsu_r=self.local_otsu_r)
        combined_mask = trench_mask*cell_mask
        
        cell_mask = sk.morphology.binary_opening(combined_mask)

        eig_kymo = kymo_handle()
        eig_kymo.import_unwrap(min_eigvals,t_tot,padding=self.wrap_pad)
        wrap_eig = eig_kymo.return_wrap()
        
        trench_mask_kymo = kymo_handle()
        trench_mask_kymo.import_unwrap(trench_mask,t_tot,padding=self.wrap_pad)
        wrap_trench_mask = trench_mask_kymo.return_wrap()
        
        mid_threshold_arr = self.get_mid_threshold_arr(wrap_eig,wrap_trench_mask,edge_threshold_scaling=self.edge_threshold_scaling,padding=self.wrap_pad)
        del wrap_eig
        del wrap_trench_mask
        del trench_mask
                
        convex_scores = self.get_scores(combined_mask,min_eigvals,mid_threshold_arr,\
                                          threshold_step_perc=self.threshold_step_perc,threshold_perc_num_steps=self.threshold_perc_num_steps,\
                                                  min_obj_size=self.min_obj_size)
        del cell_mask
        del min_eigvals
        del mid_threshold_arr

        final_mask = (convex_scores>self.convex_threshold)
        
        output_kymo = kymo_handle()
        
        output_kymo.import_unwrap(final_mask,t_tot,padding=self.wrap_pad)
        segmented = output_kymo.return_wrap()
        return segmented
    
class phase_segmentation_cluster(phase_segmentation):
    def __init__(self,headpath,seg_channel,smooth_sigma=0.75,wrap_pad=0,hess_pad=4,min_obj_size=30,global_otsu_scaling=1.,\
                 cell_otsu_scaling=1.,local_otsu_r=15,edge_threshold_scaling=1.,threshold_step_perc=0.1,threshold_perc_num_steps=2,convex_threshold=0.8,trench_mask_scaling=1.):
        
        super(phase_segmentation_cluster, self).__init__(smooth_sigma=smooth_sigma,wrap_pad=wrap_pad,hess_pad=hess_pad,min_obj_size=min_obj_size,global_otsu_scaling=global_otsu_scaling,\
            cell_otsu_scaling=cell_otsu_scaling,local_otsu_r=local_otsu_r,edge_threshold_scaling=edge_threshold_scaling,threshold_step_perc=threshold_step_perc,\
            threshold_perc_num_steps=threshold_perc_num_steps,convex_threshold=convex_threshold,trench_mask_scaling=trench_mask_scaling)

        self.headpath = headpath
        self.seg_channel = seg_channel
        self.input_path = headpath + "/kymo"
        self.output_path = headpath + "/phasesegmentation"
        self.metapath = headpath + "/metadata.hdf5"
        
    def removefile(self,filepath):
        """Removes a file at the specified path, if it exists.
        
        Args:
            filepath (str): Path to file for deletion.
        """
        if os.path.exists(filepath):
            os.remove(filepath)

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
    
    def init_fov_number(self,fov_number):
        self.fov_number = fov_number
        self.input_file_path = self.input_path+"/kymo_"+str(self.fov_number)+".hdf5"
        self.output_file_path = self.output_path+"/seg_"+str(self.fov_number)+".hdf5"

    def generate_segmentation(self,fov_number):

        self.init_fov_number(fov_number)
        self.writedir(self.output_path,overwrite=False)
        self.removefile(self.output_file_path)
        
        with h5py.File(self.input_file_path,"r") as input_file:
            with h5py.File(self.output_file_path, "w") as h5pyfile:
                for trenchid in input_file.keys():
                    trench_array = input_file[trenchid+"/"+self.seg_channel]
                    try:
                        out_array = self.segment(trench_array)
                        hdf5_dataset = h5pyfile.create_dataset(str(trenchid), data=out_array, dtype=bool)
                    except:
                        print("Failed Segmentation.")
                        pass