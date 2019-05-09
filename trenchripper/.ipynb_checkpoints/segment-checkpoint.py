import numpy as np
import skimage as sk
import h5py
import os

from skimage import measure,feature,segmentation,future,util,morphology,filters
from .utils import kymo_handle

class fluo_segmentation:
    def __init__(self,smooth_sigma=0.75,wrap_pad=3,hess_pad=4,min_obj_size=30,cell_mask_method='global',\
                 cell_otsu_scaling=1.,local_otsu_r=15,edge_threshold_scaling=1.,threshold_range=20,threshold_step=5,convex_threshold=0.8):

        self.smooth_sigma = smooth_sigma
        self.wrap_pad = wrap_pad
        self.hess_pad = hess_pad
        self.min_obj_size = min_obj_size
        self.cell_mask_method = cell_mask_method
        self.cell_otsu_scaling = cell_otsu_scaling
        self.local_otsu_r = local_otsu_r
        self.edge_threshold_scaling = edge_threshold_scaling
        self.threshold_range = threshold_range
        self.threshold_step = threshold_step
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
    
    def cell_region_mask(self,img_arr,method='global',cell_otsu_scaling=1.,t_tot=0,local_otsu_r=15):
        if method == 'global': ###Maybe combinw with mid_threshold class
            img_kymo = kymo_handle()
            img_kymo.import_unwrap(img_arr,t_tot)
            wrap_img = img_kymo.return_wrap()
            
            cell_threshold_kymo = []
            for t in range(t_tot):
                cell_threshold = sk.filters.threshold_otsu(wrap_img[:,:,t])*cell_otsu_scaling
                cell_thr_arr = cell_threshold*np.ones(wrap_img[:,:,t].shape,dtype='uint8')
                cell_threshold_kymo.append(cell_thr_arr)
            cell_threshold_kymo = np.array(cell_threshold_kymo)
            cell_threshold_kymo = np.moveaxis(cell_threshold_kymo,(0,1,2),(2,0,1))
            
            thr_kymo = kymo_handle()
            thr_kymo.import_wrap(cell_threshold_kymo)
            final_cell_mask = img_arr>thr_kymo.return_unwrap()
            
        elif method == 'local':
            otsu_selem = sk.morphology.disk(local_otsu_r)
            local_otsu = sk.filters.rank.otsu(img_arr, otsu_selem)
            cell_local_mask = img_arr>local_otsu
            del img_arr
            
            cell_region_threshold = sk.filters.threshold_otsu(local_otsu)
            cell_region_mask = local_otsu>cell_region_threshold
            del local_otsu
            
            final_cell_mask=cell_local_mask*cell_region_mask
            del cell_region_mask
            
        else:
            print("no valid cell threshold method chosen!!!")
            
        return final_cell_mask
    
    def hessian_contrast_enc(self,img_arr,edge_padding=0):
        img_arr = np.pad(img_arr, edge_padding, 'reflect')
        hessian = sk.feature.hessian_matrix(img_arr,order="rc")
        eigvals = sk.feature.hessian_matrix_eigvals(hessian)
        min_eigvals = np.min(eigvals,axis=0)
        if edge_padding>0:
            min_eigvals = min_eigvals[edge_padding:-edge_padding,edge_padding:-edge_padding]
        return min_eigvals
    
    def find_mask(self,cell_local_mask,min_eigvals,edge_threshold,min_size=30):
        edge_mask = min_eigvals>edge_threshold
        composite_mask = cell_local_mask*edge_mask
        composite_mask = sk.morphology.remove_small_objects(composite_mask,min_size=min_size)
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
        weights = []
        for obj_idx in objects:
            curr_obj = self.crop_object(conn_comp,obj_idx,padding=padding)
            weight = self.compute_convexity(curr_obj)
            weights.append(weight)
        return np.array(weights)
    
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

    def get_convex_scores(self,cell_mask,min_eigvals,mid_threshold_arr,threshold_range=20,threshold_step=5):
        edge_thresholds = [mid_threshold_arr+i for i in range(-threshold_range,threshold_range,threshold_step)]
        weight_arrs = []
        for edge_threshold in edge_thresholds:
            composite_mask = self.find_mask(cell_mask,min_eigvals,edge_threshold,min_size=30)
            conn_comp = sk.measure.label(composite_mask,neighbors=4,connectivity=2)
            weights = self.get_image_weights(conn_comp)
            weight_arr = self.make_weight_arr(conn_comp,weights)
            weight_arrs.append(weight_arr)
        weight_arrs = np.array(weight_arrs)
        max_merged = np.max(weight_arrs,axis=0)
        return max_merged 
    
    def segment(self,img_arr):
        input_kymo = kymo_handle()
        input_kymo.import_wrap(img_arr,scale=True,scale_perc=95)
        t_tot = input_kymo.kymo_arr.shape[-1]

        working_img = self.preprocess_img(input_kymo.return_unwrap(padding=self.wrap_pad),sigma=self.smooth_sigma)
        del input_kymo

        inverted = sk.util.invert(working_img)
        min_eigvals = self.to_8bit(self.hessian_contrast_enc(inverted,self.hess_pad))
        del inverted
        
        cell_mask = self.cell_region_mask(working_img,method=self.cell_mask_method,cell_otsu_scaling=self.cell_otsu_scaling,t_tot=t_tot,local_otsu_r=self.local_otsu_r)

        eig_kymo = kymo_handle()
        eig_kymo.import_unwrap(min_eigvals,t_tot,padding=self.wrap_pad)
        wrap_eig = eig_kymo.return_wrap()
        mid_threshold_arr = self.get_mid_threshold_arr(wrap_eig,edge_threshold_scaling=self.edge_threshold_scaling,padding=self.wrap_pad)
        del wrap_eig

        convex_scores = self.get_convex_scores(cell_mask,min_eigvals,mid_threshold_arr,\
                                          threshold_range=self.threshold_range,threshold_step=self.threshold_step)
        del cell_mask
        del min_eigvals
        del mid_threshold_arr

        final_mask = convex_scores>self.convex_threshold

        output_kymo = kymo_handle()
        output_kymo.import_unwrap(final_mask,t_tot,padding=self.wrap_pad)
        segmented = output_kymo.return_wrap()
        return segmented

class fluo_segmentation_cluster(fluo_segmentation):
    def __init__(self,headpath,seg_channel,smooth_sigma=0.75,wrap_pad=3,hess_pad=4,min_obj_size=30,cell_mask_method='local',\
                 cell_otsu_scaling=1.,local_otsu_r=15,edge_threshold_scaling=1.,threshold_range=20,threshold_step=5,convex_threshold=0.8):
        super(fluo_segmentation_cluster, self).__init__(smooth_sigma=smooth_sigma,wrap_pad=wrap_pad,hess_pad=hess_pad,\
            min_obj_size=min_obj_size,cell_mask_method=cell_mask_method,cell_otsu_scaling=cell_otsu_scaling,local_otsu_r=local_otsu_r,\
            edge_threshold_scaling=edge_threshold_scaling,threshold_range=threshold_range,threshold_step=threshold_step,convex_threshold=convex_threshold)

        self.headpath = headpath
        self.seg_channel = seg_channel
        self.input_path = headpath + "/kymo"
        self.output_path = headpath + "/segmentation"
        self.metapath = headpath + "/metadata.hdf5"

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
        input_data = h5py.File(self.input_file_path,"r")
        for lane in input_data.keys():
            lane_array = input_data[lane+"/"+self.seg_channel]
            with h5py.File(self.output_file_path, "a") as h5pyfile:
                if lane in list(h5pyfile.keys()):
                    hdf5_dataset = h5pyfile[lane]
                else:
                    hdf5_dataset = h5pyfile.create_dataset(lane, lane_array.shape, dtype=bool)     
                for trench in range(lane_array.shape[0]):
                    kymo_arr = lane_array[trench]
                    segmented = self.segment(kymo_arr)
                    hdf5_dataset[trench] = segmented