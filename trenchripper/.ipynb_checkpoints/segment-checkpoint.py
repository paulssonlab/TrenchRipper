# fmt: off
import numpy as np
import skimage as sk
import scipy as sp
import h5py
import os
import copy
import pickle as pkl
import pandas as pd
import dask.dataframe as dd
from scipy import ndimage as ndi
from dask.distributed import as_completed

from skimage import measure,feature,segmentation,future,util,morphology,filters,exposure,transform
from skimage.segmentation import watershed
from .utils import kymo_handle,pandas_hdf5_handler,writedir
from .trcluster import hdf5lock
from time import sleep
import scipy.ndimage.morphology as morph
# import mahotas as mh
from dask.distributed import worker_client
from pandas import HDFStore

from matplotlib import pyplot as plt

class fluo_segmentation:
    def __init__(self,maskpath=None,bit_max=0,scale_timepoints=False,scaling_percentile=0.9,img_scaling=1.,smooth_sigma=0.75,eig_sigma=2.,\
                 eig_ball_radius=20,eig_local_thr="niblack",eig_otsu_scaling=1.,eig_niblack_k=0.05,eig_window_size=7,\
                 hess_pad=6,local_thr="otsu",background_thr="triangle",global_threshold=25,window_size=15,cell_otsu_scaling=1.,niblack_k=0.2,background_scaling=1.,\
                 min_obj_size=30,distance_threshold=2,border_buffer=1,horizontal_border_only=False):

        self.maskpath = maskpath
        self.bit_max = bit_max

        self.scale_timepoints=scale_timepoints
        self.scaling_percentile=scaling_percentile

        self.img_scaling = img_scaling

        self.smooth_sigma = smooth_sigma

#         self.hess_thr_scale = hess_thr_scale
        self.hess_pad = hess_pad
        self.eig_sigma = eig_sigma
        self.eig_ball_radius = eig_ball_radius
        self.eig_local_thr = eig_local_thr
        self.eig_otsu_scaling = eig_otsu_scaling
        self.eig_niblack_k = eig_niblack_k
        self.eig_window_size = eig_window_size

        self.local_thr = local_thr
        self.background_thr = background_thr
        self.global_threshold = global_threshold
        self.window_size = window_size
        self.cell_otsu_scaling = cell_otsu_scaling
        self.niblack_k = niblack_k
        self.background_scaling = background_scaling
        self.min_obj_size = min_obj_size

        self.distance_threshold = distance_threshold

        self.border_buffer = border_buffer
        self.horizontal_border_only = horizontal_border_only

    def to_8bit(self,img_arr,bit_max=None):
        img_max = np.max(img_arr)+0.0001
        if bit_max is None:
            max_val = img_max
        else:
            max_val = max(img_max,bit_max)
        min_val = np.min(img_arr)
#         min_val = np.min(img_arr)
        norm_array = (img_arr-min_val)/(max_val-min_val)
        norm_array = np.clip(norm_array,0,1)
        norm_byte_array = sk.img_as_ubyte(norm_array)
        return norm_byte_array

    def scale_kymo(self,wrap_arr,percentile):
        perc_t = np.percentile(wrap_arr[:].reshape(wrap_arr.shape[0],-1),percentile,axis=1)
        norm_perc_t = perc_t/np.max(perc_t)
        scaled_arr = wrap_arr.astype(float)/norm_perc_t[:,np.newaxis,np.newaxis]
        scaled_arr[scaled_arr>255.] = 255.
        scaled_arr = scaled_arr.astype("uint8")
        return scaled_arr

    def get_eig_img(self,img_arr,edge_padding=6):
        inverted = sk.util.invert(img_arr)
        del img_arr
        inverted = np.pad(inverted, edge_padding, 'edge')
        hessian = sk.feature.hessian_matrix(inverted,order="rc")
        del inverted
        eig_img = sk.feature.hessian_matrix_eigvals(hessian)
        del hessian
        eig_img = np.min(eig_img,axis=0)
        eig_img = eig_img[edge_padding:-edge_padding,edge_padding:-edge_padding]
        max_val,min_val = np.max(eig_img),np.min(eig_img)
        eig_img = self.to_8bit(eig_img)
        return eig_img

    def get_background_dist(self,array,min_tail=30):
        hist_range = (0,np.percentile(array.flatten(),90))
        freq,val = np.histogram(array.flatten(),bins=50,range=hist_range)
        mu_n = val[np.argmax(freq)]
        lower_tail = array[array<mu_n].flatten()
        if len(lower_tail) > min_tail:
            std_n = sp.stats.halfnorm.fit(lower_tail)[1]
        else:
            std_n = np.std(array.flatten())
        return mu_n,std_n

    def get_eig_mask(self,eig_img,input_kymo_mask,eig_sigma=2.,eig_ball_radius=20,eig_local_thr="niblack",eig_otsu_scaling=1.,eig_niblack_k=0.05,eig_window_size=7):
        inv_eig_img = sk.util.invert(sk.filters.gaussian(eig_img,sigma=eig_sigma,preserve_range=True).astype("uint8"))

        norm_inv_hess = (inv_eig_img)/(2**8 - 1)
        inv_eig_img = (sk.restoration.rolling_ball(norm_inv_hess,radius=eig_ball_radius)*(2**8 - 1)).astype("uint8")
        del norm_inv_hess

        if eig_local_thr == "otsu":
            otsu_selem = sk.morphology.disk(eig_window_size)
            if input_kymo_mask is None:
                eig_mask = inv_eig_img<(sk.filters.rank.otsu(inv_eig_img,otsu_selem)*eig_otsu_scaling)
            else:
                eig_mask = (inv_eig_img<(sk.filters.rank.otsu(inv_eig_img,otsu_selem,mask=input_kymo_mask)*eig_otsu_scaling))*input_kymo_mask

        elif (eig_local_thr == "niblack")&(input_kymo_mask is None):
            eig_mask = inv_eig_img<sk.filters.threshold_niblack(inv_eig_img, window_size=eig_window_size, k=eig_niblack_k)

        else:
            raise ValueError("Cannot use a segmentation mask and a Niblack threshold. Use Otsu thresholding instead.")

        return eig_mask,inv_eig_img

    def get_cell_mask(self,input_kymo, input_kymo_labels, t_tot, local_thr = "otsu" , background_thr = "triangle", global_threshold = 0, window_size = 15, cell_otsu_scaling= 1., niblack_k = 0.2, background_scaling = 1., min_obj_size = 30):

        wrapped_kymo = kymo_handle()
        wrapped_kymo.import_unwrap(input_kymo,t_tot)
        wrapped_kymo = wrapped_kymo.return_wrap()

        if input_kymo_labels is not None:
            wrapped_input_kymo_labels = kymo_handle()
            wrapped_input_kymo_labels.import_unwrap(input_kymo_labels,t_tot)
            wrapped_input_kymo_labels = wrapped_input_kymo_labels.return_wrap()

        wrapped_cell_mask = []

        for t in range(wrapped_kymo.shape[0]):
            img_arr = wrapped_kymo[t]
            if input_kymo_labels is not None:
                wrapped_input_kymo = wrapped_input_kymo_labels[t]
                label_list = sorted(list(set(np.unique(wrapped_input_kymo))-set([0])))

            try:
                if local_thr == "otsu":
                    otsu_selem = sk.morphology.disk(window_size)
                    if input_kymo_labels is None:
                        cell_mask = img_arr>(sk.filters.rank.otsu(img_arr,otsu_selem)*cell_otsu_scaling)
                    else:
                        cell_mask = []
                        for c in label_list:
                            selected_cell_mask = (wrapped_input_kymo==c)
                            working_cell_mask = (img_arr>(sk.filters.rank.otsu(img_arr,otsu_selem,mask=selected_cell_mask)*cell_otsu_scaling))
                            working_cell_mask = working_cell_mask&selected_cell_mask
                            cell_mask.append(working_cell_mask)
                        if len(cell_mask) > 0:
                            cell_mask = np.any(np.stack(cell_mask,axis=0),axis=0)
                        else:
                            cell_mask = np.zeros(img_arr.shape,dtype=bool)

                elif (local_thr == "niblack")&(input_kymo_labels is None):
                    cell_mask = img_arr>sk.filters.threshold_niblack(img_arr, window_size=window_size, k=niblack_k)

                elif local_thr == "none":
                    cell_mask = img_arr>0

                else:
                    raise ValueError("Cannot use a segmentation mask and a Niblack threshold. Use Otsu thresholding instead.")

                if (background_thr == "triangle") or (background_thr == "yen"):
                    if input_kymo_labels is None:
                        if len(np.unique(img_arr))>1:
                            if background_thr == "triangle":
                                back_thr = sk.filters.threshold_triangle(img_arr)*background_scaling
                            elif background_thr == "yen":
                                back_thr = sk.filters.threshold_yen(img_arr)*background_scaling
                        else:
                            back_thr = img_arr[0,0]*background_scaling
                        back_mask = img_arr>back_thr
                    else:
                        back_mask = []
                        for c in label_list:
                            selected_cell_mask = (wrapped_input_kymo==c)
                            img_arr_masked = img_arr[selected_cell_mask]
                            if len(np.unique(img_arr_masked))>1:
                                if background_thr == "triangle":
                                    back_thr = sk.filters.threshold_triangle(img_arr_masked)*background_scaling
                                elif background_thr == "yen":
                                    back_thr = sk.filters.threshold_yen(img_arr_masked)*background_scaling
                            elif len(np.unique(img_arr_masked))==1:
                                back_thr = img_arr_masked[0]*background_scaling
                            else:
                                back_thr = 255
                            del img_arr_masked
                            working_back_mask = (img_arr>back_thr)&selected_cell_mask
                            back_mask.append(working_back_mask)
                        if len(back_mask) > 0:
                            back_mask = np.any(np.stack(back_mask,axis=0),axis=0)
                        else:
                            back_mask = np.zeros(img_arr.shape,dtype=bool)

                    cell_mask = cell_mask&back_mask&(img_arr>global_threshold)
                    del img_arr
                    cell_mask = sk.morphology.remove_small_objects(cell_mask,min_size=min_obj_size)

                elif (background_thr == "object-based")&(input_kymo_labels is None):

                    cell_mask = ~sk.morphology.binary_closing(~cell_mask) ##NEW
                    cell_mask = sk.morphology.remove_small_objects(cell_mask,min_size=min_obj_size)

                    labeled_mask = sk.morphology.label(cell_mask)
                    del cell_mask

                    background = img_arr[labeled_mask==0]
                    mu_n,std_n = self.get_background_dist(background)
                    del background

                    rps = sk.measure.regionprops(labeled_mask,intensity_image=img_arr)
                    obj_intensities = np.array([rp.mean_intensity for rp in rps])
                    del rps

                    back_thr = mu_n + background_scaling*std_n
                    init_cell_mask = obj_intensities>back_thr

                    cell_list = np.where(init_cell_mask)[0]+1
                    del init_cell_mask
                    cell_mask = np.isin(labeled_mask,cell_list)
                    del cell_list
                    del labeled_mask

                    cell_mask = cell_mask*(img_arr>global_threshold)
                    del img_arr

                elif background_thr == "none":
                    cell_mask = cell_mask&(img_arr>global_threshold)

                else:
                    raise ValueError("Cannot use a segmentation mask and object-based thresholding. Use a triangle threshold instead.")

            except:
                raise

            cell_mask = sk.morphology.remove_small_holes(cell_mask)
            wrapped_cell_mask.append(cell_mask)
            del cell_mask


        del wrapped_kymo
        wrapped_cell_mask = np.array(wrapped_cell_mask)

        cell_mask = kymo_handle()
        cell_mask.import_wrap(wrapped_cell_mask)
        del wrapped_cell_mask
        cell_mask = cell_mask.return_unwrap()

        return cell_mask

    def reorder_ids(self,array):
        unique_ids,inv = np.unique(array,return_inverse=True)
        new_ids = np.array(range(len(unique_ids)))
        lookup = dict(zip(unique_ids,new_ids))
        output = np.array([lookup[unique_id] for unique_id in unique_ids])[inv].reshape(array.shape)
        return output

    def segment(self,img_arr,label_arr=None): ## img_arr is t,y,x
        t_tot = img_arr.shape[0]
        img_arr = self.to_8bit(img_arr,self.bit_max)
        if self.scale_timepoints:
            img_arr = self.scale_kymo(img_arr,self.scaling_percentile)

        input_kymo = kymo_handle()
        input_kymo.import_wrap(img_arr)
        del img_arr
        input_kymo = input_kymo.return_unwrap()

        original_shape = input_kymo.shape
        len_per_tpt = (original_shape[1]*self.img_scaling)//t_tot
        adjusted_scale_factor = (len_per_tpt*t_tot)/(original_shape[1])

        input_kymo = transform.rescale(input_kymo,adjusted_scale_factor,anti_aliasing=False, preserve_range=True).astype("uint8")
        input_kymo = sk.filters.gaussian(input_kymo,sigma=self.smooth_sigma,preserve_range=True,mode='reflect').astype("uint8")

        eig_img = self.get_eig_img(input_kymo,edge_padding=self.hess_pad)

        if self.maskpath is not None:
            input_kymo_labels = kymo_handle()
            input_kymo_labels.import_wrap(label_arr)
            input_kymo_labels = input_kymo_labels.return_unwrap()
            input_kymo_labels = transform.rescale(input_kymo_labels,adjusted_scale_factor,anti_aliasing=False, preserve_range=True)
            input_kymo_mask = input_kymo_labels>0
        else:
            input_kymo_labels = None
            input_kymo_mask = None

        eig_mask, inv_eig_img = self.get_eig_mask(eig_img,input_kymo_mask,eig_sigma=self.eig_sigma,eig_ball_radius=self.eig_ball_radius,\
                                                eig_local_thr=self.eig_local_thr,eig_otsu_scaling=self.eig_otsu_scaling,\
                                                eig_niblack_k=self.eig_niblack_k,eig_window_size=self.eig_window_size)
        del input_kymo_mask
        del eig_img
        del inv_eig_img

        cell_mask = self.get_cell_mask(input_kymo,input_kymo_labels,t_tot,local_thr = self.local_thr,\
                    background_thr = self.background_thr,\
                    global_threshold=self.global_threshold,\
                    window_size = self.window_size,\
                    cell_otsu_scaling = self.cell_otsu_scaling,\
                    niblack_k = self.niblack_k,\
                    background_scaling=self.background_scaling,\
                    min_obj_size=self.min_obj_size)
        del input_kymo
        del input_kymo_labels

        dist_img = ndi.distance_transform_edt(cell_mask).astype("uint8")
        dist_mask = dist_img>self.distance_threshold
        marker_mask = dist_mask*eig_mask
        del dist_mask
        marker_mask = sk.measure.label(marker_mask)

        output_labels = watershed(-dist_img, markers=marker_mask, mask=cell_mask)

        del dist_img
        del marker_mask
        del cell_mask

        output_labels = sk.morphology.remove_small_objects(output_labels,min_size=self.min_obj_size)
        output_labels = sk.transform.resize(output_labels,original_shape,order=0,anti_aliasing=False, preserve_range=True).astype("uint32")

        output_kymo = kymo_handle()
        output_kymo.import_unwrap(output_labels,t_tot)
        del output_labels
        output_kymo = output_kymo.return_wrap() #mask_arr
        if self.maskpath is not None:
            reindexed_kymo = np.stack([output_kymo,label_arr],axis=3)
            relabel_fn = lambda x: int(f'{x[0]:04n}{x[1]:04n}')
            reindexed_kymo = np.apply_along_axis(relabel_fn,3,reindexed_kymo) #kyx
            reindexed_kymo[(output_kymo==0)|(label_arr==0)] = 0
            output_kymo = reindexed_kymo
            del reindexed_kymo
            del label_arr
            output_kymo = np.stack([sk.segmentation.relabel_sequential(output_kymo[i],offset=1)[0] for i in range(output_kymo.shape[0])],axis=0)

        top_bottom_mask = np.ones(output_kymo.shape[1:],dtype=bool)
        top_bottom_mask[:(self.border_buffer+1)] = False
        top_bottom_mask[-(self.border_buffer+1):] = False

        for i in range(output_kymo.shape[0]):
            if (self.border_buffer >= 0) and not self.horizontal_border_only:
                output_kymo[i] = sk.segmentation.clear_border(output_kymo[i], buffer_size=self.border_buffer) ##NEW
            elif (self.border_buffer >= 0) and self.horizontal_border_only:
                output_kymo[i] = sk.segmentation.clear_border(output_kymo[i], mask=top_bottom_mask) ##NEWER
            output_kymo[i] = self.reorder_ids(output_kymo[i])
        return output_kymo

class fluo_segmentation_cluster(fluo_segmentation):
    def __init__(self,headpath,paramfile=True,seg_channel="",segpath="fluorsegmentation",segparfilename="fluorescent_segmentation.par",\
                 maskpath=None,bit_max=0,scale_timepoints=False,scaling_percentile=0.9,\
                 img_scaling=1.,smooth_sigma=0.75,eig_sigma=2.,eig_ball_radius=20,eig_local_thr="niblack",eig_otsu_scaling=1.,\
                 eig_niblack_k=0.05,eig_window_size=7,hess_pad=6,local_thr="otsu",background_thr="triangle",\
                 global_threshold=25,window_size=15,cell_otsu_scaling=1.,niblack_k=0.2,background_scaling=1.,min_obj_size=30,\
                 distance_threshold=2,border_buffer=1,horizontal_border_only=False):
###             local_thr="otsu",background_thr="triangle",window_size=15,cell_otsu_scaling=1.,niblack_k=0.2,background_scaling=1.,border_buffer=1)


        if paramfile:
            parampath = headpath + "/" + segparfilename
            with open(parampath, 'rb') as infile:
                param_dict = pkl.load(infile)

        seg_channel = param_dict["Segmentation Channel:"]
        maskpath = param_dict["Extra Mask Path:"]
        bit_max = param_dict['8 Bit Maximum:']
        scale_timepoints = param_dict['Scale Fluorescence?']
        scaling_percentile = param_dict["Scaling Percentile:"]
        img_scaling = param_dict["Image Scaling Factor:"]
        smooth_sigma = param_dict['Gaussian Kernel Sigma:']
        local_thr = param_dict['Local Threshold Method:']
        background_thr = param_dict['Background Threshold Method:']
        global_threshold = param_dict['Global Threshold:']
        window_size = param_dict['Local Window Size:']
        cell_otsu_scaling = param_dict['Otsu Scaling:']
        niblack_k = param_dict['Niblack K:']
        background_scaling = param_dict['Background Threshold Scaling:']
        min_obj_size = param_dict['Minimum Object Size:']

        eig_sigma = param_dict['Hessian Blur Sigma:']
        eig_ball_radius = param_dict['Hessian Rolling Ball Size:']
        eig_local_thr = param_dict['Hessian Local Threshold Method:']
        eig_otsu_scaling = param_dict['Hessian Otsu Scaling:']
        eig_niblack_k = param_dict['Hessian Niblack K:']
        eig_window_size = param_dict['Hessian Local Window Size:']
#         hess_thr_scale = param_dict['Hessian Scaling:']
        distance_threshold = param_dict['Distance Threshold:']
        border_buffer = param_dict['Border Buffer:']
        horizontal_border_only = param_dict['Horizontal Border Only:']

        super(fluo_segmentation_cluster, self).__init__(maskpath=maskpath,bit_max=bit_max,scale_timepoints=scale_timepoints,scaling_percentile=scaling_percentile,img_scaling=img_scaling,\
                                                        smooth_sigma=smooth_sigma,eig_sigma=eig_sigma,eig_ball_radius=eig_ball_radius,eig_local_thr=eig_local_thr,\
                                                        eig_otsu_scaling=eig_otsu_scaling,eig_niblack_k=eig_niblack_k,eig_window_size=eig_window_size,\
                                                        hess_pad=hess_pad,local_thr=local_thr,background_thr=background_thr,\
                                                        global_threshold=global_threshold,window_size=window_size,cell_otsu_scaling=cell_otsu_scaling,niblack_k=niblack_k,\
                                                        background_scaling=background_scaling,min_obj_size=min_obj_size,distance_threshold=distance_threshold,border_buffer=border_buffer,\
                                                       horizontal_border_only=horizontal_border_only)

        self.headpath = headpath
        self.seg_channel = seg_channel
        self.kymographpath = headpath + "/kymograph"
        self.segpath = headpath + "/" + segpath
        self.metapath = headpath + "/kymograph/metadata"
        if self.maskpath is not None:
            self.fullmaskpath = headpath + "/" + self.maskpath

    def generate_segmentation(self,file_idx):
        with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5","r") as input_file:
            input_data = input_file[self.seg_channel]
            trench_output = []

            if self.maskpath is None:
                for trench_idx in range(input_data.shape[0]):
                    trench_array = input_data[trench_idx]
                    trench_array = self.segment(trench_array)
                    trench_output.append(trench_array[np.newaxis])
                    del trench_array
            else:
                with h5py.File(self.fullmaskpath + "/segmentation_" + str(file_idx) + ".hdf5","r") as input_label_file:
                    input_label_data = input_label_file["data"]
                    for trench_idx in range(input_data.shape[0]):
                        trench_array = input_data[trench_idx]
                        input_label_array = input_label_data[trench_idx]
                        trench_array = self.segment(trench_array,label_arr=input_label_array)
                        trench_output.append(trench_array[np.newaxis])
                        del trench_array

            trench_output = np.concatenate(trench_output,axis=0)
            with h5py.File(self.segpath + "/segmentation_" + str(file_idx) + ".hdf5", "w") as h5pyfile:
                hdf5_dataset = h5pyfile.create_dataset("data", data=trench_output, dtype="uint16")
            del trench_output

        return file_idx

    def segmentation_completed(self,seg_future):
        return 0

    def dask_segment(self,dask_controller,overwrite=False):

        dask_controller.futures = {}

        writedir(self.segpath,overwrite=overwrite)

        # Check for existing files in case of in progress run
        if os.path.exists(self.segpath + "/progress.pkl"):
            with open(self.segpath + "/progress.pkl", 'rb') as infile:
                finished_files = pkl.load(infile)
        else:
            finished_files = []

        kymodf = dd.read_parquet(self.metapath,calculate_divisions=True)
        file_list = kymodf["File Index"].unique().compute().tolist()

        file_list = list(set(file_list) - set(finished_files))
        num_file_jobs = len(file_list)

        random_priorities = np.random.uniform(size=(num_file_jobs,))

        segmentation_futures_list = []
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]

            future = dask_controller.daskclient.submit(self.generate_segmentation,file_idx,retries=0,priority=priority)
            dask_controller.futures["Segmentation: " + str(file_idx)] = future
            segmentation_futures_list.append(future)

        for future in as_completed(segmentation_futures_list):
            result = future.result()
            finished_files = finished_files + [result]
            with open(self.segpath + "/progress.pkl", 'wb') as infile:
                pkl.dump(finished_files,infile)
            future.cancel()

        dask_controller.reset_worker_memory()
        # os.remove(self.fluorsegmentationpath + "/progress.pkl")
#         for k,file_idx in enumerate(file_list):
#             priority = random_priorities[k]

#             future = dask_controller.daskclient.submit(self.segmentation_completed,dask_controller.futures["Segmentation: " + str(file_idx)],retries=0,priority=priority)
#             dask_controller.futures["Segmentation Completed: " + str(file_idx)] = future
#         gathered_tasks = dask_controller.daskclient.gather([dask_controller.futures["Segmentation Completed: " + str(file_idx)] for file_idx in file_list],errors="skip")


class phase_segmentation:
    """Segmentation algorithm for high mag phase images.

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
        max_perc_contrast (float): Histogram percentile above which we cap intensities
                                (makes thresholding more robust)
        wrap_pad (int):
    """
    def __init__(self, init_niblack_k=-0.5, maxima_niblack_k=-0.55, init_smooth_sigma=4, maxima_smooth_sigma=4, init_niblack_window_size=13, maxima_niblack_window_size=13, min_cell_size=150, deviation_from_median=0.3, max_perc_contrast=97, wrap_pad = 0):

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
        """Remove segmented regions that are too large.

        Args:
            conn_comp(numpy.ndarray, int): Segmented connected components where each
                                        component is numbered
            max_size(int): Throw out regions where area is greater than this

        Returns:
            out(numpy.ndarray, int): Connected components array with large regions
                                    removed
        """
        out = np.copy(conn_comp)
        component_sizes = np.bincount(conn_comp.ravel())
        too_big = component_sizes > max_size
        too_big_mask = too_big[conn_comp]
        out[too_big_mask] = 0
        return out

    def to_8bit(self,img_arr,bit_max=None):
        """Rescale image and convert to 8bit.

        Args:
            img_arr(numpy.ndarray, int): input image to be rescaled
            bit_max(int): maximum value for intensities
        Returns:
            norm_byte_array(numpy.ndarray, int): 8-bit image
        """
        # Set maximum value
        img_max = np.max(img_arr)+0.0001
        if bit_max is None:
            max_val = img_max
        else:
            max_val = max(img_max,bit_max)
        # Set minimum value
        min_val = np.min(img_arr)
        # Scale
        norm_array = (img_arr-min_val)/(max_val-min_val)
        # Cast as 8bit image
        norm_byte_array = sk.img_as_ubyte(norm_array)
        return norm_byte_array

    def preprocess_img(self,img_arr,sigma=1.,bit_max=0):
        """Convert image stack to 8bit and smooth with gaussian filter.

        Args:
            img_arr(numpy.ndarray, int): input image stack (t x y x x)
            sigma(float):
            bit_max(int): maximum value for intensities
        Returns:
            img_smooth(numpy.ndarray, int): smoothed 8-bit image stack (t x y x x)
        """
        img_smooth = copy.copy(img_arr)
        for t in range(img_arr.shape[0]):
            img_smooth[t] = self.to_8bit(sk.filters.gaussian(img_arr[t],sigma=sigma,preserve_range=True,mode='reflect'),bit_max=bit_max)
        return img_smooth

    def detect_rough_trenches(self, img, show_plots=False):
        """Get rough estimate of where the trench is using otsu thresholding.

        Args:
            img (numpy.ndarray, int): Kymograph image to segment
            show_plots (bool): Whether to display the segmentation steps
        Returns:
            trench_masks (numpy.ndarray, bool): Rough estimates of trench masks
        """

        # Fill holes using morphologicla reconstruction
        def fill_holes(img):
            seed = np.copy(img)
            seed[1:-1, 1:-1] = img.max()
            strel = sk.morphology.square(3, dtype=bool)
            img_filled = sk.morphology.reconstruction(seed, img, selem=strel, method='erosion')
            img_filled = img_filled.astype(bool)
            return img_filled

        # Throw out intensity outliers
        max_val = np.percentile(img,self.max_perc_contrast)
        img_contrast = sk.exposure.rescale_intensity(img, in_range=(0,max_val))

        # Median filter to remove speckle
#         img_median = mh.median_filter(img_contrast,Bc=np.ones((2,2)))

        img_median = sk.filters.median(img_contrast,mask=np.ones((2,2)))

        pad_size = 6
        img_median = np.pad(img_median, ((0, 0), (pad_size, pad_size)), "minimum")
        # Edge detection (will pick up trench edges and cell edges)
        img_edges = sk.filters.sobel(img_median)

        # Otsu threshold the edges
        T_otsu = sk.filters.threshold_otsu(img_edges)
        img_otsu = (img_edges > T_otsu).astype(bool)
        dilation_kernel = np.array([0]*20 + [1]*21)[:, None]
        img_dilate = morph.binary_dilation(img_otsu, structure=dilation_kernel, iterations=3)
        img_filled = fill_holes(img_dilate)
        img_opened = morph.binary_opening(img_filled, structure = np.ones((1,5)),iterations=1)
        img_closed = morph.binary_closing(img_opened, structure = np.ones((3,3)),iterations=6)
        img_drop_filled = morph.binary_dilation(img_closed,structure=dilation_kernel, iterations=5)



#         # Close gaps in between
#         img_close = morph.binary_closing(img_otsu, structure = np.ones((3,3)),iterations=6)
#         img_close = img_close[3:-3, 3:-3]

        # Fill holes, remove small particles, and join the bulk of the trench
#         img_filled = fill_holes(img_otsu)
#         img_open = morph.binary_opening(img_filled, structure = np.ones((3,3)),iterations=2)
        trench_masks = morph.binary_erosion(img_drop_filled, structure = np.ones((1,3)),iterations=3)[:, pad_size:-pad_size]

        # Display plots if needed
        if show_plots:
            _, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(1, 9, figsize=(14,10))
            ax1.imshow(img)
            ax1.set_title("Original Image")
            ax2.imshow(img_edges)
            ax2.set_title("Edge detection")
            ax3.imshow(img_otsu)
            ax3.set_title("Thresholding")
            ax4.imshow(img_dilate)
            ax4.set_title("Drop fill 1")
            ax5.imshow(img_filled)
            ax5.set_title("Fill Holes")
            ax6.imshow(img_opened)
            ax6.set_title("Horizontal Opening")
            ax7.imshow(img_closed)
            ax7.set_title("Image closing")
            ax8.imshow(img_drop_filled)
            ax8.set_title("Drop fill 2")
            ax9.imshow(trench_masks)
            ax9.set_title("Final mask")
        return trench_masks

    def detect_trenches(self, img, show_plots=False):
        """Fully detect trenches in a kymograph image.

        Args:
            img (numpy.ndarray, int): Kymograph image to segment
            show_plots (bool): Whether to display the segmentation steps
        Returns:
            trench_masks (numpy.ndarray, bool): Rough estimates of trench masks
        """
        # Detect rough areas
        trench_masks = self.detect_rough_trenches(img, show_plots)
        # Erode to account for oversegmentation
        trench_masks = morph.binary_erosion(trench_masks,structure=np.ones((1,3)),iterations=3)
        trench_masks = morph.binary_erosion(trench_masks,structure=np.ones((3,1)),iterations=1)
        # Find regions in the mask
        conn_comp = sk.measure.label(trench_masks,neighbors=4)
        rp_area = [r.area for r in sk.measure.regionprops(conn_comp, cache=False)]
        # Select the largest region as the trench mask
        if len(rp_area) > 0:
            max_region = np.argmax(rp_area)+1
            trench_masks[conn_comp != max_region] = 0
        return trench_masks

    def medianFilter(self, img):
        """Median filter helper function."""
#         img_median = mh.median_filter(img,Bc=np.ones((3,3)))
        img_median = sk.filters.median(img,mask=np.ones((3,3)))
        return img_median

    def findCellsInTrenches(self, img, mask,sigma,window_size,niblack_k):
        """Segment cell regions (not separated yet by watershedding) in a
        trench.

        Args:
            img (numpy.ndarray, int): image
            mask (numpy.ndarray, bool): trench mask
            sigma (float): Gaussian filter kernel size
            window_size (int): Size of niblack filter
            niblack_k (float): K parameter for niblack filtering (higher is more rigorous)
        Returns:
            threshed (numpy.ndarray, bool): Mask for cell regions
        """
        img_smooth = img
        # Smooth the image
        if sigma > 0:
            img_smooth = sk.filters.gaussian(img,sigma=sigma,preserve_range=True,mode='reflect')

        # Apply Niblack threshold to identify white regions
        thresh_niblack = sk.filters.threshold_niblack(img_smooth, window_size = window_size, k= niblack_k)

        # Look for black regions (cells dark under phase)
        threshed = (img <= thresh_niblack).astype(bool)

        # Apply trench mask
        if mask is not None:
            threshed = threshed*mask
        return threshed

    def findWatershedMaxima(self, img,mask):
        """ Find watershed seeds using rigorous niblack threshold
        Args:
            img (numpy.ndarray, int): image
            mask (numpy.ndarray, bool): trench mask
        Returns:
            maxima (numpy.ndarray, bool): Mask for watershed seeds
        """
        maxima = self.findCellsInTrenches(img,mask,self.maxima_smooth_sigma,self.maxima_niblack_window_size,self.maxima_niblack_k)
        img_edges = sk.filters.sobel(img)

        # Otsu threshold the edges
        T_otsu = sk.filters.threshold_otsu(img_edges)
        img_otsu = (img_edges > T_otsu*0.7).astype(bool)*mask
        img_otsu = morph.binary_dilation(img_otsu, np.ones((3,3)))
        maxima = maxima*(~img_otsu)
        reg_props = sk.measure.regionprops(sk.measure.label(maxima,neighbors=4), cache=False)
        rp_area = [r.area for r in reg_props]
        # Throw out maxima regions that are too small
        med_size = np.median(rp_area)
        cutoff_size = int(max(0,med_size/6))
        maxima = sk.morphology.remove_small_objects(maxima,min_size=cutoff_size)
        return maxima

    def findWatershedMask(self, img,mask):
        """ Find cell regions to watershed using less rigorous niblack threshold
        Args:
            img (numpy.ndarray, int): image
            mask (numpy.ndarray, bool): trench mask
        Returns:
            img_mask (numpy.ndarray, bool): Mask for cell regions
        """
        img_mask = self.findCellsInTrenches(img,mask,self.init_smooth_sigma,self.init_niblack_window_size,self.init_niblack_k)

        img_mask = sk.morphology.binary_dilation(img_mask)
        img_mask = sk.morphology.binary_dilation(img_mask,selem==np.ones((1,3),dtype=np.bool))
#         img_mask = mh.dilate(mh.dilate(img_mask),Bc=np.ones((1,3),dtype=np.bool))
        img_mask = sk.morphology.remove_small_objects(img_mask,min_size=4)
        return img_mask

    def extract_connected_components_phase(self, threshed, maxima, return_all = False, show_plots=False):
        """Phase segmentation and connected components detection algorithm.

        :param img: numpy array containing image
        :param trench_masks: you can supply your own trench_mask rather
            than computing it each time
        :param flip_trenches: if mothers are on bottom of image set to
            True
        :param cut_from_bottom: how far to crop the bottom of the
            detected trenches to avoid impacting segmentation
        :param above_trench_pad: how muching padding above the mother
            cell
        :param init_smooth_sigma: how much smoothing to apply to the
            image for the initial niblack segmentation
        :param init_niblack_window_size: size of niblack window for
            segmentation
        :param init_niblack_k: k-offset for initial niblack segmentation
        :param maxima_smooth_sigma: how much smoothing to use for image
            that determines maxima used to seed watershed
        :param maxima_niblack_window_size: size of niblack window for
            maxima determination
        :param maxima_niblack_k: k-offset for maxima determination using
            niblack
        :param min_cell_size: minimum size of cell in pixels
        :param max_perc_contrast: scale contrast before median filter
            application
        :param return_all: whether just the connected component or the
            connected component, thresholded image pre-watershed and
            maxima used in watershed
        :return: if return_all = False: a connected component matrix if
            return_all = True: connected component matrix, thresholded
            image pre-watershed and maxima used in watershed
        """

        # Get distance transforms
#         distances = mh.stretch(mh.distance(threshed))
        distances = sp.ndimage.distance_transform_edt(threshed)
        distances = sk.exposure.rescale_intensity(distances)
        # Label seeds
        spots = sk.measure.label(maxima,neighbors=4)
#         spots, _ = mh.label(maxima,Bc=np.ones((3,3)))
        surface = (distances.max() - distances)
        # Watershed
        conn_comp = sk.morphology.watershed(surface, spots, mask=threshed)
        # Label connected components
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
        """Run segmentation for a single image.

        Args:
            img (numpy.ndarray, int): image
            return_all (bool): Whether to return intermediate masks in adition
                            to final connected components
            show_plots (bool): Whether to show plots
        Returns:
            conn_comp (numpy.ndarray, int): Mask of segmented cells, each cell
                                        has a different numbered label
            trench_masks (numpy.ndarray, bool): Mask of trench regions
            img_mask (numpy.ndarray, bool): Cells pre-watershedding
            maxima (numpy.ndarray, bool): Watershed seeds
        """
        trench_masks = self.detect_trenches(img, show_plots)
#         img_median = mh.median_filter(img,Bc=np.ones((3,3)))
        img_median = sk.filters.median(img,mask=np.ones((3,3)))
        img_mask = self.findWatershedMask(img_median,trench_masks)
        maxima = self.findWatershedMaxima(img_median,img_mask)
        conn_comp = self.extract_connected_components_phase(img_mask, maxima, show_plots=show_plots)

        if return_all:
            return conn_comp, trench_masks, img_mask, maxima
        else:
            return conn_comp

    def find_trench_masks_and_median_filtered_list(self, trench_array_list):
        """Find trench masks and median filtered images for a list of
        individual kymographs.

        Args:
            trench_array_list (list): List of numpy.ndarray (t x y x x) representing each
                                    kymograph
        Returns:
            trench_masks_list (list): List of boolean masks for the identified trenches
            img_medians_list (list): List of median-filtered images
        """
        trench_masks_list = np.empty_like(trench_array_list, dtype=bool)
        img_medians_list = np.empty_like(trench_array_list, dtype=np.uint8)
        for tr in range(trench_array_list.shape[0]):
            for t in range(trench_array_list.shape[1]):
                trench_masks_list[tr,t,:,:] = self.detect_trenches(trench_array_list[tr,t,:,:])
                img_medians_list[tr,t,:,:] = self.medianFilter(trench_array_list[tr,t,:,:])
        return trench_masks_list, img_medians_list

    def find_watershed_mask_list(self, trench_mask_and_median_filtered_list):
        """Find cell regions pre-watershed for a list of kymographs.

        Args:
            trench_mask_and_median_filtered_list (list):
                Lists of boolean masks for trenches, and median-filtered images
        Returns:
            watershed_masks_list (list): List of boolean masks for cell regions
            img_array_list (list): List of median-filtered images
        """
        trench_mask_array_list, img_array_list = trench_mask_and_median_filtered_list
        watershed_masks_list = np.empty_like(img_array_list, dtype=bool)
        for tr in range(img_array_list.shape[0]):
            for t in range(img_array_list.shape[1]):
                watershed_masks_list[tr,t,:,:] = self.findWatershedMask(img_array_list[tr,t,:,:], trench_mask_array_list[tr,t,:,:])
        return watershed_masks_list, img_array_list

    def find_watershed_maxima_list(self, watershed_mask_and_median_filtered_list):
        """Find watershed seeds for a list of kymographs.

        Args:
            watershed_mask_and_median_filtered_list (list):
                Lists of boolean masks for cell regions, and median-filtered images
        Returns:
            watershed_maxima_list (list): List of boolean masks for watershed seeds
            masks_list (list): List of boolean masks for cell regions
        """
        masks_list, img_array_list = watershed_mask_and_median_filtered_list
        watershed_maxima_list = np.empty_like(img_array_list, dtype=bool)
        for tr in range(img_array_list.shape[0]):
            for t in range(img_array_list.shape[1]):
                watershed_maxima_list[tr,t,:,:] = self.findWatershedMaxima(img_array_list[tr,t,:,:], masks_list[tr,t,:,:])
        return watershed_maxima_list, masks_list

    def find_conn_comp_list(self, watershed_maxima_and_masks_list): #masks_arr is tr,t,y,x
        """Watershed cells and return final segmentations for a list of
        kymoraphs.

        Args:
            watershed_maxima_and_masks_list (list):
                Lists of boolean masks for watershed seeds and for cell regions
        Returns:
            watershed_maxima_list (list): List of boolean masks for watershed seeds
            masks_list (list): List of boolean masks for cell regions
        """
        maxima_list, masks_list = watershed_maxima_and_masks_list
        final_masks_list = np.empty_like(masks_list, dtype=np.uint8)
        for tr in range(masks_list.shape[0]):
            for t in range(masks_list.shape[1]):
                final_masks_list[tr,t,:,:] = self.extract_connected_components_phase(masks_list[tr,t,:,:], maxima_list[tr,t,:,:])
        return final_masks_list

    def get_trench_loading_fraction(self, img):
        """Calculate loading fraction, basically how much of the trench mask is
        filled with black stuff.

        Args:
            img (numpy.ndarray): Image to detect loading fraction

        Returns:
            loading_fraction (float): How full the trench is, normal
            values are between usually 0.3 and 0.75
        """
        # Get trench mask
        trench_mask = self.detect_trenches(img)
        # Find black regions
        loaded = (img < sk.filters.threshold_otsu(img)) * trench_mask
        # Find total area
        denom = np.sum(trench_mask)
        if denom == 0:
            return 0
        else:
            return np.sum(loaded)/denom

    def measure_trench_loading(self, trench_array_list):
        """Measure trench loadings for a list of kymographs.

        Args:
            trench_array_list(numpy.ndarray, int): tr x t x y x x array of kymograph images
        Returns:
            trench_loadings (numpy.ndarray, float): tr x t array of trench loading
        """
        trench_loadings = np.empty((trench_array_list.shape[0], trench_array_list.shape[1]))
        for tr in range(trench_array_list.shape[0]):
            for t in range(trench_array_list.shape[1]):
                trench_loadings[tr, t] = self.get_trench_loading_fraction(trench_array_list[tr, t, :, :])
        trench_loadings = trench_loadings.flatten()
        return trench_loadings

class phase_segmentation_cluster(phase_segmentation):
    """Class for handling cell segmentation and extraction of morphological
    properties.

    Attributes:
        headpath: base analysis directory
        seg_channel: segmentation image channel
        kymographpath: kymograph image directory
        phasesegmentationpath: directory to save segmented images
        phasedatapath: directory to save morphological data
        metapath: path to metadata
        meta_handle: helper for accessing metadata
        metadf: in memory storage of metadata pandas dataframe
        bit_max: maximum intensity value for images
    """
    def __init__(self,headpath,paramfile=True,seg_channel="", init_niblack_k=-0.45, maxima_niblack_k=-0.55, init_smooth_sigma=4, maxima_smooth_sigma=4, init_niblack_window_size=13, maxima_niblack_window_size=13, min_cell_size=100, deviation_from_median=0.3, max_perc_contrast=97, wrap_pad=0):

        super(phase_segmentation_cluster, self).__init__(init_niblack_k=init_niblack_k, init_smooth_sigma=init_smooth_sigma, maxima_smooth_sigma=maxima_smooth_sigma,maxima_niblack_k=maxima_niblack_k, init_niblack_window_size=maxima_niblack_window_size, maxima_niblack_window_size=maxima_niblack_window_size, min_cell_size=min_cell_size, deviation_from_median=deviation_from_median, max_perc_contrast=max_perc_contrast, wrap_pad = wrap_pad)

        self.headpath = headpath
        self.seg_channel = seg_channel
        self.kymographpath = headpath + "/kymograph"
        self.phasesegmentationpath = headpath + "/phasesegmentation"
        self.phasedatapath = self.phasesegmentationpath + "/cell_data"
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        self.metadf = None
        self.bit_max = None

    def get_num_trenches_timepoints(self):
        """Count number of trenches and times from metadata.

        Args:
            None
        Returns:
            num_trenchid (int): number of trenches in this dataset
            num_time (int): number of timepoints in this dataset
        """
        # Load metadata
        metadf = self.meta_handle.read_df("kymograph",read_metadata=True)
        # Get data from lengths of indices
        num_trenchid = len(metadf.index.unique("trenchid"))
        num_time = len(metadf.index.unique("timepoints"))
        return num_trenchid, num_time

    def view_kymograph(self, trench_idx, timepoint, channel):
        """Visualize a single timepoint in a kymograph.

        Args:
            trench_idx (int): Trench index within a file
            timepoint (int): Time to view
            channel (str): Image channel
        Returns:
            None
        """
        if self.metadf is None:
            self.metadf = self.meta_handle.read_df("kymograph",read_metadata=True)
        img_entry = self.metadf.loc[trench_idx,timepoint]
        file_idx = int(img_entry["File Index"])
        trench_idx = int(img_entry["File Trench Index"])
        with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5", "r") as infile:
            img_arr = infile[channel][trench_idx,timepoint,:,:]
        plt.imshow(img_arr)

    def load_trench_array_list(self, path_form, file_idx, key, to_8bit):
        """Load all the trenches in a file.

        Args:
            path_form (str): filename pattern for hdf5s
            file_idx (int): file index of hdf5 archive
            key (str): channel name
            to_8bit (bool): whether to turn the data to 8-bit
        Returns:
            trench_array_list (numpy.ndarray, int): all timepoints for all trenches (tr x t x y x x)
        """
        with h5py.File(path_form + str(file_idx) + ".hdf5","r") as input_file:
            if to_8bit:
                trench_array_list = np.empty_like(input_file[key], dtype=np.uint8)
                for tr in range(trench_array_list.shape[0]):
                    for t in range(trench_array_list.shape[1]):
                        trench_array_list[tr,t,:,:] = self.to_8bit(input_file[key][tr,t,:,:])
            else:
                trench_array_list = input_file[key][:]
            return trench_array_list

    def save_masks_to_hdf(self, file_idx, final_masks_future):
        """Save segmented data to hdf5 archives.

        Args:
            file_idx (int): file index of the hdf5 kymograph
            final_masks_future (numpy.ndarray, int): (tr x t x y x x) stack of segmented trenches
        Returns:
            "Done"
        """
        with h5py.File(self.phasesegmentationpath + "/segmentation_" + str(file_idx) + ".hdf5", "w") as h5pyfile:
            hdf5_dataset = h5pyfile.create_dataset("data", data=final_masks_future, dtype=np.uint8)
        return "Done"

    def generate_trench_loading(self, file_idx):
        """Measure trench loading for all trenches in file.

        Args:
            file_idx (int): file index of the hdf5 kymograph
        Returns:
            trench_output (numpy.ndarray): (tr x t) array of trench laoding
        """
        # Load file
        with h5py.File(self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5","r") as input_file:
            input_data = input_file[self.seg_channel]
            trench_output = []
            # Measure loading for each trench
            for trench_idx in range(input_data.shape[0]):
                trench_array = input_data[trench_idx]
                trench_loading_array = self.measure_trench_loading(trench_array)
                trench_output.append(trench_loading_array[np.newaxis])
        trench_output = np.concatenate(trench_output,axis=0)
        return trench_output

    def dask_segment(self,dask_controller, file_list=None, overwrite=True):
        """Segment kymographs in parallel using Dask.

        Args:
            dask_controller (trenchripper.dask_controller): Helper object to handle dask jobs
            file_list (list, int): Subset kymograph files
            overwrite (bool): Whether to overwrite the output directory
        Returns:
            None
        """
        # Make/overwrite output directory
        writedir(self.phasesegmentationpath,overwrite=overwrite)

        dask_controller.futures = {}
        if file_list is None:
            file_list = self.meta_handle.read_df("kymograph",read_metadata=True)["File Index"].unique().tolist()

        num_file_jobs = len(file_list)

        # Send dask jobs with increasing priority (to reduce memory usage)
        random_priorities = np.random.uniform(size=(num_file_jobs,6))
        for k,file_idx in enumerate(file_list):
            # Load trenches
            future = dask_controller.daskclient.submit(self.load_trench_array_list, self.kymographpath + "/kymograph_", file_idx, self.seg_channel, True, retries=1,priority=random_priorities[k,0]*0.1)
            # Find trench masks and median filter images
            future = dask_controller.daskclient.submit(self.find_trench_masks_and_median_filtered_list,future,retries=1,priority=random_priorities[k,1]*0.4)
            # Get cell regions
            future = dask_controller.daskclient.submit(self.find_watershed_mask_list,future,retries=1,priority=random_priorities[k,2]*1.6)
            # Get watershed seeds
            future = dask_controller.daskclient.submit(self.find_watershed_maxima_list,future,retries=1,priority=random_priorities[k,3]*6.4)
            # Get connected components
            future = dask_controller.daskclient.submit(self.find_conn_comp_list,future,retries=1,priority=random_priorities[k,4]*25.6)
            # Save to file
            future = dask_controller.daskclient.submit(self.save_masks_to_hdf,file_idx,future,retries=1,priority=random_priorities[k,5]*51.2)
            dask_controller.futures["Segmentation: " + str(file_idx)] = future

    def dask_characterize_trench_loading(self, dask_controller, file_list=None):
        """Measure trench loading for the whole dataset in parallel.

        Args:
            dask_controller (trenchripper.dask_controller): Helper object to handle dask jobs
            file_list (list, int): Subset kymograph files
        Returns:
            None
        """
        dask_controller.futures = {}
        dask_controller.futures["Trench Loading"] = []

        if file_list is None:
            file_list = self.meta_handle.read_df("kymograph",read_metadata=True)["File Index"].unique().tolist()
        num_file_jobs = len(file_list)

        random_priorities = np.random.uniform(size=(num_file_jobs,2))
        for k,file_idx in enumerate(file_list):
            # Load data
            future = dask_controller.daskclient.submit(self.load_trench_array_list, self.kymographpath + "/kymograph_", file_idx, self.seg_channel, True, retries=1,priority=random_priorities[k,0]*0.1)
            # Measure loading
            future = dask_controller.daskclient.submit(self.measure_trench_loading,future,retries=1,priority=random_priorities[k,0]*0.8)
            # Save to futures
            dask_controller.futures["Trench Loading"].append(future)

    def dask_postprocess_trench_loading(self, dask_controller):
        """Add trench loading to metadata.

        Args:
            dask_controller (trenchripper.dask_controller): Helper object to handle dask jobs
        Returns:
            None
        """
        # Concatenate future results
        trench_loadings = np.concatenate(dask_controller.daskclient.gather(dask_controller.futures["Trench Loading"]), axis=0)
        # Add to metadata frame
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        kymodf["Trench Loading"] = trench_loadings
        # Save
        self.meta_handle.write_df("kymograph", kymodf, metadata=kymodf.metadata)

    def props_to_dict(self, regionprops, props_to_grab):
        """Select properties from skimage regionprops object and turn into
        dictionary.

        Args:
            regionprops (skimage.regionprops): regionprops objects for each cell
            props_to_grab(list, str): metrics to extract from regionprops data
        Returns:
            props_dict(dict): dictionary of morphology metrics
        """
        props_dict = {}
        for prop in props_to_grab:
            props_dict[prop] = list(map(lambda x: x[prop], regionprops))
        del regionprops
        return props_dict

    def dask_extract_cell_data(self, dask_controller, props_to_grab, file_list=None, overwrite=True):
        """Extract cell morphology measurements.

        Args:
            dask_controller (trenchripper.dask_controller): Helper object to handle dask jobs
            props_to_grab(list, str): metrics to extract from regionprops data
            file_list (list, int): Subset kymograph files
            overwrite (bool): Whether to overwrite the output directory
        Returns:
            None
        """
        dask_controller.futures = {}
        # write directory
        writedir(self.phasedatapath,overwrite=overwrite)

        # load metadata
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        metadata = kymodf.metadata

        globaldf = self.meta_handle.read_df("global", read_metadata=True)
        # get pixel scaling so that measurements are in micrometers
        pixel_scaling = metadata["pixel_microns"]


        if file_list is None:
            file_list = kymodf["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
        # index according to how the final cell data should be organized
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

    def dask_extract_cell_data_mask(self, dask_controller, channels, props_to_grab, file_list=None, overwrite=False):
        """Extract cell fluorescence properties using phase segmentation graph.

        Args:
            dask_controller (trenchripper.dask_controller): Helper object to handle dask jobs
            props_to_grab(list, str): metrics to extract from regionprops data
            file_list (list, int): Subset kymograph files
            overwrite (bool): Whether to overwrite the output directory
        Returns:
            None
        """
        dask_controller.futures = {}
        # write directory
        writedir(self.phasedatapath,overwrite=overwrite)

        # load metadata
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        metadata = kymodf.metadata

        globaldf = self.meta_handle.read_df("global", read_metadata=True)
        # get pixel scaling so that measurements are in micrometers
        pixel_scaling = globaldf.metadata["pixel_microns"]


        if file_list is None:
            file_list = kymodf["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
        num_channels = len(channels)
        # index according to how the final cell data should be organized
        kymodf = kymodf.reset_index()
        kymodf = kymodf.set_index(["File Index", "File Trench Index", "timepoints"])

        random_priorities = np.random.uniform(size=(num_file_jobs,num_channels))
        for k,file_idx in enumerate(file_list):
            for k2, channel in enumerate(channels):
                segmented_masks_file = "segmentation_" + str(file_idx) + ".hdf5"
                if segmented_masks_file in os.listdir(self.phasesegmentationpath):
                    times = kymodf.loc[file_idx, "time (s)"]
                    global_trench_indices = kymodf.loc[file_idx, "trenchid"]
                    trench_loadings = kymodf.loc[file_idx, "Trench Loading"]
                    fov_idx = kymodf.loc[file_idx, "fov"]
                    dask_controller.futures["Cell Props %d: " % file_idx] = dask_controller.daskclient.submit(self.extract_cell_data_mask, file_idx, fov_idx, channel, times, global_trench_indices, trench_loadings, props_to_grab, pixel_scaling, metadata, priority=random_priorities[k, k2]*8)

    def extract_cell_data(self, file_idx, fov_idx, times, global_trench_indices, trench_loadings, props_to_grab, pixel_scaling, metadata=None):
        """Get cell morphology data from segmented trenches.

        Args:
            file_idx (int): hdf5 file index
            fov_idx (int): Field of view in the original data
            times: (list, float): Time indices to look at
            global_trench_indices (list, int): Trench IDs
            props_to_grab (list, str): list of properties to grab
            pixel_scaling (float): microns/pixel
            metadata (pandas.Dataframe): metdata dataframe
        Returns:
            "Done"
        """
        # Get segmented masks
        segmented_mask_array = self.load_trench_array_list(self.phasesegmentationpath + "/segmentation_", file_idx, "data", False)
        # Load output file
        with HDFStore(os.path.join(self.phasedatapath, "data_%d.h5" % file_idx)) as store:
            if "/metrics" in store.keys():
                store.remove("/metrics")
            trench_time_dataframes = {}
            first_dict_flag = True
            # Iterate through trenches and times
            for trench_idx in range(segmented_mask_array.shape[0]):
                for time_idx in range(segmented_mask_array.shape[1]):
                    # get regionprops
                    seg_props = sk.measure.regionprops(segmented_mask_array[trench_idx, time_idx,:,:],cache=False)
                    if len(seg_props) > 0:
                        # Convert to dict
                        seg_props = self.props_to_dict(seg_props, props_to_grab)
                        # Add metadata
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

            # Convert to dataframe
            seg_props = pd.DataFrame(trench_time_dataframes)
            # Rename label to trench index
            seg_props["trench_cell_index"] = seg_props["label"]
            seg_props = seg_props.drop("label", axis=1)

            # Convert bounding box to multiple columns
            if "bbox" in props_to_grab:
                seg_props[['min_row', 'min_col', 'max_row', 'max_col']] = pd.DataFrame(seg_props['bbox'].tolist(), index=seg_props.index)
                seg_props = seg_props.drop("bbox", axis=1)
            # Convert centroid to multiple columns
            if "centroid" in props_to_grab:
                seg_props[['centy', 'centx']] = pd.DataFrame(seg_props['centroid'].tolist(), index=seg_props.index)
                seg_props = seg_props.drop("centroid", axis=1)

            # Convert area and lengths to pixels
            length_scale_measurements = set(["major_axis_length", "equivalent_diameter", "minor_axis_length", "perimeter"])
            for prop in props_to_grab:
                if prop in length_scale_measurements:
                    seg_props[prop] = seg_props[prop]*pixel_scaling
            if "area" in props_to_grab:
                seg_props["area"] = seg_props["area"]*(pixel_scaling)**2
            # Index
            seg_props = seg_props.set_index(["file_trench_index", "time_s", "trench_cell_index"])
            # Save file
            store.put("metrics", seg_props, data_columns=True)
            if metadata is not None:
                store.get_storer("metrics").attrs.metadata = metadata
            del seg_props
            del trench_time_dataframes
        return "Done"

    def extract_cell_data_mask(self, file_idx, fov_idx, channel, times, global_trench_indices, trench_loadings, props_to_grab, pixel_scaling, metadata=None):
            """Get cell morphology data from segmented trenches.

            Args:
                file_idx (int): hdf5 file index
                fov_idx (int): Field of view in the original data
                channel (str): channel to measure intensities on
                times: (list, float): Time indices to look at
                global_trench_indices (list, int): Trench IDs
                props_to_grab (list, str): list of properties to grab
                pixel_scaling (float): microns/pixel
                metadata (pandas.Dataframe): metdata dataframe
            Returns:
                "Done"
            """
            # Get segmented masks
            segmented_mask_array = self.load_trench_array_list(self.phasesegmentationpath + "/segmentation_", file_idx, "data", False)
            # Get kymographs for other channel
            kymograph = self.load_trench_array_list(self.kymographpath + "/kymograph_", file_idx, self.seg_channel, False)
            # Load output file
            with HDFStore(os.path.join(self.phasedatapath, "data_%d.h5" % file_idx)) as store:
                if "/metrics_%s" % channel in store.keys():
                    store.remove("/metrics_%s" % channel)
                trench_time_dataframes = {}
                first_dict_flag = True
                # Iterate through trenches and times
                for trench_idx in range(segmented_mask_array.shape[0]):
                    for time_idx in range(segmented_mask_array.shape[1]):
                        # get regionprops
                        seg_props = sk.measure.regionprops(segmented_mask_array[trench_idx, time_idx,:,:], intensity_image=kymograph[trench_idx, time_idx,:,:], cache=False)
                        if len(seg_props) > 0:
                            # Convert to dict
                            seg_props = self.props_to_dict(seg_props, props_to_grab)
                            # Add metadata
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

                # Convert to dataframe
                seg_props = pd.DataFrame(trench_time_dataframes)
                # Rename label to trench index
                seg_props["trench_cell_index"] = seg_props["label"]
                seg_props = seg_props.drop("label", axis=1)

                # Convert bounding box to multiple columns
                if "bbox" in props_to_grab:
                    seg_props[['min_row', 'min_col', 'max_row', 'max_col']] = pd.DataFrame(seg_props['bbox'].tolist(), index=seg_props.index)
                    seg_props = seg_props.drop("bbox", axis=1)
                # Convert centroid to multiple columns
                if "centroid" in props_to_grab:
                    seg_props[['centy', 'centx']] = pd.DataFrame(seg_props['centroid'].tolist(), index=seg_props.index)
                    seg_props = seg_props.drop("centroid", axis=1)

                # Convert area and lengths to pixels
                length_scale_measurements = set(["major_axis_length", "equivalent_diameter", "minor_axis_length", "perimeter"])
                for prop in props_to_grab:
                    if prop in length_scale_measurements:
                        seg_props[prop] = seg_props[prop]*pixel_scaling
                if "area" in props_to_grab:
                    seg_props["area"] = seg_props["area"]*(pixel_scaling)**2
                # Index
                seg_props = seg_props.set_index(["file_trench_index", "time_s", "trench_cell_index"])
                # Save file
                store.put("/metrics_%s" % channel, seg_props, data_columns=True)
                del seg_props
                del trench_time_dataframes
            return "Done"
