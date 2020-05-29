# fmt: off
import os
import h5py
import torch
import copy
import ipywidgets as ipyw
import scipy
import pandas as pd
import datetime
import time
import itertools
import qgrid
import shutil
import subprocess

from random import shuffle
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate,ndimage

import skimage as sk
import pickle as pkl
import skimage.morphology
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .utils import pandas_hdf5_handler,kymo_handle,writedir
from .trcluster import hdf5lock,dask_controller
from .metrics import object_f_scores

from matplotlib import pyplot as plt

class weightmap_generator:
    def __init__(self,nndatapath,w0,wm_sigma):
        self.nndatapath = nndatapath
        self.w0 = w0
        self.wm_sigma = wm_sigma

    def make_weight_map(self,binary_mask):
        ttl_count = binary_mask.size
        cell_count = np.sum(binary_mask)
        background_count = ttl_count - cell_count
        class_weight = np.array([ttl_count/(background_count+1),ttl_count/(cell_count+1)])
        class_weight = class_weight/np.sum(class_weight)

        labeled = sk.measure.label(binary_mask)
        labels = np.unique(labeled)[1:]

        dist_maps = []
        borders = []
        
        num_labels = len(labels)
        
        if num_labels == 0:
            weight = np.ones(binary_mask.shape)*class_weight[0]
        elif num_labels == 1:
            cell = labeled==1
#             dilated = sk.morphology.binary_dilation(cell)
            eroded = sk.morphology.binary_dilation(cell)
            border = eroded^cell
            weight = np.ones(binary_mask.shape)*class_weight[0]
            weight[binary_mask] += class_weight[1]
#             weight[border] = 0.
        else:
            for i in labels:
                cell = labeled==i
#                 dilated = sk.morphology.binary_dilation(cell)
                eroded = sk.morphology.binary_dilation(cell)
                border = eroded^cell
                borders.append(border)
                dist_map = scipy.ndimage.morphology.distance_transform_edt(~border)
                dist_maps.append(dist_map)
            dist_maps = np.array(dist_maps)
            borders = np.array(borders)
            borders = np.max(borders,axis=0)
            dist_maps = np.sort(dist_maps,axis=0)
            weight = self.w0*np.exp(-((dist_maps[0] + dist_maps[1])**2)/(2*(self.wm_sigma**2)))
            weight[binary_mask] += class_weight[1]
            weight[~binary_mask] += class_weight[0]
#             weight[borders] = 0.
        return weight
    
    def make_weightmaps(self,seg_arr):
        num_indices = seg_arr.shape[0]
        weightmap_arr = []
        for t in range(0,num_indices):
            working_seg_arr = seg_arr[t,0].astype(bool)
            weightmap = self.make_weight_map(working_seg_arr).astype("float32")
            weightmap_arr.append(weightmap)
        weightmap_arr = np.array(weightmap_arr)[:,np.newaxis,:,:]
        return weightmap_arr
            
class data_augmentation:
    def __init__(self,p_flip=0.5,max_rot=10,min_padding=20):
        self.p_flip = p_flip
        self.max_rot = max_rot
        self.min_padding = min_padding
                
#     def make_chunked_kymograph(self,img_arr,chunksize=10):
#         pad = (chunksize - (img_arr.shape[2]%chunksize))*img_arr.shape[1]
#         chunked_arr = np.swapaxes(img_arr,1,2)
#         chunked_arr = chunked_arr.reshape(chunked_arr.shape[0],-1)
#         chunked_arr = np.pad(chunked_arr,((0,0),(0,pad)),'constant',constant_values=0)
#         chunked_arr = chunked_arr.reshape(chunked_arr.shape[0],-1,img_arr.shape[1]*chunksize)
#         chunked_arr = np.swapaxes(chunked_arr,1,2)
#         return chunked_arr
    
    def random_crop(self,img_arr,seg_arr):
        false_arr = np.zeros(img_arr.shape[2:4],dtype=bool)
        random_crop_len_y = np.random.uniform(low=0.1,high=1.,size=(1,img_arr.shape[0]))
        random_crop_len_x = np.random.uniform(low=0.4,high=1.,size=(1,img_arr.shape[0]))
        
        random_crop_len = np.concatenate([random_crop_len_y,random_crop_len_x],axis=0)
        
        random_crop_remainder = 1.-random_crop_len
        random_crop_start = (np.random.uniform(low=0.,high=1.,size=(2,img_arr.shape[0])))*random_crop_remainder
        low_crop = np.floor(random_crop_start*np.array(img_arr.shape[2:4])[:,np.newaxis]).astype('int32')
        high_crop = np.floor(low_crop+(random_crop_len*np.array(img_arr.shape[2:4])[:,np.newaxis])).astype('int32')
#         random_low_samples = np.random.uniform(low=0.,high=0.5,size=(2,img_arr.shape[0]))
#         low_crop = (random_low_samples*np.array(img_arr.shape[2:4])[:,np.newaxis]).astype('int32')
#         remainder = np.array(img_arr.shape[2:4])[:,np.newaxis]-low_crop
#         random_high_samples = np.random.uniform(low=0.5,high=1.,size=(2,img_arr.shape[0]))
#         high_crop = np.floor(random_high_samples*remainder).astype('int32')+low_crop
        out_arr = []
        out_seg_arr = []
        center = (img_arr.shape[2]//2,img_arr.shape[3]//2)
        for t in range(img_arr.shape[0]):
            mask = copy.copy(false_arr)
            working_arr = copy.copy(img_arr[t,0,:,:])
            working_seg_arr = copy.copy(seg_arr[t,0,:,:])

            dim_0_range = (high_crop[0,t] - low_crop[0,t])
            dim_1_range = high_crop[1,t] - low_crop[1,t]
            top_left = (center[0]-dim_0_range//2,center[1]-dim_1_range//2)

            dim_0_maxscale = img_arr.shape[2]/dim_0_range
            dim_1_maxscale = img_arr.shape[3]/dim_1_range

            dim_0_scale = np.clip(np.random.normal(loc=1.0,scale=0.1),0.8,dim_0_maxscale)
            dim_1_scale = np.clip(np.random.normal(loc=1.0,scale=0.1),0.8,dim_1_maxscale)

            rescaled_img = sk.transform.rescale(working_arr[low_crop[0,t]:high_crop[0,t],low_crop[1,t]:high_crop[1,t]],(dim_0_scale,dim_1_scale),preserve_range=True).astype(int)
            rescaled_seg = (sk.transform.rescale(working_seg_arr[low_crop[0,t]:high_crop[0,t],low_crop[1,t]:high_crop[1,t]]==1,(dim_0_scale,dim_1_scale))>0.5).astype("int8")
#             rescaled_border = (sk.transform.rescale(working_seg_arr[low_crop[0,t]:high_crop[0,t],low_crop[1,t]:high_crop[1,t]]==2,(dim_0_scale,dim_1_scale))>0.5)
#             rescaled_seg[rescaled_border] = 2

            top_left = (center[0]-rescaled_img.shape[0]//2,center[1]-rescaled_img.shape[1]//2)        
            working_arr[top_left[0]:top_left[0]+rescaled_img.shape[0],top_left[1]:top_left[1]+rescaled_img.shape[1]] = rescaled_img
            working_seg_arr[top_left[0]:top_left[0]+rescaled_img.shape[0],top_left[1]:top_left[1]+rescaled_img.shape[1]] = rescaled_seg

            mask[top_left[0]:top_left[0]+rescaled_img.shape[0],top_left[1]:top_left[1]+rescaled_img.shape[1]] = True
            working_arr[~mask] = 0
            working_seg_arr[~mask] = False        
            
            out_arr.append(working_arr)
            out_seg_arr.append(working_seg_arr)
        out_arr = np.expand_dims(np.array(out_arr),1)
        out_seg_arr = np.expand_dims(np.array(out_seg_arr),1)
        return out_arr,out_seg_arr
    
    def random_x_flip(self,img_arr,seg_arr,p=0.5):
        choices = np.random.choice(np.array([True,False]),size=img_arr.shape[0],p=np.array([p,1.-p]))
        out_img_arr = copy.copy(img_arr)
        out_seg_arr = copy.copy(seg_arr)
        out_img_arr[choices,0,:,:] = np.flip(img_arr[choices,0,:,:],axis=1)
        out_seg_arr[choices,0,:,:] = np.flip(seg_arr[choices,0,:,:],axis=1)
        return out_img_arr,out_seg_arr
    def random_y_flip(self,img_arr,seg_arr,p=0.5):
        choices = np.random.choice(np.array([True,False]),size=img_arr.shape[0],p=np.array([p,1.-p]))
        out_img_arr = copy.copy(img_arr)
        out_seg_arr = copy.copy(seg_arr)
        out_img_arr[choices,0,:,:] = np.flip(img_arr[choices,0,:,:],axis=2)
        out_seg_arr[choices,0,:,:] = np.flip(seg_arr[choices,0,:,:],axis=2)
        return out_img_arr,out_seg_arr
    
    def change_brightness(self,img_arr,num_control_points=3):
        out_img_arr = copy.copy(img_arr)
        for t in range(img_arr.shape[0]):
            control_points = (np.add.accumulate(np.ones(num_control_points+2))-1.)/(num_control_points+1)
            control_point_locations = (control_points*65535).astype(int)
            orig_locations = copy.copy(control_point_locations)
            random_points = np.random.uniform(low=0,high=65535,size=num_control_points).astype(int)
            sorted_points = np.sort(random_points)
            control_point_locations[1:-1] = sorted_points
            mapping = interpolate.PchipInterpolator(orig_locations, control_point_locations)
            out_img_arr[t,0,:,:] = mapping(img_arr[t,0,:,:])
        return out_img_arr
    
    
    def add_padding(self,img_arr,seg_arr,max_rot=20,min_padding=20):
        hyp_length = np.ceil((img_arr.shape[2]**2+img_arr.shape[3]**2)**(1/2)).astype(int)
        max_rads = ((90-max_rot)/360)*(2*np.pi)
        min_rads = (90/360)*(2*np.pi)
        max_y = np.maximum(np.ceil(hyp_length*np.sin(max_rads)),np.ceil(hyp_length*np.sin(min_rads))).astype(int)
        max_x = np.maximum(np.ceil(hyp_length*np.cos(max_rads)),np.ceil(hyp_length*np.cos(min_rads))).astype(int)
        delta_y = max_y-img_arr.shape[2]
        delta_x = max_x-img_arr.shape[3]
        if delta_x % 2 == 1:
            delta_x+=1
        if delta_y % 2 == 1:
            delta_y+=1
        delta_y = np.maximum(delta_y,2*min_padding)
        delta_x = np.maximum(delta_x,2*min_padding)
        padded_img_arr = np.pad(img_arr, ((0,0),(0,0),(delta_y//2,delta_y//2),(delta_x//2,delta_x//2)), 'constant', constant_values=0)
        padded_seg_arr = np.pad(seg_arr, ((0,0),(0,0),(delta_y//2,delta_y//2),(delta_x//2,delta_x//2)), 'constant', constant_values=0)
        return padded_img_arr,padded_seg_arr
    
    def translate(self,pad_img_arr,pad_seg_arr,img_arr,seg_arr):
        trans_img_arr = copy.copy(pad_img_arr)
        trans_seg_arr = copy.copy(pad_seg_arr)
        delta_y = pad_img_arr.shape[2] - img_arr.shape[2]
        delta_x = pad_img_arr.shape[3] - img_arr.shape[3]
        for t in range(pad_img_arr.shape[0]):
            trans_y = np.random.randint(-(delta_y//2),high=delta_y//2)
            trans_x = np.random.randint(-(delta_x//2),high=delta_x//2)
            trans_img_arr[t,0,delta_y//2:delta_y//2+img_arr.shape[2],delta_x//2:delta_x//2+img_arr.shape[3]] = 0
            trans_seg_arr[t,0,delta_y//2:delta_y//2+img_arr.shape[2],delta_x//2:delta_x//2+img_arr.shape[3]] = 0
            trans_img_arr[t,0,delta_y//2+trans_y:delta_y//2+img_arr.shape[2]+trans_y,delta_x//2+trans_x:delta_x//2+img_arr.shape[3]+trans_x] =\
            pad_img_arr[t,0,delta_y//2:delta_y//2+img_arr.shape[2],delta_x//2:delta_x//2+img_arr.shape[3]]
            trans_seg_arr[t,0,delta_y//2+trans_y:delta_y//2+img_arr.shape[2]+trans_y,delta_x//2+trans_x:delta_x//2+img_arr.shape[3]+trans_x] =\
            pad_seg_arr[t,0,delta_y//2:delta_y//2+img_arr.shape[2],delta_x//2:delta_x//2+img_arr.shape[3]]
        return trans_img_arr,trans_seg_arr
    
    def rotate(self,img_arr,seg_arr,max_rot=20):
        rot_img_arr = copy.copy(img_arr)
        rot_seg_arr = copy.copy(seg_arr)
        for t in range(img_arr.shape[0]):
            r = np.random.uniform(low=-max_rot,high=max_rot)
            rot_img_arr[t,0,:,:] = sk.transform.rotate(img_arr[t,0,:,:],r,preserve_range=True).astype("int32")
            rot_seg = (sk.transform.rotate(seg_arr[t,0,:,:]==1,r)>0.5).astype("int8")
#             rot_border = sk.transform.rotate(seg_arr[t,:,:]==2,r)>0.5
#             rot_seg[rot_border] = 2
            rot_seg_arr[t,0,:,:] = rot_seg
        return rot_img_arr,rot_seg_arr

    def deform_img_arr(self,img_arr,seg_arr):
        def_img_arr = copy.copy(img_arr)
        def_seg_arr = copy.copy(seg_arr)
        for t in range(img_arr.shape[0]):
            y_steps = np.linspace(0.,4.,num=img_arr.shape[2])
            x_steps = np.linspace(0.,4.,num=img_arr.shape[3])
            grid = np.random.normal(scale=1.,size=(2,4,4))
            dx = RectBivariateSpline(np.arange(4),np.arange(4),grid[0]).ev(y_steps[:,np.newaxis],x_steps[np.newaxis,:])
            dy = RectBivariateSpline(np.arange(4),np.arange(4),grid[1]).ev(y_steps[:,np.newaxis],x_steps[np.newaxis,:])
            y,x = np.meshgrid(np.arange(img_arr.shape[2]), np.arange(img_arr.shape[3]), indexing='ij')
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            elastic_img = map_coordinates(img_arr[t,0,:,:], indices, order=1).reshape(img_arr.shape[2:4])
            
            def_img_arr[t,0,:,:] = elastic_img
            
            elastic_cell = (map_coordinates(seg_arr[t,0,:,:]==1, indices, order=1).reshape(seg_arr.shape[2:4])>0.5)
            elastic_cell = sk.morphology.binary_closing(elastic_cell)
#             elastic_border = (map_coordinates(seg_arr[t,:,:]==2, indices, order=1).reshape(seg_arr.shape[1:3])>0.5)
            def_seg_arr[t,0,elastic_cell] = 1
#             def_seg_arr[t,elastic_border] = 2
        return def_img_arr,def_seg_arr

    
    def get_augmented_data(self,img_arr,seg_arr):
        img_arr,seg_arr = self.random_crop(img_arr,seg_arr)
        img_arr,seg_arr = self.random_x_flip(img_arr,seg_arr,p=self.p_flip)
        img_arr,seg_arr = self.random_y_flip(img_arr,seg_arr,p=self.p_flip)
        img_arr = self.change_brightness(img_arr)
        pad_img_arr,pad_seg_arr = self.add_padding(img_arr,seg_arr,max_rot=self.max_rot+5)
        img_arr,seg_arr = self.translate(pad_img_arr,pad_seg_arr,img_arr,seg_arr)
        del pad_img_arr
        del pad_seg_arr
        img_arr,seg_arr = self.rotate(img_arr,seg_arr,max_rot=self.max_rot)
        img_arr,seg_arr = self.deform_img_arr(img_arr,seg_arr)
        img_arr,seg_arr = (img_arr.astype("int32"),seg_arr.astype("int8"))
        return img_arr,seg_arr

class UNet_Training_DataLoader:
    def __init__(self,nndatapath="",experimentname="",trainpath="",testpath="",valpath="",augment=False):
        self.nndatapath = nndatapath
        self.experimentname = experimentname
        self.trainpath = trainpath
        self.testpath = testpath
        self.valpath = valpath
        
        self.trainname = self.trainpath.split("/")[-1]
        self.testname = self.testpath.split("/")[-1]
        self.valname = self.valpath.split("/")[-1]
        
        self.metapath = self.nndatapath + "/metadata.hdf5"
        
    def get_metadata(self,headpath):
        meta_handle = pandas_hdf5_handler(headpath + "/metadata.hdf5")
        global_handle = meta_handle.read_df("global",read_metadata=True)
        kymo_handle = meta_handle.read_df("kymograph",read_metadata=True)
        fovdf = kymo_handle.reset_index(inplace=False)
        fovdf = fovdf.set_index(["fov","row","trench"], drop=True, append=False, inplace=False)
        fovdf = fovdf.sort_index()
        
        channel_list = global_handle.metadata["channels"]
        fov_list = kymo_handle['fov'].unique().tolist()
        t_len = len(kymo_handle.index.get_level_values("timepoints").unique())
        trench_dict = {fov:len(fovdf.loc[fov]["trenchid"].unique()) for fov in fov_list}
        shape_y = kymo_handle.metadata["kymograph_params"]["ttl_len_y"]
        shape_x = kymo_handle.metadata["kymograph_params"]["trench_width_x"]
        kymograph_img_shape = tuple((shape_y,shape_x))
        return channel_list,fov_list,t_len,trench_dict,kymograph_img_shape
    
    def get_selection(self,channel,trench_dict,fov_list,t_subsample_step,t_range,max_trenches,kymograph_img_shape,selectionname):        
        fov_list = list(fov_list)
        ttl_trench_count = np.sum(np.array([trench_dict[fov] for fov in fov_list]))
        ttl_trench_count = min(ttl_trench_count,max_trenches)
        num_t = len(range(t_range[0],t_range[1]+1,t_subsample_step))
        ttl_imgs = ttl_trench_count*num_t
        print("Total Number of Trenches: " + str(ttl_trench_count))
        print("Total Number of Timepoints: " + str(num_t))
        print("Total Number of Images: " + str(ttl_imgs))
        selection = tuple((channel,fov_list,t_subsample_step,t_range,max_trenches,ttl_imgs,kymograph_img_shape))
        setattr(self, selectionname + "_selection", selection)
        
    def inter_get_selection(self,headpath,selectionname):
        channel_list,fov_list,t_len,trench_dict,kymograph_img_shape = self.get_metadata(headpath)
        selection = ipyw.interactive(self.get_selection, {"manual":True}, channel=ipyw.Dropdown(options=channel_list,value=channel_list[0],description='Feature Channel:',disabled=False),\
                trench_dict=ipyw.fixed(trench_dict),fov_list=ipyw.SelectMultiple(options=fov_list),\
                t_subsample_step=ipyw.IntSlider(value=1, min=1, max=50, step=1),\
                t_range=ipyw.IntRangeSlider(value=[0, t_len-1],min=0,max=t_len-1,step=1,disabled=False,continuous_update=False),\
                max_trenches=ipyw.IntText(value=1,description='Maximum Trenches per FOV: ',disabled=False),\
                kymograph_img_shape=ipyw.fixed(kymograph_img_shape),\
                selectionname=ipyw.fixed(selectionname));
        display(selection)
    
    def export_chunk(self,selectionname,file_idx,augment,file_trench_indices,weight_grid_list):
        selection = getattr(self,selectionname + "_selection")
        datapath = getattr(self,selectionname + "path")
        dataname = getattr(self,selectionname + "name")
        
        img_path = datapath + "/kymograph/kymograph_" + str(file_idx) + ".hdf5"
        seg_path = datapath + "/fluorsegmentation/segmentation_" + str(file_idx) + ".hdf5"
        nndatapath = self.nndatapath + "/" + selectionname + "_" + str(file_idx) + ".hdf5"
        
        with h5py.File(img_path,"r") as imgfile:
            img_arr = imgfile[selection[0]][file_trench_indices,selection[3][0]:selection[3][1]+1:selection[2]]
            img_arr = img_arr.reshape(img_arr.shape[0]*img_arr.shape[1],img_arr.shape[2],img_arr.shape[3])
            img_arr = img_arr[:,np.newaxis,:,:]
            img_arr = img_arr.astype('int32')

        with h5py.File(seg_path,"r") as segfile:
            seg_arr = segfile["data"][file_trench_indices,selection[3][0]:selection[3][1]+1:selection[2]]
            seg_arr = seg_arr.reshape(seg_arr.shape[0]*seg_arr.shape[1],seg_arr.shape[2],seg_arr.shape[3])
            seg_arr = seg_arr[:,np.newaxis,:,:]
            seg_arr = seg_arr.astype('int8')

        if augment:
            img_arr,seg_arr = self.data_augmentation.get_augmented_data(img_arr,seg_arr)
            
        chunk_shape = (1,1,img_arr.shape[2],img_arr.shape[3])
            
        with h5py.File(nndatapath,"w") as outfile:
            img_handle = outfile.create_dataset("img",data=img_arr,chunks=chunk_shape,dtype='int32')
            seg_handle = outfile.create_dataset("seg",data=seg_arr,chunks=chunk_shape,dtype='int8')
            
        for item in weight_grid_list:
            w0,wm_sigma = item
            weightmap_gen = weightmap_generator(self.nndatapath,w0,wm_sigma)
            weightmap_arr = weightmap_gen.make_weightmaps(seg_arr)
            with h5py.File(nndatapath,"a") as outfile:
                weightmap_handle = outfile.create_dataset("weight_" + str(item),data=weightmap_arr,chunks=chunk_shape,dtype='int32')
            
        return file_idx
            
    def gather_chunks(self,outputdf,output_metadata,selectionname,file_idx_list,weight_grid_list):
        nnoutputpath = self.nndatapath + "/" + selectionname + ".hdf5"
                
        tempdatapath = self.nndatapath + "/" + selectionname + "_" + str(file_idx_list[0]) + ".hdf5"
        with h5py.File(tempdatapath,"r") as infile:
            output_shape = (len(outputdf.index),1,infile["img"].shape[2],infile["img"].shape[3])
            chunk_shape = (1,1,infile["img"].shape[2],infile["img"].shape[3])
        
        with h5py.File(nnoutputpath,"w") as outfile:
            img_handle = outfile.create_dataset("img",output_shape,chunks=chunk_shape,dtype='int32')
            seg_handle = outfile.create_dataset("seg",output_shape,chunks=chunk_shape,dtype='int8')
            for item in weight_grid_list:
                weightmap_handle = outfile.create_dataset("weight_" + str(item),output_shape,chunks=chunk_shape,dtype='int32')
        
        current_idx = 0
        for file_idx in file_idx_list:
            nndatapath = self.nndatapath + "/" + selectionname + "_" + str(file_idx) + ".hdf5"
            with h5py.File(nndatapath,"r") as infile:
                img_arr = infile["img"][:]
                seg_arr = infile["seg"][:]
                weight_arr_list = []
                for item in weight_grid_list:
                    weight_arr_list.append(infile["weight_" + str(item)][:])
            num_indices = img_arr.shape[0]
            with h5py.File(nnoutputpath,"a") as outfile:
                outfile["img"][current_idx:current_idx+num_indices] = img_arr
                outfile["seg"][current_idx:current_idx+num_indices] = seg_arr
                for i,item in enumerate(weight_grid_list):
                    outfile["weight_" + str(item)][current_idx:current_idx+num_indices] = weight_arr_list[i]
            current_idx += num_indices
            os.remove(nndatapath)

    def export_data(self,selectionname,dask_controller,weight_grid_list,augment=False):
        
        dask_controller.futures = {}
        
        selection = getattr(self,selectionname + "_selection")
        datapath = getattr(self,selectionname + "path")
        dataname = getattr(self,selectionname + "name")
        
        input_meta_handle = pandas_hdf5_handler(datapath + "/metadata.hdf5")
        output_meta_handle = pandas_hdf5_handler(self.metapath)
        
        trenchdf_list = []
        
        kymodf = input_meta_handle.read_df("kymograph",read_metadata=True)
        fovdf = kymodf.reset_index(inplace=False)
        fovdf = fovdf.set_index(["fov","row","trench"], drop=True, append=False, inplace=False)
        fovdf = fovdf.sort_index()
        
        trenchdf = fovdf.loc[selection[1]]
        trenchdf = trenchdf.reset_index(inplace=False)
        trenchdf = trenchdf.set_index(["trenchid","timepoints"], drop=True, append=False, inplace=False)
        trenchdf = trenchdf.sort_index()
        
        trenches = trenchdf.index.get_level_values("trenchid").unique().tolist()
        shuffle(trenches)
        trenches = np.sort(trenches[:selection[4]])
        
        filedf = trenchdf.loc[pd.IndexSlice[trenches, selection[3][0]:selection[3][1]+1:selection[2]], :]        
        filedf = filedf.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","File Trench Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()
                
        filelist = filedf.index.get_level_values("File Index").unique().tolist()
        for file_idx in filelist:
            file_trenchdf = filedf.loc[file_idx]
            file_trench_indices = file_trenchdf.index.get_level_values("File Trench Index").unique().tolist()
            future = dask_controller.daskclient.submit(self.export_chunk,selectionname,file_idx,augment,file_trench_indices,weight_grid_list,retries=1)
            dask_controller.futures["File Number: " + str(file_idx)] = future
                
        outputdf = filedf.reset_index(inplace=False)
        outputdf = outputdf.set_index(["trenchid","timepoints"], drop=True, append=False, inplace=False)
        outputdf = outputdf.sort_index()
        
        del outputdf["File Index"]
        del outputdf["File Trench Index"]
        
        selection_keys = ["channel", "fov_list", "t_subsample_step", "t_range", "max_trenches", "ttl_imgs", "kymograph_img_shape"]
        selection = {selection_keys[i]:item for i,item in enumerate(selection)}
        selection["experiment_name"],selection["data_name"] = (self.experimentname, dataname)
        selection["W0 List"], selection["Wm Sigma List"] = (self.grid_dict['W0 (Border Region Weight):'],self.grid_dict['Wm Sigma (Border Region Spread):'])
                          
        output_metadata = {"nndataset" : selection}
        
        segparampath = datapath + "/fluorescent_segmentation.par"
        with open(segparampath, 'rb') as infile:
            seg_param_dict = pkl.load(infile)
        
        output_metadata["segmentation"] = seg_param_dict
        
        input_meta_handle = pandas_hdf5_handler(datapath + "/metadata.hdf5")
        for item in ["global","kymograph"]:
            indf = input_meta_handle.read_df(item,read_metadata=True)
            output_metadata[item] = indf.metadata
        
        output_meta_handle.write_df(selectionname,outputdf,metadata=output_metadata)
        
        file_idx_list = dask_controller.daskclient.gather([dask_controller.futures["File Number: " + str(file_idx)] for file_idx in filelist])
        self.gather_chunks(outputdf,output_metadata,selectionname,file_idx_list,weight_grid_list)
        
        
    def display_grid(self):
        tab_dict = {'W0 (Border Region Weight):':[1., 3., 5., 10.],'Wm Sigma (Border Region Spread):':[1., 2., 3., 4., 5.]}
        children = [ipyw.SelectMultiple(options=val,value=(val[1],),description=key,disabled=False) for key,val in tab_dict.items()]
        self.tab = ipyw.Tab()
        self.tab.children = children
        for i,key in enumerate(tab_dict.keys()):
            self.tab.set_title(i, key[:-1])
        return self.tab
    
    def get_grid_params(self):
        if hasattr(self,'tab'):
            self.grid_dict = {child.description:child.value for child in self.tab.children}
            delattr(self, 'tab')
        elif hasattr(self,'grid_dict'):
            pass
        else:
            raise "No selection defined."
        print("======== Grid Params ========")
        for key,val in self.grid_dict.items():
            print(key + " " + str(val))
        
    def export_all_data(self,n_workers=20,memory='4GB'):
        writedir(self.nndatapath,overwrite=True)
        
        grid_keys = self.grid_dict.keys()
        grid_combinations = list(itertools.product(*list(self.grid_dict.values())))

        self.data_augmentation = data_augmentation()
        
        dask_cont = dask_controller(walltime='01:00:00',local=False,n_workers=n_workers,memory=memory)
        dask_cont.startdask()
#         dask_cont.daskcluster.start_workers()
        dask_cont.displaydashboard()
        
        try:        
            for selectionname in ["train","test","val"]:
                if selectionname == "train":
                    self.export_data(selectionname,dask_cont,grid_combinations,augment=True)
                else:
                    self.export_data(selectionname,dask_cont,grid_combinations,augment=False)
            dask_cont.shutdown()
        except:
            dask_cont.shutdown()
            raise
                        
class GridSearch:
    def __init__(self,nndatapath,numepochs=50):
        self.nndatapath = nndatapath
        self.numepochs = numepochs
        
    def display_grid(self):
        meta_handle = pandas_hdf5_handler(self.nndatapath + "/metadata.hdf5")
        trainmeta = meta_handle.read_df("train",read_metadata=True).metadata["nndataset"]
        w0_list,wm_sigma_list = trainmeta["W0 List"],trainmeta["Wm Sigma List"]
        
        self.tab_dict = {'Batch Size:':[5, 10, 25],'Layers:':[2, 3, 4],\
           'Hidden Size:':[16, 32, 64],'Learning Rate:':[0.001, 0.005, 0.01, 0.05],\
           'Momentum:':[0.9, 0.95, 0.99],'Weight Decay:':[0.0001,0.0005, 0.001],\
           'Dropout:':[0., 0.3, 0.5, 0.7], 'w0:':w0_list, 'wm sigma':wm_sigma_list}
                  
        children = [ipyw.SelectMultiple(options=val,value=(val[1],),description=key,disabled=False) for key,val in self.tab_dict.items()]
        self.tab = ipyw.Tab()
        self.tab.children = children
        for i,key in enumerate(self.tab_dict.keys()):
            self.tab.set_title(i, key[:-1])
        return self.tab
    
    def get_grid_params(self):
        self.grid_dict = {child.description:child.value for child in self.tab.children}
        print("======== Grid Params ========")
        for key,val in self.grid_dict.items():
            print(key + " " + str(val))
            
    def generate_pyscript(self,run_idx,grid_params):
        import_line = "import trenchripper as tr"
        trainer_line = "nntrainer = tr.unet.UNet_Trainer(\"" + self.nndatapath + "\"," + str(run_idx) + \
        ",gpuon=True,numepochs=" + str(self.numepochs) + ",batch_size=" + str(grid_params[0])+",layers=" + \
        str(grid_params[1])+",hidden_size=" + str(grid_params[2]) + ",lr=" + str(grid_params[3]) + \
        ",momentum=" + str(grid_params[4]) + ",weight_decay=" + str(grid_params[5])+",dropout="+str(grid_params[6]) + \
        ",w0=" + str(grid_params[7]) + ",wm_sigma=" + str(grid_params[8]) + ")"
        train_line = "nntrainer.train_model()"
        pyscript = "\n".join([import_line,trainer_line,train_line])
        with open(self.nndatapath + "/models/scripts/" + str(run_idx) + ".py", "w") as scriptfile:
            scriptfile.write(pyscript)
    
    def generate_sbatchscript(self,run_idx,hours,cores,mem,gres):
        shebang = "#!/bin/bash"
        core_line = "#SBATCH -c " + str(cores)
        hour_line = "#SBATCH -t " + str(hours) + ":00:00"
        gpu_lines = "#SBATCH -p gpu\n#SBATCH --gres=" + gres
        mem_line = "#SBATCH --mem=" + mem
        report_lines = "#SBATCH -o " + self.nndatapath + "/models/scripts/" + str(run_idx) +\
        ".out\n#SBATCH -e " + self.nndatapath + "/models/scripts/" + str(run_idx) + ".err\n"
        
        run_line = "python -u " + self.nndatapath + "/models/scripts/" + str(run_idx) + ".py"
        
        sbatchscript = "\n".join([shebang,core_line,hour_line,gpu_lines,mem_line,report_lines,run_line])
        with open(self.nndatapath + "/models/scripts/" + str(run_idx) + ".sh", "w") as scriptfile:
            scriptfile.write(sbatchscript)
    
    def run_sbatchscript(self,run_idx):
        cmd = ["sbatch",self.nndatapath + "/models/scripts/" + str(run_idx) + ".sh"]
        subprocess.run(cmd)

    def run_grid_search(self,hours=12,cores=2,mem="8G",gres="gpu:1"):
        
        grid_keys = self.grid_dict.keys()
        grid_combinations = list(itertools.product(*list(self.grid_dict.values())))
        writedir(self.nndatapath + "/models",overwrite=True)
        writedir(self.nndatapath + "/models/scripts",overwrite=True)
        
        self.run_indices = []
        
        for run_idx,grid_params in enumerate(grid_combinations):
            self.generate_pyscript(run_idx,grid_params)
            self.generate_sbatchscript(run_idx,hours,cores,mem,gres)
            self.run_sbatchscript(run_idx)
            self.run_indices.append(run_idx)
            
    def cancel_all_runs(self,username):
        for run_idx in self.run_indices:
            cmd = ["scancel","-p","gpu","--user=" + username]
            subprocess.Popen(cmd,shell=True,stdin=None,stdout=None,stderr=None,close_fds=True)
            

class SegmentationDataset(Dataset):
    def __init__(self,filepath,weightchannel="",training=False):
        self.filepath = filepath
        self.weightchannel = weightchannel
        self.training = training
        self.chunksize = 1000
        with h5py.File(self.filepath,"r") as infile:
            self.shape = infile["img"].shape
        self.current_chunk = 0
        self.load_chunk(self.current_chunk)
    def load_chunk(self,chunk_idx):
        with h5py.File(self.filepath,"r") as infile:
            self.img_data = infile["img"][chunk_idx*self.chunksize:(chunk_idx+1)*self.chunksize]
            if self.training:
                self.seg_data = infile["seg"][chunk_idx*self.chunksize:(chunk_idx+1)*self.chunksize]
                self.weight_data = infile[self.weightchannel][chunk_idx*self.chunksize:(chunk_idx+1)*self.chunksize]
        self.current_chunk = chunk_idx
    def __len__(self):
        with h5py.File(self.filepath,"r") as infile:
            out_len = infile["img"].shape[0]
        return out_len
    def __getitem__(self,idx):
        idx_chunk = idx//self.chunksize
        subidx = idx%self.chunksize
        if idx_chunk != self.current_chunk:
            self.load_chunk(idx_chunk)
        if self.training:
            sample = {'img': self.img_data[subidx], 'seg': self.seg_data[subidx], self.weightchannel: self.weight_data[subidx]}
        else:
            sample = {'img': self.img_data[subidx]}
        return sample

class double_conv(nn.Module):
    '''(Conv => BatchNorm =>ReLU) twice'''
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.downconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.downconv(x)
        return x
    
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,layers=3,hidden_size=64,dropout=0.,withsoftmax=False):
        super().__init__()
        self.inc = inconv(n_channels, hidden_size)
        self.downlist = nn.ModuleList([down(hidden_size*(2**i), hidden_size*(2**(i+1))) for i in range(0,layers-1)] + [down(hidden_size*(2**(layers-1)), hidden_size*(2**(layers-1)))])
        self.uplist = nn.ModuleList([up(hidden_size*(2**i), hidden_size*(2**(i-2))) for i in reversed(range(2,layers+1))] + [up(hidden_size*2, hidden_size)])
        self.outc = outconv(hidden_size, n_classes)
        self.drop = nn.Dropout(p=dropout)
        self.withsoftmax = withsoftmax
    def uniforminit(self):
        for param in self.named_parameters():
            param[1].data.uniform_(-0.05,0.05)
    def forward(self, x):
        xlist = [self.inc(x)]
        for item in self.downlist:
            xlist.append(item(xlist[-1]))
        x = xlist[-1]
        x = self.drop(x)
        for i,item in enumerate(self.uplist):
            x = item(x, xlist[-(i+2)])
        x = self.outc(x)
        if self.withsoftmax:
            x = F.softmax(x,dim=1)
        return x
    
class UNet_Trainer:
    
    def __init__(self,nndatapath,model_number,numepochs=10,batch_size=100,layers=3,hidden_size=64,lr=0.005,momentum=0.95,weight_decay=0.0005,dropout=0.,\
                 w0=5.,wm_sigma=3.,gpuon=False):
        self.nndatapath = nndatapath
        self.model_number = model_number
        
        self.numepochs = numepochs
        self.batch_size = batch_size
        self.gpuon = gpuon
        
        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.w0 = w0
        self.wm_sigma = wm_sigma
        
        self.model = UNet(1,2,layers=layers,hidden_size=hidden_size,dropout=dropout,withsoftmax=True)
        self.model.uniforminit()
        if gpuon:
            self.model = self.model.cuda()
            
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
                
    def removefile(self,path):
        if os.path.exists(path):
            os.remove(path)
        
    def load_model(self,paramspath):
        if self.gpuon:
            device = torch.device("cuda")
            self.model.load_state_dict(torch.load(paramspath))
        else:
            device = torch.device('cpu')
            self.model.load_state_dict(torch.load(paramspath, map_location=device))
    
    def train(self,x,y,weightmaps):
        self.optimizer.zero_grad()
        fx = self.model.forward(x)
        fx = torch.log(fx)
        
        nll = F.nll_loss(fx,y,reduction='none')*weightmaps
        
        mean_nll = torch.mean(nll)
        mean_nll.backward()
        self.optimizer.step()
        
        nll = torch.sum(nll)
        return nll

    def test(self,x,y,weightmaps):
        fx = self.model.forward(x)
        fx = torch.log(fx)
        
        nll = F.nll_loss(fx,y,reduction='none')*weightmaps
        
        nll = torch.sum(nll)
        return nll
    
    def perepoch(self,e,train_iter,test_iter,val_iter,train_data_shape,test_data_shape,val_data_shape):
        
        now = datetime.datetime.now()
        
        print('=======epoch ' + str(e) + '=======')
        self.model.train()
        total_train_nll = 0.
        num_train_batches = len(train_iter)
        for i,b in enumerate(train_iter):
            img_arr,seg_arr,weightmaps = (b['img'].numpy(),b['seg'].numpy(),b['weight_' + str(tuple([self.w0,self.wm_sigma]))].numpy())
            
            seg_arr,weightmaps = seg_arr[:,0],weightmaps[:,0]
            x = torch.Tensor(img_arr)
            y = torch.LongTensor(seg_arr)
            weightmaps = torch.Tensor(weightmaps)
            if self.gpuon:
                x = x.cuda()
                y = y.cuda()
                weightmaps = weightmaps.cuda()
#                 weights = weights.cuda()
            nll = self.train(x,y,weightmaps)
            total_train_nll += nll.detach().cpu().numpy()
#             if (i%100 == 0) and self.saveparams:
#                 torch.save(self.model.state_dict(), self.nnpath + "/model_layers=" + str(self.layers) + "_hidden_size=" + str(self.hidden_size) +\
#                            "_dropout=" + str(self.dropout) + '_lr=' + str(self.lr) + '_momentum=' + str(self.momentum) + "_epoch_" + str(e) + "_step_" + str(i) +".pt")
            del x
            del y
            del weightmaps
            del nll
            torch.cuda.empty_cache()
        avgtrainnll = total_train_nll/(np.prod(np.array(train_data_shape)))
        print('Mean Train NLL: ' + str(avgtrainnll))
        self.model.eval()
        total_val_nll = 0.
        for i,b in enumerate(val_iter):
            img_arr,seg_arr,weightmaps = (b['img'].numpy(),b['seg'].numpy(),b['weight_' + str(tuple([self.w0,self.wm_sigma]))].numpy())
            seg_arr,weightmaps = seg_arr[:,0],weightmaps[:,0]
            x = torch.Tensor(img_arr)
            y = torch.LongTensor(seg_arr)
            weightmaps = torch.Tensor(weightmaps)
            if self.gpuon:
                x = x.cuda()
                y = y.cuda()
                weightmaps = weightmaps.cuda()
#                 weights = weights.cuda()
            nll = self.test(x,y,weightmaps)
            total_val_nll += nll.detach().cpu().numpy()
            del x
            del y
            del weightmaps
            del nll
            torch.cuda.empty_cache()
        avgvalnll = total_val_nll/(np.prod(np.array(val_data_shape)))
        print('Mean Val NLL: ' + str(avgvalnll))
        
        total_test_nll = 0.
        for i,b in enumerate(test_iter):
            img_arr,seg_arr,weightmaps = (b['img'].numpy(),b['seg'].numpy(),b['weight_' + str(tuple([self.w0,self.wm_sigma]))].numpy())
            seg_arr,weightmaps = seg_arr[:,0],weightmaps[:,0]
            x = torch.Tensor(img_arr)
            y = torch.LongTensor(seg_arr)
            weightmaps = torch.Tensor(weightmaps)
            if self.gpuon:
                x = x.cuda()
                y = y.cuda()
                weightmaps = weightmaps.cuda()
#                 weights = weights.cuda()
            nll = self.test(x,y,weightmaps)
            total_test_nll += nll.detach().cpu().numpy()
            del x
            del y
            del weightmaps
            del nll
            torch.cuda.empty_cache()
        avgtestnll = total_test_nll/(np.prod(np.array(test_data_shape)))
        print('Mean Test NLL: ' + str(avgtestnll))
        
        entry = [[self.model_number,self.batch_size,self.layers,self.hidden_size,self.lr,self.momentum,self.weight_decay,\
                  self.dropout,self.w0,self.wm_sigma,e,avgtrainnll,avgvalnll,avgtestnll,str(now)]]

        df_out = pd.DataFrame(data=entry,columns=['Model #','Batch Size','Layers','Hidden Size','Learning Rate','Momentum','Weight Decay',\
                            'Dropout',"W0 Weight","Wm Sigma",'Epoch','Train Loss','Val Loss','Test Loss','Date/Time'])
        df_out = df_out.set_index(['Model #','Epoch'], drop=True, append=False, inplace=False)
        df_out = df_out.sort_index()
        
        return df_out
        
    def write_metadata(self,filepath,iomode,df_out):
        meta_handle = pandas_hdf5_handler(filepath)
        if os.path.exists(filepath):
            ind = df_out.index[0]
            df_in = meta_handle.read_df("data")
            df_mask = ~df_in.index.isin([ind])
            df_in = df_in[df_mask]
            df_out = pd.concat([df_in, df_out])
        meta_handle.write_df("data",df_out)
        
    def get_fscore(self,iterator,data_shape):
        y_true = []
        y_scores = []
        for i,b in enumerate(iterator):
            img_arr,y = (b['img'].numpy(),b['seg'].numpy())
            x = torch.Tensor(img_arr)
            if self.gpuon:
                x = x.cuda()
            fx = self.model.forward(x).detach().cpu().numpy()
#             y_true.append(y.flatten())
#             y_scores.append(fx[:,1].flatten())
            
            y_true.append(y[:,0])
            y_scores.append(fx[:,1])
            
            del x
            del y
            torch.cuda.empty_cache()
            
        y_true = np.concatenate(y_true,axis=0)
        y_scores = np.concatenate(y_scores,axis=0)
        precisions, recalls, thresholds = precision_recall_curve(y_true.flatten(), y_scores.flatten())        
        fscores = 2*((precisions*recalls)/(precisions+recalls))
        best_idx = np.nanargmax(fscores)
        precision, recall, fscore, threshold = (precisions[best_idx], recalls[best_idx], fscores[best_idx], thresholds[best_idx])
        
        y_true = y_true.astype(bool)
        y_pred = y_scores>threshold
        
        all_f_scores = []
        for i in range(y_true.shape[0]):
            _,_,f_score = object_f_scores(y_true[i],y_pred[i])
            all_f_scores += f_score.tolist()
        all_f_scores = np.array(all_f_scores)
        all_f_scores = all_f_scores[~np.isnan(all_f_scores)]
        
        return precision, recall, fscore, threshold, all_f_scores
    
    def train_model(self):
        timestamp = datetime.datetime.now()
        start = time.time()
        writedir(self.nndatapath + "/models", overwrite=False)
        self.removefile(self.nndatapath + "/models/training_metadata_" + str(self.model_number) + ".hdf5")
        
        train_data = SegmentationDataset(self.nndatapath + "/train.hdf5",weightchannel='weight_' + str(tuple([self.w0,self.wm_sigma])),training=True)
        test_data = SegmentationDataset(self.nndatapath + "/test.hdf5",weightchannel='weight_' + str(tuple([self.w0,self.wm_sigma])),training=True)
        val_data = SegmentationDataset(self.nndatapath + "/val.hdf5",weightchannel='weight_' + str(tuple([self.w0,self.wm_sigma])),training=True)
        
        train_data_shape = train_data.shape
        test_data_shape = test_data.shape
        val_data_shape = val_data.shape
                
        for e in range(0,self.numepochs):
            train_iter = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
            test_iter = DataLoader(test_data,batch_size=self.batch_size,shuffle=True)
            val_iter = DataLoader(val_data,batch_size=self.batch_size,shuffle=True)
            df_out = self.perepoch(e,train_iter,test_iter,val_iter,train_data_shape,test_data_shape,val_data_shape)
            
            self.write_metadata(self.nndatapath + "/models/training_metadata_" + str(self.model_number) + ".hdf5","w",df_out)
        end = time.time()
        time_elapsed = (end-start)/60.
        torch.save(self.model.state_dict(), self.nndatapath + "/models/" + str(self.model_number) + ".pt")
        
        val_p, val_r, val_f, val_t, all_val_f = self.get_fscore(val_iter,val_data_shape)
        test_p, test_r, test_f, test_t, all_test_f = self.get_fscore(test_iter,test_data_shape)
                
        meta_handle = pandas_hdf5_handler(self.nndatapath + "/metadata.hdf5")
        trainmeta = meta_handle.read_df("train",read_metadata=True).metadata
        valmeta = meta_handle.read_df("val",read_metadata=True).metadata
        testmeta = meta_handle.read_df("test",read_metadata=True).metadata
        experiment_name = trainmeta["nndataset"]["experiment_name"]        
        train_dataname,train_org,train_micro,train_ttl_img = (trainmeta["nndataset"]["data_name"],trainmeta["global"]["Organism"],\
                                               trainmeta["global"]["Microscope"],trainmeta["nndataset"]["ttl_imgs"])
        val_dataname,val_org,val_micro,val_ttl_img = (valmeta["nndataset"]["data_name"],valmeta["global"]["Organism"],\
                                                      valmeta["global"]["Microscope"],valmeta["nndataset"]["ttl_imgs"])
        test_dataname,test_org,test_micro,test_ttl_img = (testmeta["nndataset"]["data_name"],testmeta["global"]["Organism"],\
                                                          testmeta["global"]["Microscope"],testmeta["nndataset"]["ttl_imgs"])
        
        train_loss,val_loss,test_loss = df_out['Train Loss'].tolist()[0],df_out['Val Loss'].tolist()[0],df_out['Test Loss'].tolist()[0]
        
        entry = [[experiment_name,self.model_number,train_dataname,train_org,train_micro,train_ttl_img,val_dataname,val_org,val_micro,val_ttl_img,\
                  test_dataname,test_org,test_micro,test_ttl_img,self.batch_size,self.layers,self.hidden_size,self.lr,self.momentum,\
                  self.weight_decay,self.dropout,self.w0,self.wm_sigma,train_loss,val_loss,val_p,val_r,val_f,val_t,all_val_f,test_loss,test_p,test_r,\
                  test_f,test_t,all_test_f,str(timestamp),self.numepochs,time_elapsed]]
                
        df_out = pd.DataFrame(data=entry,columns=['Experiment Name','Model #','Train Dataset','Train Organism','Train Microscope','Train # Images',\
                                                  'Val Dataset','Val Organism','Val Microscope','Val # Images',\
                                                  'Test Dataset','Test Organism','Test Microscope','Test # Images',\
                                                  'Batch Size','Layers','Hidden Size','Learning Rate','Momentum','Weight Decay',\
                                                  'Dropout',"W0 Weight","Wm Sigma",'Train Loss','Val Loss','Val Precision','Val Recall','Val F1 Score',\
                                                  'Val Threshold','Val F1 Cell Scores','Test Loss','Test Precision','Test Recall','Test F1 Score',\
                                                  'Test Threshold','Test F1 Cell Scores','Date/Time','# Epochs','Training Time (mins)'])
        
        df_out = df_out.set_index(['Experiment Name','Model #'], drop=True, append=False, inplace=False)
        df_out = df_out.sort_index()
        
        metalock = hdf5lock(self.nndatapath + "/model_metadata.hdf5",updateperiod=5.)
        metalock.lockedfn(self.write_metadata,"w",df_out)

class TrainingVisualizer:
    def __init__(self,trainpath,modeldbpath):
        self.trainpath = trainpath
        self.modelpath = trainpath + "/models"
        self.modeldfpath = trainpath + "/model_metadata.hdf5"
        self.modeldbpath = modeldbpath
        self.paramdbpath = modeldbpath+"/Parameters"
        self.update_dfs()
        if os.path.exists(self.modeldfpath):
            self.models_widget = qgrid.show_grid(self.model_df.sort_index())
        
    def update_dfs(self):
        df_idx_list = []
        for path in os.listdir(self.modelpath):
            if "training_metadata" in path:
                df_idx = int(path.split("_")[-1][:-5])
                df_idx_list.append(df_idx)
        df_list = []
        for df_idx in df_idx_list:
            dfpath = self.modelpath + "/training_metadata_" + str(df_idx) + ".hdf5"
            df_handle = pandas_hdf5_handler(dfpath)
            df = df_handle.read_df("data")
            df_list.append(copy.deepcopy(df))
            del df
        self.train_df = pd.concat(df_list)
        if os.path.exists(self.modeldfpath):
            modeldfhandle = pandas_hdf5_handler(self.modeldfpath)
            self.model_df = modeldfhandle.read_df("data").sort_index()
        
    def select_df_columns(self,selected_columns):
        df = copy.deepcopy(self.model_df)
        for column in df.columns.tolist():
            if column not in selected_columns:
                df = df.drop(column, 1)
        self.model_widget = qgrid.show_grid(df)

    def inter_df_columns(self):
        column_list = self.model_df.columns.tolist()
        inter = ipyw.interactive(self.select_df_columns,{"manual":True},selected_columns=ipyw.SelectMultiple(options=column_list,description='Columns to Display:',disabled=False))
        display(inter)

    def handle_filter_changed(self,event,widget):
        df = widget.get_changed_df().sort_index()
        
        all_model_indices = self.train_df.index.get_level_values("Model #").unique().tolist()
        current_model_indices = df.index.get_level_values("Model #").unique().tolist()
        
        all_epochs = []
        all_loss = []
        for model_idx in all_model_indices:
            if model_idx in current_model_indices:
                filter_df = df.loc[model_idx]
                epochs,loss = (filter_df.index.get_level_values("Epoch").tolist(),filter_df[self.losskey].tolist())
                all_epochs += epochs
                all_loss += loss
                self.line_dict[model_idx].set_data(epochs,loss)
                self.line_dict[model_idx].set_label(str(model_idx))
            else:
                epochs_empty,loss_empty = ([],[])
                self.line_dict[model_idx].set_data(epochs_empty,loss_empty)
                self.line_dict[model_idx].set_label('_nolegend_')
                
        self.ax.set_xlim(min(all_epochs), max(all_epochs)+1)
        self.ax.set_ylim(0, max(all_loss)*1.1)
        self.ax.legend()
        self.fig.canvas.draw()

    def inter_plot_loss(self,losskey):
        self.losskey = losskey
        self.fig,self.ax = plt.subplots()
        self.grid_widget = qgrid.show_grid(self.train_df.sort_index())
        current_df = self.grid_widget.get_changed_df()
        
        self.line_dict = {}
        for model_idx in current_df.index.get_level_values("Model #").unique().tolist():
            filter_df = current_df.loc[model_idx]
            epochs,loss = (filter_df.index.get_level_values("Epoch").tolist(),filter_df[losskey].tolist())
            line, = self.ax.plot(epochs,loss,label=str(model_idx))
            self.line_dict[model_idx] = line
            
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel(losskey)
        self.ax.legend()
        
    def export_models(self):
        writedir(self.modeldbpath,overwrite=False)
        writedir(self.modeldbpath+"/Parameters",overwrite=False)
        modeldbhandle = pandas_hdf5_handler(self.modeldbpath + "/Models.hdf5")
        if "Models.hdf5" in os.listdir(self.modeldbpath):
            old_df = modeldbhandle.read_df("data")
            current_df = self.models_widget.get_changed_df()
            current_df = pd.concat([old_df, current_df])
        else:
            current_df = self.models_widget.get_changed_df()
        modeldbhandle.write_df("data",current_df)
        
        indices = current_df.index.tolist()
        exp_names = [str(item[0]) for item in indices]
        model_numbers = [str(item[1]) for item in indices]
        dates = [item.replace(" ","_") for item in current_df["Date/Time"].tolist()]
        
        for i in range(len(model_numbers)):
            exp_name,model_number,date = (exp_names[i],model_numbers[i],dates[i])
            shutil.copyfile(self.modelpath + "/" + str(model_number) + ".pt",self.paramdbpath+"/"+exp_name+"_"+model_number+"_"+date+".pt")

class UNet_Segmenter:
    def __init__(self,headpath,modeldbpath):#,min_obj_size=20):#,cell_threshold_scaling=1.,border_threshold_scaling=1.,\
             #    layers=3,hidden_size=64,batch_size=100,gpuon=False):
        
        self.headpath = headpath
        self.kymopath = headpath + "/kymograph"
        self.nnoutputpath = headpath + "/phasesegmentation"
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        globaldf = self.meta_handle.read_df("global",read_metadata=True)
        self.all_channels = globaldf.metadata['channels']
        
        self.modeldbpath = modeldbpath
        self.update_dfs()
        
        if os.path.exists(self.modeldbpath):
            self.models_widget = qgrid.show_grid(self.models_df.sort_index())
        
#         self.min_obj_size = min_obj_size
#         self.border_threshold_scaling = border_threshold_scaling
#         self.gpuon = gpuon
#         self.layers = layers
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
        
    def update_dfs(self):
        dfpath = self.modeldbpath + "/Models.hdf5"
        if os.path.exists(dfpath):
            df_handle = pandas_hdf5_handler(dfpath)
            self.models_df = df_handle.read_df("data")
            
    def select_df_columns(self,selected_columns):
        df = copy.deepcopy(self.models_df)
        for column in df.columns.tolist():
            if column not in selected_columns:
                df = df.drop(column, 1)
        self.model_widget = qgrid.show_grid(df)

    def inter_df_columns(self):
        column_list = self.models_df.columns.tolist()
        inter = ipyw.interactive(self.select_df_columns,{"manual":True},selected_columns=ipyw.SelectMultiple(options=column_list,description='Columns to Display:',disabled=False))
        display(inter)
        
    def select_model(self):
        current_df = self.model_widget.get_changed_df()
        print(current_df)
        
    def choose_seg_channel(self,seg_channel):
        self.seg_channel = seg_channel
                
#     def prepare_data(self,fov_num):
#         writedir(self.nnoutputpath,overwrite=False)
#         writepath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
#         img_path = self.kymopath + "/kymo_" + str(fov_num) + ".hdf5"
#         with h5py.File(writepath,"w") as outfile:
#             with h5py.File(img_path,"r") as infile:
#                 keys = list(infile.keys())
#                 ex_data = infile[keys[0]+"/"+self.seg_channel]
#                 out_shape = (len(keys)*ex_data.shape[2],1,ex_data.shape[0],ex_data.shape[1])
#                 chunk_shape = (1,1,out_shape[2],out_shape[3])
#                 img_handle = outfile.create_dataset("img",out_shape,chunks=chunk_shape,dtype=float)
                
#                 for i,trenchid in enumerate(keys):
#                     img_arr = infile[trenchid+"/"+self.seg_channel][:]
#                     img_arr = np.moveaxis(img_arr,(0,1,2),(1,2,0))
#                     img_arr = np.expand_dims(img_arr,1)
#                     img_arr = img_arr.astype(float)
                    
#                     img_handle[i*ex_data.shape[2]:(i+1)*ex_data.shape[2]] = img_arr
    
#     def segment(self,fov_num):
#         torch.cuda.empty_cache()
#         self.model = UNet(1,3,layers=self.layers,hidden_size=self.hidden_size,withsoftmax=True)
        
#         if self.gpuon:
#             device = torch.device("cuda")
#             self.model.load_state_dict(torch.load(self.paramspath))
#             self.model.to(device)
#         else:
#             device = torch.device('cpu')
#             self.model.load_state_dict(torch.load(self.paramspath, map_location=device))
#         self.model.eval()
        
#         inputpath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
#         outputpath = self.nnoutputpath + "/nnoutput_" + str(fov_num) + ".hdf5"
#         with h5py.File(inputpath,"r") as infile:
#             out_shape = tuple((infile["img"].shape[0],3,infile["img"].shape[2],infile["img"].shape[3]))
#             chunk_shape = infile["img"].chunks
#         data = SegmentationDataset(inputpath,training=False)
#         data_iter = DataLoader(data,batch_size=self.batch_size,shuffle=False) #careful
#         with h5py.File(outputpath,"w") as outfile:
#             img_handle = outfile.create_dataset("img",out_shape,chunks=chunk_shape,dtype=float)
#             print(len(data_iter))
#             for i,b in enumerate(data_iter):
#                 print(i)
#                 x = torch.Tensor(b['img'].numpy())
#                 if self.gpuon:
#                     x = x.cuda()
#                 fx = self.model.forward(x) #N,3,y,x
#                 img_handle[i*self.batch_size:(i+1)*self.batch_size] = fx.cpu().data.numpy()
        
#     def postprocess(self,fov_num):
#         nninputpath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
#         nnoutputpath = self.nnoutputpath + "/nnoutput_" + str(fov_num) + ".hdf5"
#         segpath = self.nnoutputpath + "/seg_" + str(fov_num) + ".hdf5"
#         kymopath = self.kymopath + "/kymo_" + str(fov_num) + ".hdf5"
#         with h5py.File(kymopath,"r") as kymofile:
#             trench_num = len(kymofile.keys())
#             trenchids = list(kymofile.keys())
#         with h5py.File(segpath,"w") as outfile:
#             with h5py.File(nnoutputpath,"r") as infile:
#                 num_img = infile["img"].shape[0]
#                 y_shape,x_shape = (infile["img"].shape[2],infile["img"].shape[3])
#                 timepoints = int(num_img/trench_num)
#                 for trench in range(trench_num):
#                     print(trench)
#                     trenchid = trenchids[trench]
#                     trench_data = infile["img"][trench*timepoints:(trench+1)*timepoints] #t,3,y,x
#                     trench_output = []
#                     for t in range(timepoints):
#                         cell_otsu = sk.filters.threshold_otsu(trench_data[t,1])*self.cell_threshold_scaling
#                         border_otsu = sk.filters.threshold_otsu(trench_data[t,2])*self.border_threshold_scaling
#                         cell_mask = trench_data[t,1]>cell_otsu
#                         border_mask = trench_data[t,2]>border_otsu
#                         trench_arr = cell_mask*(~border_mask)
#                         del cell_mask
#                         del border_mask
#                         trench_arr = sk.morphology.remove_small_objects(trench_arr,min_size=self.min_obj_size)                        
#                         trench_output.append(trench_arr)
#                         del trench_arr
#                     trench_output = np.array(trench_output)
#                     trench_output = np.moveaxis(trench_output,(0,1,2),(2,0,1))
#                     outdset = outfile.create_dataset(trenchid, data=trench_output, chunks=(y_shape,x_shape,1), dtype=bool)
# #         os.remove(nninputpath)
# #         os.remove(nnoutputpath)