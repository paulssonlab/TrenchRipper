import os
import h5py
import torch
import copy
import ipywidgets as ipyw

from random import shuffle
from tensorboardX import SummaryWriter
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
from .utils import pandas_hdf5_handler,kymo_handle

from matplotlib import pyplot as plt

class data_augmentation:
    def __init__(self,p_flip=0.5,max_rot=15,min_padding=20):
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
            working_seg_arr = copy.copy(seg_arr[t,:,:])

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
        out_seg_arr = np.array(out_seg_arr)
        return out_arr,out_seg_arr
    
    def random_x_flip(self,img_arr,seg_arr,p=0.5):
        choices = np.random.choice(np.array([True,False]),size=img_arr.shape[0],p=np.array([p,1.-p]))
        out_img_arr = copy.copy(img_arr)
        out_seg_arr = copy.copy(seg_arr)
        out_img_arr[choices,0,:,:] = np.flip(img_arr[choices,0,:,:],axis=1)
        out_seg_arr[choices,:,:] = np.flip(seg_arr[choices,:,:],axis=1)
        return out_img_arr,out_seg_arr
    def random_y_flip(self,img_arr,seg_arr,p=0.5):
        choices = np.random.choice(np.array([True,False]),size=img_arr.shape[0],p=np.array([p,1.-p]))
        out_img_arr = copy.copy(img_arr)
        out_seg_arr = copy.copy(seg_arr)
        out_img_arr[choices,0,:,:] = np.flip(img_arr[choices,0,:,:],axis=2)
        out_seg_arr[choices,:,:] = np.flip(seg_arr[choices,:,:],axis=2)
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
        padded_seg_arr = np.pad(seg_arr, ((0,0),(delta_y//2,delta_y//2),(delta_x//2,delta_x//2)), 'constant', constant_values=0)
        return padded_img_arr,padded_seg_arr
    
    def translate(self,pad_img_arr,pad_seg_arr,img_arr,seg_arr):
        trans_img_arr = copy.copy(pad_img_arr)
        trans_seg_arr = copy.copy(pad_seg_arr)
        delta_y = pad_img_arr.shape[2] - img_arr.shape[2]
        delta_x = pad_img_arr.shape[3] - img_arr.shape[3]
        for t in range(pad_img_arr.shape[0]):
            trans_y = np.random.randint(-(delta_y//2),high=delta_y//2)
            trans_x = np.random.randint(-(delta_x//2),high=delta_x//2)
            trans_img_arr[t,0,delta_y//2:delta_y//2+img_arr.shape[0],delta_x//2:delta_x//2+img_arr.shape[1]] = 0
            trans_seg_arr[t,delta_y//2:delta_y//2+img_arr.shape[0],delta_x//2:delta_x//2+img_arr.shape[1]] = 0
            trans_img_arr[t,0,delta_y//2+trans_y:delta_y//2+img_arr.shape[0]+trans_y,delta_x//2+trans_x:delta_x//2+img_arr.shape[1]+trans_x] =\
            pad_img_arr[t,0,delta_y//2:delta_y//2+img_arr.shape[0],delta_x//2:delta_x//2+img_arr.shape[1]]
            trans_seg_arr[t,delta_y//2+trans_y:delta_y//2+img_arr.shape[0]+trans_y,delta_x//2+trans_x:delta_x//2+img_arr.shape[1]+trans_x] =\
            pad_seg_arr[t,delta_y//2:delta_y//2+img_arr.shape[0],delta_x//2:delta_x//2+img_arr.shape[1]]
        return trans_img_arr,trans_seg_arr
    
    def rotate(self,img_arr,seg_arr,max_rot=20):
        rot_img_arr = copy.copy(img_arr)
        rot_seg_arr = copy.copy(seg_arr)
        for t in range(img_arr.shape[0]):
            r = np.random.uniform(low=-max_rot,high=max_rot)
            rot_img_arr[t,0,:,:] = sk.transform.rotate(img_arr[t,0,:,:],r,preserve_range=True).astype("int32")
            rot_seg = (sk.transform.rotate(seg_arr[t,:,:]==1,r)>0.5).astype("int8")
#             rot_border = sk.transform.rotate(seg_arr[t,:,:]==2,r)>0.5
#             rot_seg[rot_border] = 2
            rot_seg_arr[t,:,:] = rot_seg
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
            
            elastic_cell = (map_coordinates(seg_arr[t,:,:]==1, indices, order=1).reshape(seg_arr.shape[1:3])>0.5)
            elastic_cell = sk.morphology.binary_closing(elastic_cell)
#             elastic_border = (map_coordinates(seg_arr[t,:,:]==2, indices, order=1).reshape(seg_arr.shape[1:3])>0.5)
            def_seg_arr[t,elastic_cell] = 1
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
    def __init__(self,nndatapath="",trainpath="",testpath="",valpath=""):
        self.nndatapath = nndatapath
        self.trainpath = trainpath
        self.testpath = testpath
        self.valpath = valpath
        self.metapath = self.nndatapath + "/metadata.hdf5"
        
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
    
    def get_selection(self,channel,trench_dict,fov_list,t_subsample_step,t_range,max_trench_per_fov,kymograph_img_shape,selectionname):        
        fov_list = list(fov_list)
        trench_count = np.array([min(max_trench_per_fov,trench_dict[fov]) for fov in fov_list])
        ttl_trench_count = np.sum(trench_count)
        num_t = len(range(t_range[0],t_range[1],t_subsample_step))
        ttl_imgs = ttl_trench_count*num_t
        print("Total Number of Trenches: " + str(ttl_trench_count))
        print("Total Number of Timepoints: " + str(num_t))
        print("Total Number of Images: " + str(ttl_imgs))
        selection = tuple((channel,fov_list,t_subsample_step,t_range,max_trench_per_fov,ttl_imgs,kymograph_img_shape))
        setattr(self, selectionname + "_selection", selection)
        
    def inter_get_selection(self,headpath,selectionname):
        channel_list,fov_list,t_len,trench_dict,kymograph_img_shape = self.get_metadata(headpath)
        self.selection = ipyw.interactive(self.get_selection, {"manual":True}, channel=ipyw.Dropdown(options=channel_list,value=channel_list[0],description='Feature Channel:',disabled=False),\
                trench_dict=ipyw.fixed(trench_dict),fov_list=ipyw.SelectMultiple(options=fov_list),\
                t_subsample_step=ipyw.IntSlider(value=1, min=1, max=50, step=1),\
                t_range=ipyw.IntRangeSlider(value=[0, t_len-1],min=0,max=t_len-1,step=1,disabled=False,continuous_update=False),\
                max_trench_per_fov=ipyw.IntText(value=1,description='Maximum Trenches per FOV: ',disabled=False),\
                kymograph_img_shape=ipyw.fixed(kymograph_img_shape),\
                selectionname=ipyw.fixed(selectionname));
        display(self.selection)
    
    def export_data(self,selectionname):
        self.writedir(self.nndatapath,overwrite=False)
        selection = getattr(self,selectionname + "_selection")
        datapath = getattr(self,selectionname + "path")
        input_meta_handle = pandas_hdf5_handler(datapath + "/metadata.hdf5")
        output_meta_handle = pandas_hdf5_handler(self.metapath)
        nndatapath = self.nndatapath + "/" + selectionname + ".hdf5"
        
        kymodf = input_meta_handle.read_df("kymograph",read_metadata=True)
        fovdf = kymodf.reset_index(inplace=False)
        fovdf = fovdf.set_index(["fov","row","trench"], drop=True, append=False, inplace=False)
        fovdf = fovdf.sort_index()
        
        fovdf["fov"] = fovdf.index.get_level_values("fov")
        trenchdf_list = []
        
        with h5py.File(nndatapath,"w") as outfile:
            out_shape = (selection[5],1,selection[6][0],selection[6][1])
            chunk_shape = (1,1,selection[6][0],selection[6][1])

            img_handle = outfile.create_dataset("img",out_shape,chunks=chunk_shape,dtype='int32')
            seg_handle = outfile.create_dataset("seg",out_shape,chunks=chunk_shape,dtype='int8')
            
            img_idx = 0
            img_step = len(range(selection[3][0],selection[3][1],selection[2]))
            
            fov_list = []
            for idx,fov in enumerate(selection[1]):
                working_fovdf = fovdf.loc[fov]
                trenches = working_fovdf.index.unique().tolist()
                shuffle(trenches)
                trenches = trenches[:selection[4]]
                
                for trench in trenches:
                    working_trenchdf = working_fovdf.loc[trench[0],trench[1]]
                    file_idx = working_trenchdf["File Index"].values[0]
                    file_trench_idx = working_trenchdf["File Trench Index"].values[0]
                    
                    img_path = datapath + "/kymograph/kymograph_" + str(file_idx) + ".hdf5"
                    seg_path = datapath + "/fluorsegmentation/segmentation_" + str(file_idx) + ".hdf5"
                    
                    with h5py.File(img_path,"r") as imgfile:
                        with h5py.File(seg_path,"r") as segfile:
                            img_arr = imgfile[selection[0]][file_trench_idx,selection[3][0]:selection[3][1]:selection[2]]
                            seg_arr = segfile["data"][file_trench_idx,selection[3][0]:selection[3][1]:selection[2]]
                            
                            img_arr = img_arr[:,np.newaxis,:,:] #t,y,x -> t,1,y,x
                            img_arr = img_arr.astype('int32')
                            img_handle[img_idx:img_idx+img_step] = img_arr
                            
                            seg_arr = seg_arr[:,np.newaxis,:,:]
                            seg_arr = seg_arr.astype('int8')
                            seg_handle[img_idx:img_idx+img_step] = seg_arr
                            
                            img_idx += img_step
                    trenchdf_list.append(working_trenchdf)
        outputdf = pd.concat(trenchdf_list)
        outputdf = outputdf.reset_index(inplace=False)
        outputdf = outputdf.set_index(["fov","trench"], drop=True, append=False, inplace=False)
        outputdf = outputdf.sort_index()
                
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
        
class SegmentationDataset(Dataset):
    def __init__(self,filepath,training=False):
        self.filepath = filepath
        self.training = training
        with h5py.File(self.filepath,"r") as infile:
            self.shape = infile["img"].shape
            self.img_data = infile["img"][:]
            if self.training:
                self.seg_data = infile["seg"][:]
    def __len__(self):
        with h5py.File(self.filepath,"r") as infile:
            out_len = infile["img"].shape[0]
        return out_len
    def __getitem__(self,idx):
        if self.training:
            sample = {'img': self.img_data[idx], 'seg': self.seg_data[idx]}
        else:
            sample = {'img': self.img_data[idx]}
        return sample
# class SegmentationDataset(Dataset):
#     def __init__(self,filepath,training=False):
#         self.filepath = filepath
#         self.training = training
#         with h5py.File(self.filepath,"r") as infile:
#             self.shape = infile["img"].shape
#     def __len__(self):
#         with h5py.File(self.filepath,"r") as infile:
#             out_len = infile["img"].shape[0]
#         return out_len
#     def __getitem__(self,idx):
#         with h5py.File(self.filepath,"r") as infile:
#             if self.training:
#                 sample = {'img': infile["img"][idx], 'seg': infile["seg"][idx]}
#             else:
#                 sample = {'img': infile["img"][idx]}
#         return sample

#https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

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
    
    def __init__(self,headpath,layers=3,hidden_size=64,lr=0.005,momentum=0.9,weight_decay=0.0005,dropout=0.,batch_size=100,\
                 gpuon=False,saveparams=False,writetotb=False,augment=True,p_flip=0.5,max_rot=15,min_padding=20,\
                wm_sigma=3.,w0=10.):
        self.headpath = headpath
        self.nnpath = headpath + "/nn"
        self.nnoutputpath = headpath + "/nnsegmentation"
        
        self.batch_size = batch_size
        self.gpuon = gpuon
        self.saveparams = saveparams
        self.writetotb = writetotb
        self.augment = augment
        
        if augment:
            self.data_augmentor = data_augmentation(p_flip=p_flip,max_rot=max_rot,min_padding=min_padding)
        
        self.wm_sigma = wm_sigma
        self.w0 = w0
        
        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.momentum = momentum
        
        if writetotb:
            self.writer = SummaryWriter('runs/layers='+str(layers)+'_hidden_size='+str(hidden_size)+'_dropout='+str(dropout)+'_lr='+str(lr)+'_momentum='+str(momentum))
#         self.model = UNet(1,2,layers=layers,hidden_size=hidden_size,dropout=dropout,withsoftmax=False)
        self.model = UNet(1,2,layers=layers,hidden_size=hidden_size,dropout=dropout,withsoftmax=True)
        self.model.uniforminit()
        if gpuon:
            self.model = self.model.cuda()
            
        self.optimizer = optim.SGD(self.model.parameters(), lr = lr,momentum=momentum,weight_decay=weight_decay)
        
    def load_model(self,paramspath):
        if self.gpuon:
            device = torch.device("cuda")
            self.model.load_state_dict(torch.load(paramspath))
        else:
            device = torch.device('cpu')
            self.model.load_state_dict(torch.load(paramspath, map_location=device))
        
    def make_weight_map(binary_mask):
        ttl_count = binary_mask.size
        cell_count = np.sum(binary_mask)
        background_count = ttl_count - cell_count
        class_weight = np.array([ttl_count/(background_count+1),ttl_count/(cell_count+1)])
        class_weight = class_weight/np.sum(class_weight)

        labeled = sk.measure.label(binary_mask)
        labels = np.unique(labeled)[1:]

        dist_maps = []
        borders = []
        for i in labels:
            cell = labeled==i
            dilated = sk.morphology.binary_dilation(cell)
            border = dilated^cell
            borders.append(border)
            dist_map = sp.ndimage.morphology.distance_transform_edt(~cell)
            dist_maps.append(dist_map)
        dist_maps = np.array(dist_maps)
        borders = np.array(borders)
        borders = np.max(borders,axis=0)
        dist_maps = np.sort(dist_maps,axis=0)
        weight = self.w0*np.exp(-((dist_maps[0] + dist_maps[1])**2)/(2*(self.wm_sigma**2)))
        weight[binary_mask] = class_weight[1]
        weight[~binary_mask] += class_weight[0]
        weight[borders] = 0.
        return weight
    
    def train(self,x,y,weightmaps):
        self.optimizer.zero_grad()
        fx = self.model.forward(x)
        fx = torch.log(fx)
        
        nll = F.nll_loss(fx,y,reduction='none')*weightmaps
        
        mean_nll = torch.mean(nll)
        mean_nll.backward()
        self.optimizer.step()
        return mean_nll

    def test(self,x,y,weightmaps):
        fx = self.model.forward(x)
        fx = torch.log(fx)
        
        nll = F.nll_loss(fx,y,reduction='none')*weightmaps
        
        nll = torch.sum(nll)
        return nll

    def perepoch(self,e,train_iter,val_iter,train_data_shape,val_data_shape):
        
        print('=======epoch ' + str(e) + '=======')
        self.model.train()
        num_train_batches = len(train_iter)
        for i,b in enumerate(train_iter):
            img_arr,seg_arr = (b['img'].numpy(),b['seg'].numpy())
            seg_arr = seg_arr[:,0] #dirty fix move to data prep when scratch stabalizes
            if self.augment:
                img_arr,seg_arr = self.data_augmentor.get_augmented_data(img_arr,seg_arr)
            weightmaps = []
            for t in range(seg_arr.shape[0]):
                binary = seg_arr[t].astype(bool)
                weightmap = self.make_weight_map(binary)
                weightmaps.append(weightmap)
                
                
#                 binary = seg_arr[t].astype(bool)
#                 dilated = sk.morphology.binary_dilation(binary)
#                 dilated = sk.morphology.binary_dilation(dilated)
#                 border = dilated^binary
#                 inner = sk.morphology.binary_erosion(seg_arr[t])^seg_arr[t]
#                 border = outer+inner
#                 seg_arr[t,border] = 2

#             if self.augment:
#                 img_arr,seg_arr = self.data_augmentor.get_augmented_data(img_arr,seg_arr)

#             background_count = int(np.sum(seg_arr==0))
#             cell_count = int(np.sum(seg_arr==1))
#             border_count = int(np.sum(seg_arr==2))
#             ttl_count = seg_arr.size
            
#             weights = torch.Tensor(np.array([ttl_count/(background_count+1),ttl_count/(cell_count+1),ttl_count/(border_count+1)]))
            
            x = torch.Tensor(img_arr)
            y = torch.LongTensor(seg_arr)
            weightmaps = torch.Tensor(np.array(weightmaps))
            if self.gpuon:
                x = x.cuda()
                y = y.cuda()
                weightmaps = weightmaps.cuda()
#                 weights = weights.cuda()
            mean_nll = self.train(x,y,weightmaps)
            mean_nll = mean_nll.cpu().data.numpy()
            if self.writetotb:
                self.writer.add_scalar('Mean Train NLL', mean_nll, i+e*num_train_batches)
            if i%25 == 0:
                print('train_iter: ' + str(i) + ' Mean Train NLL: ' + str(mean_nll))
            if (i%100 == 0) and self.saveparams:
                torch.save(self.model.state_dict(), self.nnpath + "/model_layers=" + str(self.layers) + "_hidden_size=" + str(self.hidden_size) +\
                           "_dropout=" + str(self.dropout) + '_lr=' + str(self.lr) + '_momentum=' + str(self.momentum) + "_epoch_" + str(e) + "_step_" + str(i) +".pt")
            del x
            del y
            del weightmaps
            del mean_nll
            torch.cuda.empty_cache()
        self.model.eval()
        total_test_nll = 0.
        for i,b in enumerate(val_iter):
            img_arr,seg_arr = (b['img'].numpy(),b['seg'].numpy())
            seg_arr = seg_arr[:,0]
            weightmaps = []
            for t in range(seg_arr.shape[0]):
                binary = seg_arr[t].astype(bool)
                weightmap = self.make_weight_map(binary)
                weightmaps.append(weightmap)
#             for t in range(seg_arr.shape[0]):
#                 border = sk.morphology.binary_dilation(seg_arr[t])^sk.morphology.binary_erosion(seg_arr[t])
#                 binary = seg_arr[t].astype(bool)
#                 dilated = sk.morphology.binary_dilation(binary)
#                 dilated = sk.morphology.binary_dilation(dilated)
#                 border = dilated^binary
#                 seg_arr[t,border] = 2
            
#             background_count = int(np.sum(seg_arr==0))
#             cell_count = int(np.sum(seg_arr==1))
#             border_count = int(np.sum(seg_arr==2))
#             ttl_count = seg_arr.size
            
#             weights = torch.Tensor(np.array([ttl_count/(background_count+1),ttl_count/(cell_count+1),ttl_count/(border_count+1)]))

            x = torch.Tensor(img_arr)
            y = torch.LongTensor(seg_arr)
            weightmaps = torch.Tensor(np.array(weightmaps))
            if self.gpuon:
                x = x.cuda()
                y = y.cuda()
                weightmaps = weightmaps.cuda()
#                 weights = weights.cuda()
            nll = self.test(x,y,weightmaps)
            total_test_nll += nll.cpu().data.numpy()
            del x
            del y
            del weightmaps
            del nll
            torch.cuda.empty_cache()
        avgtestnll = total_test_nll/(np.prod(np.array(val_data_shape)))
        if self.writetotb:
            self.writer.add_scalar('Mean Test NLL', avgtestnll, e)
        print('Mean Test NLL: ' + str(avgtestnll))
        return avgtestnll
    
    def train_model(self,numepochs,train_data,val_data):
        train_data_shape = train_data.shape
        val_data_shape = val_data.shape
        for e in range(0,numepochs):
            train_iter = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=4)
            val_iter = DataLoader(val_data,batch_size=self.batch_size,shuffle=True,num_workers=4)
            self.perepoch(e,train_iter,val_iter,train_data_shape,val_data_shape)
            if self.saveparams:
                torch.save(self.model.state_dict(), self.nnpath + "/model_layers=" + str(self.layers) + "_hidden_size=" + str(self.hidden_size) +\
                           "_dropout=" + str(self.dropout) + '_lr=' + str(self.lr) + '_momentum=' + str(self.momentum) + "_epoch_" + str(e)+".pt")
    
    def get_test_pr(self,test_data,samples=1000):
        test_data_shape = test_data.shape
        test_iter = DataLoader(test_data,batch_size=samples,shuffle=True,num_workers=4)
        for i,b in enumerate(test_iter):
            x = Variable(b['img'].float())
            y = b['seg'].float().numpy()
            if self.gpuon:
                x = x.cuda()
            fx = self.model.forward(x).cpu().data.numpy()
            y_true = y.flatten()
            y_scores = fx.flatten()
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            break
        return precision, recall, thresholds
    
# class UNet_DataLoader:
#     def __init__(self,headpath,seg_channel):
#         self.headpath = headpath
#         self.kymopath = headpath + "/kymo"
#         self.segpath = headpath + "/segmentation"
#         self.nnpath = headpath + "/nn"
#         self.nnoutputpath = headpath + "/nnsegmentation"
#         self.metapath = headpath + "/metadata.hdf5"
#         self.seg_channel = seg_channel
    
#     def writedir(self,directory,overwrite=False):
#         """Creates an empty directory at the specified location. If a directory is
#         already at this location, it will be overwritten if 'overwrite' is true,
#         otherwise it will be left alone.
        
#         Args:
#             directory (str): Path to directory to be overwritten/created.
#             overwrite (bool, optional): Whether to overwrite a directory that
#             already exists in this location.
#         """
#         if overwrite:
#             if os.path.exists(directory):
#                 shutil.rmtree(directory)
#             os.makedirs(directory)
#         else:
#             if not os.path.exists(directory):
#                 os.makedirs(directory)
        
#     def prepare_data(self,fov_num):
#         self.writedir(self.nnoutputpath,overwrite=False)
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
     
#     def postprocess(self,fov_num,threshold=0.5):
#         threshold = 0.5
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
#                     trenchid = trenchids[trench]
#                     trench_arr = (infile["img"][trench*timepoints:(trench+1)*timepoints,0]>threshold)
#                     trench_arr = np.moveaxis(trench_arr,(0,1,2),(2,0,1))
#                     outdset = outfile.create_dataset(trenchid, data=trench_arr, chunks=(y_shape,x_shape,1), dtype=bool)
#         os.remove(nninputpath)
#         os.remove(nnoutputpath)
            
class UNet_Segmenter:
    
    def __init__(self,headpath,seg_channel,paramspath,min_obj_size=20,cell_threshold_scaling=1.,border_threshold_scaling=1.,\
                 layers=3,hidden_size=64,batch_size=100,gpuon=False):
        
        self.headpath = headpath
        self.kymopath = headpath + "/kymo"
        self.segpath = headpath + "/segmentation"
        self.nnpath = headpath + "/nn"
        self.metapath = headpath + "/metadata.hdf5"
        self.nnoutputpath = headpath + "/nnsegmentation"
        self.seg_channel = seg_channel
        self.paramspath = paramspath
        self.min_obj_size = min_obj_size
        self.cell_threshold_scaling = cell_threshold_scaling
        self.border_threshold_scaling = border_threshold_scaling
        self.gpuon = gpuon
        self.layers = layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
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
                
    def prepare_data(self,fov_num):
        self.writedir(self.nnoutputpath,overwrite=False)
        writepath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
        img_path = self.kymopath + "/kymo_" + str(fov_num) + ".hdf5"
        with h5py.File(writepath,"w") as outfile:
            with h5py.File(img_path,"r") as infile:
                keys = list(infile.keys())
                ex_data = infile[keys[0]+"/"+self.seg_channel]
                out_shape = (len(keys)*ex_data.shape[2],1,ex_data.shape[0],ex_data.shape[1])
                chunk_shape = (1,1,out_shape[2],out_shape[3])
                img_handle = outfile.create_dataset("img",out_shape,chunks=chunk_shape,dtype=float)
                
                for i,trenchid in enumerate(keys):
                    img_arr = infile[trenchid+"/"+self.seg_channel][:]
                    img_arr = np.moveaxis(img_arr,(0,1,2),(1,2,0))
                    img_arr = np.expand_dims(img_arr,1)
                    img_arr = img_arr.astype(float)
                    
                    img_handle[i*ex_data.shape[2]:(i+1)*ex_data.shape[2]] = img_arr
    
    def segment(self,fov_num):
        torch.cuda.empty_cache()
        self.model = UNet(1,3,layers=self.layers,hidden_size=self.hidden_size,withsoftmax=True)
        
        if self.gpuon:
            device = torch.device("cuda")
            self.model.load_state_dict(torch.load(self.paramspath))
            self.model.to(device)
        else:
            device = torch.device('cpu')
            self.model.load_state_dict(torch.load(self.paramspath, map_location=device))
        self.model.eval()
        
        inputpath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
        outputpath = self.nnoutputpath + "/nnoutput_" + str(fov_num) + ".hdf5"
        with h5py.File(inputpath,"r") as infile:
            out_shape = tuple((infile["img"].shape[0],3,infile["img"].shape[2],infile["img"].shape[3]))
            chunk_shape = infile["img"].chunks
        data = SegmentationDataset(inputpath,training=False)
        data_iter = DataLoader(data,batch_size=self.batch_size,shuffle=False) #careful
        with h5py.File(outputpath,"w") as outfile:
            img_handle = outfile.create_dataset("img",out_shape,chunks=chunk_shape,dtype=float)
            print(len(data_iter))
            for i,b in enumerate(data_iter):
                print(i)
                x = torch.Tensor(b['img'].numpy())
                if self.gpuon:
                    x = x.cuda()
                fx = self.model.forward(x) #N,3,y,x
                img_handle[i*self.batch_size:(i+1)*self.batch_size] = fx.cpu().data.numpy()
        
    def postprocess(self,fov_num):
        nninputpath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
        nnoutputpath = self.nnoutputpath + "/nnoutput_" + str(fov_num) + ".hdf5"
        segpath = self.nnoutputpath + "/seg_" + str(fov_num) + ".hdf5"
        kymopath = self.kymopath + "/kymo_" + str(fov_num) + ".hdf5"
        with h5py.File(kymopath,"r") as kymofile:
            trench_num = len(kymofile.keys())
            trenchids = list(kymofile.keys())
        with h5py.File(segpath,"w") as outfile:
            with h5py.File(nnoutputpath,"r") as infile:
                num_img = infile["img"].shape[0]
                y_shape,x_shape = (infile["img"].shape[2],infile["img"].shape[3])
                timepoints = int(num_img/trench_num)
                for trench in range(trench_num):
                    print(trench)
                    trenchid = trenchids[trench]
                    trench_data = infile["img"][trench*timepoints:(trench+1)*timepoints] #t,3,y,x
                    trench_output = []
                    for t in range(timepoints):
                        cell_otsu = sk.filters.threshold_otsu(trench_data[t,1])*self.cell_threshold_scaling
                        border_otsu = sk.filters.threshold_otsu(trench_data[t,2])*self.border_threshold_scaling
                        cell_mask = trench_data[t,1]>cell_otsu
                        border_mask = trench_data[t,2]>border_otsu
                        trench_arr = cell_mask*(~border_mask)
                        del cell_mask
                        del border_mask
                        trench_arr = sk.morphology.remove_small_objects(trench_arr,min_size=self.min_obj_size)                        
                        trench_output.append(trench_arr)
                        del trench_arr
                    trench_output = np.array(trench_output)
                    trench_output = np.moveaxis(trench_output,(0,1,2),(2,0,1))
                    outdset = outfile.create_dataset(trenchid, data=trench_output, chunks=(y_shape,x_shape,1), dtype=bool)
#         os.remove(nninputpath)
#         os.remove(nnoutputpath)