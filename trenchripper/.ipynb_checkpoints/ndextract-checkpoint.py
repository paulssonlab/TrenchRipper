import h5py
import os
import shutil
import copy
import pickle as pkl
import numpy as np
import pandas as pd

from nd2reader import ND2Reader
from tifffile import imsave

class hdf5_fov_extractor:
    def __init__(self,nd2filename,hdf5path,chunk_shape=(256,256,1)): #note this chunk size has a large role in downstream steps...make sure is less than 1 MB
        self.nd2filename = nd2filename
        self.hdf5path = hdf5path
        self.chunk_shape = chunk_shape
        self.writedir(hdf5path)
        
        meta_handle = nd_metadata_handler(self.nd2filename)
        self.exp_metadata,self.fov_metadata = meta_handle.get_metadata()
            
    def writedir(self,directory,overwrite=False):
        if overwrite:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def writemetadata(self):
        exp_meta_path = self.hdf5path + "/exp_metadata.pkl"
        fov_meta_path = self.hdf5path + "/fov_metadata.pkl"
        with open(exp_meta_path, 'wb') as outfile:
            pkl.dump(self.exp_metadata, outfile)
        self.fov_metadata.to_pickle(fov_meta_path)
        
    def extract_fov(self,fovnum):
        nd2file = ND2Reader(self.nd2filename)
        num_fovs = len(nd2file.metadata["fields_of_view"])
        
        with h5py.File(self.hdf5path + "/fov_" + str(fovnum) + ".hdf5", "w") as h5pyfile:
            for i,channel in enumerate(nd2file.metadata["channels"]):
                y_dim = nd2file.metadata['height']
                x_dim = nd2file.metadata['width']
                t_dim = len(nd2file.metadata['frames'])
                hdf5_dataset = h5pyfile.create_dataset("channel_" + str(channel),\
                                (x_dim,y_dim,t_dim), chunks=self.chunk_shape, dtype='uint16')
                for frame in nd2file.metadata['frames']:
                    nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                    hdf5_dataset[:,:,int(frame)] = nd2_image
        nd2file.close()

class tiff_fov_extractor: ###needs some work
    def __init__(self,nd2filename,tiffpath):
        self.nd2filename = nd2filename
        self.tiffpath = tiffpath
    def writedir(self,directory,overwrite=False):
        if overwrite:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
    def extract_fov(self,fovnum):
        nd2file = ND2Reader(self.nd2filename)
        metadata = nd2file.metadata
        for i,channel in enumerate(nd2file.metadata["channels"]):
            t_dim = len(nd2file.metadata['frames'])
            dirpath = self.tiffpath + "/fov_" + str(fovnum) + "/" + channel + "/"
            self.writedir(dirpath,overwrite=True)
            for frame in nd2file.metadata['frames']:
                filepath = dirpath + "t_" + str(frame) + ".tif"
                nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                imsave(filepath, nd2_image)
        nd2file.close()

class nd_metadata_handler:
    def __init__(self,nd2filename):
        self.nd2filename = nd2filename
        
    def decode_unidict(self,unidict):
        outdict = {}
        for key, val in unidict.items():
            if type(key) == bytes:
                key = key.decode('utf8')
            if type(val) == bytes:
                val = val.decode('utf8')
            outdict[key] = val
        return outdict

    def get_imaging_settings(self,nd2file):
        raw_metadata = nd2file.parser._raw_metadata
        imaging_settings = {}
        for key,meta in raw_metadata.image_metadata_sequence[b'SLxPictureMetadata'][b'sPicturePlanes'][b'sSampleSetting'].items():
            camera_settings = meta[b'pCameraSetting']
            camera_name = camera_settings[b'CameraUserName'].decode('utf-8')
            channel_name = camera_settings[b'Metadata'][b'Channels'][b'Channel_0'][b'Name'].decode('utf-8')
            obj_settings = self.decode_unidict(meta[b'pObjectiveSetting'])
            imaging_settings[channel_name] = {'camera_name':camera_name,'obj_settings':obj_settings}
        return imaging_settings
    
    def make_fov_df(self,nd2file): #only records values for single timepoints, does not seperate between channels....
        img_metadata = nd2file.parser._raw_metadata

        num_fovs = len(nd2file.parser.metadata['fields_of_view'])

        acq_times = np.reshape(np.array(list(img_metadata.acquisition_times)),(-1,num_fovs)).T
        exp_times = np.reshape(img_metadata.camera_exposure_time,(-1,num_fovs)).T
        x = np.reshape(img_metadata.x_data,(-1,num_fovs)).T
        y = np.reshape(img_metadata.y_data,(-1,num_fovs)).T
        z = np.reshape(img_metadata.z_data,(-1,num_fovs)).T

        time_points = acq_times.shape[1]
        pos_label = np.repeat(np.expand_dims(np.add.accumulate(np.ones(num_fovs,dtype=int)),1),time_points,1)

        output = pd.DataFrame({'pos':pos_label.flatten(),'t':acq_times.flatten(),'x':x.flatten(),'y':y.flatten(),'z':z.flatten(),'exp':exp_times.flatten()})
        return output
    
    def get_metadata(self):
        nd2file = ND2Reader(self.nd2filename)
        exp_metadata = copy.copy(nd2file.metadata)
        exp_metadata["num_fovs"] = len(exp_metadata['fields_of_view'])
        exp_metadata["settings"] = self.get_imaging_settings(nd2file)
        fov_metadata = self.make_fov_df(nd2file)
        nd2file.close()
        return exp_metadata,fov_metadata