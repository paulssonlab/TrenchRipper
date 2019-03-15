import h5py
import os
import shutil
import copy
import pickle

from nd2reader import ND2Reader
from tifffile import imsave

class hdf5_fov_extractor:
    def __init__(self,nd2filename,hdf5path,num_cols=None,chunk_shape=(256,256,1)): #note this chunk size has a large role in downstream steps...make sure is less than 1 MB
        self.nd2filename = nd2filename
        self.hdf5path = hdf5path
        self.chunk_shape = chunk_shape
        self.writedir(hdf5path)
        
        meta_handle = nd_metadata_handler(self.nd2filename)
        self.num_fovs = meta_handle.num_fovs
        self.metadata = meta_handle.get_metadata(num_cols=num_cols)
            
    def writedir(self,directory,overwrite=False):
        if overwrite:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def writemetadata(self):
        with open(self.hdf5path + "/metadata.pkl", 'wb') as outfile:
            pickle.dump(self.metadata, outfile)
    
    def extract_fov(self,fovnum):
        nd2file = ND2Reader(self.nd2filename)
        metadata = nd2file.metadata
        num_fovs = len(metadata['fields_of_view'])
        
        with h5py.File(self.hdf5path + "/fov_" + str(fovnum) + ".hdf5", "w") as h5pyfile:
            for i,channel in enumerate(nd2file.metadata["channels"]):
                y_dim = metadata['height']
                x_dim = metadata['width']
                t_dim = len(nd2file.metadata['frames'])
                hdf5_dataset = h5pyfile.create_dataset("channel_" + str(channel),\
                                (x_dim,y_dim,t_dim), chunks=self.chunk_shape, dtype='uint16')
                for frame in nd2file.metadata['frames']:
                    nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                    hdf5_dataset[:,:,int(frame)] = nd2_image
        nd2file.close()

class tiff_fov_extractor:
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
        nd2file = ND2Reader(self.nd2filename)
        self.metadata = copy.copy(nd2file.metadata)
        nd2file.close()
        self.num_fovs = len(self.metadata['fields_of_view'])
        
    def __get_fovnames(self,num_cols):
        fov_names = []
        for fov in range(self.num_fovs):
            i = fov//num_cols
            if i%2 == 0:
                j = (fov%(num_cols))+1
            else:
                j = num_cols-(fov%num_cols)
            fov_name = str(chr(i+65)) + str(j).zfill(2)
            fov_names.append(fov_name)
        return fov_names
    
    def get_metadata(self,num_cols=None):
        if num_cols == None:
            num_cols = self.num_fovs
        else:
            num_cols = num_cols
        fov_names = self.__get_fovnames(num_cols)
        self.metadata["num_fovs"] = self.num_fovs
        self.metadata["fov_names"] = fov_names
        return self.metadata