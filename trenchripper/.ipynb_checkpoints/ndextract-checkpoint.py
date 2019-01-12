import h5py
import os
import shutil

from nd2reader import ND2Reader
from tifffile import imsave

class hdf5_fov_extractor:
    def __init__(self,nd2filename,hdf5path):
        self.nd2filename = nd2filename
        self.hdf5path = hdf5path
        self.writedir(hdf5path)
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
        with h5py.File(self.hdf5path + "/fov_" + str(fovnum) + ".hdf5", "w") as h5pyfile:
            for i,channel in enumerate(nd2file.metadata["channels"]):
                y_dim = metadata['height']
                x_dim = metadata['width']
                t_dim = len(nd2file.metadata['frames'])
                hdf5_dataset = h5pyfile.create_dataset("channel_" + str(channel),\
                                (x_dim,y_dim,t_dim), chunks=(x_dim,y_dim,1), dtype='uint16')
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