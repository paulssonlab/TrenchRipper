import numpy as np
import h5py
import shutil
import os
import ast
import h5py_cache

import pandas as pd
from copy import deepcopy
    
class multifov():
    def __init__(self,fov_list):
        """Write later...
            
        Args:
            input_file_prefix (string): File prefix for all input hdf5 files of the form
            [input_file_prefix][number].hdf5 
            all_channels (list): list of strings corresponding to the different image channels
            available in the input hdf5 file, with the channel used for segmenting trenches in
            the first position. NOTE: these names must match those of the input hdf5 file datasets.
            fov_list (list): List of ints corresponding to fovs of interest.
        """
        self.fov_list = fov_list
        self.num_fovs = len(fov_list)

    def map_to_fovs(self,func,*args,**kargs):
        """Handler for performing steps of analysis across multiple fovs. Appends output
        of a function to a list of outputs for each fov.
        
        Args:
            func (function): Function to apply to each fov. NOTE: Must be written
            to accept the fov index i as the first argument.
            *args: Arguments to pass to the function.
            **kargs: Keyword arguments to pass to the function.
        
        Returns:
            list: List of function outputs, one for each fov.
        """
        output_list = []
        for i in range(self.num_fovs):
            output = func(i,*args,**kargs)
            output_list.append(output)
        return output_list

class kymo_handle():
    def __init__(self):
        return
    def _scale_kymo(self,wrap_arr,percentile):
        perc_t = np.percentile(wrap_arr[:].reshape(-1,wrap_arr.shape[2]),percentile,axis=0)
        norm_perc_t = perc_t/np.max(perc_t)
        scaled_arr = wrap_arr/norm_perc_t[np.newaxis,np.newaxis,:]
        return scaled_arr
    def import_wrap(self,wrap_arr,scale=False,scale_perc=80):
        self.kymo_arr = wrap_arr
        if scale:
            self.kymo_arr = self._scale_kymo(self.kymo_arr,scale_perc)
    def import_unwrap(self,unwrap_arr,t_tot,padding=0,scale=False,scale_perc=80):
        self.kymo_arr = unwrap_arr.reshape(unwrap_arr.shape[0], t_tot, -1)
        self.kymo_arr = np.swapaxes(self.kymo_arr,1,2) #yxt
        if padding > 0:
            self.kymo_arr = self.kymo_arr[:,padding:-padding]
        if scale:
            self.kymo_arr = self._scale_kymo(self.kymo_arr,scale_perc)
    def return_unwrap(self,padding=0):
        padded_arr = np.pad(self.kymo_arr,((0,0),(padding,padding),(0,0)),'edge')
        wrapped_arr = np.swapaxes(padded_arr,1,2)
        unwrapped_arr = wrapped_arr.reshape(wrapped_arr.shape[0], -1)
        return unwrapped_arr[:]
    def return_wrap(self):
        return self.kymo_arr[:]

class pandas_hdf5_handler:
    def __init__(self,hdf5_path):
        self.hdf5_path = hdf5_path
        
    def keys(self):
        with pd.HDFStore(self.hdf5_path,"r") as store:
            return store.keys()
        
    def write_df(self,key,df,metadata=None):
        with pd.HDFStore(self.hdf5_path) as store:
            if "/" + key in store.keys():
                store.remove(key)
            store.put(key, df)
            if metadata is not None:
                store.get_storer(key).attrs.metadata = metadata
    def read_df(self,key,read_metadata=False):           
        with pd.HDFStore(self.hdf5_path,"r") as store:
            df = store.get(key)
            if read_metadata:
                df.metadata = store.get_storer(key).attrs.metadata
            return df