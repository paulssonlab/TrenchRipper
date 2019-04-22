import numpy as np
import h5py
import shutil
import os
import ast
from copy import deepcopy

class timechunker():
    def __init__(self,input_file_prefix,output_path,fov_number,all_channels,t_chunk=1,img_chunk_size=128):
        """Write later...
            
        Args:
            input_file_prefix (str): File prefix for all input hdf5 files of the form
            [input_file_prefix][fov_number].hdf5 
            output_path (str): Directory to write output files to.
            fov_number (int): The fov number to process.
            all_channels (list): list of strings corresponding to the different image channels
            available in the input hdf5 file, with the channel used for segmenting trenches in
            the first position. NOTE: these names must match those of the input hdf5 file dataset keys.
            
            chunk_shape (tuple, optional): The chunk size to use when perfoming time-chunked computation.
        """
        self.input_file_prefix = input_file_prefix
        self.output_path = output_path
        self.temp_dir_prefix = output_path + "/tempfiles_"   
        self.fov_number = fov_number
        self.input_path = self.input_file_prefix + str(self.fov_number) + ".hdf5"
        self.temp_path = self.temp_dir_prefix + str(self.fov_number) + "/"
        self.output_file_path = self.output_path+"/"+str(self.fov_number)+".hdf5"

        self.all_channels = all_channels
        self.seg_channel = self.all_channels[0]
        
        self.img_chunk_size = img_chunk_size
        self.t_chunk = t_chunk ##this might scale poorly?

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
    
    def removefile(self,filepath):
        """Removes a file at the specified path, if it exists.
        
        Args:
            filepath (str): Path to file for deletion.
        """
        if os.path.exists(filepath):
            os.remove(filepath)

    def reassign_idx(self,array,values,indices,axis):
        """Performs in-line value reassignment on numpy arrays, normally handled
        with the "array[:,indices] = values" syntax, with the ability to supply
        the axis as an argument.
        
        Args:
            array (array): Input array to have values reassigned.
            values (array): New value positions in array.
            indices (array): Positions in the input array to be reassigned.
            axis (int): Axis along which to reassign values.
        """
        str_constructor = "".join((len(array.shape)*[":,"]))[:-1]
        str_constructor = "[" + str_constructor + "]"
        str_constructor = "array"+ str_constructor[:axis*2+1] + "indices" +\
                        str_constructor[axis*2+2:] + " = values"
        exec(str_constructor)
        
    def hdf5_getrange(self,array,ti,tf,axis):
        ### THIS IS A DIRTY FIX...repair later by reworking chunking function to take input for shape size
        # I might have to re-write to have time dim in last slot...
        if type(array) == list:
            output = array[ti:tf]
        else:
            ttl_axes = len(array.shape)
            slc = [slice(None)] * ttl_axes
            slc[axis] = slice(ti,tf)
            output = array[tuple(slc)]
        return output
        
    def write_hdf5(self,file_name,array,ti,t_len,t_dim_out,dataset_name):
        """Writes an array to a particular dataset in an hdf5 file. Positions
        in time are left variable to enable chunking the dataset in time.
        
        Args:
            file_name (str): Name of the hdf5 file, assumed to be in the temp folder
            initialized by this class.
            array (array): Array to be written.
            ti (int): Initial time position to write array values to.
            t_len (int): Total size of the target time dimension.
            t_dim_out (int): Axis of the target time dimension.
            dataset_name (str): The name of the hdf5 dataset to write to.
        """
        with h5py.File(self.temp_path + file_name + ".hdf5", "r+") as h5pyfile:
            indices = list(range(ti,min(ti+self.t_chunk,t_len))) ## this is clearly wrong
            self.reassign_idx(h5pyfile[dataset_name],array,indices,t_dim_out)

    def delete_hdf5(self,file_handle):
        """Deletes an hdf5 file, given its file handle, and closes the handle
        itself.
        
        Args:
            file_handle (hdf5file): Hdf5 file handle.
        """
        filepath = file_handle.filename
        file_handle.close()
        self.removefile(filepath)
        
    def init_hdf5(self,file_name,dataset_name,array,t_len,t_dim_out,dtype='uint16'):
        """Initializes an empty hdf5 file and dataset to write to, given an array
        with the target shape in all axes but the time axis. The time axis
        is then specified by t_len.

        Args:
            file_name (str): Name of the hdf5 file, assumed to be in the temp folder
            initialized by this class.
            dataset_name (str): The name of the hdf5 dataset to initialize.
            array (array): Array which is of the same size as the dataset,
            except in the time dimension.
            t_len (int): Total size of the dataset time dimension.
            t_dim_out (int): Axis of the dataset time dimension.

            dtype(str, optional): Specifies the array datatype to initialize an
            hdf5 file for. A 16 bit unsigned integer by default.
        """
        out_shape = list(array.shape)   
        out_shape[t_dim_out] = t_len
        out_shape = tuple(out_shape)
        
        chunk_shape = np.array(list(out_shape))
        min_arr = np.ones(chunk_shape.shape,dtype=int)*self.img_chunk_size
        chunk_shape = np.minimum(chunk_shape,min_arr)
        chunk_shape[t_dim_out] = 1
        chunk_shape = tuple(chunk_shape)
        
        with h5py.File(self.temp_path + file_name + ".hdf5", "a") as h5pyfile:
            hdf5_dataset = h5pyfile.create_dataset(dataset_name , out_shape, chunks=chunk_shape, dtype=dtype)
            
    def chunk_t(self,hdf5_array_tuple,t_dim_in_tuple,t_dim_out,function,file_name,dataset_name,*args,dtype='uint16',t_range_tuple=None,**kwargs):
        """Applies a given function to any number of input hdf5 arrays, chunking this processing in the
        time dimension, and outputs another hdf5 file.
        
        Args:
            hdf5_array_tuple (tuple): Tuple of input arrays to be operated on by the function.
            t_dim_in_tuple (tuple): Tuple of ints that specify the time axis of each input array.
            t_dim_out (int): Specifies the time axis of the output array.
            function (func): Function to apply to the input arrays. The function must be of the form
            func(array_tuple,*args,**kwargs).
            file_name (str): Name of the output hdf5 file, assumed to be in the temp folder
            initialized by this class.
            dataset_name (str): The name of the hdf5 dataset to write to.
            *args: Extra arguments to be passed to the function, that will be static across time chunks.
            *kwargs: Extra keyword arguments to be passed to the function, that will be static across time chunks.

        Returns:
            hdf5file: Hdf5 file handle corresponding to the output array.
        """
        
        if t_range_tuple == None:
            t_range_tuple = tuple([(0,-1) for i in range(len(hdf5_array_tuple))])
        
        t_initial_list = [item[0] for item in t_range_tuple]
        if t_range_tuple[0][1] == -1:
            t_final_0 = hdf5_array_tuple[0].shape[t_dim_in_tuple[0]]-1
        else:
            t_final_0 = t_range_tuple[0][1]-1
        t_len = t_final_0 - t_initial_list[0] + 1

        for ti in range(0,t_len,self.t_chunk):
            tis = [ti+offset for offset in t_initial_list]
            tf = min(ti+self.t_chunk,t_len)
            tfs = [tf+offset for offset in t_initial_list]
            
            chunk_tuple = tuple(self.hdf5_getrange(hdf5_array_tuple[i],tis[i],tfs[i],t_dim_in_tuple[i])\
                                for i in range(len(hdf5_array_tuple)))
                        
            f_chunk = function(chunk_tuple,*args,**kwargs)
            del chunk_tuple
            if ti == 0:
                self.init_hdf5(file_name,dataset_name,f_chunk,t_len,t_dim_out,dtype=dtype)
            self.write_hdf5(file_name,f_chunk,ti,t_len,t_dim_out,dataset_name)
            del f_chunk
        out_hdf5_handle = h5py.File(self.temp_path + file_name + ".hdf5", "r")
        return out_hdf5_handle

class multifov():
    def __init__(self,input_file_prefix,all_channels,fov_list):
        """Write later...
            
        Args:
            input_file_prefix (string): File prefix for all input hdf5 files of the form
            [input_file_prefix][number].hdf5 
            all_channels (list): list of strings corresponding to the different image channels
            available in the input hdf5 file, with the channel used for segmenting trenches in
            the first position. NOTE: these names must match those of the input hdf5 file datasets.
            fov_list (list): List of ints corresponding to fovs of interest.
        """
        self.input_file_prefix = input_file_prefix
        self.all_channels = all_channels
        self.seg_channel = self.all_channels[0]
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
        self.kymo_arr = np.swapaxes(self.kymo_arr,1,2)[:,padding:-padding]
        if scale:
            self.kymo_arr = self._scale_kymo(self.kymo_arr,scale_perc)
    def return_unwrap(self,padding=0):
        padded_arr = np.pad(self.kymo_arr,((0,0),(padding,padding),(0,0)),'edge')
        wrapped_arr = np.swapaxes(padded_arr,1,2)
        unwrapped_arr = wrapped_arr.reshape(wrapped_arr.shape[0], -1)
        return unwrapped_arr[:]
    def return_wrap(self):
        return self.kymo_arr[:]