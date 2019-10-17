import h5py
import os
import shutil
import copy
import h5py_cache
import pickle as pkl
import numpy as np
import pandas as pd
import ipywidgets as ipyw

from nd2reader import ND2Reader
from tifffile import imsave
from .utils import pandas_hdf5_handler,writedir

class hdf5_fov_extractor:
    def __init__(self,nd2filename,headpath,tpts_per_file=100): #note this chunk size has a large role in downstream steps...make sure is less than 1 MB
        self.nd2filename = nd2filename
        self.headpath = headpath
        self.metapath = self.headpath + "/metadata.hdf5"
        self.hdf5path = self.headpath + "/hdf5"
        self.tpts_per_file = tpts_per_file
                
    def assignidx(self,metadf):
        outdf = copy.deepcopy(metadf)
        ttllen = len(metadf)
        numfovs = len(metadf.index.get_level_values("fov").unique())
        timepoints_per_fov = len(metadf.index.get_level_values("timepoints").unique())
        files_per_fov = (timepoints_per_fov//self.tpts_per_file) + 1
        remainder = timepoints_per_fov%self.tpts_per_file
        ttlfiles = numfovs*files_per_fov
        
        fov_file_idx = np.repeat(list(range(files_per_fov)), self.tpts_per_file)[:-(self.tpts_per_file-remainder)]
        file_idx = np.concatenate([fov_file_idx+(fov_idx*files_per_fov) for fov_idx in range(numfovs)])
        
        fov_img_idx = np.repeat(np.array(list(range(self.tpts_per_file)))[np.newaxis,:],files_per_fov,axis=0)
        fov_img_idx = fov_img_idx.flatten()[:-(self.tpts_per_file-remainder)]
        img_idx = np.concatenate([fov_img_idx for fov_idx in range(numfovs)])
        outdf["File Index"] = file_idx
        outdf["Image Index"] = img_idx
        return outdf
    
    def get_notes(self,organism,microscope,notes):
        self.organism = organism
        self.microscope = microscope
        self.notes = notes
        
    def inter_get_notes(self):
        selection = ipyw.interactive(self.get_notes, {"manual":True}, organism=ipyw.Textarea(value='',\
                placeholder='Organism imaged in this experiment.',description='Organism:',disabled=False),\
                microscope=ipyw.Textarea(value='',placeholder='Microscope used in this experiment.',\
                description='Microscope:',disabled=False),notes=ipyw.Textarea(value='',\
                placeholder='General experiment notes.',description='Notes:',disabled=False),)
        display(selection)
        
    def writemetadata(self):
        ndmeta_handle = nd_metadata_handler(self.nd2filename)
        exp_metadata,fov_metadata = ndmeta_handle.get_metadata()
        
        self.chunk_shape = (1,exp_metadata["height"],exp_metadata["width"])
        chunk_bytes = (2*np.multiply.accumulate(np.array(self.chunk_shape))[-1])
        self.chunk_cache_mem_size = 2*chunk_bytes
        
        exp_metadata["chunk_shape"],exp_metadata["chunk_cache_mem_size"] = (self.chunk_shape,self.chunk_cache_mem_size)
        exp_metadata["Organism"],exp_metadata["Microscope"],exp_metadata["Notes"] = (self.organism,self.microscope,self.notes)
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        
        assignment_metadata = self.assignidx(fov_metadata)
        assignment_metadata.astype({"t":float,"x": float,"y":float,"z":float,"File Index":int,"Image Index":int})
        
        self.meta_handle.write_df("global",assignment_metadata,metadata=exp_metadata)
                                          
    def extract(self,dask_controller):
        writedir(self.hdf5path,overwrite=True)
        self.writemetadata()
        
        dask_controller.futures = {}
        metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        metadf = metadf.sort_index()
        
        def writehdf5(fovnum,num_entries,timepoint_list,file_idx):
            with ND2Reader(self.nd2filename) as nd2file:
                y_dim = self.metadata['height']
                x_dim = self.metadata['width']
                with h5py_cache.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",chunk_cache_mem_size=self.chunk_cache_mem_size) as h5pyfile:
                    for i,channel in enumerate(self.metadata["channels"]):
                        hdf5_dataset = h5pyfile.create_dataset(str(channel),\
                        (num_entries,y_dim,x_dim), chunks=self.chunk_shape, dtype='uint16')

                        for j in range(len(timepoint_list)):
                            frame = timepoint_list[j]
#                             frame = dataframe[j:j+1]["timepoints"].values[0]
    #                         frame = self.metadf['frames'][timepoint] #not sure if this is necessary...
                            nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                            hdf5_dataset[j,:,:] = nd2_image
            return "Done."
        
        file_list = metadf.index.get_level_values("File Index").unique().values
        num_jobs = len(file_list)
        random_priorities = np.random.uniform(size=(num_jobs,))

        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            filedf = metadf.loc[file_idx]
            
            fovnum = filedf[0:1]["fov"].values[0]
            num_entries = len(filedf.index.get_level_values("Image Index").values)
            timepoint_list = filedf["timepoints"].tolist()
                                    
            future = dask_controller.daskclient.submit(writehdf5,fovnum,num_entries,timepoint_list,file_idx,retries=1,priority=priority)
            dask_controller.futures["extract file: " + str(file_idx)] = future

class tiff_fov_extractor: ###needs some work
    def __init__(self,nd2filename,tiffpath):
        self.nd2filename = nd2filename
        self.tiffpath = tiffpath
    def extract_fov(self,fovnum):
        nd2file = ND2Reader(self.nd2filename)
        metadata = nd2file.metadata
        for i,channel in enumerate(nd2file.metadata["channels"]):
            t_dim = len(nd2file.metadata['frames'])
            dirpath = self.tiffpath + "/fov_" + str(fovnum) + "/" + channel + "/"
            writedir(dirpath,overwrite=True)
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
    
    def read_specsettings(self,SpecSettings):
        spec_list = SpecSettings.decode('utf-8').split('\r\n')[1:]
        spec_list = [item for item in spec_list if ":" in item]
        spec_dict = {item.split(": ")[0].replace(" ", "_"):item.split(": ")[1].replace(" ", "_") for item in spec_list}
        return spec_dict

    def get_imaging_settings(self,nd2file):
        raw_metadata = nd2file.parser._raw_metadata
        imaging_settings = {}
        for key,meta in raw_metadata.image_metadata_sequence[b'SLxPictureMetadata'][b'sPicturePlanes'][b'sSampleSetting'].items():
            camera_settings = meta[b'pCameraSetting']
            camera_name = camera_settings[b'CameraUserName'].decode('utf-8')
            channel_name = camera_settings[b'Metadata'][b'Channels'][b'Channel_0'][b'Name'].decode('utf-8')
            obj_settings = self.decode_unidict(meta[b'pObjectiveSetting'])
            spec_settings = self.read_specsettings(meta[b'sSpecSettings'])
            imaging_settings[channel_name] = {'camera_name':camera_name,'obj_settings':obj_settings,**spec_settings}
        return imaging_settings
    
    def make_fov_df(self,nd2file): #only records values for single timepoints, does not seperate between channels....
        img_metadata = nd2file.parser._raw_metadata

        num_fovs = len(nd2file.parser.metadata['fields_of_view'])
        x = np.reshape(img_metadata.x_data,(-1,num_fovs)).T
        y = np.reshape(img_metadata.y_data,(-1,num_fovs)).T
        z = np.reshape(img_metadata.z_data,(-1,num_fovs)).T

        time_points = x.shape[1]
        acq_times = np.reshape(np.array(list(img_metadata.acquisition_times)),(-1,num_fovs)).T #quick fix for inconsistancies beteen the number of timepoints recorded in acquisition times and the x/y/z positions
        acq_times = acq_times[:,:time_points]
        pos_label = np.repeat(np.expand_dims(np.add.accumulate(np.ones(num_fovs,dtype=int))-1,1),time_points,1) ##???
        time_point_labels = np.repeat(np.expand_dims(np.add.accumulate(np.ones(time_points,dtype=int))-1,1),num_fovs,1).T

        output = pd.DataFrame({'fov':pos_label.flatten(),'timepoints':time_point_labels.flatten(),'t':acq_times.flatten(),'x':x.flatten(),'y':y.flatten(),'z':z.flatten()})
        output = output.astype({'fov': int, 'timepoints':int, 't': float, 'x': float,'y': float,'z': float})
        
        output = output[~((output['x'] == 0.)&(output['y'] == 0.)&(output['z'] == 0.))].reset_index(drop=True) ##bootstrapped to fix issue when only some FOVs are selected (return if it causes problems in the future)
        output = output.set_index(["fov","timepoints"], drop=True, append=False, inplace=False)
        
        return output
    
    def get_metadata(self):
        nd2file = ND2Reader(self.nd2filename)
        exp_metadata = copy.copy(nd2file.metadata)
        exp_metadata["num_fovs"] = len(exp_metadata['fields_of_view'])
        exp_metadata["settings"] = self.get_imaging_settings(nd2file)
        fov_metadata = self.make_fov_df(nd2file)
        nd2file.close()
        return exp_metadata,fov_metadata