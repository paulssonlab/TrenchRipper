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
from tifffile import imsave, imread
from .utils import pandas_hdf5_handler,writedir
from parse import compile

class hdf5_fov_extractor:
    def __init__(self,nd2filename,headpath,tpts_per_file=100,ignore_fovmetadata=False): #note this chunk size has a large role in downstream steps...make sure is less than 1 MB
        self.nd2filename = nd2filename
        self.headpath = headpath
        self.metapath = self.headpath + "/metadata.hdf5"
        self.hdf5path = self.headpath + "/hdf5"
        self.tpts_per_file = tpts_per_file
        self.ignore_fovmetadata = ignore_fovmetadata
                
    def assignidx(self,expmeta,metadf=None):
        numfovs = len(expmeta["fields_of_view"])
        timepoints_per_fov = len(expmeta["frames"])
        files_per_fov = (timepoints_per_fov//self.tpts_per_file) + 1
        remainder = timepoints_per_fov%self.tpts_per_file
        ttlfiles = numfovs*files_per_fov
        fov_file_idx = np.repeat(list(range(files_per_fov)), self.tpts_per_file)[:-(self.tpts_per_file-remainder)]
        file_idx = np.concatenate([fov_file_idx+(fov_idx*files_per_fov) for fov_idx in range(numfovs)])
        fov_img_idx = np.repeat(np.array(list(range(self.tpts_per_file)))[np.newaxis,:],files_per_fov,axis=0)
        fov_img_idx = fov_img_idx.flatten()[:-(self.tpts_per_file-remainder)]
        img_idx = np.concatenate([fov_img_idx for fov_idx in range(numfovs)])
        
        if metadf is None:
            fov_idx = np.repeat(list(range(numfovs)), timepoints_per_fov)
            timepoint_idx = np.repeat(np.array(list(range(timepoints_per_fov)))[np.newaxis,:],numfovs,axis=0).flatten()
            
            data = {"fov" : fov_idx,"timepoints" : timepoint_idx,"File Index" : file_idx, "Image Index" : img_idx}
            outdf = pd.DataFrame(data)
            outdf = outdf.set_index(["fov","timepoints"], drop=True, append=False, inplace=False)
            
        else:
            outdf = copy.deepcopy(metadf)
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
        
    def writemetadata(self, manual_num_frames=None, manual_num_fovs=None):
        ndmeta_handle = nd_metadata_handler(self.nd2filename,ignore_fovmetadata=self.ignore_fovmetadata)
        if self.ignore_fovmetadata:
            exp_metadata = ndmeta_handle.get_metadata()
        else:
            exp_metadata,fov_metadata = ndmeta_handle.get_metadata(manual_num_frames=manual_num_frames, manual_num_fovs=manual_num_fovs)
        
        self.chunk_shape = (1,exp_metadata["height"],exp_metadata["width"])
        chunk_bytes = (2*np.multiply.accumulate(np.array(self.chunk_shape))[-1])
        self.chunk_cache_mem_size = 2*chunk_bytes
        exp_metadata["chunk_shape"],exp_metadata["chunk_cache_mem_size"] = (self.chunk_shape,self.chunk_cache_mem_size)
        exp_metadata["Organism"],exp_metadata["Microscope"],exp_metadata["Notes"] = (self.organism,self.microscope,self.notes)
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        
        if self.ignore_fovmetadata:
            assignment_metadata = self.assignidx(exp_metadata,metadf=None)
            assignment_metadata.astype({"File Index":int,"Image Index":int})
        else:
            assignment_metadata = self.assignidx(exp_metadata,metadf=fov_metadata)
            assignment_metadata.astype({"t":float,"x": float,"y":float,"z":float,"File Index":int,"Image Index":int})
        
        self.meta_handle.write_df("global",assignment_metadata,metadata=exp_metadata)
                                          
    def extract(self,dask_controller, manual_num_frames=None, manual_num_fovs=None):
        writedir(self.hdf5path,overwrite=True)
        self.writemetadata(manual_num_frames=manual_num_frames, manual_num_fovs=manual_num_fovs)
        
        dask_controller.futures = {}
        metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        metadf = metadf.sort_index()
        
        def writehdf5(fovnum,num_entries,timepoint_list,file_idx, num_fovs):
            with ND2Reader(self.nd2filename) as nd2file:
                y_dim = self.metadata['height']
                x_dim = self.metadata['width']
                with h5py_cache.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",chunk_cache_mem_size=self.chunk_cache_mem_size) as h5pyfile:
                    for i,channel in enumerate(self.metadata["channels"]):
                        hdf5_dataset = h5pyfile.create_dataset(str(channel),\
                        (num_entries,y_dim,x_dim), chunks=self.chunk_shape, dtype='uint16')
                        if self.metadata["failed_file"]:
                            for j in range(len(timepoint_list)):
                                frame = timepoint_list[j]
                                nd2_image = nd2file.get_frame_2D(c=i, t=0, v=fovnum+frame*num_fovs)
                                hdf5_dataset[j,:,:] = nd2_image
                        else:
                            for j in range(len(timepoint_list)):
                                frame = timepoint_list[j]
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
                                    
            future = dask_controller.daskclient.submit(writehdf5,fovnum,num_entries,timepoint_list,file_idx,self.metadata["num_fovs"],retries=1,priority=priority)
            dask_controller.futures["extract file: " + str(file_idx)] = future
            
        extracted_futures = [dask_controller.futures["extract file: " + str(file_idx)] for file_idx in file_list]
        pause_for_extract = dask_controller.daskclient.gather(extracted_futures,errors='skip')
        
        futures_name_list = ["extract file: " + str(file_idx) for file_idx in file_list]
        failed_files = [futures_name_list[k] for k,item in enumerate(extracted_futures) if item.status is not "finished"]
        failed_file_idx = [int(item.split(":")[1]) for item in failed_files]
        outdf = self.meta_handle.read_df("global",read_metadata=False)
        
        tempmeta = outdf.reset_index(inplace=False)
        tempmeta = tempmeta.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        failed_fovs = tempmeta.loc[failed_file_idx]["fov"].unique().tolist()
        
        outdf  = outdf.drop(failed_fovs)
        self.meta_handle.write_df("global",outdf,metadata=self.metadata)

class tiff_to_hdf5_extractor:
    def __init__(self, headpath, tiffpath, tpts_per_file=100):
        self.tiffpath = tiffpath
        self.headpath = headpath
        self.metapath = self.headpath + "/metadata.hdf5"
        self.hdf5path = self.headpath + "/hdf5"
        self.tpts_per_file = tpts_per_file

    
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
    
    def assignidx(self,metadf):
        outdf = copy.deepcopy(metadf)
        numchannels = len(pd.unique(metadf["channel"]))
        numfovs = len(metadf.index.get_level_values("fov").unique())
        timepoints_per_fov = len(metadf.index.get_level_values("timepoints").unique())
        files_per_fov = (timepoints_per_fov//self.tpts_per_file) + 1
        remainder = timepoints_per_fov%self.tpts_per_file
        
        fov_file_idx = np.repeat(list(range(files_per_fov)), self.tpts_per_file*numchannels)[:-(self.tpts_per_file-remainder)*numchannels]
        file_idx = np.concatenate([fov_file_idx+(fov_idx*files_per_fov) for fov_idx in range(numfovs)])
        
        fov_img_idx = np.repeat(np.repeat(np.array(list(range(self.tpts_per_file))), numchannels)[np.newaxis,:],files_per_fov,axis=0)
        fov_img_idx = fov_img_idx.flatten()[:-(self.tpts_per_file-remainder)*numchannels]
        img_idx = np.concatenate([fov_img_idx for fov_idx in range(numfovs)])
        outdf["File Index"] = file_idx
        outdf["Image Index"] = img_idx
        return outdf

    def writemetadata(self, parser, tiff_files):
        fov_metadata = {}
        exp_metadata = {}
        assignment_metadata = {}
        first_img = imread(tiff_files[0])
        exp_metadata["height"] = first_img.shape[0]
        exp_metadata["width"] = first_img.shape[1]
        exp_metadata["Organism"] = self.organism
        exp_metadata["Microscope"] = self.microscope
        exp_metadata["Notes"] = self.notes

        self.chunk_shape = (1,exp_metadata["height"],exp_metadata["width"])
        chunk_bytes = (2*np.multiply.accumulate(np.array(self.chunk_shape))[-1])
        self.chunk_cache_mem_size = 2*chunk_bytes
        exp_metadata["chunk_shape"],exp_metadata["chunk_cache_mem_size"] = (self.chunk_shape,self.chunk_cache_mem_size)

        fov_metadata = dict([(key, [value]) for key, value in parser.search(tiff_files[0]).named.items()])
        fov_metadata["Image Path"] = [tiff_files[0]]
        for f in tiff_files[1:]:
            fov_frame_dict = parser.search(f).named
            for key, value in fov_frame_dict.items():
                fov_metadata[key].append(value)
            fov_metadata["Image Path"].append(f)
        if "lane" not in fov_metadata:
            fov_metadata["lane"] = [1]*len(tiff_files)
            
        fov_metadata = pd.DataFrame(fov_metadata)
        
        exp_metadata["frames"] = sorted(list(pd.unique(fov_metadata["timepoints"])))
        exp_metadata["num_frames"] = len(exp_metadata["frames"])
        exp_metadata["channels"] = list(pd.unique(fov_metadata["channel"]))
        
        fov_metadata = fov_metadata.set_index(["lane", "fov"]).sort_values("timepoints").sort_index()
        old_labels = [list(frozen_array) for frozen_array in fov_metadata.index.unique().labels]
        old_labels = list(zip(old_labels[0], old_labels[1]))
        label_mapping = {}
        for i in range(len(old_labels)):
            label_mapping[old_labels[i]] = i
        old_labels = np.array(fov_metadata.index.labels).T
        new_labels = np.empty(old_labels.shape[0])
        for i in range(old_labels.shape[0]):
            old_label = (old_labels[i, 0], old_labels[i, 1])
            new_labels[i] = label_mapping[old_label]
        fov_metadata = fov_metadata.reset_index()
        fov_metadata["fov"] = new_labels
        exp_metadata["fields_of_view"] = sorted(list(pd.unique(fov_metadata["fov"])))
        exp_metadata["num_fovs"] = len(exp_metadata["fields_of_view"])
        
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        
        assignment_metadata = self.assignidx(fov_metadata.set_index(["fov", "timepoints"]))
        channel_paths_by_file_index = assignment_metadata.reset_index()[["File Index", "channel", "Image Path"]].set_index("File Index")
        channel_paths_by_file_index = [(file_index, list(channel_paths_by_file_index.loc[file_index]["channel"]), list(channel_paths_by_file_index.loc[file_index]["Image Path"])) for file_index in channel_paths_by_file_index.index.unique("File Index")]
        assignment_metadata = assignment_metadata.drop_duplicates(subset=["File Index", "Image Index"])
        assignment_metadata = assignment_metadata[["lane", "File Index", "Image Index"]]

        self.meta_handle.write_df("global",assignment_metadata,metadata=exp_metadata)
        return channel_paths_by_file_index

    def extract(self, dask_controller, filename_format_string):
        writedir(self.hdf5path,overwrite=True)
        parser = compile(filename_format_string)
        tiff_files = []
        for root, _, files in os.walk(self.tiffpath):
            tiff_files.extend([os.path.join(root, f) for f in files if ".tif" in os.path.splitext(f)[1]])
        
        channel_paths_by_file_index = self.writemetadata(parser, tiff_files)
        dask_controller.futures = {}
        metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = metadf.metadata

        def writehdf5(fidx_channels_paths):
            y_dim = self.metadata['height']
            x_dim = self.metadata['width']
            num_channels = len(self.metadata["channels"])
            
            file_idx, channels, filepaths = fidx_channels_paths
            datasets = {}
            with h5py_cache.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",chunk_cache_mem_size=self.chunk_cache_mem_size) as h5pyfile:
                for i,channel in enumerate(self.metadata["channels"]):
                    hdf5_dataset = h5pyfile.create_dataset(str(channel),\
                    (len(filepaths)/num_channels,y_dim,x_dim), chunks=self.chunk_shape, dtype='uint16')
                    datasets[channel] = hdf5_dataset
                for i in range(len(filepaths)):
                    curr_channel = channels[i]
                    curr_file = filepaths[i]
                    datasets[curr_channel][i//num_channels,:,:] = imread(curr_file)
            return "Done."
        dask_controller.futures["extract file"] = dask_controller.daskclient.map(writehdf5, channel_paths_by_file_index)


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
    def __init__(self,nd2filename,ignore_fovmetadata=False):
        self.nd2filename = nd2filename
        self.ignore_fovmetadata = ignore_fovmetadata
        
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
    
    def make_fov_df(self,nd2file, exp_metadata): #only records values for single timepoints, does not seperate between channels....
        img_metadata = nd2file.parser._raw_metadata
        num_fovs = exp_metadata['num_fovs']
        num_frames = exp_metadata['num_frames']
        num_images_expected = num_fovs*num_frames
        
        if img_metadata.x_data is not None:
            x = np.reshape(img_metadata.x_data,(-1,num_fovs)).T
            y = np.reshape(img_metadata.y_data,(-1,num_fovs)).T
            z = np.reshape(img_metadata.z_data,(-1,num_fovs)).T
        else:
            positions = img_metadata.image_metadata[b'SLxExperiment'][b'ppNextLevelEx'][b''][b'uLoopPars'][b'Points'][b'']
            x = []
            y = []
            z = []
            for position in positions:
                x.append([position[b'dPosX']]*num_frames)
                y.append([position[b'dPosY']]*num_frames)
                z.append([position[b'dPosZ']]*num_frames)
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            

        time_points = x.shape[1]
        if img_metadata.x_data is not None:
            acq_times = np.reshape(np.array(list(img_metadata.acquisition_times)),(-1,num_fovs)).T #quick fix for inconsistancies beteen the number of timepoints recorded in acquisition times and the x/y/z positions
            acq_times = acq_times[:,:time_points]
        else:
            acq_times = np.reshape(np.array(list(img_metadata.acquisition_times)[:num_images_expected]),(-1,num_fovs)).T
        pos_label = np.repeat(np.expand_dims(np.add.accumulate(np.ones(num_fovs,dtype=int))-1,1),time_points,1) ##???
        time_point_labels = np.repeat(np.expand_dims(np.add.accumulate(np.ones(time_points,dtype=int))-1,1),num_fovs,1).T

        output = pd.DataFrame({'fov':pos_label.flatten(),'timepoints':time_point_labels.flatten(),'t':acq_times.flatten(),'x':x.flatten(),'y':y.flatten(),'z':z.flatten()})
        output = output.astype({'fov': int, 'timepoints':int, 't': float, 'x': float,'y': float,'z': float})
        
        output = output[~((output['x'] == 0.)&(output['y'] == 0.)&(output['z'] == 0.))].reset_index(drop=True) ##bootstrapped to fix issue when only some FOVs are selected (return if it causes problems in the future)
        output = output.set_index(["fov","timepoints"], drop=True, append=False, inplace=False)
        
        return output
    
    def get_metadata(self, manual_num_frames=None, manual_num_fovs=None):
        nd2file = ND2Reader(self.nd2filename)
        exp_metadata = copy.copy(nd2file.metadata)
        wanted_keys = ['height', 'width', 'date', 'fields_of_view', 'frames', 'z_levels', 'total_images_per_channel', 'channels', 'pixel_microns', 'num_frames', 'experiment']
        exp_metadata = dict([(k, exp_metadata[k]) for k in wanted_keys if k in exp_metadata])
        exp_metadata["failed_file"] = False
        if manual_num_frames is not None:
            exp_metadata["frames"] = list(range(manual_num_frames))
            exp_metadata["num_frames"] = len(exp_metadata["frames"])
            exp_metadata["failed_file"] = True
        if manual_num_fovs is not None:
            exp_metadata["fields_of_view"] = list(range(manual_num_fovs))
            exp_metadata["failed_file"] = True
        exp_metadata["num_fovs"] = len(exp_metadata['fields_of_view'])
        exp_metadata["settings"] = self.get_imaging_settings(nd2file)
        if not self.ignore_fovmetadata:
            fov_metadata = self.make_fov_df(nd2file)
            nd2file.close()
            return exp_metadata,fov_metadata
        else:
            nd2file.close()
            return exp_metadata
