# fmt: off
import h5py
import os
import shutil
import copy
# import h5py_cache #not using this anymore
import tifffile
import pickle as pkl
import numpy as np
import pandas as pd
import ipywidgets as ipyw
import skimage as sk

from nd2reader import ND2Reader
from .utils import pandas_hdf5_handler,writedir
from parse import compile

def get_registration_shifts(img_stack):
    shift_coords = [[0.,0.]]
    for i in range(1,img_stack.shape[0]):
        shift = sk.registration.phase_cross_correlation(img_stack[i-1],img_stack[i],return_error=False,normalization=None)
        shift_coords.append(shift)
    shift_coords = np.array(shift_coords)
    cumulative_shift_coords = np.add.accumulate(shift_coords,axis=0)
    return cumulative_shift_coords

def register_image_stack(img_stack,cumulative_shift_coords): #performs a basic image registration on a given image stack, should include all timepoints
    pad_val = np.median(img_stack)

    registered = []
    for i in range(0,img_stack.shape[0]):
        tform = sk.transform.AffineTransform(translation=(-cumulative_shift_coords[i][1],-cumulative_shift_coords[i][0]))
        shifted = sk.transform.warp(img_stack[i], tform, mode='constant', preserve_range=True, cval=pad_val).astype(img_stack[i].dtype)
        registered.append(shifted)

    registered_stack = np.array(registered)
    return registered_stack

def generate_flatfield(flatfieldpath,outputpath): #can add dark image correction to this
    img_arr = []
    with ND2Reader(flatfieldpath) as infile:
        if 'v' in infile.sizes.keys():
            for j in range(infile.sizes['v']):
                nd2_image = infile.get_frame_2D(c=0, t=0, v=j)
                in_arr = np.array(nd2_image)
                img_arr.append(np.array(in_arr))
        else:
            nd2_image = infile.get_frame_2D(c=0, t=0, v=0)
            in_arr = np.array(nd2_image)
            img_arr.append(np.array(in_arr))
    img_arr = np.array(img_arr)
    aggregated_img = np.median(img_arr,axis=0)
    aggregated_img = aggregated_img/np.max(aggregated_img)
    tifffile.imsave(outputpath,data=aggregated_img)

def apply_flatfield(img,flatfieldimg,darkimg):
    outimg = (img - darkimg)/flatfieldimg
    outimg = np.clip(outimg,0.,65535.)
    outimg = outimg.astype("uint16")
    return outimg

class hdf5_fov_extractor:
    def __init__(self,nd2filename,headpath,tpts_per_file=100,ignore_fovmetadata=False,generate_thumbnails=True,thumbnail_rescale=0.05,register_images=False,reg_channel=None,nd2reader_override={}): #note this chunk size has a large role in downstream steps...make sure is less than 1 MB
        self.nd2filename = nd2filename
        self.headpath = headpath
        self.metapath = self.headpath + "/metadata.hdf5"
        self.hdf5path = self.headpath + "/hdf5"
        self.hdf5thumbpath = self.headpath + "/hdf5_thumbnails"
        self.tempregpath = self.headpath + "/registration_temp"
        self.tpts_per_file = tpts_per_file
        self.ignore_fovmetadata = ignore_fovmetadata
        self.nd2reader_override = nd2reader_override
        self.generate_thumbnails = generate_thumbnails
        self.thumbnail_rescale = thumbnail_rescale
        self.register_images = register_images
        self.reg_channel = reg_channel

        self.organism = ''
        self.microscope = ''
        self.notes = ''

        self.channel_to_flat_dict = {}

    def writemetadata(self,t_range=None,fov_list=None):
        ndmeta_handle = nd_metadata_handler(self.nd2filename,ignore_fovmetadata=self.ignore_fovmetadata,nd2reader_override=self.nd2reader_override)
        if self.ignore_fovmetadata:
            exp_metadata = ndmeta_handle.get_metadata()
        else:
            exp_metadata,fov_metadata = ndmeta_handle.get_metadata()

        if t_range != None:
            exp_metadata["frames"] = exp_metadata["frames"][t_range[0]:t_range[1]+1]
            exp_metadata["num_frames"] = len(exp_metadata["frames"])
            fov_metadata = fov_metadata.loc[pd.IndexSlice[:,slice(t_range[0],t_range[1])],:]  #4 -> 70

        if fov_list != None:
            fov_metadata = fov_metadata.loc[list(fov_list)]
            exp_metadata["fields_of_view"] = list(fov_list)

        self.chunk_shape = (1,exp_metadata["height"],exp_metadata["width"])
        self.thumb_chunk_shape = (1,int(exp_metadata["height"]*self.thumbnail_rescale),int(exp_metadata["width"]*self.thumbnail_rescale))
        chunk_bytes = (2*np.multiply.accumulate(np.array(self.chunk_shape))[-1])
        self.chunk_cache_mem_size = 2*chunk_bytes
        exp_metadata["chunk_shape"],exp_metadata["chunk_cache_mem_size"] = (self.chunk_shape,self.chunk_cache_mem_size)
        exp_metadata["Images Registered?"],exp_metadata["Registration Channel"],exp_metadata["Organism"],exp_metadata["Microscope"],exp_metadata["Notes"] = \
        (self.register_images,self.reg_channel,self.organism,self.microscope,self.notes)
        self.meta_handle = pandas_hdf5_handler(self.metapath)

        if self.ignore_fovmetadata:
            assignment_metadata = self.assignidx(exp_metadata,metadf=None)
            assignment_metadata.astype({"File Index":int,"Image Index":int})
        else:
            assignment_metadata = self.assignidx(exp_metadata,metadf=fov_metadata)
            assignment_metadata.astype({"t":float,"x": float,"y":float,"z":float,"File Index":int,"Image Index":int})

        self.meta_handle.write_df("global",assignment_metadata,metadata=exp_metadata)

    def assignidx(self,expmeta,metadf=None):

        if metadf is None:
            numfovs = len(expmeta["fields_of_view"])
            timepoints_per_fov = len(expmeta["frames"])

        else:
            numfovs = len(metadf.index.get_level_values(0).unique().tolist())
            timepoints_per_fov = len(metadf.index.get_level_values(1).unique().tolist())

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

    def read_metadata(self):
        writedir(self.hdf5path,overwrite=True)
        self.writemetadata()
        metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        self.metadf = metadf.sort_index()

    def set_params(self,fov_list,t_range,generate_thumbnails,thumbnail_rescale,register_images,reg_channel,organism,microscope,notes):
        self.fov_list = fov_list
        self.t_range = t_range
        self.generate_thumbnails = generate_thumbnails
        self.thumbnail_rescale = thumbnail_rescale
        self.register_images = register_images
        self.reg_channel = reg_channel
        self.organism = organism
        self.microscope = microscope
        self.notes = notes

    def inter_set_params(self):
        self.read_metadata()
        channels_list = self.metadata["channels"]
        ext_channel_list = channels_list + ["Dark_Image"]
        self.channel_to_flat_dict = {channel:None for channel in ext_channel_list}

        t0,tf = (self.metadata['frames'][0],self.metadata['frames'][-1])
        available_fov_list = self.metadf["fov"].unique().tolist()

        selection = ipyw.interactive(self.set_params, {"manual":True}, fov_list=ipyw.SelectMultiple(options=available_fov_list),\
                t_range=ipyw.IntRangeSlider(value=[t0, tf],\
                min=t0,max=tf,step=1,description='Time Range:',disabled=False),\
                generate_thumbnails=ipyw.Dropdown(options=[True,False],value=True),\
                thumbnail_rescale=ipyw.FloatSlider(value=0.05, min=0., max=1., step=0.01),\
                register_images=ipyw.Dropdown(options=[True,False],value=False,description='Register Images?'),\
                reg_channel=ipyw.Dropdown(options=channels_list,value=channels_list[0],description='Registration Channel:'),organism=ipyw.Textarea(value='',\
                placeholder='Organism imaged in this experiment.',description='Organism:',disabled=False),\
                microscope=ipyw.Textarea(value='',placeholder='Microscope used in this experiment.',\
                description='Microscope:',disabled=False),notes=ipyw.Textarea(value='',\
                placeholder='General experiment notes.',description='Notes:',disabled=False),
                )
        display(selection)

    def set_flatfieldpath(self,channel,path):
        self.channel_to_flat_dict[channel] = path

    def inter_set_flatfieldpaths(self):
        channels_list = self.metadata["channels"]
        ext_channel_list = channels_list + ["Dark_Image"]

        channel_children = [ipyw.interactive(self.set_flatfieldpath,channel=ipyw.fixed(channel),\
                            path=ipyw.Text(description=channel + " Flatfield Path", value='')) for channel in ext_channel_list]
        channel_tab = ipyw.Tab()

        channel_tab.children = channel_children
        for i,channel in enumerate(ext_channel_list):
            channel_tab.set_title(i, channel)

        return channel_tab

    def extract(self,dask_controller,retries=1):
        dask_controller.futures = {}

        if self.generate_thumbnails:
            writedir(self.hdf5thumbpath,overwrite=True)

        self.writemetadata(t_range=self.t_range,fov_list=self.fov_list)
        metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        self.metadf = metadf.sort_index()

        def writehdf5(fovnum,num_entries,timepoint_list,file_idx,num_fovs):
            #### open flatfield images
            flatfield_img_dict = {}
            for channel,path in self.channel_to_flat_dict.items():
                if path != "":
                    flatfield_img_dict[channel] = tifffile.imread(path)

            #### start reading nd2 files
            with ND2Reader(self.nd2filename) as nd2file:
                for key,item in self.nd2reader_override.items():
                    nd2file.metadata[key] = item
                y_dim = self.metadata['height']
                x_dim = self.metadata['width']
                with h5py.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",rdcc_nbytes=self.chunk_cache_mem_size) as h5pyfile:
                # with h5py_cache.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",chunk_cache_mem_size=self.chunk_cache_mem_size) as h5pyfile:

                    if self.generate_thumbnails:
                        with h5py.File(self.hdf5thumbpath + "/hdf5_" + str(file_idx) + ".hdf5","w") as h5pythumbfile:
                            for i,channel in enumerate(self.metadata["channels"]):
                                hdf5_dataset = h5pyfile.create_dataset(str(channel),\
                                (num_entries,y_dim,x_dim), chunks=self.chunk_shape, dtype='uint16')

                                thumbnail_dataset = h5pythumbfile.create_dataset(str(channel),\
                                (num_entries,self.thumb_chunk_shape[1],self.thumb_chunk_shape[2]), chunks=self.thumb_chunk_shape, dtype='uint16')

                                if self.channel_to_flat_dict[channel] != '': ##flatfielding channels
                                    for j in range(len(timepoint_list)):
                                        frame = timepoint_list[j]
                                        nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                                        nd2_image = np.array(nd2_image)
                                        nd2_image = apply_flatfield(nd2_image,flatfield_img_dict[channel],flatfield_img_dict["Dark_Image"])
                                        thumb_image = sk.transform.resize(nd2_image,(self.thumb_chunk_shape[1],self.thumb_chunk_shape[2]),anti_aliasing=False,preserve_range=True)
                                        hdf5_dataset[j,:,:] = nd2_image
                                        thumbnail_dataset[j,:,:] = thumb_image

                                else:
                                    for j in range(len(timepoint_list)): ##not flatfielding channels
                                        frame = timepoint_list[j]
                                        nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                                        thumb_image = sk.transform.resize(nd2_image,(self.thumb_chunk_shape[1],self.thumb_chunk_shape[2]),anti_aliasing=False,preserve_range=True)
                                        hdf5_dataset[j,:,:] = nd2_image
                                        thumbnail_dataset[j,:,:] = thumb_image

                    else:
                        for i,channel in enumerate(self.metadata["channels"]):
                            hdf5_dataset = h5pyfile.create_dataset(str(channel),\
                            (num_entries,y_dim,x_dim), chunks=self.chunk_shape, dtype='uint16')

                            if self.channel_to_flat_dict[channel] != '': ##flatfielding channels
                                for j in range(len(timepoint_list)):
                                    frame = timepoint_list[j]
                                    nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                                    nd2_image = np.array(nd2_image)
                                    nd2_image = apply_flatfield(nd2_image,flatfield_img_dict[channel],flatfield_img_dict["Dark_Image"])
                                    hdf5_dataset[j,:,:] = nd2_image

                            else:
                                for j in range(len(timepoint_list)): ##not flatfielding channels
                                    frame = timepoint_list[j]
                                    nd2_image = nd2file.get_frame_2D(c=i, t=frame, v=fovnum)
                                    hdf5_dataset[j,:,:] = nd2_image

            return "Done."

        file_list = self.metadf.index.get_level_values("File Index").unique().values
        num_jobs = len(file_list)
        random_priorities = np.random.uniform(size=(num_jobs,))

        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            filedf = self.metadf.loc[file_idx]

            fovnum = filedf[0:1]["fov"].values[0]
            num_entries = len(filedf.index.get_level_values("Image Index").values)
            timepoint_list = filedf["timepoints"].tolist()

            future = dask_controller.daskclient.submit(writehdf5,fovnum,num_entries,timepoint_list,file_idx,self.metadata["num_fovs"],retries=retries,priority=priority)
            dask_controller.futures["extract file: " + str(file_idx)] = future

        extracted_futures = [dask_controller.futures["extract file: " + str(file_idx)] for file_idx in file_list]
        pause_for_extract = dask_controller.daskclient.gather(extracted_futures,errors='skip')

        futures_name_list = ["extract file: " + str(file_idx) for file_idx in file_list]
        failed_files = [futures_name_list[k] for k,item in enumerate(extracted_futures) if item.status != "finished"]
        failed_file_idx = [int(item.split(":")[1]) for item in failed_files]
        outdf = self.meta_handle.read_df("global",read_metadata=False)

        tempmeta = outdf.reset_index(inplace=False)
        tempmeta = tempmeta.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        failed_fovs = tempmeta.loc[failed_file_idx]["fov"].unique().tolist()

        outdf  = outdf.drop(failed_fovs)

        if self.t_range != None:
            outdf = outdf.reset_index(inplace=False)
            outdf["timepoints"] = outdf["timepoints"] - self.t_range[0]
            outdf = outdf.set_index(["fov","timepoints"], drop=True, append=False, inplace=False)

        self.meta_handle.write_df("global",outdf,metadata=self.metadata)

        ###optional registration
        ###not optimized for memory, can be reworked fairly easily, but only if it becomes necessary

        if self.register_images:

            writedir(self.tempregpath,overwrite=True)
            self.metadf = self.meta_handle.read_df("global",read_metadata=True)
            self.metadata = self.metadf.metadata

            def registerhdf5(file_idx_list,reg_channel):
                reg_stack = []
                for file_idx in file_idx_list:
                    with h5py.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5", "r") as infile:
                        channels = list(infile.keys())
                        reg_stack.append(infile[reg_channel][:])
                reg_stack = np.concatenate(reg_stack)

                cumulative_shift_coords = get_registration_shifts(reg_stack)
                del reg_stack

                y_dim = self.metadata['height']
                x_dim = self.metadata['width']

                for channel in channels:
                    img_stack = []
                    for file_idx in file_idx_list:
                        with h5py.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5", "r") as infile:
                            img_stack.append(infile[channel][:])
                    stack_borders = np.add.accumulate([0] + [item.shape[0] for item in img_stack])
                    img_stack = np.concatenate(img_stack)
                    img_stack = register_image_stack(img_stack,cumulative_shift_coords)
                    for idx,file_idx in enumerate(file_idx_list):
                        with h5py.File(self.tempregpath + "/hdf5_" + str(file_idx) + ".hdf5", "a") as outfile:
                            stack_i = stack_borders[idx]
                            stack_f = stack_borders[idx+1]
                            stack_len = stack_f-stack_i

                            hdf5_dataset = outfile.create_dataset(str(channel),(stack_len,y_dim,x_dim), chunks=self.chunk_shape, dtype='uint16')
                            hdf5_dataset[:] = img_stack[stack_i:stack_f]

            fov_file_idx_list = self.metadf.reset_index().groupby("fov").apply(lambda x: sorted(list(x["File Index"].unique()))).tolist()
            num_jobs = len(fov_file_idx_list)
            random_priorities = np.random.uniform(size=(num_jobs,))

            for k,file_idx_list in enumerate(fov_file_idx_list):
                priority = random_priorities[k]

                future = dask_controller.daskclient.submit(registerhdf5,file_idx_list,self.reg_channel,retries=retries,priority=priority)
                dask_controller.futures["register file: " + str(k)] = future

            extracted_futures = [dask_controller.futures["register file: " + str(k)] for k in range(len(fov_file_idx_list))]
            pause_for_extract = dask_controller.daskclient.gather(extracted_futures,errors='skip')

            shutil.rmtree(self.hdf5path)
            os.rename(self.tempregpath,self.hdf5path)



class nd_metadata_handler:
    def __init__(self,nd2filename,ignore_fovmetadata=False,nd2reader_override={}):
        self.nd2filename = nd2filename
        self.ignore_fovmetadata = ignore_fovmetadata
        self.nd2reader_override = nd2reader_override

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
            out_dict = {}
            camera_settings = meta[b'pCameraSetting']
            channel_name = camera_settings[b'Metadata'][b'Channels'][b'Channel_0'][b'Name'].decode('utf-8')
            try:
                camera_name = camera_settings[b'CameraUserName'].decode('utf-8')
                out_dict['camera_name'] = camera_name
            except:
                print("No camera name detected!")
            try:
                obj_settings = self.decode_unidict(meta[b'pObjectiveSetting'])
                out_dict['obj_settings'] = obj_settings
            except:
                print("No objective setting detected!")
            try:
                spec_settings = self.read_specsettings(meta[b'sSpecSettings'])
                out_dict.update({**spec_settings})
            except:
                print("No spec settings detected!")
            imaging_settings[channel_name] = out_dict
        return imaging_settings

    def make_fov_df(self,nd2file, exp_metadata): #only records values for single timepoints, does not seperate between channels....
        img_metadata = nd2file.parser._raw_metadata
        num_fovs = exp_metadata['num_fovs']
        num_frames = exp_metadata['num_frames']
        num_images_expected = num_fovs*num_frames

        if img_metadata.x_data != None:
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
        acq_times = np.reshape(np.array(list(img_metadata.acquisition_times)[:num_images_expected]),(-1,num_fovs)).T
        pos_label = np.repeat(np.expand_dims(np.add.accumulate(np.ones(num_fovs,dtype=int))-1,1),time_points,1) ##???
        time_point_labels = np.repeat(np.expand_dims(np.add.accumulate(np.ones(time_points,dtype=int))-1,1),num_fovs,1).T

        output = pd.DataFrame({'fov':pos_label.flatten(),'timepoints':time_point_labels.flatten(),'t':acq_times.flatten(),'x':x.flatten(),'y':y.flatten(),'z':z.flatten()})
        output = output.astype({'fov': int, 'timepoints':int, 't': float, 'x': float,'y': float,'z': float})

        output = output[~((output['x'] == 0.)&(output['y'] == 0.)&(output['z'] == 0.))].reset_index(drop=True) ##bootstrapped to fix issue when only some FOVs are selected (return if it causes problems in the future)
        output = output.set_index(["fov","timepoints"], drop=True, append=False, inplace=False)

        return output

    def get_metadata(self):
        # Manual numbers are for broken .nd2 files (from when Elements crashes)
        nd2file = ND2Reader(self.nd2filename)
        for key,item in self.nd2reader_override.items():
            nd2file.metadata[key] = item
        exp_metadata = copy.copy(nd2file.metadata)
        wanted_keys = ['height', 'width', 'date', 'fields_of_view', 'frames', 'z_levels', 'z_coordinates', 'total_images_per_channel', 'channels', 'pixel_microns', 'num_frames', 'experiment']
        exp_metadata = dict([(k, exp_metadata[k]) for k in wanted_keys if k in exp_metadata])
        exp_metadata["num_fovs"] = len(exp_metadata['fields_of_view'])
        exp_metadata["settings"] = self.get_imaging_settings(nd2file)
        if not self.ignore_fovmetadata:
            fov_metadata = self.make_fov_df(nd2file, exp_metadata)
            nd2file.close()
            return exp_metadata,fov_metadata
        else:
            nd2file.close()
            return exp_metadata

def get_tiff_tags(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        tiff_tags = {}
        for tag in tiff.pages[0].tags.values():
            name, value = tag.name, tag.value
            tiff_tags[name] = value
    return tiff_tags

class tiff_extractor:
    def __init__(self,tiffpath,headpath,channels,tpts_per_file=100,parsestr="t{timepoints:d}xy{fov:d}c{channel:d}.tif",zero_base_keys=["timepoints","fov","channel"],\
                constant_key=None): #note this chunk size has a large role in downstream steps...make sure is less than 1 MB
        """Utility to convert individual tiff files to hdf5 archives.

        Attributes:
            headpath (str): base directory for data analysis
            tiffpath (str): directory where tiff files are located
            metapath (str): metadata path
            hdf5path (str): where to store hdf5 data
            tpts_per_file (int): number of timepoints to put in each hdf5 file
            parsestr (str): format of filenames from which to extract metadata (using parse library)
        """
        self.tiffpath = tiffpath
        self.headpath = headpath
        self.channels = channels
        self.metapath = self.headpath + "/metadata.hdf5"
        self.hdf5path = self.headpath + "/hdf5"
        self.tpts_per_file = tpts_per_file
        self.parsestr = parsestr
        self.zero_base_keys = zero_base_keys
        self.constant_key = constant_key

        self.organism = ''
        self.microscope = ''
        self.notes = ''

        self.channel_to_flat_dict = {}

    def get_metadata(self,tiffpath,channels,parsestr="t{timepoints:d}xy{fov:d}c{channel:d}.tif",zero_base_keys=["timepoints","fov","channel"],constant_key=None):
        parser = compile(parsestr)
        parse_keys = [item.split("}")[0].split(":")[0] for item in parsestr.split("{")[1:]] + ["image_paths"]

        exp_metadata = {}
        fov_metadata = {key:[] for key in parse_keys}
        if constant_key is not None:
            for key in constant_key.keys():
                fov_metadata[key] = []

        tiff_files = []
        for root, _, files in os.walk(tiffpath):
            tiff_files.extend([os.path.join(root, f) for f in files if ".tif" in os.path.splitext(f)[1]])

        tags = get_tiff_tags(tiff_files[0])
        exp_metadata["height"] = tags['ImageLength']
        exp_metadata["width"] = tags['ImageWidth']
        try:
            exp_metadata['pixel_microns'] = tags['65326']
        except:
            exp_metadata['pixel_microns'] = 1
            print("Pixel microns not detected. Global position annotations will be invalid.")
        exp_metadata["channels"] = channels

        for f in tiff_files:
            match = parser.search(f)
            # ignore any files that don't match the regex
            if match != None:
                # Add to dictionary
                fov_frame_dict = match.named
                for key, value in fov_frame_dict.items():
                    fov_metadata[key].append(value)
                if constant_key is not None:
                    for key in constant_key.keys():
                        fov_metadata[key].append(constant_key[key])
                fov_metadata["image_paths"].append(f)

        for zero_base_key in zero_base_keys:
            if 0 not in fov_metadata[zero_base_key]:
                fov_metadata[zero_base_key] = [item-1 for item in fov_metadata[zero_base_key]]

        exp_metadata["num_fovs"] = len(set(fov_metadata['fov']))
        exp_metadata["frames"] = list(set(fov_metadata['timepoints']))
        exp_metadata["num_frames"] = len(exp_metadata["frames"])

        fov_metadata = pd.DataFrame(fov_metadata)
        fov_metadata["channel"] = fov_metadata["channel"].apply(lambda x: channels[x])
        fov_metadata = fov_metadata.set_index(["fov","timepoints"]).sort_index()

        output_fov_metadata = []
        step = len(channels)
        for i in range(0,len(fov_metadata),step):
            rows = fov_metadata[i:i+step]
            channel_path_entry = dict(zip(rows["channel"],rows["image_paths"]))
            fov_entry = rows.index.get_level_values("fov").unique()[0]
            timepoint_entry = rows.index.get_level_values("timepoints").unique()[0]
            fov_metadata_entry = {"fov":fov_entry,"timepoints":timepoint_entry,"channel_paths":channel_path_entry}
            output_fov_metadata.append(fov_metadata_entry)
        fov_metadata = pd.DataFrame(output_fov_metadata).set_index(["fov","timepoints"])

        return exp_metadata,fov_metadata

    def assignidx(self,fov_metadata):
        numfovs = len(fov_metadata.index.get_level_values("fov").unique().tolist())
        timepoints_per_fov = len(fov_metadata.index.get_level_values("timepoints").unique().tolist())

        files_per_fov = (timepoints_per_fov//self.tpts_per_file) + 1
        remainder = timepoints_per_fov%self.tpts_per_file
        ttlfiles = numfovs*files_per_fov
        fov_file_idx = np.repeat(list(range(files_per_fov)), self.tpts_per_file)[:-(self.tpts_per_file-remainder)]
        file_idx = np.concatenate([fov_file_idx+(fov_idx*files_per_fov) for fov_idx in range(numfovs)])
        fov_img_idx = np.repeat(np.array(list(range(self.tpts_per_file)))[np.newaxis,:],files_per_fov,axis=0)
        fov_img_idx = fov_img_idx.flatten()[:-(self.tpts_per_file-remainder)]
        img_idx = np.concatenate([fov_img_idx for fov_idx in range(numfovs)])

        fov_idx = np.repeat(list(range(numfovs)), timepoints_per_fov)
        timepoint_idx = np.repeat(np.array(list(range(timepoints_per_fov)))[np.newaxis,:],numfovs,axis=0).flatten()

        outdf = copy.deepcopy(fov_metadata)
        outdf["File Index"] = file_idx
        outdf["Image Index"] = img_idx
        return outdf

    def writemetadata(self,t_range=None,fov_list=None):

        exp_metadata,fov_metadata = self.get_metadata(self.tiffpath,self.channels,parsestr=self.parsestr,zero_base_keys=self.zero_base_keys,constant_key=self.constant_key)

        if t_range != None:
            exp_metadata["frames"] = exp_metadata["frames"][t_range[0]:t_range[1]+1]
            exp_metadata["num_frames"] = len(exp_metadata["frames"])
            fov_metadata = fov_metadata.loc[pd.IndexSlice[:,slice(t_range[0],t_range[1])],:]  #4 -> 70

        if fov_list != None:
            fov_metadata = fov_metadata.loc[list(fov_list)]
            exp_metadata["fields_of_view"] = list(fov_list)

        self.chunk_shape = (1,exp_metadata["height"],exp_metadata["width"])
        chunk_bytes = (2*np.multiply.accumulate(np.array(self.chunk_shape))[-1])
        self.chunk_cache_mem_size = 2*chunk_bytes
        exp_metadata["chunk_shape"],exp_metadata["chunk_cache_mem_size"] = (self.chunk_shape,self.chunk_cache_mem_size)
        exp_metadata["Organism"],exp_metadata["Microscope"],exp_metadata["Notes"] = (self.organism,self.microscope,self.notes)
        self.meta_handle = pandas_hdf5_handler(self.metapath)

        assignment_metadata = self.assignidx(fov_metadata)
        assignment_metadata.astype({"File Index":int,"Image Index":int})

        self.meta_handle.write_df("global",assignment_metadata,metadata=exp_metadata)

    def read_metadata(self):
        writedir(self.hdf5path,overwrite=True)
        self.writemetadata()
        metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        self.metadf = metadf.sort_index()

    def set_params(self,fov_list,t_range,organism,microscope,notes):
        self.fov_list = fov_list
        self.t_range = t_range
        self.organism = organism
        self.microscope = microscope
        self.notes = notes

    def inter_set_params(self):
        self.read_metadata()
        t0,tf = (self.metadata['frames'][0],self.metadata['frames'][-1])
        available_fov_list = self.metadf["fov"].unique().tolist()
        selection = ipyw.interactive(self.set_params, {"manual":True}, fov_list=ipyw.SelectMultiple(options=available_fov_list),\
                t_range=ipyw.IntRangeSlider(value=[t0, tf],\
                min=t0,max=tf,step=1,description='Time Range:',disabled=False), organism=ipyw.Textarea(value='',\
                placeholder='Organism imaged in this experiment.',description='Organism:',disabled=False),\
                microscope=ipyw.Textarea(value='',placeholder='Microscope used in this experiment.',\
                description='Microscope:',disabled=False),notes=ipyw.Textarea(value='',\
                placeholder='General experiment notes.',description='Notes:',disabled=False),)
        display(selection)

    def set_flatfieldpath(self,channel,path):
        self.channel_to_flat_dict[channel] = path

    def inter_set_flatfieldpaths(self):
        channels_list = self.metadata["channels"]
        ext_channel_list = channels_list + ["Dark_Image"]

        channel_children = [ipyw.interactive(self.set_flatfieldpath,channel=ipyw.fixed(channel),\
                            path=ipyw.Text(description=channel + " Flatfield Path", value='')) for channel in ext_channel_list]
        channel_tab = ipyw.Tab()

        channel_tab.children = channel_children
        for i,channel in enumerate(ext_channel_list):
            channel_tab.set_title(i, channel)

        return channel_tab

    def extract(self,dask_controller,retries=1):
        dask_controller.futures = {}

        self.writemetadata(t_range=self.t_range,fov_list=self.fov_list)
        metadf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        self.metadf = metadf.sort_index()

        def writehdf5(fovnum,num_entries,timepoint_list,file_idx):
            #### open flatfield images
            flatfield_img_dict = {}
            for channel,path in self.channel_to_flat_dict.items():
                if path != "":
                    flatfield_img_dict[channel] = tifffile.imread(path)

            y_dim = self.metadata['height']
            x_dim = self.metadata['width']
            filedf = self.metadf.loc[file_idx].reset_index(inplace=False)
            filedf = filedf.set_index(["timepoints"], drop=True, append=False, inplace=False)
            filedf = filedf.sort_index()

            with h5py.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",rdcc_nbytes=self.chunk_cache_mem_size) as h5pyfile:
            # with h5py_cache.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",chunk_cache_mem_size=self.chunk_cache_mem_size) as h5pyfile:
                for i,channel in enumerate(self.metadata["channels"]):
                    hdf5_dataset = h5pyfile.create_dataset(str(channel),\
                    (num_entries,y_dim,x_dim), chunks=self.chunk_shape, dtype='uint16')

                    if self.channel_to_flat_dict[channel] != '': ##flatfielding channels
                        for j in range(len(timepoint_list)):
                            frame = timepoint_list[j]
                            entry = filedf.loc[frame]["channel_paths"]
                            file_path = entry[channel]
                            img = tifffile.imread(file_path)
                            img = apply_flatfield(img,flatfield_img_dict[channel],flatfield_img_dict["Dark_Image"])
                            hdf5_dataset[j,:,:] = img
                    else:
                        for j in range(len(timepoint_list)):
                            frame = timepoint_list[j]
                            entry = filedf.loc[frame]["channel_paths"]
                            file_path = entry[channel]
                            img = tifffile.imread(file_path)
                            hdf5_dataset[j,:,:] = img

            return "Done."

        file_list = self.metadf.index.get_level_values("File Index").unique().values
        num_jobs = len(file_list)
        random_priorities = np.random.uniform(size=(num_jobs,))

        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            filedf = self.metadf.loc[file_idx]

            fovnum = filedf[0:1]["fov"].values[0]
            num_entries = len(filedf.index.get_level_values("Image Index").values)
            timepoint_list = filedf["timepoints"].tolist()

            future = dask_controller.daskclient.submit(writehdf5,fovnum,num_entries,timepoint_list,file_idx,retries=retries,priority=priority)
            dask_controller.futures["extract file: " + str(file_idx)] = future

        extracted_futures = [dask_controller.futures["extract file: " + str(file_idx)] for file_idx in file_list]
        pause_for_extract = dask_controller.daskclient.gather(extracted_futures,errors='skip')

        futures_name_list = ["extract file: " + str(file_idx) for file_idx in file_list]
        failed_files = [futures_name_list[k] for k,item in enumerate(extracted_futures) if item.status != "finished"]
        failed_file_idx = [int(item.split(":")[1]) for item in failed_files]
        outdf = self.meta_handle.read_df("global",read_metadata=False)

        tempmeta = outdf.reset_index(inplace=False)
        tempmeta = tempmeta.set_index(["File Index","Image Index"], drop=True, append=False, inplace=False)
        failed_fovs = tempmeta.loc[failed_file_idx]["fov"].unique().tolist()

        outdf  = outdf.drop(failed_fovs)

        if self.t_range != None:
            outdf = outdf.reset_index(inplace=False)
            outdf["timepoints"] = outdf["timepoints"] - self.t_range[0]
            outdf = outdf.set_index(["fov","timepoints"], drop=True, append=False, inplace=False)

        self.meta_handle.write_df("global",outdf,metadata=self.metadata)
