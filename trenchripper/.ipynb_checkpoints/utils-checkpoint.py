# fmt: off
import numpy as np
import h5py
import shutil
import os
import ast

import pandas as pd
import pickle as pkl
import dask.dataframe as dd
from copy import deepcopy

class multifov():
    def __init__(self,selected_fov_list):
        """Write later...

        Args:
            input_file_prefix (string): File prefix for all input hdf5 files of the form
            [input_file_prefix][number].hdf5
            all_channels (list): list of strings corresponding to the different image channels
            available in the input hdf5 file, with the channel used for segmenting trenches in
            the first position. NOTE: these names must match those of the input hdf5 file datasets.
            selected_fov_list (list): List of ints corresponding to fovs of interest.
        """
        self.selected_fov_list = selected_fov_list
        self.num_fovs = len(selected_fov_list)

    def map_to_fovs(self,func,*args,**kargs):
        """Handler for performing steps of analysis across multiple fovs.
        Appends output of a function to a list of outputs for each fov.

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
        perc_t = np.percentile(wrap_arr[:].reshape(wrap_arr.shape[0],-1),percentile,axis=1)
        norm_perc_t = perc_t/np.max(perc_t)
        scaled_arr = wrap_arr/norm_perc_t[:,np.newaxis,np.newaxis]
        return scaled_arr
    def import_wrap(self,wrap_arr,scale=False,scale_perc=80):
        self.kymo_arr = wrap_arr
        if scale:
            self.kymo_arr = self._scale_kymo(self.kymo_arr,scale_perc)
    def import_unwrap(self,unwrap_arr,t_tot,padding=0,scale=False,scale_perc=80):
        self.kymo_arr = unwrap_arr.reshape(unwrap_arr.shape[0], t_tot, -1)
        self.kymo_arr = np.swapaxes(self.kymo_arr,0,1) #tyx
        if padding > 0:
            self.kymo_arr = self.kymo_arr[:,:,padding:-padding]
        if scale:
            self.kymo_arr = self._scale_kymo(self.kymo_arr,scale_perc)
    def return_unwrap(self,padding=0):
        padded_arr = np.pad(self.kymo_arr,((0,0),(0,0),(padding,padding)),'edge')
        wrapped_arr = np.swapaxes(padded_arr,0,1)
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

def writedir(directory,overwrite=False):
    if overwrite:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
class dataset_time_cropper:
    def __init__(self,headpath,subsample_headpath,segpath):
        
        self.headpath = headpath
        self.subsample_headpath = subsample_headpath
        self.segpath = segpath
        
    def reset_daughters(self,df):
        min_tpts = df.groupby(['Global CellID'])['timepoints'].idxmin().tolist()
        init_cells = df.loc[min_tpts]
        cellid_list = init_cells['Global CellID'].tolist()
        daughter_1_mask = df['Daughter CellID 1'].isin(cellid_list)
        daughter_2_mask = df['Daughter CellID 2'].isin(cellid_list)
        df.loc[~daughter_1_mask,'Daughter CellID 1'] = -1
        df.loc[~daughter_2_mask,'Daughter CellID 2'] = -1
        return df
        
    def crop_timepoints_kymograph(self,file_idx,timepoint_list):
        kymographpath = self.headpath+"/kymograph"
        subsample_kymographpath = self.subsample_headpath+"/kymograph"

        with h5py.File(subsample_kymographpath+"/kymograph_" + str(file_idx) + ".hdf5", "w") as outfile:
            with h5py.File(kymographpath+"/kymograph_" + str(file_idx) + ".hdf5", "r") as infile:
                for channel in infile.keys():
                    cropped_data = infile[channel][:,timepoint_list]
                    hdf5_dataset = outfile.create_dataset(str(channel), data=cropped_data, dtype="uint16")

        return file_idx

    def crop_timepoints_segmentation(self,file_idx,timepoint_list):
        segmentationpath = self.headpath + "/" + self.segpath
        subsample_segmentationpath = self.subsample_headpath + "/" + self.segpath

        with h5py.File(subsample_segmentationpath+"/segmentation_" + str(file_idx) + ".hdf5", "w") as outfile:
            with h5py.File(segmentationpath+"/segmentation_" + str(file_idx) + ".hdf5", "r") as infile:
                cropped_data = infile['data'][:,timepoint_list]
                hdf5_dataset = outfile.create_dataset("data", data=cropped_data, dtype="uint16")

        return file_idx

    def clone_kymograph_metadata(self,timepoint_list):
        kymo_meta = dd.read_parquet(self.headpath + "/kymograph/metadata",calculate_divisions=True)

        kymograph_metadata = pd.read_pickle(self.headpath + "/kymograph/metadata.pkl")

        with open(self.subsample_headpath + "/kymograph/metadata.pkl", 'wb') as handle:
            pkl.dump(kymograph_metadata, handle)

        timepoint_remap = {timepoint:i for i,timepoint in enumerate(timepoint_list)}

        kymo_meta_filtered = kymo_meta[kymo_meta["timepoints"].isin(timepoint_list)]
        kymo_meta_filtered["timepoints"] = kymo_meta_filtered["timepoints"].apply(lambda x: timepoint_remap[x], meta=('timepoints', int)).persist()

        kymo_meta_filtered = kymo_meta_filtered.reset_index()
        ##FOV Parquet Index
        kymo_meta_filtered["FOV Parquet Index"] = kymo_meta_filtered.apply(lambda x: int(f'{x["fov"]:04n}{x["row"]:04n}{x["trench"]:04n}{x["timepoints"]:04n}'), axis=1, meta=(None, 'int64')).persist()
        ##File Parquet Index
        kymo_meta_filtered["File Parquet Index"] = kymo_meta_filtered.apply(lambda x: int(f'{int(x["File Index"]):08n}{int(x["File Trench Index"]):04n}{int(x["timepoints"]):04n}'), axis=1, meta=(None, 'int64')).persist()
        ##Trenchid Timepoint Index
        kymo_meta_filtered["Trenchid Timepoint Index"] = kymo_meta_filtered.apply(lambda x: int(f'{x["trenchid"]:08n}{x["timepoints"]:04n}'), axis=1, meta=(None, 'int64')).persist()

        kymo_meta_filtered = kymo_meta_filtered.set_index("FOV Parquet Index",sorted=True)
        
        dd.to_parquet(kymo_meta_filtered, self.subsample_headpath + "/kymograph/metadata",engine='pyarrow',compression='gzip',write_metadata_file=True)

    def clone_lineage_metadata(self,dask_controller,timepoint_list):
        lineage_meta = dd.read_parquet(self.headpath + "/lineage/output",engine="pyarrow",calculate_divisions=True)

        timepoint_remap = {timepoint:i for i,timepoint in enumerate(timepoint_list)}
        lineage_meta_filtered = lineage_meta[lineage_meta["timepoints"].isin(timepoint_list)]
        lineage_meta_filtered["timepoints"] = lineage_meta_filtered["timepoints"].apply(lambda x: timepoint_remap[x], meta=('timepoints', int)).persist()

        lineage_meta_filtered = lineage_meta_filtered.reset_index()

        #Kymograph File Parquet Index
        lineage_meta_filtered["Kymograph File Parquet Index"] = lineage_meta_filtered.apply(lambda x: int(f'{int(x["File Index"]):08n}{int(x["File Trench Index"]):04n}{int(x["timepoints"]):04n}'), axis=1, meta=(None, 'int64')).persist()
        #Kymograph FOV Parquet Index
        lineage_meta_filtered["Kymograph FOV Parquet Index"] = lineage_meta_filtered.apply(lambda x: int(f'{x["fov"]:04n}{x["row"]:04n}{x["trench"]:04n}{x["timepoints"]:04n}'), axis=1, meta=(None, 'int64')).persist()
        #Trenchid Timepoint Index
        lineage_meta_filtered["Trenchid Timepoint Index"] = lineage_meta_filtered.apply(lambda x: int(f'{x["trenchid"]:08n}{x["timepoints"]:04n}'), axis=1, meta=(None, 'int64')).persist()
        #FOV Parquet Index
        lineage_meta_filtered["FOV Parquet Index"] = lineage_meta_filtered.apply(lambda x: int(f'{int(x["fov"]):04n}{int(x["row"]):04n}{int(x["trench"]):04n}{int(x["timepoints"]):04n}{int(x["CellID"]):04n}'), axis=1, meta=(None, 'int64'))
        #File Parquet Index
        lineage_meta_filtered["File Parquet Index"] = lineage_meta_filtered.apply(lambda x: int(f'{int(x["File Index"]):08n}{int(x["File Trench Index"]):04n}{int(x["timepoints"]):04n}{int(x["CellID"]):04n}'), axis=1, meta=(None, 'int64'))

        lineage_meta_filtered = lineage_meta_filtered.set_index("File Parquet Index",sorted=True)
        
        dd.to_parquet(lineage_meta_filtered, self.subsample_headpath + "/lineage/output_temp",engine='pyarrow',compression='gzip',write_metadata_file=True)
        ## set daughters outside of observation window to -1
        lineage_meta_filtered = dd.read_parquet(self.subsample_headpath + "/lineage/output_temp",engine="pyarrow",calculate_divisions=True)
        
        input_test_partition = lineage_meta_filtered.get_partition(0).compute()
        test_partition_1 = self.reset_daughters(input_test_partition)
        lineage_meta_filtered_daughters_reset = dd.map_partitions(self.reset_daughters,lineage_meta_filtered,meta=test_partition_1)

        dd.to_parquet(lineage_meta_filtered_daughters_reset, self.subsample_headpath + "/lineage/output",engine='pyarrow',compression='gzip',write_metadata_file=True)
        dask_controller.daskclient.cancel(lineage_meta_filtered)
        dask_controller.daskclient.cancel(lineage_meta_filtered_daughters_reset)
        
        shutil.rmtree(self.subsample_headpath + "/lineage/output_temp")
                
        
    def clone_global_metadata(self,timepoint_list):
        meta_handle = pandas_hdf5_handler(self.headpath + '/metadata.hdf5')
        output_meta_handle = pandas_hdf5_handler(self.subsample_headpath + '/metadata.hdf5')

        global_df = meta_handle.read_df("global")
        global_meta = meta_handle.read_df("global", read_metadata=True).metadata

        timepoint_remap = {timepoint:i for i,timepoint in enumerate(timepoint_list)}

        filtered_global_df = global_df.reset_index()
        filtered_global_df = filtered_global_df[filtered_global_df["timepoints"].isin(timepoint_list)]
        filtered_global_df["timepoints"] = filtered_global_df["timepoints"].apply(lambda x: timepoint_remap[x])

        filtered_global_df = filtered_global_df.set_index(["fov","timepoints"])

        output_meta_handle.write_df("global",filtered_global_df,global_meta)

    def clone_par_files(self):
        par_filepaths = [filepath for filepath in os.listdir(self.headpath) if ".par" in filepath]
        for par_filepath in par_filepaths:
            shutil.copyfile(self.headpath + "/" + par_filepath, self.subsample_headpath + "/" + par_filepath)
        #also copy eliminated rows        
        shutil.copyfile(self.headpath + "/kymograph/global_rows.pkl", self.subsample_headpath + "/kymograph/global_rows.pkl")
    
    def crop_dataset(self,dask_controller,timepoint_list,overwrite=False):
        writedir(self.subsample_headpath,overwrite=overwrite)
        writedir(self.subsample_headpath + "/kymograph",overwrite=False)
        writedir(self.subsample_headpath + "/" + self.segpath,overwrite=False)

        kymo_meta = dd.read_parquet(self.headpath + "/kymograph/metadata",calculate_divisions=True)

        all_kymograph_indices = kymo_meta["File Index"].unique().compute().tolist()
        num_files = len(all_kymograph_indices)

        random_priorities = np.random.uniform(size=(num_files,))
        for k in range(0,num_files):
            priority = random_priorities[k]
            file_idx = all_kymograph_indices[k]

            future = dask_controller.daskclient.submit(self.crop_timepoints_kymograph,k,timepoint_list,retries=1,priority=priority)
            dask_controller.futures["Kymograph Cropped: " + str(k)] = future

        random_priorities = np.random.uniform(size=(num_files,))
        for k in range(0,num_files):
            priority = random_priorities[k]
            file_idx = all_kymograph_indices[k]

            future = dask_controller.daskclient.submit(self.crop_timepoints_segmentation,k,timepoint_list,retries=1,priority=priority)
            dask_controller.futures["Segmentation Cropped: " + str(k)] = future

        all_futures = [dask_controller.futures["Kymograph Cropped: " + str(k)] for k in range(num_files)] +\
        [dask_controller.futures["Segmentation Cropped: " + str(k)] for k in range(num_files)]

        dask_controller.daskclient.gather(all_futures);

        self.clone_kymograph_metadata(timepoint_list)
        self.clone_lineage_metadata(dask_controller,timepoint_list)
        self.clone_global_metadata(timepoint_list)
        self.clone_par_files()
        print("Finished.")