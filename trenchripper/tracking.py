import numpy as np
import skimage as sk
import h5py
import os
import copy
import pickle
import shutil
import cv2
import pandas as pd

import scipy.signal as signal
from .utils import kymo_handle,pandas_hdf5_handler,writedir
from .cluster import hdf5lock
from time import sleep
from dask.distributed import worker_client
from pandas import HDFStore

class mother_tracker():
    
    def get_doubling_times(self, times, peak_series):
        peaks = detect_peaks(peak_series, mpd=5)
        time_of_doubling = times[peaks]
        doubling_time_s = times[1:]-times[0:len(times)-1]
        return np.array([times[0:len(times)-1], doubling_time_s]).T
        
    def get_mother_cell_growth_props(self, mother_data_frame):
        loading_fractions = np.array(mother_data_frame["loading_fractions"])
        cutoff_index = len(loading_fractions)
        if cutoff_index < 5:
            return None, None, None
        for i in range(len(loading_fractions)-5):
            if np.all(~((loading_fractions[i:i+5] > 0.35)*(loading_fractions[i:i+5] < 0.6))):
                cutoff_index = i
        if cutoff_index < 5:
            return None, None, None
        times = np.array(mother_data_frame["time_s"])[:cutoff_index]
        major_axis_length = np.array(mother_data_frame["major_axis_length"])[:cutoff_index]
        area = np.array(mother_data_frame["area"])[:cutoff_index]
        mal_smoothed = signal.wiener(major_axis_length)
        area_smoothed = signal.wiener(area)
        
        doubling_time = self.get_doubling_times(times, major_axis_length)
        doubling_time_smoothed = self.get_doubling_times(times, mal_smoothed)
        
        instantaneous_growth_rate_length = np.gradient(major_axis_length, times)[1:]
        instantaneous_growth_rate_area = np.gradient(area, times)[1:]
        instantaneous_growth_rate_length_smoothed = np.gradient(mal_smoothed, times)[1:]
        instantaneous_growth_rate_area_smoothed = np.gradient(major_axis_length, times)[1:]
        
        growth_rate_data = np.array([times[1:], instantaneous_growth_rate_length, instantaneous_growth_rate_length_smoothed, instantaneous_growth_rate_area, instantaneous_growth_rate_area_smoothed]).T
        
        doubling_time_dataframe = pd.DataFrame(doubling_time, index=None, columns=["time_s", "doubling_time_s"])
        doubling_time_smoothed_dataframe = pd.DataFrame(doubling_time_smoothed, index=None, columns=["time_s", "doubling_time_s"])
        growth_rate_dataframe = pd.DataFrame(growth_rate_data, index=None, columns=["time_s", "igr_length", "igr_length_smoothed", "igr_area", "igr_area_smoothed"])
                                               
        return doubling_time_dataframe, doubling_time_smoothed_dataframe, growth_rate_dataframe
        

class mother_tracker_cluster(mother_tracker):
    def __init__(self, headpath):
        super(mother_tracker_cluster).__init__()
        self.headpath = headpath
        self.phasesegmentationpath = headpath + "/phasesegmentation"
        self.phasedatapath = self.phasesegmentationpath + "/cell_data"
        self.growthdatapath = headpath +"/growth_data.hdf5"
        
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
    
    def get_growth_props(self, file_idx):
        file_df = pd.read_hdf(os.path.join(self.phasedatapath, "data_%d.h5" % file_idx))
        trenches = file_df.index.unique("file_trench_idx")
        dt_dfs = []
        dt_smoothed_dfs = []
        growth_rate_dfs = []
        for trench in trenches:
            mother_df = file_df.loc[trench,:,1]
            dt_df, dt_smoothed_df, growth_rate_df = self.get_mother_cell_growth_props(mother_df)
            if dt_df is not None:
                dt_df["file_trench_idx"] = trench
                dt_smoothed_df["file_trench_idx"] = trench
                dt_smoothed_df["file_trench_idx"] = trench
                dt_dfs.append(dt_df)
                dt_smoothed_dfs.append(dt_smoothed_df)
                growth_rate_df.append(growth_rate_df)
        dt_dfs = pd.concatenate(dt_dfs)
        dt_smoothed_dfs = pd.concatenate(dt_smoothed_dfs)
        growth_rate_dfs = pd.concatenate(growth_rate_dfs)
        dt_dfs["file_idx"] = file_idx
        dt_smoothed_dfs["file_idx"] = file_idx
        growth_rate_dfs["file_idx"] = file_idx
        dt_dfs.set_index(["file_idx", "file_trench_idx"])
        dt_smoothed_dfs.set_index(["file_idx", "file_trench_idx"])
        growth_rate_dfs.set_index(["file_idx", "file_trench_idx"])
        return dt_dfs, dt_smoothed_dfs, growth_rate_dfs
        
    def growth_props_cluster(self, daskcontroller):
        dask_controller.futures = []
        writedir(self.phasedatapath,overwrite=True)
        kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        file_list = kymodf["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
                                               
        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for file_idx in file_list:
            dask_controller.futures.append(dask_controller.daskclient.submit(self.get_growth_props, file_idx))
    
    def postprocess_growth_props(self, dask_controller):
        store = HDFStore(self.growthdatapath)
        for idx, future in enumerate(dask_controller.futures):
            dt_df, dt_smoothed_df, gr_df = future.result()
            if idx == 0:
                store.put("doubling_time", dt_df, format='table', data_columns=True)
                store.put("doubling_time_smoothed", dt_smoothed_df, format='table', data_columns=True)
                store.put("growth_rate", gr_df, format='table', data_columns=True)
            else:
                store.append("doubling_time", dt_df)
                store.append("doubling_time_smoothed", dt_smoothed_df)
                store.append("growth_rate", gr_df)
        store.close()