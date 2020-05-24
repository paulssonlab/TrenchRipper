# fmt: off
from .utils import pandas_hdf5_handler
from .trcluster import dask_controller
import h5py
import skimage as sk
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt

class regionprops_extractor:
    def __init__(self,headpath,segmentationdir,intensity_channel_list=None,props=['centroid','area','mean_intensity']):
        self.headpath = headpath
        self.intensity_channel_list = intensity_channel_list
        self.kymographpath = headpath + "/kymograph"
        self.segmentationpath = headpath + "/" + segmentationdir
        self.metapath = headpath + "/metadata.hdf5"
        self.analysispath = headpath + "/analysis.pkl"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        self.props = props

    def get_file_regionprops(self,file_idx):
        segmentation_file = self.segmentationpath + "/segmentation_" + str(file_idx) + ".hdf5"
        kymograph_file = self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5"

        with h5py.File(segmentation_file,"r") as segfile:
            seg_arr = segfile["data"][:]
        if self.intensity_channel_list is not None:
            kymo_arr_list = []
            with h5py.File(kymograph_file,"r") as kymofile:
                for intensity_channel in self.intensity_channel_list:
                    kymo_arr_list.append(kymofile[intensity_channel][:])
        all_props_list = []
        for k in range(seg_arr.shape[0]):
            for t in range(seg_arr.shape[1]):
                labels = sk.measure.label(seg_arr[k,t])
                if self.intensity_channel_list is not None:
                    for i,intensity_channel in enumerate(self.intensity_channel_list):
                        rps = sk.measure.regionprops(labels, kymo_arr_list[i][k,t])
                        props_list = [[file_idx, k, t, obj, intensity_channel]+[getattr(rp, prop_key) for prop_key in self.props] for obj,rp in enumerate(rps)]
                        all_props_list+=props_list
                else:
                    rps = sk.measure.regionprops(labels)
                    props_list = [[file_idx, k, t, obj]+[getattr(rp, prop_key) for prop_key in self.props] for obj,rp in enumerate(rps)]
                    all_props_list+=props_list

        if self.intensity_channel_list is not None:
            column_list = ['File Index','File Trench Index','timepoints','Objectid','Intensity Channel'] + self.props
            df_out = pd.DataFrame(all_props_list, columns=column_list).reset_index()
        else:
            column_list = ['File Index','File Trench Index','timepoints','Objectid'] + self.props
            df_out = pd.DataFrame(all_props_list, columns=column_list).reset_index()

        df_out = df_out.set_index(['File Index','File Trench Index','timepoints','Objectid'], drop=True, append=False, inplace=False)
        temp_df_path = self.kymographpath + "/temp_df_" + str(file_idx) + ".pkl"
        df_out.to_pickle(temp_df_path)
        return file_idx

    def analyze_all_files(self,dask_cont):
        kymo_meta = self.meta_handle.read_df("kymograph")
        file_list = kymo_meta["File Index"].unique().tolist()
        num_file_jobs = len(file_list)

        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            future = dask_cont.daskclient.submit(self.get_file_regionprops,file_idx,retries=1,priority=priority)
            dask_cont.futures["File Index: " + str(file_idx)] = future

    def compile_data(self,dask_cont):
        kymo_meta = self.meta_handle.read_df("kymograph")
        file_list = kymo_meta["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
        file_idx_list = dask_cont.daskclient.gather([dask_cont.futures["File Index: " + str(file_idx)] for file_idx in file_list],errors="skip")

        ttl_indices = len(file_idx_list)

        df_out = []
        for file_idx in file_idx_list:
            temp_df_path = self.kymographpath + "/temp_df_" + str(file_idx) + ".pkl"
            temp_df = pd.read_pickle(temp_df_path)
            df_out.append(temp_df)
            os.remove(temp_df_path)
        df_out = pd.concat(df_out)

        kymo_meta = kymo_meta.reset_index(inplace=False)
        kymo_meta = kymo_meta.set_index(["File Index","File Trench Index","timepoints"], drop=True, append=False, inplace=False)
        kymo_meta = kymo_meta.sort_index()

        df_out = df_out.reset_index(inplace=False)
        df_out = df_out.set_index(["File Index","File Trench Index","timepoints"], drop=True, append=False, inplace=False)
        df_out = df_out.sort_index()

        mergeddf = df_out.join(kymo_meta)
        mergeddf = mergeddf.reset_index(inplace=False)
        mergeddf = mergeddf.set_index(["File Index","File Trench Index","timepoints","Intensity Channel","Objectid"], drop=True, append=False, inplace=False)
        del mergeddf["index"]
        mergeddf = mergeddf.sort_index()

        mergeddf.to_pickle(self.analysispath)

    def export_all_data(self,n_workers=20,memory='4GB'):

        dask_cont = dask_controller(walltime='01:00:00',local=False,n_workers=n_workers,memory=memory,working_directory=self.headpath+"/dask")
        dask_cont.startdask()
#         dask_cont.daskcluster.start_workers()
        dask_cont.displaydashboard()
        dask_cont.futures = {}

        try:
            self.analyze_all_files(dask_cont)
            self.compile_data(dask_cont)
            dask_cont.shutdown()
        except:
            dask_cont.shutdown()
            raise

class kymograph_viewer:
    def __init__(self,headpath,channel,segdir):
        self.headpath = headpath
        self.kymopath = headpath + "/kymograph"
        self.segpath = headpath + "/" + segdir
        self.channel = channel

    def get_kymograph_data(self,file_idx,trench_idx):
        with h5py.File(self.kymopath + "/kymograph_"+str(file_idx)+".hdf5", "r") as infile:
            kymodat = infile[self.channel][trench_idx]
        with h5py.File(self.segpath+"/segmentation_"+str(file_idx)+".hdf5", "r") as infile:
            segdat = infile["data"][trench_idx]
        segdat = np.array([sk.morphology.label(segdat[t],connectivity=1) for t in range(segdat.shape[0])])
        return kymodat,segdat

    def plot_kymograph(self,kymograph):
        """Helper function for plotting kymographs. Takes a kymograph array of
        shape (y_dim,x_dim,t_dim).

        Args:
            kymograph (array): kymograph array of shape (y_dim,x_dim,t_dim).
        """
        list_in_t = [kymograph[t,:,:] for t in range(kymograph.shape[0])]
        img_arr = np.concatenate(list_in_t,axis=1)
        plt.imshow(img_arr)

    def plot_kymograph_data(self,kymodat,segdat,x_size=20,y_size=6):
        fig=plt.figure(figsize=(x_size, y_size))
        fig.add_subplot(2, 1, 1)
        self.plot_kymograph(kymodat)
        fig.add_subplot(2, 1, 2)
        self.plot_kymograph(segdat)
        plt.show()

    def inspect_trench(self,file_idx,trench_idx,x_size=20,y_size=6):
        kymodat,segdat = self.get_kymograph_data(file_idx,trench_idx)
        self.plot_kymograph_data(kymodat,segdat,x_size=x_size,y_size=y_size)
