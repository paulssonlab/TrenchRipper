# fmt: off
from .utils import pandas_hdf5_handler
from .trcluster import dask_controller

import h5py
import os

import skimage as sk
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.array as da
import dask.delayed as delayed
import xarray as xr
import holoviews as hv
import panel as pn
import scipy as sp
import scipy.stats

from distributed.client import futures_of
from time import sleep

from matplotlib import pyplot as plt

from .metrics import get_cell_dimensions_spherocylinder_ellipse,get_cell_dimensions_spherocylinder_perimeter_area,width_length_from_permeter_area
## HERE

class regionprops_extractor:
    def __init__(self,headpath,segmentationdir,intensity_channel_list=None,\
                 size_estimation=False,size_estimation_method="Perimeter/Area",\
                 include_background=False,props_list=['centroid','area'],custom_props_list=[],\
                 intensity_props_list=['mean_intensity'],custom_intensity_props_list=[],props_to_unpack={'centroid':["centroid_y","centroid_x"]},\
                 pixel_scaling_factors={'centroid_y': 1,'centroid_x': 1}):
        self.headpath = headpath
        self.intensity_channel_list = intensity_channel_list
        self.intensity_channel_dict = {channel:i for i,channel in enumerate(intensity_channel_list)}

        self.size_estimation = size_estimation
        self.size_estimation_method = size_estimation_method
        self.include_background = include_background
        self.kymographpath = headpath + "/kymograph"
        self.segmentationpath = headpath + "/" + segmentationdir
        self.metapath = self.kymographpath + "/metadata"

        self.global_metapath = self.headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.global_metapath)
        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = fovdf.metadata

        self.analysispath = headpath + "/analysis"
        self.props_list = props_list
        self.custom_props_list = custom_props_list
        self.custom_props_str_list = [item.__name__ for item in self.custom_props_list]
        self.intensity_props_list = intensity_props_list
        self.custom_intensity_props_list = custom_intensity_props_list
        self.custom_intensity_props_str_list = [item.__name__ for item in self.custom_intensity_props_list]
        self.props_to_unpack = props_to_unpack
        self.pixel_scaling_factors = pixel_scaling_factors
        
        if self.size_estimation:
            est_pixel_scaling_factors = {"Length":1,"Width":1,"Volume":3,"Surface Area":2}
            self.pixel_scaling_factors = {**self.pixel_scaling_factors, **est_pixel_scaling_factors}

    def get_file_regionprops(self,file_idx):
        pixel_microns = self.metadata['pixel_microns']

        segmentation_file = self.segmentationpath + "/segmentation_" + str(file_idx) + ".hdf5"
        kymograph_file = self.kymographpath + "/kymograph_" + str(file_idx) + ".hdf5"

        with h5py.File(segmentation_file,"r") as segfile:
            seg_arr = segfile["data"][:]
        if self.intensity_channel_list is not None:
            kymo_arr_list = []
            with h5py.File(kymograph_file,"r") as kymofile:
                for intensity_channel in self.intensity_channel_list:
                    kymo_arr_list.append(kymofile[intensity_channel][:])
        props_output = []
        for k in range(seg_arr.shape[0]):
            for t in range(seg_arr.shape[1]):
                labels = sk.measure.label(seg_arr[k,t])
                ## Measure regionprops of background pixels; will always be marked as the first object
                if self.include_background:
                    labels += 1

                #non intensity info first
                non_intensity_rps = sk.measure.regionprops(labels,extra_properties=self.custom_props_list)

                if self.size_estimation:
                    if self.size_estimation_method == "Ellipse":
                        cell_dimensions = get_cell_dimensions_spherocylinder_ellipse(non_intensity_rps)
                    elif self.size_estimation_method == "Perimeter/Area":
                        cell_dimensions = get_cell_dimensions_spherocylinder_perimeter_area(non_intensity_rps)

                if self.intensity_channel_list is not None:
                    intensity_rps_list = []
                    for i,intensity_channel in enumerate(self.intensity_channel_list):
                        intensity_rps = sk.measure.regionprops(labels, kymo_arr_list[i][k,t],extra_properties=self.custom_intensity_props_list)
                        intensity_rps_list.append(intensity_rps)

                for idx in range(len(non_intensity_rps)):
                    rp = non_intensity_rps[idx]
                    props_entry = [file_idx, k, t, idx]
                    for prop_key in (self.props_list+self.custom_props_str_list):
                        prop = getattr(rp, prop_key)
                        if prop_key in self.props_to_unpack.keys():
                            prop_out = dict(zip(self.props_to_unpack[prop_key],list(prop)))
                        else:
                            prop_out = {prop_key:prop}
                        for key,value in prop_out.items():
                            if key in self.pixel_scaling_factors.keys():
                                output = value*(pixel_microns**self.pixel_scaling_factors[key])
                            else:
                                output = value
                            props_entry.append(output)
                    if self.size_estimation: ## not super effecient
                        single_cell_dimensions = cell_dimensions[idx]
                        for key,value in single_cell_dimensions.items():
                            if key in self.pixel_scaling_factors.keys():
                                output = value*(pixel_microns**self.pixel_scaling_factors[key])
                            else:
                                output = value
                            props_entry.append(output)

                    if self.intensity_channel_list is not None:
                        for i,intensity_channel in enumerate(self.intensity_channel_list):
                            intensity_rps=intensity_rps_list[i]
                            inten_rp = intensity_rps[idx]
                            for prop_key in (self.intensity_props_list+self.custom_intensity_props_str_list):
                                prop = getattr(inten_rp, prop_key)
                                if prop_key in self.props_to_unpack.keys():
                                    prop_out = dict(zip(self.props_to_unpack[prop_key],list(prop)))
                                else:
                                    prop_out = {prop_key:prop}
                                for key,value in prop_out.items():
                                    if key in self.pixel_scaling_factors.keys():
                                        output = value*(pixel_microns**self.pixel_scaling_factors[key])
                                    else:
                                        output = value
                                    props_entry.append(output)

                    props_output.append(props_entry)

        base_list = ['File Index','File Trench Index','timepoints','Objectid']

        unpacked_props_list = []
        for prop_key in (self.props_list+self.custom_props_str_list):
            if prop_key in self.props_to_unpack.keys():
                unpacked_names = self.props_to_unpack[prop_key]
                unpacked_props_list += unpacked_names
            else:
                unpacked_props_list.append(prop_key)
        if self.size_estimation:
            for prop_key in ["Length","Width","Volume","Surface Area"]:
                unpacked_props_list.append(prop_key)

        if self.intensity_channel_list is not None:
            for channel in self.intensity_channel_list:
                for prop_key in (self.intensity_props_list+self.custom_intensity_props_str_list):
                    if prop_key in self.props_to_unpack.keys():
                        unpacked_names = self.props_to_unpack[prop_key]
                        unpacked_props_list += [channel + " " + item for item in unpacked_names]
                    else:
                        unpacked_props_list.append(channel + " " + prop_key)

        column_list = base_list + unpacked_props_list

        df_out = pd.DataFrame(props_output, columns=column_list).reset_index()
        file_idx = df_out.apply(lambda x: int(f'{x["File Index"]:08n}{x["File Trench Index"]:04n}{x["timepoints"]:04n}{x["Objectid"]:02n}'), axis=1)

        df_out["File Parquet Index"] = [item for item in file_idx]
        df_out = df_out.set_index("File Parquet Index").sort_index()
        del df_out["index"]

        return df_out

    def analyze_all_files(self,dask_cont):
        df = dd.read_parquet(self.metapath,calculate_divisions=True)
        file_list = df["File Index"].unique().compute().tolist()
#         kymo_meta = dd.read_parquet(self.metapath)
#         file_list = kymo_meta["File Index"].unique().tolist()

        delayed_list = []
        for file_idx in file_list:
            df_delayed = delayed(self.get_file_regionprops)(file_idx)
            delayed_list.append(df_delayed.persist())

        ## filtering out non-failed dataframes ##
        all_delayed_futures = []
        for item in delayed_list:
            all_delayed_futures += futures_of(item)
        while any(future.status == "pending" for future in all_delayed_futures):
            sleep(0.1)

        good_delayed = []
        for item in delayed_list:
            if all([future.status == "finished" for future in futures_of(item)]):
                good_delayed.append(item)

        ## compiling output dataframe ##
        df_out = dd.from_delayed(good_delayed).persist()
        df_out["File Parquet Index"] = df_out.index
        df_out = df_out.set_index("File Parquet Index", drop=True, sorted=False)
        df_out = df_out.repartition(partition_size="25MB").persist()

        kymo_df = dd.read_parquet(self.metapath,calculate_divisions=True)
        kymo_df["File Merge Index"] = kymo_df["File Parquet Index"]
        kymo_df = kymo_df.set_index("File Merge Index", sorted=True)
        kymo_df = kymo_df.drop(["File Index","File Trench Index","timepoints","File Parquet Index"], axis=1)

        df_out["File Merge Index"] = df_out.apply(lambda x: int(f'{x["File Index"]:08n}{x["File Trench Index"]:04n}{x["timepoints"]:04n}'), axis=1)
        df_out = df_out.reset_index(drop=True)
        df_out = df_out.set_index("File Merge Index", sorted=True)

        df_out = df_out.join(kymo_df)
        df_out = df_out.set_index("File Parquet Index",sorted=True)

        dd.to_parquet(
            df_out,
            self.analysispath,
            engine="fastparquet",
            compression="gzip",
            write_metadata_file=True,
        )

    def export_all_data(self,n_workers=20,memory='8GB'):

        dask_cont = dask_controller(walltime='01:00:00',local=False,n_workers=n_workers,memory=memory,working_directory=self.headpath+"/dask")
        dask_cont.startdask()
        dask_cont.displaydashboard()
        dask_cont.futures = {}

        try:
            self.analyze_all_files(dask_cont)
            dask_cont.shutdown(delete_files=False)
        except:
            dask_cont.shutdown(delete_files=False)
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

def get_image_measurements(
    kymographpath, n_partitions, divisions, channels, file_idx, output_name, img_fn, *args, **kwargs
):

    df = dd.read_parquet(kymographpath + "/metadata",calculate_divisions=True)
    df = df.set_index("File Parquet Index",sorted=True,npartitions=n_partitions,divisions=divisions)

    start_idx = int(str(file_idx) + "00000000")
    end_idx = int(str(file_idx) + "99999999")

    working_dfs = []

    proc_file_path = kymographpath + "/kymograph_" + str(file_idx) + ".hdf5"
    with h5py.File(proc_file_path, "r") as infile:
        working_filedf = df.loc[start_idx:end_idx].compute(scheduler='threads')
        trench_idx_list = working_filedf["File Trench Index"].unique().tolist()
        for trench_idx in trench_idx_list:
            trench_df = working_filedf[working_filedf["File Trench Index"] == trench_idx]
            for channel in channels:
                kymo_arr = infile[channel][trench_idx]
                fn_out = [
                img_fn(kymo_arr[i], *args, **kwargs)
                    for i in range(kymo_arr.shape[0])
                ]
                trench_df[channel + " " + output_name] = fn_out
            working_dfs.append(trench_df)


    out_df = pd.concat(working_dfs)
    return out_df


def get_all_image_measurements(dask_controller, headpath, output_path, channels, output_name, img_fn, *args, **kwargs):
    kymographpath = headpath + "/kymograph"
    df = dd.read_parquet(kymographpath + "/metadata",calculate_divisions=True)
    df = df.set_index("File Parquet Index",sorted=True)
    n_partitions,divisions = (df.npartitions,df.divisions)

    file_list = df["File Index"].unique().compute().tolist()

    df_futures = []
    for file_idx in file_list:
        df_future = dask_controller.daskclient.submit(get_image_measurements,kymographpath,n_partitions,divisions,channels,file_idx,output_name,img_fn,*args,retries=0,**kwargs)

#         df_delayed = delayed(get_image_measurements)(
#             kymographpath, channels, file_idx, output_name, img_fn, *args, **kwargs
#         )
#         delayed_list.append(df_delayed.persist())
        df_futures.append(df_future)

    while any(future.status == 'pending' for future in df_futures):
        sleep(0.1)

    good_futures = []
    for future in df_futures:
        if future.status == 'finished':
            good_futures.append(future)

    good_delayed = [delayed(good_future.result)() for good_future in good_futures]

#     ## filtering out non-failed dataframes ##
#     all_delayed_futures = []
#     for item in delayed_list:
#         all_delayed_futures += futures_of(item)
#     while any(future.status == "pending" for future in all_delayed_futures):
#         sleep(0.1)

#     good_delayed = []
#     for item in delayed_list:
#         if all([future.status == "finished" for future in futures_of(item)]):
#             good_delayed.append(item)

    ## compiling output dataframe ##
    df_out = dd.from_delayed(good_delayed).persist()
    df_out["FOV Parquet Index"] = df_out.index
    df_out = df_out.set_index("FOV Parquet Index", drop=True, sorted=False)
    df_out = df_out.repartition(partition_size="25MB").persist()

    dd.to_parquet(
        df_out,
        output_path,
        engine="fastparquet",
        compression="gzip",
        write_metadata_file=True,
    )

def get_mapping(df):
    df = df.apply(lambda x: eval(x)) ## dirty fix to type issue
    df1_xy = np.array(df["Frame 1 Position"]).T
    df2_xy = np.array(df["Frame 2 Position"]).T
    ymat = np.subtract.outer(df1_xy[:,0],df2_xy[:,0])
    xmat = np.subtract.outer(df1_xy[:,1],df2_xy[:,1])
    distmat = (ymat**2+xmat**2)**(1/2)

    #ensuring map is one-to-one
    mapping = np.argmin(distmat,axis=1)
    invmapping = np.argmin(distmat,axis=0)

    mapping = {idx:map_idx for idx,map_idx in enumerate(mapping) if invmapping[map_idx]==idx}

    df1_trenchids = df["Frame 1 Trenchid"]
    df2_trenchids = df["Frame 2 Trenchid"]

    trenchid_map = {trenchid: df2_trenchids[mapping[i]] for i,trenchid in enumerate(df1_trenchids)\
                        if i in mapping.keys()}

    return trenchid_map

def get_trenchid_map(kymodf1,kymodf2,offset_x=0.,offset_y=0.):

    ## Generates trenchid map from two kymograph dataframes
    ## Generally inputs should be kymograph dataframes containing
    ## the two timepoints to be used to establish the mapping

    fovset1 = set(kymodf1["fov"].unique().compute().tolist())
    fovset2 = set(kymodf2["fov"].unique().compute().tolist())
    fov_intersection = fovset1.intersection(fovset2)

    fovdf1_groupby = kymodf1.set_index("fov",sorted=True).groupby("fov")
    fovdf1_pos = fovdf1_groupby.apply(lambda x:[list(x["y (local)"]),list(x["x (local)"])],meta=list).loc[list(fov_intersection)].to_frame()
    fovdf1_pos.columns=["Frame 1 Position"]
    fovdf1_trenchid = fovdf1_groupby.apply(lambda x:list(x["trenchid"]),meta=list).loc[list(fov_intersection)].to_frame()
    fovdf1_trenchid.columns=["Frame 1 Trenchid"]

    fovdf2_groupby = kymodf2.set_index("fov",sorted=True).groupby("fov")
    if (offset_x == 0.) and (offset_y == 0.):
        fovdf2_pos = fovdf2_groupby.apply(lambda x:[list(x["y (local)"]),list(x["x (local)"])],meta=list).loc[list(fov_intersection)].to_frame()
    else:
        fovdf2_pos = fovdf2_groupby.apply(lambda x:[list(x["y (local)"]-offset_y),list(x["x (local)"]-offset_x)],meta=list).loc[list(fov_intersection)].to_frame()
    fovdf2_pos.columns=["Frame 2 Position"]
    fovdf2_trenchid = fovdf2_groupby.apply(lambda x:list(x["trenchid"]),meta=list).loc[list(fov_intersection)].to_frame()
    fovdf2_trenchid.columns=["Frame 2 Trenchid"]

    combined_df = fovdf1_pos.join([fovdf1_trenchid,fovdf2_pos,fovdf2_trenchid])
    mapping_df = combined_df.apply(get_mapping,axis=1,meta=dict).compute()
    mapping_df = mapping_df.apply(lambda x: eval(x))

    trenchid_map = {key:val for item in mapping_df.to_list() for key,val in item.items()}

    return trenchid_map

def files_to_trenchid_map(phenotype_kymopath,barcode_kymopath,offset_x=0.,offset_y=0.):

    ### Utility for establishing a trenchid map from phenotype data to barcode data
    ### Using the last and first timepoints, respectively
    ### offsets defined as shift to move phenotype trench to barcode trench position (reason for the sign flip)

    pheno_kymo_df = dd.read_parquet(phenotype_kymopath,calculate_divisions=True)
    barcode_kymo_df = dd.read_parquet(barcode_kymopath,calculate_divisions=True)

    max_pheno_tpt = pheno_kymo_df.get_partition(0)["timepoints"].max().compute()
    min_barcode_tpt = barcode_kymo_df.get_partition(0)["timepoints"].min().compute()

    last_pheno_tpt_df = pheno_kymo_df[pheno_kymo_df["timepoints"] == max_pheno_tpt]
    first_barcode_tpt_df = barcode_kymo_df[barcode_kymo_df["timepoints"] == min_barcode_tpt]

    trenchid_map = get_trenchid_map(first_barcode_tpt_df,last_pheno_tpt_df,offset_x=-offset_x,offset_y=-offset_y)

    return trenchid_map

def get_called_df(barcode_df, scalar_dict, trenchid_map):

    ## Maps a dictionary of measurements from each trench with a trenchid index
    ## (all in pandas series format) into a barcode_df, using a given trenchid_map
    ## of the form barcode_trenchid:phenotype_trenchid (from scalar_dict)

    init_scalar_df = scalar_dict[list(scalar_dict.keys())[0]]
    init_scalar_df_idx = init_scalar_df.index.compute().to_list()

    valid_barcode_df = barcode_df[barcode_df["trenchid"].isin(trenchid_map.keys())].compute()
    barcode_df_mapped_trenchids = valid_barcode_df["trenchid"].apply(lambda x: trenchid_map[x])
    valid_init_scalar_df_indices = barcode_df_mapped_trenchids.isin(init_scalar_df_idx)
    barcode_df_mapped_trenchids = barcode_df_mapped_trenchids[valid_init_scalar_df_indices]
    final_valid_barcode_df_indices = barcode_df_mapped_trenchids.index.to_list()
    called_df = barcode_df.loc[final_valid_barcode_df_indices]
    called_df["phenotype trenchid"] = barcode_df_mapped_trenchids
    called_df = called_df.set_index("phenotype trenchid").compute()

    for key,val in scalar_dict.items():
        called_df[key] = val.compute().loc[called_df.index]

    called_df = called_df.reset_index(drop=False)
    called_df = dd.from_pandas(called_df,chunksize=50000).persist()

    return called_df

def get_scalar_mean_median_mode(df,groupby,key): ## defined bc of a global variable issue with the key not playing nice with dask groupby

    ## Gets basic stats given trenchwise variable from scalar_dict

    df[key + ": Median"] = groupby.apply(lambda x: np.median(x[key]),meta=float)
    df[key + ": Mean"] = groupby.apply(lambda x: np.mean(x[key]),meta=float)
    df[key + ": SEM"] = groupby.apply(lambda x: scipy.stats.sem(x[key]),meta=float) #ADDING THIS MESSED UP THE LABELS?

    return df

def stats_from_called_df(called_df,scalar_dict):

    called_df_barcode = called_df.set_index("Barcode",sorted=False)
    called_df_barcode_groupby = called_df.groupby(["Barcode"])
    barcode_only_df = called_df_barcode_groupby.apply(lambda x: x.iloc[0])

    barcode_only_df["phenotype trenchids"] = called_df_barcode_groupby.apply(lambda x: list(x["phenotype trenchid"]),meta=list)
    barcode_only_df["trenchids"] = called_df_barcode_groupby.apply(lambda x: list(x["trenchid"]),meta=list)

    for key in scalar_dict.keys():
        barcode_only_df = get_scalar_mean_median_mode(barcode_only_df,called_df_barcode_groupby,key)

    for key,_ in scalar_dict.items():
        del barcode_only_df[key]

    barcode_only_df["N Trenches"] = barcode_only_df["trenchids"].apply(lambda x: len(x), meta=int)

    del barcode_only_df["trenchid"]
    del barcode_only_df["phenotype trenchid"]
    del barcode_only_df["Barcode Signal"]
    del barcode_only_df["Barcode"]
    del barcode_only_df["barcode"]

    return barcode_only_df

def fetch_hdf5(filename,channel):
    with h5py.File(filename,'r') as infile:
        data = infile[channel][:]
    return data

def put_index_first(df,col_name):
    df_cols = list(df)
    df_cols.insert(0, df_cols.pop(df_cols.index(col_name)))
    df = df[df_cols]
#     df = df[.reindex(columns=df_cols)]
    return df

def linked_scatter(df,x_dim_label,y_dim_label,trenchids_as_list=False,trenchid_column='trenchid',minperc=0,maxperc=99,height=400,**scatterkwargs):

    #put trenchid first
    df = put_index_first(df,trenchid_column)

    dataset = hv.Dataset(df)
    x_data_vals,y_data_vals = df[x_dim_label],df[y_dim_label]

    x_low,x_high = np.percentile(x_data_vals,minperc),np.percentile(x_data_vals,maxperc)
    y_low,y_high = np.percentile(y_data_vals,minperc),np.percentile(y_data_vals,maxperc)

    scatter = hv.Scatter(data=dataset,vdims=[y_dim_label],kdims=[x_dim_label])
    # some toy code to try datashading in the future, need to figure out linked brushing for this to work
#     if px_size == None:
#         ## Typical datashade mode
#         shaded_scatter = dynspread(rasterize(scatter, cmap=cmap, cnorm="linear"))
#         shaded_scatter = shaded_scatter.opts(colorbar=True, colorbar_position="bottom")
#     else:
#         shaded_scatter = spread(rasterize(scatter), px=px_size, shape='circle').opts(cmap=cmap, cnorm="linear")
#         shaded_scatter = shaded_scatter.opts(colorbar=True, colorbar_position="bottom")
# #         shaded_scatter = spread(rasterize(scatter, cmap="kbc_r", cnorm="linear"), px=px_size, shape='circle')

    scatter = scatter.opts(tools=["hover","doubletap","lasso_select"],xlim=(x_low,x_high),ylim=(y_low,y_high),height=height,responsive=True,**scatterkwargs)

    select_scatter = hv.streams.Selection1D(source=scatter,index=[0],rename={'index': 'scatterselect'})

    def get_scatter_trenchids(scatterselect, dataset=dataset):

        filtered_dataset = dataset.iloc[scatterselect]

        return hv.Table(filtered_dataset)

    trenchid_table = hv.DynamicMap(get_scatter_trenchids, streams=[select_scatter])
    trenchid_table = trenchid_table.opts(height=height)
    select_trenchid = hv.streams.Selection1D(source=trenchid_table,index=[0],rename={'index': 'trenchid_index'})

    if trenchids_as_list:

        def unpack_trenchids(scatterselect, trenchid_index, dataset=dataset):

            filtered_dataset = dataset.iloc[scatterselect]
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])

            return hv.Table(double_filtered_dataset)

        unpack_trenchid_table = hv.DynamicMap(unpack_trenchids, streams=[select_scatter,select_trenchid])
        unpack_trenchid_table = unpack_trenchid_table.opts(height=height)
        select_unpacked_trenchid = hv.streams.Selection1D(source=unpack_trenchid_table,index=[0],rename={'index': 'unpack_trenchid_index'})
        return scatter,trenchid_table,unpack_trenchid_table,select_scatter,select_trenchid,select_unpacked_trenchid

    else:
        return scatter,trenchid_table,select_scatter,select_trenchid

def linked_kymograph_for_scatter(xrstack,df,x_dim_label,y_dim_label,select_scatter,select_trenchid,select_unpacked_trenchid=None,trenchid_column='trenchid',y_scale=3,x_window_size=300):
    ### stream must return trenchid value
    ### df must have trenchid lookups
    width,height = xrstack.shape[3],int(xrstack.shape[2]*y_scale)
    x_window_scale = x_window_size/xrstack.shape[3]
    x_size = int(xrstack.shape[3]*(y_scale*x_window_scale))

    dataset = hv.Dataset(df)

    if select_unpacked_trenchid != None:
        def select_pt_load_image(channel,scatterselect,trenchid_index,unpack_trenchid_index,width=width,height=height):
            filtered_dataset = dataset.iloc[scatterselect]
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            arr = xrstack.loc[channel,trenchid].values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(scatterselect,trenchid_index,unpack_trenchid_index):
            filtered_dataset = dataset.iloc[scatterselect]
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)

    else:
        def select_pt_load_image(channel,scatterselect,trenchid_index,width=width,height=height):
            filtered_dataset = dataset.iloc[scatterselect]
            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            arr = xrstack.loc[channel,trenchid].values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(scatterselect,trenchid_index):
            filtered_dataset = dataset.iloc[scatterselect]
            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)

    def set_bounds(fig, element, y_dim=height, x_dim=width, x_window_size=x_window_size):
        sy = y_dim-0.5
        sx = x_dim-0.5

        fig.state.y_range.bounds = (-0.5, sy)
        fig.state.x_range.bounds = (0, sx)
        fig.state.x_range.start = 0
        fig.state.x_range.reset_start = 0
        fig.state.x_range.end = x_window_size
        fig.state.x_range.reset_end = x_window_size

        fig.state.y_range.start = 0
        fig.state.y_range.reset_start = 0
        fig.state.y_range.end = y_dim
        fig.state.y_range.reset_end = y_dim


    if select_unpacked_trenchid != None:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_scatter,select_trenchid,select_unpacked_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_scatter,select_trenchid,select_unpacked_trenchid])
    else:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_scatter,select_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_scatter,select_trenchid])

    kymograph_display = image_stack.opts(plot={'Image': dict(colorbar=True, tools=['hover'],hooks=[set_bounds],aspect='equal'),})
    kymograph_display = kymograph_display.opts(cmap='Greys_r',height=height,width=x_size)

    kymograph_display = kymograph_display.redim.range(trenchid=(0,xrstack.shape[1]))
    kymograph_display = kymograph_display.redim.values(Channel=xrstack.coords["Channel"].values.tolist())

    trenchid_display = trenchid_display.opts(text_align="left",text_color="white")

    output_display = kymograph_display*trenchid_display

    return output_display

def linked_histogram(df,label,trenchids_as_list=False,trenchid_column='trenchid',bins=50,minperc=0,maxperc=99,height=400,**histkwargs):

    #put trenchid first
    df = put_index_first(df,trenchid_column)

    dataset = hv.Dataset(df)
    data_vals = df[label]

    x_low = np.percentile(data_vals,minperc)
    x_high = np.percentile(data_vals,maxperc)

    frequencies, edges = np.histogram(data_vals,bins=50,range=(x_low,x_high))
    hist = hv.Histogram((edges,frequencies))

    hist = hist.opts(tools=["hover","doubletap","box_select"],height=height,responsive=True,**histkwargs)

    select_histcolumn = hv.streams.Selection1D(source=hist,index=[0],rename={'index': 'histcolumn'})

    def get_hist_trenchids(histcolumn, df=df, label=label, edges=edges):
        min_col,max_col = np.min(histcolumn),np.max(histcolumn)
        all_edges = edges[min_col:max_col+2]
        filtered_df = df[(df[label]<all_edges[-1])&(df[label]>all_edges[0])]
        filtered_dataset = hv.Dataset(filtered_df)

        return hv.Table(filtered_dataset)

    trenchid_table = hv.DynamicMap(get_hist_trenchids, streams=[select_histcolumn], responsive=True)
    trenchid_table = trenchid_table.opts(height=height)
    select_trenchid = hv.streams.Selection1D(source=trenchid_table,index=[0],rename={'index': 'trenchid_index'})

    if trenchids_as_list:

        def unpack_trenchids(histcolumn, trenchid_index, df=df, label=label, edges=edges):
            min_col,max_col = np.min(histcolumn),np.max(histcolumn)
            all_edges = edges[min_col:max_col+2]
            filtered_df = df[(df[label]<all_edges[-1])&(df[label]>all_edges[0])]
            filtered_dataset = hv.Dataset(filtered_df)
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])

            return hv.Table(double_filtered_dataset)

        unpack_trenchid_table = hv.DynamicMap(unpack_trenchids, streams=[select_histcolumn,select_trenchid], responsive=True)
        unpack_trenchid_table = unpack_trenchid_table.opts(height=height)
        select_unpacked_trenchid = hv.streams.Selection1D(source=unpack_trenchid_table,index=[0],rename={'index': 'unpack_trenchid_index'})

        return hist,trenchid_table,unpack_trenchid_table,edges,select_histcolumn,select_trenchid,select_unpacked_trenchid

    else:
        return hist,trenchid_table,edges,select_histcolumn,select_trenchid

def linked_kymograph_for_hist(xrstack,df,label,edges,select_histcolumn,select_trenchid,select_unpacked_trenchid=None,trenchid_column='trenchid',y_scale=3,x_window_size=300):
    ### stream must return trenchid value
    ### df must have trenchid lookups
    width,height = xrstack.shape[3],int(xrstack.shape[2]*y_scale)
    x_window_scale = x_window_size/xrstack.shape[3]
    x_size = int(xrstack.shape[3]*(y_scale*x_window_scale))

    dataset = hv.Dataset(df)

    if select_unpacked_trenchid != None:

        def select_pt_load_image(channel,histcolumn,trenchid_index,unpack_trenchid_index,width=width,height=height):

            min_col,max_col = np.min(histcolumn),np.max(histcolumn)
            all_edges = edges[min_col:max_col+2]
            filtered_df = df[(df[label]<all_edges[-1])&(df[label]>all_edges[0])]

            filtered_dataset = hv.Dataset(filtered_df)
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            arr = xrstack.loc[channel,trenchid].values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(histcolumn,trenchid_index,unpack_trenchid_index):
            min_col,max_col = np.min(histcolumn),np.max(histcolumn)
            all_edges = edges[min_col:max_col+2]
            filtered_df = df[(df[label]<all_edges[-1])&(df[label]>all_edges[0])]

            filtered_dataset = hv.Dataset(filtered_df)
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)


    else:

        def select_pt_load_image(channel,histcolumn,trenchid_index,width=width,height=height):
            min_col,max_col = np.min(histcolumn),np.max(histcolumn)
            all_edges = edges[min_col:max_col+2]
            filtered_df = df[(df[label]<all_edges[-1])&(df[label]>all_edges[0])]
            filtered_dataset = hv.Dataset(filtered_df)

            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            arr = xrstack.loc[channel,trenchid].values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(histcolumn,trenchid_index):
            min_col,max_col = np.min(histcolumn),np.max(histcolumn)
            all_edges = edges[min_col:max_col+2]
            filtered_df = df[(df[label]<all_edges[-1])&(df[label]>all_edges[0])]
            filtered_dataset = hv.Dataset(filtered_df)

            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)

    def set_bounds(fig, element, y_dim=height, x_dim=width, x_window_size=x_window_size):
        sy = y_dim-0.5
        sx = x_dim-0.5

        fig.state.y_range.bounds = (-0.5, sy)
        fig.state.x_range.bounds = (0, sx)
        fig.state.x_range.start = 0
        fig.state.x_range.reset_start = 0
        fig.state.x_range.end = x_window_size
        fig.state.x_range.reset_end = x_window_size

        fig.state.y_range.start = 0
        fig.state.y_range.reset_start = 0
        fig.state.y_range.end = y_dim
        fig.state.y_range.reset_end = y_dim

    if select_unpacked_trenchid != None:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_histcolumn,select_trenchid,select_unpacked_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_histcolumn,select_trenchid,select_unpacked_trenchid])
    else:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_histcolumn,select_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_histcolumn,select_trenchid])

    kymograph_display = image_stack.opts(plot={'Image': dict(colorbar=True, tools=['hover'],hooks=[set_bounds],aspect='equal'),})
    kymograph_display = kymograph_display.opts(cmap='Greys_r',height=height,width=x_size)

    kymograph_display = kymograph_display.redim.range(trenchid=(0,xrstack.shape[1]))
    kymograph_display = kymograph_display.redim.values(Channel=xrstack.coords["Channel"].values.tolist())

    trenchid_display = trenchid_display.opts(text_align="left",text_color="white")

    output_display = kymograph_display*trenchid_display

    return output_display

def linked_gene_table(df,trenchids_as_list=False,trenchid_column='trenchid',height=400,**histkwargs):
    df = df.reset_index(drop=False).set_index("Gene",drop=False).sort_index()
    df.index = df.index.rename("Gene Index")
    df_genes_only = df[["Gene"]].groupby("Gene").apply(lambda x: x.iloc[0])
    dataset = hv.Dataset(df)
    gene_dataset = hv.Dataset(df_genes_only)

    def get_gene_list(gene_dataset=gene_dataset):
        return hv.Table(gene_dataset)

    gene_list_dmap = hv.DynamicMap(get_gene_list)
    select_gene = hv.streams.Selection1D(source=gene_list_dmap,index=[0],rename={'index': 'select_gene_index'})

    def get_gene_table(select_gene_index,df=df):
        gene_name = gene_dataset.iloc[select_gene_index]["Gene"][0]
        dataset = hv.Dataset(df.loc[gene_name:gene_name])
        return hv.Table(dataset)

    trenchid_table = hv.DynamicMap(get_gene_table, streams=[select_gene])
    select_trenchid = hv.streams.Selection1D(source=trenchid_table,index=[0],rename={'index': 'trenchid_index'})

    if trenchids_as_list:
        def unpack_trenchids(select_gene_index,trenchid_index,dataset=dataset):
            gene_name = gene_dataset.iloc[select_gene_index]["Gene"][0]
            dataset = hv.Dataset(df.loc[gene_name:gene_name])
            filtered_dataset = hv.Dataset({"trenchid": dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            return hv.Table(filtered_dataset)

        unpack_trenchid_table = hv.DynamicMap(unpack_trenchids, streams=[select_gene,select_trenchid], responsive=True)
        unpack_trenchid_table = unpack_trenchid_table.opts(height=height)
        select_unpacked_trenchid = hv.streams.Selection1D(source=unpack_trenchid_table,index=[0],rename={'index': 'unpack_trenchid_index'})

        gene_table_layout = gene_list_dmap + trenchid_table + unpack_trenchid_table

        return gene_table_layout,select_gene,select_trenchid,select_unpacked_trenchid

    else:
        gene_table_layout = autocomplete + trenchid_table

        return gene_table_layout,select_gene,select_trenchid

def linked_kymograph_for_gene_table(xrstack,wrapped_xrstack,df,select_gene,select_trenchid,select_unpacked_trenchid=None,trenchid_column='trenchid',
                                    y_scale=3,x_window_size=300,rescale_videos=True,fps=10,dpi=200):
    ### stream must return trenchid value
    #### df must have trenchid lookups
    width,height = xrstack.shape[3],int(xrstack.shape[2]*y_scale)
    x_window_scale = x_window_size/xrstack.shape[3]
    x_size = int(xrstack.shape[3]*(y_scale*x_window_scale))

    df = df.reset_index(drop=False).set_index("Gene",drop=False).sort_index()
    df.index = df.index.rename("Gene Index")
    df_genes_only = df[["Gene"]].groupby("Gene").apply(lambda x: x.iloc[0])
    dataset = hv.Dataset(df)
    gene_dataset = hv.Dataset(df_genes_only)

    if select_unpacked_trenchid != None:
        def select_pt_load_image(channel,select_gene_index,trenchid_index,unpack_trenchid_index,width=width,height=height):
            gene_name = gene_dataset.iloc[select_gene_index]["Gene"][0]
            filtered_dataset = hv.Dataset(df.loc[gene_name:gene_name])
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            arr = xrstack.loc[channel,trenchid].compute(scheduler="threads").values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(select_gene_index,trenchid_index,unpack_trenchid_index):
            gene_name = gene_dataset.iloc[select_gene_index]["Gene"][0]
            filtered_dataset = hv.Dataset(df.loc[gene_name:gene_name])
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)

    else:
        def select_pt_load_image(channel,select_gene_index,trenchid_index,width=width,height=height):
            gene_name = gene_dataset.iloc[select_gene_index]["Gene"][0]
            filtered_dataset = hv.Dataset(df.loc[gene_name:gene_name])
            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            arr = xrstack.loc[channel,trenchid].compute(scheduler="threads").values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(select_gene_index,trenchid_index):
            gene_name = gene_dataset.iloc[select_gene_index]["Gene"][0]
            filtered_dataset = hv.Dataset(df.loc[gene_name:gene_name])
            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)

    def set_bounds(fig, element, y_dim=height, x_dim=width, x_window_size=x_window_size):
        sy = y_dim-0.5
        sx = x_dim-0.5

        x_zoom_pos = (fig.state.x_range.start,fig.state.x_range.end)
        xdata = element.dataset.data['x']-0.5
        xdata_range = (xdata.min(), xdata.max()+1.)
        init_test = x_zoom_pos==xdata_range

        fig.state.x_range.bounds = (0, sx)
        fig.state.y_range.bounds = (-0.5, sy)

        if init_test:
            fig.state.x_range.start = 0
            fig.state.x_range.end = x_window_size
        else:
            fig.state.x_range.start = x_zoom_pos[0]
            fig.state.x_range.end = x_zoom_pos[0]+x_window_size

        fig.state.y_range.start = 0
        fig.state.y_range.end = y_dim
        fig.state.x_range.reset_start = 0
        fig.state.x_range.reset_end = x_window_size
        fig.state.y_range.reset_start = 0
        fig.state.y_range.reset_end = y_dim

    if select_unpacked_trenchid != None:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_gene,select_trenchid,select_unpacked_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_gene,select_trenchid,select_unpacked_trenchid])
    else:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_gene,select_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_gene,select_trenchid])

    kymograph_display = image_stack.opts(plot={'Image': dict(colorbar=True, tools=['hover'],hooks=[set_bounds],aspect='equal'),})
    kymograph_display = kymograph_display.opts(cmap='Greys_r',height=height,width=x_size)

    kymograph_display = kymograph_display.redim.range(trenchid=(0,xrstack.shape[1]))
    kymograph_display = kymograph_display.redim.values(Channel=xrstack.coords["Channel"].values.tolist())

    if select_unpacked_trenchid != None:

        def save_video(event):
            hv.extension('matplotlib')
            channel = kymograph_display.dimension_values("Channel")[0]
            gene_name = gene_dataset.iloc[select_gene.index[0]]["Gene"][0]
            filtered_dataset = hv.Dataset(df.loc[gene_name:gene_name])
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[select_trenchid.index[0]][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[select_unpacked_trenchid.index[0]]["trenchid"][0]
            arr = wrapped_xrstack.loc[channel,trenchid].values
            if rescale_videos:
                arr = np.array([sk.exposure.rescale_intensity(arr[t]) for t in range(arr.shape[0])])
            hmap = hv.HoloMap([(t, hv.Raster(arr[t])) for t in range(0,arr.shape[0])], kdims=['t'])
            hmap = hmap.opts(plot={'Raster': dict(tools=['hover'],aspect='equal',cmap='Greys_r'),})
            hmap = hmap.opts(cmap='Greys_r')
            hv.save(hmap, gene_name + '_trenchid=' + str(trenchid) + '.mp4', fmt='mp4', fps=fps, dpi=dpi , backend='matplotlib')
            hv.extension('bokeh')

    else:

        def save_video(event):
            hv.extension('matplotlib')
            channel = kymograph_display.dimension_values("Channel")[0]
            gene_name = gene_dataset.iloc[select_gene.index[0]]["Gene"][0]
            filtered_dataset = hv.Dataset(df.loc[gene_name:gene_name])
            trenchid = filtered_dataset.iloc[select_trenchid.index[0]][trenchid_column][0]
            arr = wrapped_xrstack.loc[channel,trenchid].values
            if rescale_videos:
                arr = np.array([sk.exposure.rescale_intensity(arr[t]) for t in range(arr.shape[0])])
            hmap = hv.HoloMap([(t, hv.Raster(arr[t])) for t in range(0,arr.shape[0])], kdims=['t'])
            hmap = hmap.opts(plot={'Raster': dict(tools=['hover'],aspect='equal',cmap='Greys_r'),})
            hmap = hmap.opts(cmap='Greys_r')
            hv.save(hmap, gene_name + '_trenchid=' + str(trenchid) + '.mp4', fmt='mp4', fps=fps, dpi=dpi , backend='matplotlib')
            hv.extension('bokeh')

    trenchid_display = trenchid_display.opts(text_align="left",text_color="white")

    output_display = kymograph_display*trenchid_display

    button = pn.widgets.button.Button(name='Save Video', width=60)
    button.on_click(save_video)

    return output_display, button

def linked_table(df,index_key="Gene",trenchids_as_list=False,trenchid_column='trenchid',height=400,**histkwargs):
    df = df.reset_index(drop=False).set_index(index_key,drop=False).sort_index()
    df.index = df.index.rename(index_key + " Index")
    df_index_only = df[[index_key]].groupby(index_key).apply(lambda x: x.iloc[0])
    dataset = hv.Dataset(df)
    index_dataset = hv.Dataset(df_index_only)

    def get_index_list(index_dataset=index_dataset):
        return hv.Table(index_dataset)

    index_list_dmap = hv.DynamicMap(get_index_list)
    select_index = hv.streams.Selection1D(source=index_list_dmap,index=[0],rename={'index': 'select_index'})

    def get_index_table(select_index,df=df,index_key=index_key):
        index_name = index_dataset.iloc[select_index][index_key][0]
        dataset = hv.Dataset(df.loc[index_name:index_name])
        return hv.Table(dataset)

    trenchid_table = hv.DynamicMap(get_index_table, streams=[select_index])
    select_trenchid = hv.streams.Selection1D(source=trenchid_table,index=[0],rename={'index': 'trenchid_index'})

    if trenchids_as_list:
        def unpack_trenchids(select_index,trenchid_index,dataset=dataset,index_key=index_key):
            index_name = index_dataset.iloc[select_index][index_key][0]
            dataset = hv.Dataset(df.loc[index_name:index_name])
            filtered_dataset = hv.Dataset({"trenchid": dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            return hv.Table(filtered_dataset)

        unpack_trenchid_table = hv.DynamicMap(unpack_trenchids, streams=[select_index,select_trenchid], responsive=True)
        unpack_trenchid_table = unpack_trenchid_table.opts(height=height)
        select_unpacked_trenchid = hv.streams.Selection1D(source=unpack_trenchid_table,index=[0],rename={'index': 'unpack_trenchid_index'})

        index_table_layout = index_list_dmap + trenchid_table + unpack_trenchid_table

        return index_table_layout,select_index,select_trenchid,select_unpacked_trenchid

    else:
        index_table_layout = index_list_dmap + trenchid_table

        return index_table_layout,select_index,select_trenchid

def linked_kymograph_for_table(xrstack,wrapped_xrstack,df,select_index,select_trenchid,index_key="Gene",select_unpacked_trenchid=None,trenchid_column='trenchid',
                                    y_scale=3,x_window_size=300,rescale_videos=True,fps=10,dpi=200):
    ### stream must return trenchid value
    #### df must have trenchid lookups
    width,height = xrstack.shape[3],int(xrstack.shape[2]*y_scale)
    x_window_scale = x_window_size/xrstack.shape[3]
    x_size = int(xrstack.shape[3]*(y_scale*x_window_scale))

    df = df.reset_index(drop=False).set_index(index_key,drop=False).sort_index()
    df.index = df.index.rename(index_key + " Index")
    df_index_only = df[[index_key]].groupby(index_key).apply(lambda x: x.iloc[0])
    dataset = hv.Dataset(df)
    index_dataset = hv.Dataset(df_index_only)

    if select_unpacked_trenchid != None:
        def select_pt_load_image(channel,select_index,trenchid_index,unpack_trenchid_index,width=width,height=height,index_key=index_key):
            index_name = index_dataset.iloc[select_index][index_key][0]
            filtered_dataset = hv.Dataset(df.loc[index_name:index_name])
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            arr = xrstack.loc[channel,trenchid].values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(select_index,trenchid_index,unpack_trenchid_index,index_key=index_key):
            index_name = index_dataset.iloc[select_index][index_key][0]
            filtered_dataset = hv.Dataset(df.loc[index_name:index_name])
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[trenchid_index][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[unpack_trenchid_index]["trenchid"][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)

    else:
        def select_pt_load_image(channel,select_index,trenchid_index,width=width,height=height,index_key=index_key):
            index_name = index_dataset.iloc[select_index][index_key][0]
            filtered_dataset = hv.Dataset(df.loc[index_name:index_name])
            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            arr = xrstack.loc[channel,trenchid].values
            return hv.Image(arr,bounds=(0,0,width,height))

        def print_trenchid(select_index,trenchid_index,index_key=index_key):
            index_name = index_dataset.iloc[select_index][index_key][0]
            filtered_dataset = hv.Dataset(df.loc[index_name:index_name])
            trenchid = filtered_dataset.iloc[trenchid_index][trenchid_column][0]
            return hv.Text(3.,20.,str(trenchid),fontsize=30)

    def set_bounds(fig, element, y_dim=height, x_dim=width, x_window_size=x_window_size):
        sy = y_dim-0.5
        sx = x_dim-0.5

        fig.state.y_range.bounds = (-0.5, sy)
        fig.state.x_range.bounds = (0, sx)
        fig.state.x_range.start = 0
        fig.state.x_range.reset_start = 0
        fig.state.x_range.end = x_window_size
        fig.state.x_range.reset_end = x_window_size

        fig.state.y_range.start = 0
        fig.state.y_range.reset_start = 0
        fig.state.y_range.end = y_dim
        fig.state.y_range.reset_end = y_dim


    if select_unpacked_trenchid != None:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_index,select_trenchid,select_unpacked_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_index,select_trenchid,select_unpacked_trenchid])
    else:
        image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel'], streams=[select_index,select_trenchid])
        trenchid_display = hv.DynamicMap(print_trenchid, streams=[select_index,select_trenchid])

    kymograph_display = image_stack.opts(plot={'Image': dict(colorbar=True, tools=['hover'],hooks=[set_bounds],aspect='equal'),})
    kymograph_display = kymograph_display.opts(cmap='Greys_r',height=height,width=x_size)

    kymograph_display = kymograph_display.redim.range(trenchid=(0,xrstack.shape[1]))
    kymograph_display = kymograph_display.redim.values(Channel=xrstack.coords["Channel"].values.tolist())

    if select_unpacked_trenchid != None:

        def save_video(event,index_key=index_key):
            hv.extension('matplotlib')
            channel = kymograph_display.dimension_values("Channel")[0]
            index_name = index_dataset.iloc[select_index.index[0]][index_key][0]
            filtered_dataset = hv.Dataset(df.loc[index_name:index_name])
            double_filtered_dataset = hv.Dataset({"trenchid": filtered_dataset.iloc[select_trenchid.index[0]][trenchid_column][0]},['trenchid'])
            trenchid = double_filtered_dataset.iloc[select_unpacked_trenchid.index[0]]["trenchid"][0]
            arr = wrapped_xrstack.loc[channel,trenchid].values
            if rescale_videos:
                arr = np.array([sk.exposure.rescale_intensity(arr[t]) for t in range(arr.shape[0])])
            hmap = hv.HoloMap([(t, hv.Raster(arr[t])) for t in range(0,arr.shape[0])], kdims=['t'])
            hmap = hmap.opts(plot={'Raster': dict(tools=['hover'],aspect='equal',cmap='Greys_r'),})
            hmap = hmap.opts(cmap='Greys_r')
            hv.save(hmap, index_key + '_trenchid=' + str(trenchid) + '.mp4', fmt='mp4', fps=fps, dpi=dpi , backend='matplotlib')
            hv.extension('bokeh')

    else:

        def save_video(event,index_key=index_key):
            hv.extension('matplotlib')
            channel = kymograph_display.dimension_values("Channel")[0]
            index_name = index_dataset.iloc[select_index.index[0]][index_key][0]
            filtered_dataset = hv.Dataset(df.loc[index_name:index_name])
            trenchid = filtered_dataset.iloc[select_trenchid.index[0]][trenchid_column][0]
            arr = wrapped_xrstack.loc[channel,trenchid].values
            if rescale_videos:
                arr = np.array([sk.exposure.rescale_intensity(arr[t]) for t in range(arr.shape[0])])
            hmap = hv.HoloMap([(t, hv.Raster(arr[t])) for t in range(0,arr.shape[0])], kdims=['t'])
            hmap = hmap.opts(plot={'Raster': dict(tools=['hover'],aspect='equal',cmap='Greys_r'),})
            hmap = hmap.opts(cmap='Greys_r')
            hv.save(hmap, index_key + '_trenchid=' + str(trenchid) + '.mp4', fmt='mp4', fps=fps, dpi=dpi , backend='matplotlib')
            hv.extension('bokeh')

    trenchid_display = trenchid_display.opts(text_align="left",text_color="white")

    output_display = kymograph_display*trenchid_display

    button = pn.widgets.button.Button(name='Save Video', width=60)
    button.on_click(save_video)

    return output_display, button
    #
def interactive_scatter_for_estimators(estimator_df,x_dim_label,y_dim_label,estimator_name="Mean (Robust)",extra_columns=["oDEPool7_id","Gene","N Mismatch","N Observations"],minperc=0,maxperc=99,height=400,**scatterkwargs):

    estimate_df = estimator_df.loc[estimator_name]
    pivoted_df = pd.pivot(estimate_df.reset_index(),index=["oDEPool7_id"],columns=["Variable(s)"],values=["Value"])["Value"]
    pivoted_df = pivoted_df[[x_dim_label,y_dim_label]]
    extra_column_df = estimate_df.reset_index().groupby("oDEPool7_id").apply(lambda x: x.iloc[0])[extra_columns]
    pivoted_df = pivoted_df.join(extra_column_df)

    dataset = hv.Dataset(pivoted_df,vdims=[y_dim_label,x_dim_label],kdims=extra_columns)

    x_data_vals,y_data_vals = pivoted_df[x_dim_label],pivoted_df[y_dim_label]

    x_low,x_high = np.percentile(x_data_vals,minperc),np.percentile(x_data_vals,maxperc)
    y_low,y_high = np.percentile(y_data_vals,minperc),np.percentile(y_data_vals,maxperc)

    scatter = hv.Scatter(data=dataset,vdims=[y_dim_label]+extra_columns,kdims=[x_dim_label])
    scatter = scatter.opts(tools=['hover','lasso_select', 'box_select'],xlim=(x_low,x_high),ylim=(y_low,y_high),height=height,responsive=True,**scatterkwargs)
    return scatter

def unlinked_kymograph(xrstack,y_scale=3,x_window_size=300):
    #### stream must return trenchid value
    #### df must have trenchid lookups
    width,height = xrstack.shape[3],int(xrstack.shape[2]*y_scale)
    x_window_scale = x_window_size/xrstack.shape[3]
    x_size = int(xrstack.shape[3]*(y_scale*x_window_scale))

    def select_pt_load_image(channel,trenchid,width=width,height=height):
        arr = xrstack.loc[channel,trenchid].values
        return hv.Image(arr,bounds=(0,0,width,height))

    def print_trenchid(trenchid):
        return hv.Text(3.,20.,str(trenchid),fontsize=30)

    def set_bounds(fig, element, y_dim=height, x_dim=width, x_window_size=x_window_size):
        sy = y_dim-0.5
        sx = x_dim-0.5

        fig.state.y_range.bounds = (-0.5, sy)
        fig.state.x_range.bounds = (0, sx)
        fig.state.x_range.start = 0
        fig.state.x_range.reset_start = 0
        fig.state.x_range.end = x_window_size
        fig.state.x_range.reset_end = x_window_size

        fig.state.y_range.start = 0
        fig.state.y_range.reset_start = 0
        fig.state.y_range.end = y_dim
        fig.state.y_range.reset_end = y_dim


    image_stack = hv.DynamicMap(select_pt_load_image, kdims=['Channel','trenchid'])
    trenchid_display = hv.DynamicMap(print_trenchid, kdims=['trenchid'])

    kymograph_display = image_stack.opts(plot={'Image': dict(colorbar=True, tools=['hover'],hooks=[set_bounds],aspect='equal'),})
    kymograph_display = kymograph_display.opts(cmap='Greys_r',height=height,width=x_size)

    kymograph_display = kymograph_display.redim.range(trenchid=(0,xrstack.shape[1]))
    kymograph_display = kymograph_display.redim.values(Channel=xrstack.coords["Channel"].values.tolist(),trenchid=xrstack.coords["trenchid"].values.tolist())

    trenchid_display = trenchid_display.opts(text_align="left",text_color="white")

    output_display = kymograph_display*trenchid_display

    return output_display

def kymo_xarr(headpath,subset=None,in_memory=False,in_distributed_memory=False,unwrap=True):
    data_parquet = dd.read_parquet(headpath + "/kymograph/metadata",calculate_divisions=True)
    meta_handle = pandas_hdf5_handler(headpath+"/metadata.hdf5")
    metadata = meta_handle.read_df("global",read_metadata=True).metadata
    channels = metadata["channels"]
    file_indices = data_parquet["File Index"].unique().compute().to_list()

    delayed_fetch_hdf5 = delayed(fetch_hdf5)
    filenames = [headpath + '/kymograph/kymograph_'+str(file_idx)+'.hdf5' for file_idx in file_indices]

    sample = fetch_hdf5(filenames[0],channels[0])

    channel_arr = []
    for channel in channels:
        filenames = [headpath + '/kymograph/kymograph_'+str(file_idx)+'.hdf5' for file_idx in file_indices]
        delayed_arrays = [delayed_fetch_hdf5(fn,channel) for fn in filenames]
        da_file_arrays = [da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype) for delayed_reader in delayed_arrays]
        da_file_index_arr = da.concatenate(da_file_arrays, axis=0)
        channel_arr.append(da_file_index_arr)
    da_channel_arr = da.stack(channel_arr, axis=0)

    if unwrap:
        da_channel_arr = da_channel_arr.swapaxes(3,4).reshape(da_channel_arr.shape[0],da_channel_arr.shape[1],-1,da_channel_arr.shape[3]).swapaxes(2,3)
        if subset != None:
            da_channel_arr = da_channel_arr[:,subset]
        if in_memory:
            da_channel_arr = da_channel_arr.compute()
        elif in_distributed_memory:
            da_channel_arr = da_channel_arr.persist()

        # defining xarr
        dims = ['Channel','trenchid', 'y', 'xt']
        coords = {d: np.arange(s) for d, s in zip(dims, da_channel_arr.shape)}
        coords['Channel'] = np.array(channels)
        kymo_xarr = xr.DataArray(da_channel_arr, dims=dims, coords=coords, name="Data").astype("uint16")

        return kymo_xarr

    else:
        if subset != None:
            da_channel_arr = da_channel_arr[:,subset]
        if in_memory:
            da_channel_arr = da_channel_arr.compute()
        elif in_distributed_memory:
            da_channel_arr = da_channel_arr.persist()

        # defining xarr
        dims = ['Channel','trenchid','t','y','x']
        coords = {d: np.arange(s) for d, s in zip(dims, da_channel_arr.shape)}
        coords['Channel'] = np.array(channels)
        kymo_xarr = xr.DataArray(da_channel_arr, dims=dims, coords=coords, name="Data").astype("uint16")

        return kymo_xarr
