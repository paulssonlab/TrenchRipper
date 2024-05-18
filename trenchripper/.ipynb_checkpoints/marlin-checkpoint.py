# fmt: off
import itertools
import json
import h5py
import os
import copy
# import h5py_cache # not using anymore
import tifffile
import shutil
import dask
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as skl
import ipywidgets as ipyw
import dask.array as da
import dask.dataframe as dd
import scipy.signal
import sklearn.mixture

from matplotlib import pyplot as plt
from .utils import pandas_hdf5_handler, writedir
from .ndextract import get_registration_shifts,register_image_stack,apply_flatfield
from parse import compile


####

## This might be broken at the moment, waiting for an oppertunity to test it.
## Also does not do a good job of keeping metadata at the moment...
class marlin_extractor:
    def __init__(
        self,
        hdf5inputpath,
        headpath,
        register_images=False,
        reg_channel=None,
        pixel_microns=0.2125, ##hack assuming ti5 20x
        tpts_per_file=100,
        parsestr="fov={fov:d}_config={channel}_t={timepoints:d}.hdf5",
        metaparsestr="metadata_t={timepoint:d}.hdf5",
        zero_base_keys=["timepoints"],
    ):  # note this chunk size has a large role in downstream steps...make sure is less than 1 MB
        """Utility to import hdf5 format files from MARLIN Runs.

        Attributes:
            headpath (str): base directory for data analysis
            tiffpath (str): directory where tiff files are located
            metapath (str): metadata path
            hdf5path (str): where to store hdf5 data
            tpts_per_file (int): number of timepoints to put in each hdf5 file
            parsestr (str): format of filenames from which to extract metadata (using parse library)
        """
        self.hdf5inputpath = hdf5inputpath
        self.headpath = headpath
        self.metapath = self.headpath + "/metadata.hdf5"
        self.hdf5path = self.headpath + "/hdf5"
        self.tempregpath = self.headpath + "/registration_temp"
        self.tpts_per_file = tpts_per_file
        self.parsestr = parsestr
        self.metaparsestr = metaparsestr
        self.zero_base_keys = zero_base_keys
        self.register_images = register_images
        self.reg_channel = reg_channel

        self.pixel_microns = pixel_microns

        self.organism = ""
        self.microscope = ""
        self.notes = ""

        self.channel_to_flat_dict = {}

    def get_metadata(
        self,
        hdf5inputpath,
        parsestr="fov={fov:d}_config={channel}_t={timepoints:d}.hdf5",
        metaparsestr="metadata_t={timepoint:d}.hdf5",
        zero_base_keys=["timepoints"],
    ):
        parser = compile(parsestr)
        parse_keys = [
            item.split("}")[0].split(":")[0] for item in parsestr.split("{")[1:]
        ] + ["image_paths"]

        exp_metadata = {}
        fov_metadata = {key: [] for key in parse_keys}

        self.hdf5_files = []
        self.metadata_files = []
        for root, _, files in os.walk(hdf5inputpath):
            self.hdf5_files.extend(
                [
                    os.path.join(root, f)
                    for f in files
                    if "config" in os.path.splitext(f)[0]
                ]
            )
            self.metadata_files.extend(
                [
                    os.path.join(root, f)
                    for f in files
                    if "metadata" in os.path.splitext(f)[0]
                ]
            )

        with h5py.File(self.hdf5_files[0], "r") as infile:
            hdf5_shape = infile["data"].shape
        exp_metadata["height"] = hdf5_shape[0]
        exp_metadata["width"] = hdf5_shape[1]
        #     exp_metadata['pixel_microns'] = tags['65326']

        for f in self.hdf5_files:
            match = parser.search(f)
            # ignore any files that don't match the regex
            if match is not None:
                # Add to dictionary
                fov_frame_dict = match.named
                for key, value in fov_frame_dict.items():
                    fov_metadata[key].append(value)
                fov_metadata["image_paths"].append(f)

        for zero_base_key in zero_base_keys:
            if 0 not in fov_metadata[zero_base_key]:
                fov_metadata[zero_base_key] = [
                    item - 1 for item in fov_metadata[zero_base_key]
                ]

        channels = list(set(fov_metadata["channel"]))
        exp_metadata["channels"] = channels
        exp_metadata["num_fovs"] = len(set(fov_metadata["fov"]))
        exp_metadata["frames"] = list(set(fov_metadata["timepoints"]))
        exp_metadata["num_frames"] = len(exp_metadata["frames"])
        exp_metadata["pixel_microns"] = self.pixel_microns
        fov_metadata = pd.DataFrame(fov_metadata)
        fov_metadata = fov_metadata.set_index(["fov", "timepoints"]).sort_index()

        output_fov_metadata = []
        step = len(channels)
        for i in range(0, len(fov_metadata), step):
            rows = fov_metadata[i : i + step]
            channel_path_entry = dict(zip(rows["channel"], rows["image_paths"]))
            fov_entry = rows.index.get_level_values("fov").unique()[0]
            timepoint_entry = rows.index.get_level_values("timepoints").unique()[0]
            fov_metadata_entry = {
                "fov": fov_entry,
                "timepoints": timepoint_entry,
                "channel_paths": channel_path_entry,
            }
            output_fov_metadata.append(fov_metadata_entry)
        fov_metadata = pd.DataFrame(output_fov_metadata).set_index(
            ["fov", "timepoints"]
        ).sort_index()

        metaparser = compile(metaparsestr)
        meta_df_out = []
        for metadata_file in self.metadata_files:
            match = metaparser.search(metadata_file)
            if match is not None:
                timepoint = match.named["timepoint"]
                meta_df = pd.read_hdf(metadata_file)
                meta_df["timepoints"] = timepoint
                meta_df_out.append(meta_df)
        meta_df_out = pd.concat(meta_df_out)
        if 0 not in meta_df_out["timepoints"].unique().tolist():
            meta_df_out["timepoints"] = meta_df_out["timepoints"] - 1
        meta_df_out = meta_df_out.groupby(["fov", "timepoints"], as_index=False)
        meta_df_out = meta_df_out.apply(lambda x: x[0:1])
        meta_df_out = meta_df_out.set_index(["fov", "timepoints"], drop=True)
        fov_metadata = fov_metadata.join(meta_df_out)

        return exp_metadata, fov_metadata

    def assignidx(self, fov_metadata):
        numfovs = len(fov_metadata.index.get_level_values("fov").unique().tolist())
        timepoints_per_fov = len(
            fov_metadata.index.get_level_values("timepoints").unique().tolist()
        )

        files_per_fov = (timepoints_per_fov // self.tpts_per_file) + 1
        remainder = timepoints_per_fov % self.tpts_per_file
        ttlfiles = numfovs * files_per_fov
        fov_file_idx = np.repeat(list(range(files_per_fov)), self.tpts_per_file)[
            : -(self.tpts_per_file - remainder)
        ]
        file_idx = np.concatenate(
            [fov_file_idx + (fov_idx * files_per_fov) for fov_idx in range(numfovs)]
        )
        fov_img_idx = np.repeat(
            np.array(list(range(self.tpts_per_file)))[np.newaxis, :],
            files_per_fov,
            axis=0,
        )
        fov_img_idx = fov_img_idx.flatten()[: -(self.tpts_per_file - remainder)]
        img_idx = np.concatenate([fov_img_idx for fov_idx in range(numfovs)])

        fov_idx = np.repeat(list(range(numfovs)), timepoints_per_fov)
        timepoint_idx = np.repeat(
            np.array(list(range(timepoints_per_fov)))[np.newaxis, :], numfovs, axis=0
        ).flatten()

        outdf = copy.deepcopy(fov_metadata)
        outdf["File Index"] = file_idx
        outdf["Image Index"] = img_idx
        return outdf

    def writemetadata(self, t_range=None, fov_list=None):

        exp_metadata, fov_metadata = self.get_metadata(
            self.hdf5inputpath,
            parsestr=self.parsestr,
            metaparsestr=self.metaparsestr,
            zero_base_keys=self.zero_base_keys,
        )

        if t_range is not None:
            exp_metadata["frames"] = exp_metadata["frames"][t_range[0] : t_range[1] + 1]
            exp_metadata["num_frames"] = len(exp_metadata["frames"])
            fov_metadata = fov_metadata.loc[
                pd.IndexSlice[:, slice(t_range[0], t_range[1])], :
            ]  # 4 -> 70

        if fov_list is not None:
            fov_metadata = fov_metadata.loc[list(fov_list)]
            exp_metadata["fields_of_view"] = list(fov_list)

        self.chunk_shape = (1, exp_metadata["height"], exp_metadata["width"])
        chunk_bytes = 2 * np.multiply.accumulate(np.array(self.chunk_shape))[-1]
        self.chunk_cache_mem_size = 2 * chunk_bytes
        exp_metadata["chunk_shape"], exp_metadata["chunk_cache_mem_size"] = (
            self.chunk_shape,
            self.chunk_cache_mem_size,
        )
        exp_metadata["Organism"], exp_metadata["Microscope"], exp_metadata["Notes"] = (
            self.organism,
            self.microscope,
            self.notes,
        )
        self.meta_handle = pandas_hdf5_handler(self.metapath)

        assignment_metadata = self.assignidx(fov_metadata)
        assignment_metadata.astype({"File Index": int, "Image Index": int})

        self.meta_handle.write_df("global", assignment_metadata, metadata=exp_metadata)

    def read_metadata(self):
        writedir(self.hdf5path, overwrite=True)
        self.writemetadata()
        metadf = self.meta_handle.read_df("global", read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index", "Image Index"], drop=True, append=False, inplace=False)
        return metadf

    def set_params(self, fov_list, t_range, register_images, reg_channel, organism, microscope, notes):
        self.fov_list = fov_list
        self.t_range = t_range
        self.register_images = register_images
        self.reg_channel = reg_channel
        self.organism = organism
        self.microscope = microscope
        self.notes = notes

    def inter_set_params(self):
        metadf = self.read_metadata()
        t0, tf = (self.metadata["frames"][0], self.metadata["frames"][-1])
        available_fov_list = metadf["fov"].unique().tolist()
        channels_list = self.metadata["channels"]

        selection = ipyw.interactive(self.set_params,{"manual": True},fov_list=ipyw.SelectMultiple(options=available_fov_list),\
            t_range=ipyw.IntRangeSlider(value=[t0, tf],min=t0,max=tf,step=1,description="Time Range:",disabled=False),\
            register_images=ipyw.Dropdown(options=[True,False],value=False,description='Register Images?'),\
            reg_channel=ipyw.Dropdown(options=channels_list,value=channels_list[0],description='Registration Channel:'),\
            organism=ipyw.Textarea(value="",placeholder="Organism imaged in this experiment.",description="Organism:",disabled=False),\
            microscope=ipyw.Textarea(value="",placeholder="Microscope used in this experiment.",description="Microscope:",disabled=False),\
            notes=ipyw.Textarea(value="",placeholder="General experiment notes.",description="Notes:",disabled=False),\
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

    def extract(self, dask_controller, retries=1):
        dask_controller.futures = {}

        self.writemetadata(t_range=self.t_range, fov_list=self.fov_list)

        metadf = self.meta_handle.read_df("global", read_metadata=True)
        self.metadata = metadf.metadata
        metadf = metadf.reset_index(inplace=False)
        metadf = metadf.set_index(["File Index", "Image Index"], drop=True, append=False, inplace=False).sort_index()

        def writehdf5(fovnum, num_entries, timepoint_list, file_idx):

            #### open flatfield images
            flatfield_img_dict = {}
            for channel,path in self.channel_to_flat_dict.items():
                if path != "":
                    flatfield_img_dict[channel] = tifffile.imread(path)

            metadf = self.meta_handle.read_df("global", read_metadata=True)
            metadf = metadf.reset_index(inplace=False)
            metadf = metadf.set_index(["File Index", "Image Index"], drop=True, append=False, inplace=False).sort_index()

            y_dim = self.metadata["height"]
            x_dim = self.metadata["width"]
            filedf = metadf.loc[file_idx].reset_index(inplace=False)
            filedf = filedf.set_index(["timepoints"], drop=True, append=False, inplace=False)
            filedf = filedf.sort_index()

            with h5py.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",rdcc_nbytes=self.chunk_cache_mem_size) as h5pyfile:
            # with h5py_cache.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5","w",chunk_cache_mem_size=self.chunk_cache_mem_size) as h5pyfile:
                for i, channel in enumerate(self.metadata["channels"]):
                    hdf5_dataset = h5pyfile.create_dataset(str(channel),(num_entries, y_dim, x_dim),chunks=self.chunk_shape,dtype="uint16")

                    if self.channel_to_flat_dict[channel] != '': ##flatfielding channels
                        for j in range(len(timepoint_list)):
                            frame = timepoint_list[j]
                            entry = filedf.loc[frame]["channel_paths"]
                            file_path = entry[channel]
                            with h5py.File(file_path, "r") as infile:
                            # with h5py_cache.File(file_path, "r") as infile:
                                img = infile["data"][:]
                            img = apply_flatfield(img,flatfield_img_dict[channel],flatfield_img_dict["Dark_Image"])
                            hdf5_dataset[j, :, :] = img
                    else:
                        for j in range(len(timepoint_list)):
                            frame = timepoint_list[j]
                            entry = filedf.loc[frame]["channel_paths"]
                            file_path = entry[channel]
                            with h5py.File(file_path, "r") as infile:
                            # with h5py_cache.File(file_path, "r") as infile:
                                img = infile["data"][:]
                            hdf5_dataset[j, :, :] = img
            return "Done."

        file_list = metadf.index.get_level_values("File Index").unique().values
        num_jobs = len(file_list)
        random_priorities = np.random.uniform(size=(num_jobs,))

        for k, file_idx in enumerate(file_list):
            priority = random_priorities[k]
            filedf = metadf.loc[file_idx]

            fovnum = filedf[0:1]["fov"].values[0]
            num_entries = len(filedf.index.get_level_values("Image Index").values)
            timepoint_list = filedf["timepoints"].tolist()

            future = dask_controller.daskclient.submit(
                writehdf5,
                fovnum,
                num_entries,
                timepoint_list,
                file_idx,
                retries=retries,
                priority=priority,
            )
            dask_controller.futures["extract file: " + str(file_idx)] = future

        extracted_futures = [
            dask_controller.futures["extract file: " + str(file_idx)]
            for file_idx in file_list
        ]
        pause_for_extract = dask_controller.daskclient.gather(
            extracted_futures, errors="skip"
        )

        futures_name_list = ["extract file: " + str(file_idx) for file_idx in file_list]
        failed_files = [
            futures_name_list[k]
            for k, item in enumerate(extracted_futures)
            if item.status != "finished"
        ]
        failed_file_idx = [int(item.split(":")[1]) for item in failed_files]
        outdf = self.meta_handle.read_df("global", read_metadata=False)

        tempmeta = outdf.reset_index(inplace=False)
        tempmeta = tempmeta.set_index(
            ["File Index", "Image Index"], drop=True, append=False, inplace=False
        )
        failed_fovs = tempmeta.loc[failed_file_idx]["fov"].unique().tolist()

        outdf = outdf.drop(failed_fovs)

        if self.t_range != None:
            outdf = outdf.reset_index(inplace=False)
            outdf["timepoints"] = outdf["timepoints"] - self.t_range[0]
            outdf = outdf.set_index(
                ["fov", "timepoints"], drop=True, append=False, inplace=False
            )

        self.meta_handle.write_df("global", outdf, metadata=self.metadata)

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

                with h5py.File(self.tempregpath + "/hdf5_" + str(file_idx) + ".hdf5", "w") as outfile:

                    for channel in channels:
                        img_stack = []
                        for file_idx in file_idx_list:
                            with h5py.File(self.hdf5path + "/hdf5_" + str(file_idx) + ".hdf5", "r") as infile:
                                img_stack.append(infile[channel][:])
                        stack_borders = np.add.accumulate([0] + [item.shape[0] for item in img_stack])
                        img_stack = np.concatenate(img_stack)
                        img_stack = register_image_stack(img_stack,cumulative_shift_coords)
                        for idx,file_idx in enumerate(file_idx_list):
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


def remove_bits_from_array(array,removed_bit_cycles=[]):
    out_array = copy.copy(array)
    out_array[removed_bit_cycles] = 0.
    return out_array

def get_signal_sum(df, barcode_len=30, removed_bit_cycles=[], \
    channel_list=["RFP 98th Percentile","Cy5 98th Percentile","Cy7 98th Percentile"],epsilon=0.1):
    trench_group = df.groupby(["trenchid"])
    barcodes = trench_group.apply(lambda x: np.array([x[channel].tolist() for channel in channel_list]).flatten().astype(float))

    short = barcodes.apply(lambda x: len(x) != barcode_len)
    for idx in np.where(short)[0]:
        barcodes[idx] = np.array([0. for i in range(barcode_len)])

    barcodes_arr = np.array(barcodes.to_list())
    barcodes_arr[:,removed_bit_cycles] = 0.
    barcodes_arr_no_short = np.array(barcodes[~short].to_list())

    barcodes_median = np.median(barcodes_arr_no_short,axis=0)+epsilon

    barcodes_arr = barcodes_arr/barcodes_median[np.newaxis,:]
    barcodes_arr_no_short = barcodes_arr_no_short/barcodes_median[np.newaxis,:]

    signal_sum = np.sum(barcodes_arr, axis=1)
    signal_sum_no_short = np.sum(barcodes_arr_no_short, axis=1)

    signal_filter_thr = get_background_thr(signal_sum_no_short, get_background_dist_peak)

    return signal_sum, signal_filter_thr, barcodes_median, barcodes

def get_background_thr(values, background_fn, background_scaling=10.0):
    mu_n, std_n = background_fn(values)
    back_thr = mu_n + background_scaling * std_n
    return back_thr

def get_background_dist_peak(values):
    hist_range = (0, np.percentile(values, 90))
    hist_count, hist_vals = np.histogram(values, bins=100, range=hist_range)
    peaks = sp.signal.find_peaks(hist_count, distance=20)[0]
    mu_n = hist_vals[peaks[0]]
    lower_tail = values[values < mu_n]
    std_n = sp.stats.halfnorm.fit(-lower_tail)[1]
    return mu_n, std_n

def get_gmm_hard_assign(values):
    gmm = skl.mixture.GaussianMixture(n_components=2, n_init=10)
    gmm.fit(values.reshape(-1, 1))
    lower_mean_idx = np.argmin(gmm.means_)
    assign = gmm.predict(values.reshape(-1, 1))
    if lower_mean_idx == 1:
        assign = (-assign) + 1
    return assign

def barcode_to_FISH(barcodestr,cycleorder=[0,1,6,7,12,13,18,19,24,25,2,3,8,9,14,15,20,21,26,27,4,5,10,11,16,17,22,23,28,29]):

    barcode = [bool(int(item)) for item in list(barcodestr)]
    FISH_barcode = np.array([barcode[i] for i in cycleorder])
    FISH_barcode = "".join(FISH_barcode.astype(int).astype(str))

    return FISH_barcode


def remove_bits_from_barcode(barcode,removed_bits=[]):
    # Will convert removed bits to 0 values

    barcode = np.array(list(barcode))
    barcode[removed_bits] = 0
    barcode = "".join(barcode.tolist())

    return barcode


def remove_bits(df,removed_bits):
    # Will convert removed bits to 0 values

    in_df = copy.deepcopy(df)

    subsampled_barcode = in_df["barcode"].apply(remove_bits_from_barcode,removed_bits=removed_bits)
    in_df["subsampled_barcode"] = subsampled_barcode
    unique_barcodes, n_occurances = np.unique(subsampled_barcode,return_counts=True)
    redundant_barcodes = unique_barcodes[n_occurances>1]
    redundant_indices = subsampled_barcode[subsampled_barcode.isin(redundant_barcodes.tolist())].index

    in_df = in_df.drop(redundant_indices).reset_index(drop=True).drop(['barcodeid','barcode'], axis=1)
    in_df = in_df.rename({"subsampled_barcode": "barcode"},axis=1)
    in_df['barcodeid'] = in_df.index

    return in_df

class fish_analysis():
    def __init__(self,headpath,nanoporedfpath,subsample=5000,barcode_len=30,remove_bit_list=[],hamming_thr=0,\
                 channel_names=["RFP 98th Percentile","Cy5 98th Percentile","Cy7 98th Percentile"],\
                cycleorder=[0,1,6,7,12,13,18,19,24,25,2,3,8,9,14,15,20,21,26,27,4,5,10,11,16,17,22,23,28,29]):
        ## the index in cycleorder is a flat representation of a channel x cycle array s.t. each block of N entries is a single color channel
        ## the number in cycleorder specifies the bit index on the encoded RNA molecule
        ## remove_bit_list specifies a list of bit indices on the encoded RNA molecule to remove
        ## trying to fix this, I think it is broken
        
        self.headpath = headpath
        self.nanoporedfpath = nanoporedfpath
        self.subsample = subsample
        self.barcode_len = barcode_len
        self.hamming_thr = hamming_thr
        self.channel_names = channel_names
        self.cycleorder = cycleorder
        self.percentilepath = headpath + "/percentiles"

        self.removed_bits = remove_bit_list
        self.removed_bit_cycles = [cycleorder.index(bit) for bit in self.removed_bits]

        kymograph_metadata = dd.read_parquet(self.percentilepath,calculate_divisions=True)
        kymograph_metadata = kymograph_metadata.set_index("trenchid", drop=True, sorted=True)

        self.trenchids = list(kymograph_metadata.index.compute().get_level_values(0).unique())
        if (self.subsample != None) and (self.subsample<len(self.trenchids)):
            selected_trenchids = sorted(list(np.random.choice(self.trenchids,(self.subsample,),replace=False)))
            kymograph_metadata = kymograph_metadata.loc[selected_trenchids]

        self.kymograph_metadata_subsample = kymograph_metadata.compute().reset_index().set_index(["trenchid","timepoints"],drop=True)
        self.signal_sum, self.signal_filter_thr, self.barcodes_median, self.barcodes = get_signal_sum(self.kymograph_metadata_subsample,\
                                                                barcode_len=self.barcode_len,removed_bit_cycles=self.removed_bit_cycles,\
                                                                channel_list=self.channel_names)

    def plot_signal_threshold(self,signal_filter_thr,figsize=(12,8)):
        self.signal_filter_thr = signal_filter_thr
        high_signal_mask = self.signal_sum>self.signal_filter_thr
        high_signal_barcodes_series = self.barcodes[high_signal_mask]
        self.high_signal_barcodes = np.array([item for item in high_signal_barcodes_series])

        max_v = np.percentile(self.signal_sum,99)

        fig = plt.figure(figsize=figsize)
        plt.hist(self.signal_sum[high_signal_mask], bins=100, range=(0, max_v),color="steelblue")
        plt.hist(self.signal_sum[~high_signal_mask], bins=100, range=(0, max_v),color="grey")
        plt.title("Sum of all barcode signal",fontsize=20)
        plt.xlabel("Summed Intensity",fontsize=20)
        plt.ylabel("Number of Trenches",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig("./Barcode_Signal_Sum.png",dpi=150)
        plt.show()

    def plot_signal_threshold_inter(self):
        signal_list_int = ipyw.interactive(
            self.plot_signal_threshold,
            {"manual": True},
            signal_filter_thr=ipyw.FloatSlider(
                value=self.signal_filter_thr,
                description="Signal Threshold:",
                min=0.,
                max=100.,
                step=0.5,
                disabled=False,
            ),
            figsize=ipyw.fixed((12,8)
            ),
        )

        display(signal_list_int)

    def get_bit_thresholds(self):
        bit_threshold_list = []
        omit_bit_list = []
        for i in range(self.barcode_len):
            values = self.high_signal_barcodes[:, i]
            assign = get_gmm_hard_assign(values).astype(bool)
            threshold = (np.min(values[assign])+np.max(values[~assign]))/2
            bit_threshold_list.append(threshold)
            omit_bit_list.append(False)
        self.bit_threshold_list = bit_threshold_list

    def plot_bit_threshold(self,idx,bit_filter_thr,figsize=(12,8)):
        self.bit_threshold_list[idx] = bit_filter_thr

        high_signal_mask = self.high_signal_barcodes[:, idx]>self.bit_threshold_list[idx]
        on_arr = self.high_signal_barcodes[:, idx][high_signal_mask]
        off_arr = self.high_signal_barcodes[:, idx][~high_signal_mask]

        max_val = np.percentile(self.high_signal_barcodes[:, idx].flatten(), 99)
        bins = np.linspace(0, max_val, num=50)
        on_frq, on_edges = np.histogram(on_arr, bins)
        off_frq, off_edges = np.histogram(off_arr, bins)

        plt.bar(off_edges[:-1], off_frq, width=np.diff(off_edges), align="edge", color="steelblue")
        plt.bar(on_edges[:-1], on_frq, width=np.diff(on_edges), align="edge", color="grey")
        plt.title("Barcode Signal",fontsize=20)
        plt.xlabel("Intensity",fontsize=20)
        plt.ylabel("Number of Trenches",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    def plot_bit_threshold_inter(self):
        timepoints = self.kymograph_metadata_subsample.index.get_level_values("timepoints").unique().tolist()

        idx = 0
        channel_children = []
        for j,channel_name in enumerate(self.channel_names):

            tpt_children = []
            for i,tpt in enumerate(timepoints):
                tpt_children.append(ipyw.interactive(self.plot_bit_threshold,idx=ipyw.fixed(idx),bit_filter_thr=ipyw.IntText(
                value=self.bit_threshold_list[idx],
                description="Bit " + str(idx),
                min=0,
                max=10000000,
                disabled=False,),
                figsize=ipyw.fixed((12,8))))
                idx+=1
            tpt_tab = ipyw.Tab()
            tpt_tab.children = tpt_children
            for i in range(len(tpt_children)):
                tpt_tab.set_title(i, "Cycle " + str(i))

            channel_children.append(tpt_tab)

        channel_tab = ipyw.Tab()
        channel_tab.children = channel_children
        for i in range(len(channel_children)):
            channel_tab.set_title(i, self.channel_names[i])

        display(channel_tab)

    def export_bit_thresholds(self,figsize=(24, 10),colors = ["salmon", "violet", "firebrick"], ylabel="Number of Trenches", xlabel="98th Percentile Intensity", small_fontsize=16, large_fontsize=24):

        timepoints = self.kymograph_metadata_subsample.index.get_level_values("timepoints").unique().tolist()
        n_channels,n_timepoints = len(self.channel_names),len(timepoints)

        fig, axes = plt.subplots(n_channels, n_timepoints, figsize=figsize)

        for idx in range(n_channels*n_timepoints):
            row_idx = idx // n_timepoints
            column_idx = idx % n_timepoints
            color = colors[row_idx]

            max_val = np.percentile(self.high_signal_barcodes[:, idx].flatten(), 99)

            bins = np.linspace(0, max_val, num=50)
            high_signal_mask = self.high_signal_barcodes[:, idx]>self.bit_threshold_list[idx]
            on_arr = self.high_signal_barcodes[:, idx][high_signal_mask]
            off_arr = self.high_signal_barcodes[:, idx][~high_signal_mask]

            on_frq, on_edges = np.histogram(on_arr, bins)
            off_frq, off_edges = np.histogram(off_arr, bins)
#             ttl_frq = np.sum(on_frq) + np.sum(off_frq)
#             on_frq, off_frq = (on_frq / ttl_frq, off_frq / ttl_frq)

            ax = axes[row_idx, column_idx]

            ax.bar(off_edges[:-1], off_frq, width=np.diff(off_edges), align="edge", color="grey")
            ax.bar(on_edges[:-1], on_frq, width=np.diff(on_edges), align="edge", color=color)
            ax.tick_params(axis='x', labelsize=small_fontsize)
            ax.tick_params(axis='y', labelsize=small_fontsize)

        plt.title("Barcode Signal",fontsize=large_fontsize)
        fig.text(0.5, -0.04, xlabel, ha="center", size=large_fontsize)
        fig.text(-0.01, 0.5, ylabel, va="center", rotation="vertical", size=large_fontsize)

        plt.tight_layout()
        plt.savefig("./Bit_Thresholds.png",dpi=300,bbox_inches="tight")
        plt.show()

    def get_barcode_df(self,epsilon=0.1):
        print("Getting Barcode df...")
        self.kymograph_metadata = dd.read_parquet(self.percentilepath,calculate_divisions=True)
        trench_group = self.kymograph_metadata.groupby(["trenchid"])
        barcodes = trench_group.apply(lambda x: list(itertools.chain.from_iterable([x[channel].tolist()\
                                    for channel in self.channel_names]))).compute()
        barcodes = barcodes.apply(lambda x: np.array(eval(x)))
        short = barcodes.apply(lambda x: len(x) != self.barcode_len)
        barcodes[short] = [np.array([0. for i in range(self.barcode_len)]) for i in range(len(short))]

        barcodes = barcodes.apply(remove_bits_from_array,removed_bit_cycles=self.removed_bit_cycles)
        signal_sum = barcodes.apply(lambda x: np.sum(x/self.barcodes_median))

        high_signal_mask = signal_sum>self.signal_filter_thr
        high_signal_barcodes_series = barcodes[high_signal_mask]

        assign_arr = np.array(high_signal_barcodes_series.apply(lambda x: x>np.array(self.bit_threshold_list)).to_list()).T
        assign_arr[self.removed_bit_cycles] = False

        assign_strs = np.apply_along_axis(lambda x: "".join(x.astype(int).astype(str)),0,assign_arr)
        barcode_df = pd.DataFrame(high_signal_barcodes_series, columns=["Barcode Signal"])
        barcode_df["Barcode"] = assign_strs
        self.barcode_df = barcode_df.reset_index(drop=False)

    def get_nanopore_df(self):
        print("Getting Nanopore df...")
        nanopore_df = pd.read_csv(self.nanoporedfpath,delimiter="\t",index_col=0)
        nanopore_df = remove_bits(nanopore_df,self.removed_bits)

        nanopore_lookup = {}
        for _,row in nanopore_df.iterrows():
            nanopore_lookup[barcode_to_FISH(row["barcode"],cycleorder=self.cycleorder)] = row
        del nanopore_df
        self.nanopore_lookup_df = pd.DataFrame(nanopore_lookup).T
        del nanopore_lookup

    def get_merged_df(self,dask_cont):
        print("Merging...")
        mergeddf = []

        if self.hamming_thr == 0:
            for _,row in self.barcode_df.iterrows():
                try:
                    nanopore_row = self.nanopore_lookup_df.loc[row["Barcode"]]
                    entry = pd.concat([row,nanopore_row])
                    mergeddf.append(entry)
                except:
                    pass

        else:
            nanopore_idx = np.array([np.array(list(item)).astype(bool) for item in self.nanopore_lookup_df.index.tolist()]).astype(bool)
            nanopore_idx = da.from_array(nanopore_idx,chunks=(10000,30))
            queries = np.array([np.array(list(item)).astype(bool) for item in self.barcode_df["Barcode"].tolist()]).astype(bool)
            queries = da.from_array(queries,chunks=(10000,30))
            match_1 = (nanopore_idx.astype("uint8")@queries.T.astype("uint8"))
            match_0 = ((~nanopore_idx).astype("uint8")@(~queries).T.astype("uint8"))

            match_1.to_zarr(self.headpath + '/match_1.zarr',overwrite=True)
            match_0.to_zarr(self.headpath + '/match_0.zarr',overwrite=True)

            match_1 = da.from_zarr(self.headpath + '/match_1.zarr',inline_array=True)
            match_0 = da.from_zarr(self.headpath + '/match_0.zarr',inline_array=True)
            match = match_1 + match_0
            hamming_dist = self.barcode_len-match
            hamming_dist.to_zarr(self.headpath + '/hamming_dist.zarr',overwrite=True,inline_array=True)
            shutil.rmtree(self.headpath + '/match_1.zarr')
            shutil.rmtree(self.headpath + '/match_0.zarr')
            dask_cont.daskclient.cancel([match_1,match_0])

            hamming_dist = da.from_zarr(self.headpath + '/hamming_dist.zarr')
            closest_match_thr = (da.min(hamming_dist,axis=0)<=self.hamming_thr).compute()
            closest_match = da.argmin(hamming_dist,axis=0).compute()

            dask_cont.daskclient.cancel([hamming_dist])
            shutil.rmtree(self.headpath + '/hamming_dist.zarr')

            closest_match[~closest_match_thr] = -1
            filtered_lookup = {query_idx:target_idx for query_idx,target_idx in enumerate(closest_match) \
                               if target_idx != -1}
            mergeddf = []
            for i,row in self.barcode_df.iterrows():
                try:
                    nanopore_row = self.nanopore_lookup_df.iloc[filtered_lookup[i]]
                    entry = pd.concat([row,nanopore_row])
                    mergeddf.append(entry)
                except:
                    pass

        del self.nanopore_lookup_df
        self.mergeddf = pd.DataFrame(mergeddf)
        del mergeddf


    def output_barcode_df(self,dask_cont,fishanalysispath):
        self.get_barcode_df()
        self.get_nanopore_df()
        self.get_merged_df(dask_cont)

        ttl_trenches = len(self.kymograph_metadata["trenchid"].unique().compute())
        ttl_trenches_w_cells = len(self.barcode_df)

        output_handle = pandas_hdf5_handler(fishanalysispath)
        output_handle.write_df("barcodes",self.mergeddf,metadata=\
        {"Total Trenches":ttl_trenches,"Total Trenches With Cells":ttl_trenches_w_cells,\
         "Removed Bits":self.removed_bits,"Removed Bit Cycles":self.removed_bit_cycles,\
        "Cycle Order":self.cycleorder,"Barcode Length":self.barcode_len,"Hamming Threshold": self.hamming_thr,\
        "Channel Names":self.channel_names,"Summed Intensity Threshold":self.signal_filter_thr,"Bit Threshold List":self.bit_threshold_list})

        del self.mergeddf

# def get_trenchid_map(kymodf1,kymodf2):
#     trenchid_map = {}

#     fovset1 = set(kymodf1["fov"].unique().tolist())
#     fovset2 = set(kymodf2["fov"].unique().tolist())
#     fov_intersection = fovset1.intersection(fovset2)

#     for fov in fov_intersection:
#         df1_chunk = kymodf1[kymodf1["fov"] == fov]
#         df2_chunk = kymodf2[kymodf2["fov"] == fov]

#         df1_xy = df1_chunk[["y (local)","x (local)"]].values
#         df2_xy = df2_chunk[["y (local)","x (local)"]].values

#         ymat = np.subtract.outer(df1_xy[:,0],df2_xy[:,0])
#         xmat = np.subtract.outer(df1_xy[:,1],df2_xy[:,1])
#         distmat = (ymat**2+xmat**2)**(1/2)

#         #ensuring map is one-to-one
#         mapping = np.argmin(distmat,axis=1)
#         invmapping = np.argmin(distmat,axis=0)
#         mapping = {idx:map_idx for idx,map_idx in enumerate(mapping) if invmapping[map_idx]==idx}

#         df1_trenchids = df1_chunk["trenchid"].tolist()
#         df2_trenchids = df2_chunk["trenchid"].tolist()

#         trenchid_map.update({trenchid: df2_trenchids[mapping[i]] for i,trenchid in enumerate(df1_trenchids)\
#                             if i in mapping.keys()})

#     return trenchid_map

def map_Series(x,series,trenchid_map,dtype=str):
    if x["trenchid"] in trenchid_map.keys():
        if trenchid_map[x["trenchid"]] in series.index:
            return series.loc[trenchid_map[x["trenchid"]]]
        else:
            return "Unknown"
    else:
        return "Unknown"

def get_barcode_pheno_df(phenotype_df, barcode_df, trenchid_map, output_index="File Parquet Index"):
    ##phenotype_df must contain trenchids column and a File Parquet Index

    valid_barcode_df = barcode_df[barcode_df["trenchid"].isin(trenchid_map.keys())].compute()
    barcode_df_mapped_trenchids = valid_barcode_df["trenchid"].apply(lambda x: trenchid_map[x])
    phenotype_df_idx = phenotype_df["trenchid"].unique().compute().tolist()

    valid_init_df_indices = barcode_df_mapped_trenchids.isin(phenotype_df_idx)
    barcode_df_mapped_trenchids = barcode_df_mapped_trenchids[valid_init_df_indices]
    barcode_df_mapped_trenchids_list = barcode_df_mapped_trenchids.tolist()
    final_valid_barcode_df_indices = barcode_df_mapped_trenchids.index.to_list()

    called_df = barcode_df.loc[final_valid_barcode_df_indices]
    called_df["phenotype trenchid"] = barcode_df_mapped_trenchids
    called_df["phenotype trenchid"] = called_df["phenotype trenchid"].astype(int)
    called_df = called_df.drop(["Barcode Signal"], axis=1)
    called_df = called_df.reset_index().set_index("phenotype trenchid",drop=True,sorted=False)

    output_df = phenotype_df.rename(columns={"trenchid":"phenotype trenchid"})
    output_df = output_df.reset_index().set_index("phenotype trenchid",drop=True,sorted=True)
    output_df = output_df.loc[barcode_df_mapped_trenchids_list]

    called_df = called_df.repartition(divisions = output_df.divisions).persist()
    output_df = output_df.merge(called_df,how='inner',left_index=True,right_index=True)
    output_df = output_df.reset_index().set_index(output_index)

    return output_df
