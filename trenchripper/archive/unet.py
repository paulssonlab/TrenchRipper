# fmt: off
import os
import h5py
import torch
import copy
import datetime
import time
import itertools
import qgrid
import shutil
import subprocess

import skimage.morphology

import pandas as pd
import numpy as np
import ipywidgets as ipyw
import skimage as sk
import sklearn as skl
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dask.dataframe as dd

from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from scipy.ndimage import convolve1d
from torch._six import container_abcs, string_classes, int_classes

from .utils import pandas_hdf5_handler,kymo_handle,writedir
from .trcluster import hdf5lock,dask_controller
from .metrics import object_f_scores

from matplotlib import pyplot as plt



def get_border(labeled):
    mask = sk.segmentation.find_boundaries(labeled)
    return mask

def get_mask(labeled):
    mask = np.zeros(labeled.shape,dtype=bool)
    mask[labeled>0] = True
    return mask

def get_background(labeled):
    mask = np.zeros(labeled.shape,dtype=bool)
    mask[labeled==0] = True
    return mask

def get_masknoborder(labeled):
    mask = get_mask(labeled)
    border = get_border(labeled)
    mask[border] = False
    return mask

def get_segmentation(labeled,mode_list=['background','mask','border']):
    segmentation = np.zeros(labeled.shape,dtype="uint8")
    for i,mode in enumerate(mode_list):
        if mode == 'background':
            segmentation[get_background(labeled)] = i
        elif mode == 'mask':
            segmentation[get_mask(labeled)] = i
        elif mode == 'border':
            segmentation[get_border(labeled)] = i
        elif mode == 'masknoborder':
            segmentation[get_masknoborder(labeled)] = i
        else:
            raise
    return segmentation

def get_standard_weightmap(segmentation):
    num_labels = len(np.unique(segmentation))
    label_count = []
    label_masks = []
    for label in range(num_labels):
        label_mask = segmentation==label
        num_label = np.sum(label_mask)
        label_masks.append(label_mask)
        label_count.append(num_label)

    ttl_count = np.sum(label_count)
    class_weight = ttl_count/(np.array(label_count)+1)
    class_weight = class_weight / np.sum(class_weight)
    weight_map = np.zeros(segmentation.shape, dtype=np.float32)
    for i,label_mask in enumerate(label_masks):
        weight_map[label_mask] = class_weight[i]

    return weight_map

def get_unet_weightmap(labeled,W0=5.0,Wsigma=2.0):
    mask = labeled > 0

    ttl_count = mask.size
    mask_count = np.sum(mask)
    background_count = ttl_count - mask_count

    class_weight = np.array([ttl_count / (background_count + 1), ttl_count / (mask_count + 1)])
    class_weight = class_weight / np.sum(class_weight)

    labels = np.unique(labeled)[1:]
    num_labels = len(labels)

    if num_labels == 0:
        weight_map = np.ones(labeled.shape) * class_weight[0]
    elif num_labels == 1:
        weight_map = np.ones(labeled.shape) * class_weight[0]
        weight_map[mask] += class_weight[1]
    else:
        dist_maps = []
        borders = []
        for i in labels:
            cell = labeled == i
            eroded = sk.morphology.binary_dilation(cell)
            border = eroded ^ cell
            borders.append(border)
            dist_map = ndimage.morphology.distance_transform_edt(~border)
            dist_maps.append(dist_map)
        dist_maps = np.array(dist_maps)
        borders = np.array(borders)
        borders = np.max(borders, axis=0)
        dist_maps = np.sort(dist_maps, axis=0)
        weight_map = W0 * np.exp(
            -((dist_maps[0] + dist_maps[1]) ** 2) / (2 * (Wsigma ** 2))
        )
        weight_map[mask] += class_weight[1]
        weight_map[~mask] += class_weight[0]

    return weight_map

def get_flows(labeled,eps=0.00001):
    rps = sk.measure.regionprops(labeled)
    centers = np.array([np.round(rp.centroid).astype("uint16") for rp in rps])
    y_lens = np.array([rp.bbox[2]-rp.bbox[0] for rp in rps])
    x_lens = np.array([rp.bbox[3]-rp.bbox[1] for rp in rps])
    N_arr = 2*(y_lens+x_lens)
    kernel = np.ones(3, float) / 3.

    x_grad_arr = np.zeros(labeled.shape, dtype=np.float32)
    y_grad_arr = np.zeros(labeled.shape, dtype=np.float32)

    for cell_idx in range(1,len(rps)+1):
        cell_mask = labeled==cell_idx
        cell_center = centers[cell_idx-1]
        diffusion_arr = np.zeros(cell_mask.shape,dtype=np.float32)
        for i in range(N_arr[cell_idx-1]):
            diffusion_arr[cell_center] += 1.
            diffusion_arr = convolve1d(convolve1d(diffusion_arr, kernel, axis=0), kernel, axis=1)
            diffusion_arr[~cell_mask] = 0.

        y_grad,x_grad = np.gradient(diffusion_arr)

        norm = np.sqrt(y_grad**2 + x_grad**2)

        y_grad,x_grad = (y_grad/(norm+eps)),(x_grad/(norm+eps))
        y_grad[~cell_mask] = 0.
        x_grad[~cell_mask] = 0.

        y_grad_arr += y_grad
        x_grad_arr += x_grad

    return y_grad_arr,x_grad_arr

def get_two_class(labeled):
    segmentation = get_segmentation(labeled,mode_list=['background','mask','border'])
    weightmap = get_standard_weightmap(segmentation)
    if np.any(np.isnan(segmentation)) or np.any(np.isnan(weightmap)):
        segmentation = np.zeros(labeled.shape,dtype="uint8")
        weightmap = np.ones(segmentation.shape, dtype=np.float32)
    return segmentation, weightmap

def get_one_class(labeled,W0=5.0,Wsigma=2.0):
    segmentation = get_segmentation(labeled,mode_list=['background','masknoborder']).astype(bool)
    weightmap = get_unet_weightmap(labeled,W0=W0,Wsigma=Wsigma)
    if np.any(np.isnan(segmentation)) or np.any(np.isnan(weightmap)):
        segmentation = np.zeros(labeled.shape,dtype="uint8")
        weightmap = np.ones(segmentation.shape, dtype=np.float32)
    return segmentation, weightmap

def get_cellpose(labeled):
    segmentation = get_segmentation(labeled,mode_list=['background','mask']).astype(bool)
    y_grad_arr,x_grad_arr = get_flows(labeled)
    if np.any(np.isnan(segmentation)) or np.any(np.isnan(y_grad_arr)) or np.any(np.isnan(x_grad_arr)):
        segmentation = np.zeros(labeled.shape,dtype="uint8")
        x_grad_arr = np.zeros(labeled.shape, dtype=np.float32)
        y_grad_arr = np.zeros(labeled.shape, dtype=np.float32)
    return segmentation,y_grad_arr,x_grad_arr

def numpy_collate(batch): #modified version of torch default
    r"""Puts each data field into a numpy array with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, np.ndarray):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return np.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise

            return numpy_collate([np.array(b) for b in batch])
        elif elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(elem, float):
        return np.array(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return np.array(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(numpy_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    raise

def get_cellpose_labels(fx,step_size=1.,n_iter=10): #N, (y,x,mask), y, x
    labeled = np.zeros((fx.shape[0],fx.shape[2],fx.shape[3]),dtype="uint8")
    for n in range(fx.shape[0]):
        y_grad_arr = fx[n,0]
        x_grad_arr = fx[n,1]
        mask = fx[n,2]>0.5
        pixel_arr = np.array(np.where(mask)).T
        final_pixels = []
        for pixel_idx in range(pixel_arr.shape[0]):
            pixel_coord = pixel_arr[pixel_idx].astype(np.float32)
            for N in range(n_iter):
                near_coord = pixel_coord.astype(int)
                near_coord[0] = np.clip(near_coord[0],0,y_grad_arr.shape[0]-1)
                near_coord[1] = np.clip(near_coord[1],0,y_grad_arr.shape[1]-1)

                y_grad = y_grad_arr[near_coord[0],near_coord[1]]
                x_grad = x_grad_arr[near_coord[0],near_coord[1]]
                pixel_coord += np.array([y_grad,x_grad])*step_size
            final_pixels.append(pixel_coord)

        if len(final_pixels) > 0:
            dbsc = skl.cluster.DBSCAN(eps=2.)
            cluster_assign = dbsc.fit_predict(final_pixels)

            labeled[n,pixel_arr[:,0],pixel_arr[:,1]] = cluster_assign+1
    return labeled

def get_class_labels(segmentation,mask_label_dim=1):
    segmentation = np.argmax(segmentation,axis=1)
    mask = (segmentation==mask_label_dim)
    labeled = np.zeros((segmentation.shape),dtype="uint8")
    for n in range(segmentation.shape[0]):
        labeled[n] = sk.measure.label(mask[n])
    return labeled

class UNet_Training_DataLoader:
    def __init__(
        self,
        nndatapath="",
        experimentname="",
        output_names=["train", "test", "val"],
        output_modes=["class", "multiclass", "cellpose"],
        input_paths=[],
        W0_list=[5.],
        Wsigma_list=[2.],
    ):
        self.nndatapath = nndatapath
        self.metapath = nndatapath + "/metadata.hdf5"
        self.experimentname = experimentname
        self.output_names = output_names
        self.output_modes = output_modes
        self.input_paths = input_paths
        self.W0_list = W0_list
        self.Wsigma_list = Wsigma_list

    def get_metadata(self, headpath):
        meta_handle = pandas_hdf5_handler(headpath + "/metadata.hdf5")
        global_handle = meta_handle.read_df("global", read_metadata=True)

        kymodf = dd.read_parquet(headpath + "/kymograph/metadata",calculate_divisions=True).persist()
        kymodf = kymodf.set_index("fov").persist()

        channel_list = global_handle.metadata["channels"]
        fov_list = kymodf.index.compute().get_level_values("fov").unique().tolist()
        t_len = len(kymodf.loc[fov_list[0]]["timepoints"].unique().compute())
        ttl_trenches = len(kymodf["trenchid"].unique().compute())

        trench_dict = {
            fov: len(kymodf.loc[fov]["trenchid"].unique().compute()) for fov in fov_list
        }
        with open(headpath + "/kymograph/metadata.pkl", 'rb') as handle:
            ky_metadata = pkl.load(handle)

        shape_y = ky_metadata["kymograph_params"]["ttl_len_y"]
        shape_x = ky_metadata["kymograph_params"]["trench_width_x"]
        kymograph_img_shape = tuple((shape_y, shape_x))
        return (
            channel_list,
            fov_list,
            t_len,
            trench_dict,
            ttl_trenches,
            kymograph_img_shape,
        )

    def inter_get_selection(self):
        output_tabs = []
        for i in range(len(self.output_names)):
            dset_tabs = []
            for j in range(len(self.input_paths)):
                (
                    channel_list,
                    fov_list,
                    t_len,
                    trench_dict,
                    ttl_trenches,
                    kymograph_img_shape,
                ) = self.get_metadata(self.input_paths[j])

                feature_dropdown = ipyw.Dropdown(
                    options=channel_list,
                    value=channel_list[0],
                    description="Feature Channel:",
                    disabled=False,
                )
                max_samples = ipyw.IntText(
                    value=0, description="Maximum Samples per Dataset:", disabled=False
                )
                t_range = ipyw.IntRangeSlider(
                    value=[0, t_len - 1],
                    description="Timepoint Range:",
                    min=0,
                    max=t_len - 1,
                    step=1,
                    disabled=False,
                    continuous_update=False,
                )

                working_tab = ipyw.VBox(
                    children=[feature_dropdown, max_samples, t_range]
                )
                dset_tabs.append(working_tab)

            dset_ipy_tabs = ipyw.Tab(children=dset_tabs)
            for j in range(len(self.input_paths)):
                dset_ipy_tabs.set_title(j, self.input_paths[j].split("/")[-1])
            output_tabs.append(dset_ipy_tabs)
        output_ipy_tabs = ipyw.Tab(children=output_tabs)
        for i, output_name in enumerate(self.output_names):
            output_ipy_tabs.set_title(i, output_name)
        self.tab = output_ipy_tabs

        return self.tab

    def get_import_params(self):
        self.import_param_dict = {}
        for i, output_name in enumerate(self.output_names):
            self.import_param_dict[output_name] = {}
            for j, input_path in enumerate(self.input_paths):
                working_vbox = self.tab.children[i].children[j]
                self.import_param_dict[output_name][input_path] = {
                    child.description: child.value for child in working_vbox.children
                }

        print("======== Import Params ========")
        for i, output_name in enumerate(self.output_names):
            print(str(output_name))
            for j, input_path in enumerate(self.input_paths):
                (
                    channel_list,
                    fov_list,
                    t_len,
                    trench_dict,
                    ttl_trenches,
                    kymograph_img_shape,
                ) = self.get_metadata(input_path)
                ttl_possible_samples = t_len * ttl_trenches
                param_dict = self.import_param_dict[output_name][input_path]
                requested_samples = param_dict["Maximum Samples per Dataset:"]
                if requested_samples > 0:
                    print(str(input_path))
                    for key, val in param_dict.items():
                        print(key + " " + str(val))
                    print(
                        "Requested Samples / Total Samples: "
                        + str(requested_samples)
                        + "/"
                        + str(ttl_possible_samples)
                    )

        del self.tab

    def export_chunk(self, output_name, init_idx, chunk_size, chunk_idx):
        output_meta_handle = pandas_hdf5_handler(self.metapath)
        output_df = output_meta_handle.read_df(output_name)
        working_df = output_df[init_idx : init_idx + chunk_size]
        nndatapath = self.nndatapath + "/" + output_name + "_" + str(chunk_idx) + ".hdf5"

        try:
            os.remove(nndatapath)
        except OSError:
            pass

        dset_paths = working_df.index.get_level_values(0).unique().tolist()
        for dset_path in dset_paths:
            dset_path_key = dset_path.split("/")[-1]
            dset_df = working_df.loc[dset_path]

            if isinstance(dset_df, pd.Series):
                dset_df = dset_df.to_frame().T

            param_dict = self.import_param_dict[output_name][dset_path]
            feature_channel = param_dict["Feature Channel:"]

            img_arr_list = []
            seg_arr_list = []

            dset_df = dset_df.set_index("File Index")
            dset_df = dset_df.sort_index()
            file_indices = dset_df.index.get_level_values(0).unique().tolist()

            for file_idx in file_indices:
                file_df = dset_df.loc[file_idx:file_idx]

                img_path = dset_path + "/kymograph/kymograph_" + str(file_idx) + ".hdf5"
                seg_path = dset_path + "/fluorsegmentation/segmentation_" + str(file_idx) + ".hdf5"

                with h5py.File(img_path, "r") as imgfile:
                    working_arr = imgfile[feature_channel][:]

                trench_df = file_df.set_index("File Trench Index")
                trench_df = trench_df.sort_index()

                for trench_idx, row in trench_df.iterrows():
                    img_arr = working_arr[trench_idx, row["timepoints"]][np.newaxis, np.newaxis, :, :]  # 1,1,y,x img
                    img_arr = img_arr.astype("uint16")
                    img_arr_list.append(img_arr)

                with h5py.File(seg_path, "r") as segfile:
                    working_arr = segfile["data"][:]

                for trench_idx, row in trench_df.iterrows():
                    seg_arr = working_arr[trench_idx, row["timepoints"]][np.newaxis, np.newaxis, :, :]
                    seg_arr = seg_arr.astype("int8")
                    seg_arr_list.append(seg_arr)

            output_img_arr = np.concatenate(img_arr_list, axis=0)
            output_seg_arr = np.concatenate(seg_arr_list, axis=0) #N,1,y,x
            chunk_shape = (1, 1, output_img_arr.shape[2], output_img_arr.shape[3])

            with h5py.File(nndatapath, "a") as outfile:
                img_handle = outfile.create_dataset(dset_path_key + "/img",data=output_img_arr,chunks=chunk_shape,dtype="uint16")
                seg_handle = outfile.create_dataset(dset_path_key + "/seg",data=output_seg_arr,chunks=chunk_shape,dtype="int8")

                for output_mode in self.output_modes:

                    if output_mode == "class":
                        output_seg_arr_class = []
                        for i,W0 in enumerate(self.W0_list):
                            for j,Wsigma in enumerate(self.Wsigma_list):
                                output_weight_arr = []
                                for l in range(output_seg_arr.shape[0]):
                                    labeled = output_seg_arr[l,0]
                                    segmentation, weightmap = get_one_class(labeled,W0=W0,Wsigma=Wsigma)
                                    output_weight_arr.append(weightmap[np.newaxis,np.newaxis,:,:])
                                    if i+j==0:
                                        output_seg_arr_class.append(segmentation[np.newaxis,np.newaxis,:,:])
                                output_weight_arr = np.concatenate(output_weight_arr,axis=0)
                                weight_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/W0=" + str(W0) +\
                                "_Wsigma=" + str(Wsigma) + "/weight", data=output_weight_arr, chunks=chunk_shape, dtype=np.float32)

                        output_seg_arr_class = np.concatenate(output_seg_arr_class,axis=0)
                        seg_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/seg",
                            data=output_seg_arr_class, chunks=chunk_shape, dtype="int8")

                    elif output_mode == "multiclass":
                        output_seg_arr_multiclass = []
                        output_weight_arr = []
                        for l in range(output_seg_arr.shape[0]):
                            labeled = output_seg_arr[l,0]
                            segmentation, weightmap = get_two_class(labeled)
                            output_seg_arr_multiclass.append(segmentation[np.newaxis,np.newaxis,:,:])
                            output_weight_arr.append(weightmap[np.newaxis,np.newaxis,:,:])
                        output_seg_arr_multiclass = np.concatenate(output_seg_arr_multiclass,axis=0)
                        output_weight_arr = np.concatenate(output_weight_arr,axis=0)

                        seg_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/seg",
                        data=output_seg_arr_multiclass, chunks=chunk_shape, dtype="int8")

                        weight_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/weight",
                        data=output_weight_arr, chunks=chunk_shape, dtype=np.float32)

                    elif output_mode == "cellpose":
                        output_seg_arr_cellpose = []
                        output_y_grad_arr = []
                        output_x_grad_arr = []
                        for l in range(output_seg_arr.shape[0]):
                            labeled = output_seg_arr[l,0]
                            segmentation,y_grad_arr,x_grad_arr = get_cellpose(labeled)
                            output_seg_arr_cellpose.append(segmentation[np.newaxis,np.newaxis,:,:])
                            output_y_grad_arr.append(y_grad_arr[np.newaxis,np.newaxis,:,:])
                            output_x_grad_arr.append(x_grad_arr[np.newaxis,np.newaxis,:,:])
                        output_seg_arr_cellpose = np.concatenate(output_seg_arr_cellpose,axis=0)
                        output_y_grad_arr = np.concatenate(output_y_grad_arr,axis=0)
                        output_x_grad_arr = np.concatenate(output_x_grad_arr,axis=0)

                        seg_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/seg",
                        data=output_seg_arr_cellpose, chunks=chunk_shape, dtype="int8")

                        y_grad_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/y_grad",
                        data=output_y_grad_arr, chunks=chunk_shape, dtype=np.float32)

                        x_grad_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/x_grad",
                        data=output_x_grad_arr, chunks=chunk_shape, dtype=np.float32)

                    else:
                        raise

        return init_idx

    def gather_chunks(self, output_name, init_idx_list, chunk_idx_list, chunk_size):

        #                       outputdf,output_metadata,selectionname,file_idx_list,weight_grid_list):
        nnoutputpath = self.nndatapath + "/" + output_name + ".hdf5"
        output_meta_handle = pandas_hdf5_handler(self.metapath)
        output_df = output_meta_handle.read_df(output_name)

        dset_paths = output_df.index.get_level_values(0).unique().tolist()

        with h5py.File(nnoutputpath, "w") as outfile:
            for dset_path in dset_paths:
                dset_path_key = dset_path.split("/")[-1]
                dset_df = output_df.loc[dset_path]
                dset_chunk_idx = output_df.index.get_loc(dset_path).start//chunk_size

                tempdatapath = self.nndatapath + "/" + output_name + "_" + str(dset_chunk_idx) + ".hdf5"
                with h5py.File(tempdatapath, "r") as infile:
                    img_shape = infile[dset_path_key + "/img"].shape
                output_shape = (len(dset_df.index), 1, img_shape[2], img_shape[3])
                chunk_shape = (1, 1, img_shape[2], img_shape[3])

                img_handle = outfile.create_dataset(dset_path_key + "/img", output_shape, chunks=chunk_shape, dtype="uint16")
                seg_handle = outfile.create_dataset(dset_path_key + "/seg", output_shape, chunks=chunk_shape, dtype="int8")
                for output_mode in self.output_modes:
                    if output_mode == "class":
                        for i,W0 in enumerate(self.W0_list):
                            for j,Wsigma in enumerate(self.Wsigma_list):
                                weight_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/W0=" + str(W0) +\
                                    "_Wsigma=" + str(Wsigma) + "/weight", output_shape, chunks=chunk_shape, dtype=np.float32)
                        seg_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/seg",
                            output_shape, chunks=chunk_shape, dtype="int8")
                    elif output_mode == "multiclass":
                        seg_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/seg",
                            output_shape, chunks=chunk_shape, dtype="int8")
                        weight_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/weight",
                            output_shape, chunks=chunk_shape, dtype=np.float32)
                    elif output_mode == "cellpose":
                        seg_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/seg",
                            output_shape, chunks=chunk_shape, dtype="int8")
                        y_grad_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/y_grad",
                            output_shape, chunks=chunk_shape, dtype=np.float32)
                        x_grad_handle = outfile.create_dataset(dset_path_key + "/" + output_mode + "/x_grad",
                            output_shape, chunks=chunk_shape, dtype=np.float32)
                    else:
                        raise

        current_dset_path = ""
        for i, init_idx in enumerate(init_idx_list):
            chunk_idx = chunk_idx_list[i]
            nndatapath = (self.nndatapath + "/" + output_name + "_" + str(chunk_idx) + ".hdf5")
            working_df = output_df[init_idx : init_idx + chunk_size]
            dset_paths = working_df.index.get_level_values(0).unique().tolist()

            with h5py.File(nndatapath, "r") as infile:

                for dset_path in dset_paths:
                    if dset_path != current_dset_path:
                        current_idx = 0
                        current_dset_path = dset_path

                    dset_path_key = dset_path.split("/")[-1]
                    dset_df = output_df.loc[dset_path]

                    with h5py.File(nnoutputpath, "a") as outfile:
                        img_arr = infile[dset_path_key + "/img"][:]
                        num_indices = img_arr.shape[0]
                        outfile[dset_path_key + "/img"][current_idx : current_idx + num_indices] = img_arr

                        seg_arr = infile[dset_path_key + "/seg"][:]
                        outfile[dset_path_key + "/seg"][current_idx : current_idx + num_indices] = seg_arr

                        for output_mode in self.output_modes:
                            if output_mode == "class":
                                seg_arr = infile[dset_path_key + "/" + output_mode + "/seg"][:]
                                outfile[dset_path_key + "/" + output_mode + "/seg"][current_idx : current_idx + num_indices] = seg_arr
                                for i,W0 in enumerate(self.W0_list):
                                    for j,Wsigma in enumerate(self.Wsigma_list):
                                        weight_arr = infile[dset_path_key + "/" + output_mode + "/W0=" + str(W0) +"_Wsigma=" + str(Wsigma) +\
                                                            "/weight"][:]
                                        outfile[dset_path_key + "/" + output_mode + "/W0=" + str(W0) +"_Wsigma=" + str(Wsigma) + "/weight"]\
                                        [current_idx : current_idx + num_indices] = weight_arr

                            elif output_mode == "multiclass":
                                seg_arr = infile[dset_path_key + "/" + output_mode + "/seg"][:]
                                outfile[dset_path_key + "/" + output_mode + "/seg"][current_idx : current_idx + num_indices] = seg_arr
                                weight_arr = infile[dset_path_key + "/" + output_mode + "/weight"][:]
                                outfile[dset_path_key + "/" + output_mode + "/weight"][current_idx : current_idx + num_indices] = weight_arr

                            elif output_mode == "cellpose":
                                seg_arr = infile[dset_path_key + "/" + output_mode + "/seg"][:]
                                outfile[dset_path_key + "/" + output_mode + "/seg"][current_idx : current_idx + num_indices] = seg_arr
                                y_grad_arr = infile[dset_path_key + "/" + output_mode + "/y_grad"][:]
                                outfile[dset_path_key + "/" + output_mode + "/y_grad"][current_idx : current_idx + num_indices] = y_grad_arr
                                x_grad_arr = infile[dset_path_key + "/" + output_mode + "/x_grad"][:]
                                outfile[dset_path_key + "/" + output_mode + "/x_grad"][current_idx : current_idx + num_indices] = x_grad_arr

                            else:
                                raise

                    current_idx += num_indices

            os.remove(nndatapath)

    def export_data(self, dask_controller, chunk_size=250):
        dask_controller.futures = {}
        output_meta_handle = pandas_hdf5_handler(self.metapath)
        all_output_dfs = {}

        for output_name, _ in self.import_param_dict.items():
            output_df = []

            for input_path, param_dict in self.import_param_dict[output_name].items():

                kymodf = dd.read_parquet(input_path + "/kymograph/metadata",calculate_divisions=True).persist()

                num_samples = param_dict["Maximum Samples per Dataset:"]
                feature_channel = param_dict["Feature Channel:"]
                t_range = param_dict["Timepoint Range:"]

                kymodf["filepath"] = input_path
                kymodf = kymodf.reset_index()
                timedf = kymodf.set_index("timepoints").persist()
                timedf = timedf.loc[t_range[0] : t_range[1]].persist()

                frac = (1.1*num_samples)/len(timedf)
                frac = min(frac,1.)

                timedf_subset = timedf.sample(frac=frac).compute()
                adjusted_num_samples = min(len(timedf_subset),num_samples)

                timedf_subset = timedf_subset.sample(n=adjusted_num_samples)
                timedf_subset = timedf_subset.reset_index()
                filedf_subset = timedf_subset.set_index("filepath")
                output_df.append(filedf_subset[:num_samples])
            output_df = pd.concat(output_df)
            output_df = output_df.sort_index()
            output_meta_handle.write_df(output_name, output_df)
            all_output_dfs[output_name] = output_df

        for output_name in all_output_dfs.keys():

            output_df = all_output_dfs[output_name]

            ## split into equal computation chunks here

            chunk_idx_list = []
            for chunk_idx, init_idx in enumerate(range(0, len(output_df), chunk_size)):
                future = dask_controller.daskclient.submit(self.export_chunk,output_name,init_idx,chunk_size,chunk_idx,retries=1)
                dask_controller.futures[str(output_name) + " Chunk Number: " + str(chunk_idx)] = future
                chunk_idx_list.append(chunk_idx)

            init_idx_list = dask_controller.daskclient.gather([dask_controller.futures[str(output_name) + " Chunk Number: " + str(chunk_idx)]\
                                                               for chunk_idx in chunk_idx_list])
            self.gather_chunks(output_name, init_idx_list, chunk_idx_list, chunk_size)

        output_meta_handle = pandas_hdf5_handler(self.metapath)

        for output_name in self.import_param_dict.keys():
            for input_path in self.import_param_dict[output_name].keys():
                input_meta_handle = pandas_hdf5_handler(input_path + "/metadata.hdf5")
                indf = input_meta_handle.read_df("global",read_metadata=True)
                global_meta = indf.metadata
                del indf
                self.import_param_dict[output_name][input_path]["global"] = global_meta

                kymo_meta_path = input_path + "/kymograph/metadata.pkl"
                with open(kymo_meta_path, 'rb') as infile:
                    kymo_meta = pkl.load(infile)
                self.import_param_dict[output_name][input_path]["kymograph"] = kymo_meta

                segparampath = input_path + "/fluorescent_segmentation.par"
                with open(segparampath, 'rb') as infile:
                    seg_param_dict = pkl.load(infile)
                self.import_param_dict[output_name][input_path]["segmentation"] = seg_param_dict

            output_metadata = {"nndataset":{"experimentname":self.experimentname,"output_names":self.output_names,"output_modes":self.output_modes,\
                               "input_paths":self.input_paths,"W0_list":self.W0_list,"Wsigma_list":self.Wsigma_list}}
            output_metadata = {**output_metadata,**self.import_param_dict[output_name]}

            output_meta_handle.write_df(output_name,all_output_dfs[output_name],metadata=output_metadata)

class SegmentationDataset(Dataset):
    def __init__(self,filepath,mode="run",chunksize=1000,W0=5.,Wsigma=2.):
        self.filepath = filepath
        self.mode = mode
        self.chunksize = chunksize
        self.W0 = W0
        self.Wsigma = Wsigma
        self.dset_shapes = {}
        with h5py.File(self.filepath,"r") as infile:
            for dset_name in infile.keys():
                shape = infile[dset_name + "/img"].shape
                self.dset_shapes[dset_name] = shape

        index_ranges = [0] + np.add.accumulate([value[0] for value in self.dset_shapes.values()]).tolist()
        self.chunk_index_ranges = self.fill_index_gaps(index_ranges)

        start_chunks = (np.add.accumulate((np.array([0] + [value[0] for value in self.dset_shapes.values()])//(chunksize+1))+1)-1).tolist()
        self.chunk_ranges = [range(start_chunks[i],start_chunks[i+1]) for i in range(len(start_chunks)-1)]

        self.chunk_dsets = {}
        for i,item in enumerate(self.dset_shapes.keys()):
            chunk_range = self.chunk_ranges[i]
            for chunk in chunk_range:
                self.chunk_dsets[chunk] = item

        self.current_chunk = 0
        self.load_chunk(self.current_chunk)

        self.size = 0
        with h5py.File(self.filepath,"r") as infile:
            for dset_name in infile.keys():
                self.size += (np.prod(infile[dset_name+"/img"].shape))

    def load_chunk(self,chunk_idx):
        self.current_dset = self.chunk_dsets[chunk_idx]
        chunk_offset = [item.start for item in self.chunk_ranges if chunk_idx in item][0]
        working_chunk_idx = chunk_idx-chunk_offset

        with h5py.File(self.filepath,"r") as infile:
            self.img_data = infile[self.current_dset + "/img"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
            if self.mode == "run":
                pass
            elif self.mode == "class":
                self.gt_data = infile[self.current_dset + "/seg"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
                self.seg_data = infile[self.current_dset + "/class/seg"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
                self.weight_data = infile[self.current_dset + "/class/W0=" + str(self.W0) + "_Wsigma=" + str(self.Wsigma) + "/weight"]\
                [working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
            elif self.mode == "multiclass":
                self.gt_data = infile[self.current_dset + "/seg"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
                self.seg_data = infile[self.current_dset + "/multiclass/seg"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
                self.weight_data = infile[self.current_dset + "/multiclass/weight"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
            elif self.mode == "cellpose":
                self.gt_data = infile[self.current_dset + "/seg"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
                self.seg_data = infile[self.current_dset + "/cellpose/seg"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
                self.y_grad_data = infile[self.current_dset + "/cellpose/y_grad"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
                self.x_grad_data = infile[self.current_dset + "/cellpose/x_grad"][working_chunk_idx*self.chunksize:(working_chunk_idx+1)*self.chunksize]
            else:
                raise
        self.current_chunk = chunk_idx

    def fill_index_gaps(self,index_ranges):
        chunk_index_ranges = [index_ranges[0]]
        for i in range(len(index_ranges)-1):
            last_index = index_ranges[i]
            not_gap_filled = True
            while not_gap_filled:
                del_index = index_ranges[i+1] - last_index
                if del_index > self.chunksize:
                    chunk_index_ranges.append(last_index+self.chunksize)
                    last_index = last_index+self.chunksize
                else:
                    chunk_index_ranges.append(index_ranges[i+1])
                    not_gap_filled = False
        chunk_index_ranges = [range(chunk_index_ranges[i],chunk_index_ranges[i+1]) for i in range(len(chunk_index_ranges)-1)]
        return chunk_index_ranges

    def __len__(self):
        out_len = 0
        with h5py.File(self.filepath,"r") as infile:
            for dset_name in infile.keys():
                out_len += infile[dset_name+"/img"].shape[0]
        return out_len
    def __getitem__(self,idx):
        idx_chunk = [i for i, interval in enumerate(self.chunk_index_ranges) if idx in interval][0]
        subidx = idx%self.chunksize
        if idx_chunk != self.current_chunk:
            self.load_chunk(idx_chunk)

        sample = {'img': self.img_data[subidx]}
        if self.mode == "run":
            pass
        elif self.mode == "class" or self.mode == "multiclass":
            sample['gt'] = self.gt_data[subidx]
            sample['seg'] = self.seg_data[subidx]
            sample['weight'] = self.weight_data[subidx]
        elif self.mode == "cellpose":
            sample['gt'] = self.gt_data[subidx]
            sample['seg'] = self.seg_data[subidx]
            sample['y_grad'] = self.y_grad_data[subidx]
            sample['x_grad'] = self.x_grad_data[subidx]
        else:
            raise
        return sample

class double_conv(nn.Module):
    """(Conv => BatchNorm =>ReLU) twice."""
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.downconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.downconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,layers=3,hidden_size=64,dropout=0.,withsoftmax=False):
        super().__init__()
        self.inc = inconv(n_channels, hidden_size)
        self.downlist = nn.ModuleList([down(hidden_size*(2**i), hidden_size*(2**(i+1))) for i in range(0,layers-1)] + [down(hidden_size*(2**(layers-1)), hidden_size*(2**(layers-1)))])
        self.uplist = nn.ModuleList([up(hidden_size*(2**i), hidden_size*(2**(i-2))) for i in reversed(range(2,layers+1))] + [up(hidden_size*2, hidden_size)])
        self.outc = outconv(hidden_size, n_classes)
        self.drop = nn.Dropout(p=dropout)
        self.withsoftmax = withsoftmax
    def uniforminit(self):
        for param in self.named_parameters():
            param[1].data.uniform_(-0.05,0.05)
    def forward(self, x):
        xlist = [self.inc(x)]
        for item in self.downlist:
            xlist.append(item(xlist[-1]))
        x = xlist[-1]
        x = self.drop(x)
        for i,item in enumerate(self.uplist):
            x = item(x, xlist[-(i+2)])
        x = self.outc(x)
        if self.withsoftmax:
            x = F.softmax(x,dim=1)
        return x

class UNet_Trainer:

    def __init__(self,nndatapath,model_number,mode,numepochs=100,batch_size=100,layers=3,hidden_size=64,lr=0.005,momentum=0.95,weight_decay=0.0005,dropout=0.,\
                 W0=5.,Wsigma=2.,warm_epochs=10,cool_epochs=30,gpuon=False,padding=10,**kwargs):
        self.nndatapath = nndatapath
        self.model_number = model_number
        self.mode = mode
        self.padding = padding
        self.aug_seq = self.define_aug_seq(**kwargs)

        self.numepochs = numepochs
        self.batch_size = batch_size
        self.gpuon = gpuon

        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.W0 = W0
        self.Wsigma = Wsigma

        self.warm_epochs = warm_epochs
        self.cool_epochs = cool_epochs
        self.warm_lambda = 1./self.warm_epochs

        if self.mode == "class":
            self.model = UNet(1,2,layers=layers,hidden_size=hidden_size,dropout=dropout,withsoftmax=True)
        elif self.mode == "multiclass":
            self.model = UNet(1,3,layers=layers,hidden_size=hidden_size,dropout=dropout,withsoftmax=True)
        elif self.mode == "cellpose":
            self.model = UNet(1,3,layers=layers,hidden_size=hidden_size,dropout=dropout,withsoftmax=False)

        self.model.uniforminit()
        if gpuon:
            self.model = self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.annealfn)

    def annealfn(self,epoch):
        if epoch<self.warm_epochs:
            return self.warm_lambda*epoch
        elif epoch>(self.numepochs-self.cool_epochs):
            current_cool_epoch = epoch-(self.numepochs-self.cool_epochs)
            num_cool_steps = current_cool_epoch//10
            return (1./2.)**(num_cool_steps)
        else:
            return 1.

    def pad_fn(self,array):
        padded = np.pad(array,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        return padded

    def define_aug_seq(self,pad_perc_width=0.25,pad_perc_height=0.15,max_width=40,max_height=180,flip_perc=0.5,blur_freq=0.5,blur_sigma=0.5,contrast_range=(0.8,1.2),\
                noise=0.05,mult_freq=0.2,mult_range=(0.8,1.2),scale_range=(0.7,1.3),translate_range=(-0.15,0.15),rotate_range=(-15,15),\
               x_shear_range=(-10, 10),y_shear_range=(-5,5)):
        seq = iaa.Sequential([
            iaa.CropToFixedSize(width=max_width, height=max_height),
            iaa.Pad(percent=(pad_perc_height,pad_perc_width,pad_perc_height,pad_perc_width),keep_size=False),
            iaa.Fliplr(flip_perc), # vertically flip 50% of the images
            iaa.Flipud(flip_perc), # horizontally flip 50% of the images
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
#             iaa.Sometimes(
#                 blur_freq,
#                 iaa.GaussianBlur(sigma=(0, blur_sigma))
#             ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast(alpha=contrast_range),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, noise*255)),
#             # Make some images brighter and some darker.
#             # In 20% of all cases, we sample the multiplier once per channel,
#             # which can end up changing the color of the images.
            iaa.Multiply(mult_range, per_channel=mult_freq),
            iaa.Affine(
                scale={"x": scale_range, "y": scale_range},
                translate_percent={"x": translate_range, "y": translate_range},
                rotate=rotate_range,
                shear={'x':x_shear_range,'y':y_shear_range}
            ),
        ])
        return seq

    def removefile(self,path):
        if os.path.exists(path):
            os.remove(path)

    def load_model(self,paramspath):
        if self.gpuon:
            device = torch.device("cuda")
            self.model.load_state_dict(torch.load(paramspath))
        else:
            device = torch.device('cpu')
            self.model.load_state_dict(torch.load(paramspath, map_location=device))

    def class_aug(self,img_arr,seg_arr,weight_arr): #N,1,y,x
        in_bounds_arr = np.ones(img_arr.shape[2:],dtype="uint8")

        img_aug_arr = []
        seg_aug_arr = []
        weight_aug_arr = []
        for n in range(img_arr.shape[0]):
            segmap = SegmentationMapsOnImage(seg_arr[n,0], shape=img_arr[n,0].shape)
            heatmap = HeatmapsOnImage(weight_arr[n,0], shape=img_arr[n,0].shape, min_value=min(0.,np.min(weight_arr[n,0])),max_value=max(1.,np.max(weight_arr[n,0])))
            img_aug, seg_aug, weight_aug = self.aug_seq(image=img_arr[n,0], segmentation_maps=segmap, heatmaps=heatmap)

            img_aug_arr.append(img_aug)
            seg_aug_arr.append(seg_aug.get_arr())
            weight_aug_arr.append(weight_aug.get_arr())

        img_aug_arr,seg_aug_arr,weight_aug_arr = np.stack(img_aug_arr,axis=0),np.stack(seg_aug_arr,axis=0),np.stack(weight_aug_arr,axis=0)
        img_aug_arr,seg_aug_arr,weight_aug_arr = img_aug_arr[:,np.newaxis],seg_aug_arr[:,np.newaxis],weight_aug_arr[:,np.newaxis]

        return img_aug_arr,seg_aug_arr,weight_aug_arr

    def cellpose_aug(self,img_arr,seg_arr,y_grad_arr,x_grad_arr):
        in_bounds_arr = np.ones(img_arr.shape[2:],dtype=bool)
        in_bounds_arr = sk.morphology.binary_erosion(in_bounds_arr).astype("uint8")

        img_aug_arr = []
        seg_aug_arr = []
        y_grad_aug_arr = []
        x_grad_aug_arr = []
        for n in range(img_arr.shape[0]):
            segmap = SegmentationMapsOnImage(np.stack([seg_arr[n,0],in_bounds_arr],axis=2), shape=(img_arr.shape[2],img_arr.shape[3],2))
#             segmap = SegmentationMapsOnImage(seg_arr[n,0], shape=img_arr[n,0].shape)
            grad_arr = np.stack([y_grad_arr[n,0],x_grad_arr[n,0]],axis=2)
            grad_map = HeatmapsOnImage(grad_arr, shape=img_arr[n,0].shape, min_value=-1,max_value=1)
            img_aug, seg_aug, grad_aug = self.aug_seq(image=img_arr[n,0], segmentation_maps=segmap, heatmaps=grad_map)

            img_aug_arr.append(img_aug)

            seg_aug = seg_aug.get_arr()
            grad_aug = grad_aug.get_arr()
            seg_out,in_bounds,y_grad_out,x_grad_out = seg_aug[:,:,0],seg_aug[:,:,1],grad_aug[:,:,0],grad_aug[:,:,1]

            seg_out[in_bounds!=1] = 0
            y_grad_out[in_bounds!=1] = 0.
            x_grad_out[in_bounds!=1] = 0.

            seg_aug_arr.append(seg_out)
            y_grad_aug_arr.append(y_grad_out)
            x_grad_aug_arr.append(x_grad_out)

        img_aug_arr,seg_aug_arr,y_grad_aug_arr,x_grad_aug_arr = np.stack(img_aug_arr,axis=0),np.stack(seg_aug_arr,axis=0),np.stack(y_grad_aug_arr,axis=0),np.stack(x_grad_aug_arr,axis=0)
        img_aug_arr,seg_aug_arr,y_grad_aug_arr,x_grad_aug_arr = img_aug_arr[:,np.newaxis],seg_aug_arr[:,np.newaxis],y_grad_aug_arr[:,np.newaxis],x_grad_aug_arr[:,np.newaxis]

        return img_aug_arr,seg_aug_arr,y_grad_aug_arr,x_grad_aug_arr

    def train(self,x,y,weightmaps):

        self.optimizer.zero_grad()
        fx = self.model.forward(x)
        fx = torch.log(fx)

        loss = F.nll_loss(fx,y,reduction='none')*weightmaps
        mean_loss = torch.mean(loss)
        mean_loss.backward()
        self.optimizer.step()

        loss = torch.sum(loss)
        return loss

    def test(self,x,y,weightmaps):
        fx = self.model.forward(x)
        fx = torch.log(fx)
        loss = F.nll_loss(fx,y,reduction='none')*weightmaps
        loss = torch.sum(loss)
        return loss

    def cellpose_train(self,x,y):
        self.optimizer.zero_grad()
        fx = self.model.forward(x)
        mask_pred = F.sigmoid(fx[:,2])

        mse = F.mse_loss(fx[:,:2],y[:,:2],reduction='none') ## N,* to N,*
        cross_entropy = F.binary_cross_entropy(mask_pred,y[:,2],reduction='none')

        loss = cross_entropy + 5.*mse[:,0] + 5.*mse[:,1]

#         loss[loss != loss] = 0 ##gets rid of NaNs and infs
#         loss[loss == float("Inf")] = 0

        mean_loss = torch.mean(loss)

        mean_loss.backward()
        self.optimizer.step()

        loss = torch.sum(loss)

        return loss

    def cellpose_test(self,x,y):
        fx = self.model.forward(x)
        mse = F.mse_loss(fx[:,:2],y[:,:2],reduction='none') ## N,* to N,*
        mask_pred = F.sigmoid(fx[:,2])
        cross_entropy = F.binary_cross_entropy(mask_pred,y[:,2],reduction='none')
        loss = cross_entropy + 5.*mse[:,0] + 5.*mse[:,1]

#         loss[loss != loss] = 0 ##gets rid of NaNs and infs
#         loss[loss == float("Inf")] = 0

        loss = torch.sum(loss)
        return loss

    def perepoch(self,e,train_iter,test_iter,val_iter):

        now = datetime.datetime.now()

        print('=======epoch ' + str(e) + '=======')
        self.model.train()
        total_train_loss = 0.
        train_data_size = 0
        num_train_batches = len(train_iter)
        for i,b in enumerate(train_iter):

            if self.mode == "class" or self.mode == "multiclass":
                img_arr,seg_arr,weight_arr = (b['img'],b['seg'],b['weight'])
                img_arr,seg_arr,weight_arr = self.pad_fn(img_arr),self.pad_fn(seg_arr),self.pad_fn(weight_arr)
                img_arr,seg_arr,weight_arr = self.class_aug(img_arr,seg_arr,weight_arr)
                num_pixels = np.prod(img_arr.shape)
                train_data_size += num_pixels

                seg_arr,weight_arr = seg_arr[:,0],weight_arr[:,0]
                x = torch.Tensor(img_arr.astype(float))
                y = torch.LongTensor(seg_arr)
                weight_arr = torch.Tensor(weight_arr)
                if self.gpuon:
                    x = x.cuda()
                    y = y.cuda()
                    weight_arr = weight_arr.cuda()
                loss = float(self.train(x,y,weight_arr))
                del weight_arr

            elif self.mode == "cellpose":
                img_arr,seg_arr,y_grad_arr,x_grad_arr = (b['img'],b['seg'],b['y_grad'],b['x_grad'])
                img_arr,seg_arr,y_grad_arr,x_grad_arr = self.pad_fn(img_arr),self.pad_fn(seg_arr),self.pad_fn(y_grad_arr),self.pad_fn(x_grad_arr)
                img_arr,seg_arr,y_grad_arr,x_grad_arr = self.cellpose_aug(img_arr,seg_arr,y_grad_arr,x_grad_arr)
                num_pixels = np.prod(img_arr.shape)
                train_data_size += num_pixels

                y_grad_arr,x_grad_arr,seg_arr = y_grad_arr[:,0],x_grad_arr[:,0],seg_arr[:,0]
                x = torch.Tensor(img_arr.astype(float))
                y = np.stack([y_grad_arr,x_grad_arr,seg_arr],axis=1)
                y = torch.Tensor(y)

                if self.gpuon:
                    x = x.cuda()
                    y = y.cuda()
                loss = float(self.cellpose_train(x,y))

            else:
                raise

            total_train_loss += loss
            del x
            del y
            del loss
            torch.cuda.empty_cache()

        self.scheduler.step()
        avgtrainnll = total_train_loss/train_data_size
        print('Mean Train NLL: ' + str(avgtrainnll))
        self.model.eval()

        total_val_loss = 0.
        val_data_size = 0
        for i,b in enumerate(val_iter):
            if self.mode == "class" or self.mode == "multiclass":
                img_arr,seg_arr,weight_arr = (b['img'],b['seg'],b['weight'])
                img_arr,seg_arr,weight_arr = self.pad_fn(img_arr),self.pad_fn(seg_arr),self.pad_fn(weight_arr)
                num_pixels = np.prod(img_arr.shape)
                val_data_size += num_pixels

                seg_arr,weight_arr = seg_arr[:,0],weight_arr[:,0]
                x = torch.Tensor(img_arr.astype(float))
                y = torch.LongTensor(seg_arr)
                weight_arr = torch.Tensor(weight_arr)
                if self.gpuon:
                    x = x.cuda()
                    y = y.cuda()
                    weight_arr = weight_arr.cuda()
                loss = float(self.test(x,y,weight_arr))
                del weight_arr

            elif self.mode == "cellpose":
                img_arr,seg_arr,y_grad_arr,x_grad_arr = (b['img'],b['seg'],b['y_grad'],b['x_grad'])
                img_arr,seg_arr,y_grad_arr,x_grad_arr = self.pad_fn(img_arr),self.pad_fn(seg_arr),self.pad_fn(y_grad_arr),self.pad_fn(x_grad_arr)
                y_grad_arr,x_grad_arr,seg_arr = y_grad_arr[:,0],x_grad_arr[:,0],seg_arr[:,0]
                num_pixels = np.prod(img_arr.shape)
                val_data_size += num_pixels

                x = torch.Tensor(img_arr.astype(float))
                y = np.stack([y_grad_arr,x_grad_arr,seg_arr],axis=1)
                y = torch.Tensor(y)
                if self.gpuon:
                    x = x.cuda()
                    y = y.cuda()
                loss = float(self.cellpose_test(x,y))
            else:
                raise

            total_val_loss += loss

            del x
            del y
            del loss
            torch.cuda.empty_cache()

        avgvalnll = total_val_loss/val_data_size
        print('Mean Val NLL: ' + str(avgvalnll))

        total_test_loss = 0.
        test_data_size = 0
        for i,b in enumerate(test_iter):
            if self.mode == "class" or self.mode == "multiclass":
                img_arr,seg_arr,weight_arr = (b['img'],b['seg'],b['weight'])
                img_arr,seg_arr,weight_arr = self.pad_fn(img_arr),self.pad_fn(seg_arr),self.pad_fn(weight_arr)
                num_pixels = np.prod(img_arr.shape)
                test_data_size += num_pixels

                seg_arr,weight_arr = seg_arr[:,0],weight_arr[:,0]
                x = torch.Tensor(img_arr.astype(float))
                y = torch.LongTensor(seg_arr)
                weight_arr = torch.Tensor(weight_arr)
                if self.gpuon:
                    x = x.cuda()
                    y = y.cuda()
                    weight_arr = weight_arr.cuda()
                loss = float(self.test(x,y,weight_arr))
                del weight_arr

            elif self.mode == "cellpose":
                img_arr,seg_arr,y_grad_arr,x_grad_arr = (b['img'],b['seg'],b['y_grad'],b['x_grad'])
                img_arr,seg_arr,y_grad_arr,x_grad_arr = self.pad_fn(img_arr),self.pad_fn(seg_arr),self.pad_fn(y_grad_arr),self.pad_fn(x_grad_arr)
                num_pixels = np.prod(img_arr.shape)
                test_data_size += num_pixels

                y_grad_arr,x_grad_arr,seg_arr = y_grad_arr[:,0],x_grad_arr[:,0],seg_arr[:,0]
                x = torch.Tensor(img_arr.astype(float))
                y = np.stack([y_grad_arr,x_grad_arr,seg_arr],axis=1)
                y = torch.Tensor(y)
                if self.gpuon:
                    x = x.cuda()
                    y = y.cuda()
                loss = float(self.cellpose_test(x,y))
            else:
                raise

            total_test_loss += loss

            del x
            del y
            del loss
            torch.cuda.empty_cache()

        avgtestnll = total_test_loss/test_data_size
        print('Mean Test NLL: ' + str(avgtestnll))

        entry = [[self.model_number,self.mode,self.batch_size,self.layers,self.hidden_size,self.lr,self.momentum,self.weight_decay,\
                  self.dropout,self.W0,self.Wsigma,self.warm_epochs,self.cool_epochs,e,avgtrainnll,avgvalnll,avgtestnll,str(now)]]

        df_out = pd.DataFrame(data=entry,columns=['Model #',"Mode",'Batch Size','Layers','Hidden Size','Learning Rate','Momentum','Weight Decay',\
                            'Dropout',"W0","Wsigma","Warm Epochs","Cool Epochs",'Epoch','Train Loss','Val Loss','Test Loss','Date/Time'])
        df_out = df_out.set_index(['Model #','Epoch'], drop=True, append=False, inplace=False)
        df_out = df_out.sort_index()
        return df_out

    def write_metadata(self,filepath,iomode,df_out):
        meta_handle = pandas_hdf5_handler(filepath)
        if os.path.exists(filepath):
            ind = df_out.index[0]
            df_in = meta_handle.read_df("data")
            df_mask = ~df_in.index.isin([ind])
            df_in = df_in[df_mask]
            df_out = pd.concat([df_in, df_out])
        meta_handle.write_df("data",df_out)

    def get_class_fscore(self,iterator,mask_label_dim=1):
        y_true = []
        y_pred = []
        for i,b in enumerate(iterator):
            img_arr,y = (b['img'],b['gt'])
            x = torch.Tensor(img_arr.astype(float))
            if self.gpuon:
                x = x.cuda()
            fx = self.model.forward(x).detach().cpu().numpy()

            y_true.append(y[:,0]) # N,H,W
            y_pred.append(get_class_labels(fx,mask_label_dim=mask_label_dim)) # N,H,W

            del x
            del y
            torch.cuda.empty_cache()

        y_true = np.concatenate(y_true,axis=0)
        y_pred = np.concatenate(y_pred,axis=0)

        all_f_scores = []
        for i in range(y_true.shape[0]):
            _,_,f_score = object_f_scores(y_true[i],y_pred[i])
            all_f_scores += f_score.tolist()
        all_f_scores = np.array(all_f_scores)
        all_f_scores = all_f_scores[~np.isnan(all_f_scores)]

        return all_f_scores

    def get_cellpose_fscore(self,iterator,mask_label_dim=1):
        y_true = []
        y_pred = []
        for i,b in enumerate(iterator):
            img_arr,y = (b['img'],b['gt'])
            x = torch.Tensor(img_arr.astype(float))
            if self.gpuon:
                x = x.cuda()
            fx = self.model.forward(x).detach().cpu().numpy()

            y_true.append(y[:,0]) # N,H,W
            y_pred.append(get_cellpose_labels(fx,step_size=1.,n_iter=10))

            del x
            del y
            torch.cuda.empty_cache()

        y_true = np.concatenate(y_true,axis=0)
        y_pred = np.concatenate(y_pred,axis=0)

        all_f_scores = []
        for i in range(y_true.shape[0]):
            _,_,f_score = object_f_scores(y_true[i],y_pred[i])
            all_f_scores += f_score.tolist()
        all_f_scores = np.array(all_f_scores)
        all_f_scores = all_f_scores[~np.isnan(all_f_scores)]

        return all_f_scores

    def train_model(self):
        timestamp = datetime.datetime.now()
        start = time.time()
        writedir(self.nndatapath + "/models", overwrite=False)
        self.removefile(self.nndatapath + "/models/training_metadata_" + str(self.model_number) + ".hdf5")

        train_data = SegmentationDataset(self.nndatapath + "train.hdf5",mode=self.mode,W0=self.W0,Wsigma=self.Wsigma)
        test_data = SegmentationDataset(self.nndatapath + "test.hdf5",mode=self.mode,W0=self.W0,Wsigma=self.Wsigma)
        val_data = SegmentationDataset(self.nndatapath + "val.hdf5",mode=self.mode,W0=self.W0,Wsigma=self.Wsigma)

        for e in range(0,self.numepochs):
            train_iter = DataLoader(train_data,batch_size=self.batch_size,shuffle=False,collate_fn=numpy_collate)
            test_iter = DataLoader(test_data,batch_size=self.batch_size,shuffle=False,collate_fn=numpy_collate)
            val_iter = DataLoader(val_data,batch_size=self.batch_size,shuffle=False,collate_fn=numpy_collate)
            df_out = self.perepoch(e,train_iter,test_iter,val_iter)

            self.write_metadata(self.nndatapath + "/models/training_metadata_" + str(self.model_number) + ".hdf5","w",df_out)
        end = time.time()
        time_elapsed = (end-start)/60.
        torch.save(self.model.state_dict(), self.nndatapath + "/models/" + str(self.model_number) + ".pt")

        try:
            if self.mode == "class" or self.mode == "multiclass":
                val_f = self.get_class_fscore(val_iter)
                test_f = self.get_class_fscore(test_iter)
            elif self.mode == "cellpose":
                val_f = self.get_cellpose_fscore(val_iter)
                test_f = self.get_cellpose_fscore(test_iter)
        except:
            print("Failed to compute F-scores")
            val_f = [np.NaN]
            test_f = [np.NaN]

        meta_handle = pandas_hdf5_handler(self.nndatapath + "/metadata.hdf5")
        traindf = meta_handle.read_df("train",read_metadata=True)
        valdf = meta_handle.read_df("val",read_metadata=True)
        testdf = meta_handle.read_df("test",read_metadata=True)
        trainmeta = traindf.metadata
        valmeta = valdf.metadata
        testmeta = testdf.metadata
        experiment_name = trainmeta["nndataset"]["experimentname"]

        train_data_list = traindf.index.get_level_values(0).unique().tolist()
        val_data_list = valdf.index.get_level_values(0).unique().tolist()
        test_data_list = testdf.index.get_level_values(0).unique().tolist()

        train_orgs = [trainmeta[data_name]["global"]["Organism"] for data_name in train_data_list]
        train_micros = [trainmeta[data_name]["global"]["Microscope"] for data_name in train_data_list]
        val_orgs = [trainmeta[data_name]["global"]["Organism"] for data_name in val_data_list]
        val_micros = [trainmeta[data_name]["global"]["Microscope"] for data_name in val_data_list]
        test_orgs = [trainmeta[data_name]["global"]["Organism"] for data_name in test_data_list]
        test_micros = [trainmeta[data_name]["global"]["Microscope"] for data_name in test_data_list]

        train_ttl_img = len(traindf)
        val_ttl_img = len(valdf)
        test_ttl_img = len(testdf)

        train_loss,val_loss,test_loss = df_out['Train Loss'].tolist()[0],df_out['Val Loss'].tolist()[0],df_out['Test Loss'].tolist()[0]

        entry = [[experiment_name,self.model_number,self.mode,train_data_list,train_orgs,train_micros,train_ttl_img,val_data_list,val_orgs,val_micros,val_ttl_img,\
                  test_data_list,test_orgs,test_micros,test_ttl_img,self.batch_size,self.layers,self.hidden_size,self.lr,self.momentum,\
                  self.weight_decay,self.dropout,self.W0,self.Wsigma,self.warm_epochs,self.cool_epochs,train_loss,val_loss,val_f,\
                  test_loss,test_f,str(timestamp),self.numepochs,time_elapsed]]

        df_out = pd.DataFrame(data=entry,columns=['Experiment Name','Model #','NN Mode','Train Datasets','Train Organisms','Train Microscopes','Train # Images',\
                                                  'Val Datasets','Val Organisms','Val Microscopes','Val # Images','Test Datasets','Test Organisms',\
                                                  'Test Microscopes','Test # Images','Batch Size','Layers','Hidden Size','Learning Rate','Momentum',\
                                                  'Weight Decay','Dropout',"W0 Weight (if applicable)","W Sigma (if applicable)","Warm Epochs","Cool Epochs",\
                                                  'Train Loss','Val Loss','Val F1 Cell Scores','Test Loss','Test F1 Cell Scores','Date/Time','# Epochs',\
                                                  'Training Time (mins)'])

        df_out = df_out.set_index(['Experiment Name','Model #'], drop=True, append=False, inplace=False)
        df_out = df_out.sort_index()

        metalock = hdf5lock(self.nndatapath + "/model_metadata.hdf5",updateperiod=5.)
        metalock.lockedfn(self.write_metadata,"w",df_out)

class GridSearch:
    def __init__(self,nndatapath,numepochs=50):
        self.nndatapath = nndatapath
        self.numepochs = numepochs

    def display_grid(self):
        meta_handle = pandas_hdf5_handler(self.nndatapath + "/metadata.hdf5")
        trainmeta = meta_handle.read_df("train",read_metadata=True).metadata["nndataset"]
        W0_list,Wsigma_list = trainmeta["W0_list"],trainmeta["Wsigma_list"]

        self.tab_dict = {'Mode':["class","multiclass","cellpose"],'Batch Size:':[5, 10, 25, 50],'Layers:':[2, 3, 4],\
           'Hidden Size:':[16, 32, 64],'Learning Rate:':[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4],\
           'Momentum:':[0.9, 0.95, 0.99],'Weight Decay:':[0.0001,0.0005, 0.001],\
           'Dropout:':[0., 0.3, 0.5, 0.7], 'W0:':W0_list, 'Wsigma':Wsigma_list,\
            'Warm Epochs':[1,5,10,20],'Cool Epochs':[10,20,50,100]}

        children = [ipyw.SelectMultiple(options=val,value=(val[0],),description=key,disabled=False) for key,val in self.tab_dict.items()]
        self.tab = ipyw.Tab()
        self.tab.children = children
        for i,key in enumerate(self.tab_dict.keys()):
            self.tab.set_title(i, key[:-1])
        return self.tab

    def get_grid_params(self):
        self.grid_dict = {child.description:child.value for child in self.tab.children}
        print("======== Grid Params ========")
        for key,val in self.grid_dict.items():
            print(key + " " + str(val))

    def generate_pyscript(self,run_idx,grid_params):
        import_line = "import paulssonlab.deaton.trenchripper.trenchripper as tr"
        trainer_line = "nntrainer = tr.unet.UNet_Trainer(\"" + self.nndatapath + "/\"," + str(run_idx) + \
        ",\"" + str(grid_params[0]) + "\",gpuon=True,numepochs=" + str(self.numepochs) + ",batch_size=" + str(grid_params[1])+",layers=" + \
        str(grid_params[2])+",hidden_size=" + str(grid_params[3]) + ",lr=" + str(grid_params[4]) + \
        ",momentum=" + str(grid_params[5]) + ",weight_decay=" + str(grid_params[6])+",dropout="+str(grid_params[7]) + \
        ",W0=" + str(grid_params[8]) + ",Wsigma=" + str(grid_params[9]) + ",warm_epochs=" + str(grid_params[10]) + ",cool_epochs=" + \
        str(grid_params[11]) + ")"
        train_line = "nntrainer.train_model()"
        pyscript = "\n".join([import_line,trainer_line,train_line])
        with open(self.nndatapath + "/models/scripts/" + str(run_idx) + ".py", "w") as scriptfile:
            scriptfile.write(pyscript)

    def generate_sbatchscript(self,run_idx,hours,cores,mem,gres):
        shebang = "#!/bin/bash"
        core_line = "#SBATCH -c " + str(cores)
        hour_line = "#SBATCH -t " + str(hours) + ":00:00"
        gpu_lines = "#SBATCH -p gpu\n#SBATCH --gres=" + gres
        mem_line = "#SBATCH --mem=" + mem
        report_lines = "#SBATCH -o " + self.nndatapath + "/models/scripts/" + str(run_idx) +\
        ".out\n#SBATCH -e " + self.nndatapath + "/models/scripts/" + str(run_idx) + ".err\n"

        run_line = "python -u " + self.nndatapath + "/models/scripts/" + str(run_idx) + ".py"

        sbatchscript = "\n".join([shebang,core_line,hour_line,gpu_lines,mem_line,report_lines,run_line])
        with open(self.nndatapath + "/models/scripts/" + str(run_idx) + ".sh", "w") as scriptfile:
            scriptfile.write(sbatchscript)

    def run_sbatchscript(self,run_idx):
        cmd = ["sbatch",self.nndatapath + "/models/scripts/" + str(run_idx) + ".sh"]
        subprocess.run(cmd)

    def run_grid_search(self,hours=12,cores=2,mem="8G",gres="gpu:1"):

        grid_keys = self.grid_dict.keys()
        grid_combinations = list(itertools.product(*list(self.grid_dict.values())))
        writedir(self.nndatapath + "/models",overwrite=True)
        writedir(self.nndatapath + "/models/scripts",overwrite=True)

        self.run_indices = []

        for run_idx,grid_params in enumerate(grid_combinations):
            self.generate_pyscript(run_idx,grid_params)
            self.generate_sbatchscript(run_idx,hours,cores,mem,gres)
            self.run_sbatchscript(run_idx)
            self.run_indices.append(run_idx)

    def cancel_all_runs(self,username):
        for run_idx in self.run_indices:
            cmd = ["scancel","-p","gpu","--user=" + username]
            subprocess.Popen(cmd,shell=True,stdin=None,stdout=None,stderr=None,close_fds=True)

class TrainingVisualizer:
    def __init__(self,trainpath,modeldbpath):
        self.trainpath = trainpath
        self.modelpath = trainpath + "/models"
        self.modeldfpath = trainpath + "/model_metadata.hdf5"
        self.modeldbpath = modeldbpath
        self.paramdbpath = modeldbpath+"/Parameters"
        self.update_dfs()
        if os.path.exists(self.modeldfpath):
            self.models_widget = qgrid.show_grid(self.model_df.sort_index())

    def update_dfs(self):
        df_idx_list = []
        for path in os.listdir(self.modelpath):
            if "training_metadata" in path:
                df_idx = int(path.split("_")[-1][:-5])
                df_idx_list.append(df_idx)
        df_list = []
        for df_idx in df_idx_list:
            dfpath = self.modelpath + "/training_metadata_" + str(df_idx) + ".hdf5"
            df_handle = pandas_hdf5_handler(dfpath)
            df = df_handle.read_df("data")
            df_list.append(copy.deepcopy(df))
            del df
        self.train_df = pd.concat(df_list)
        if os.path.exists(self.modeldfpath):
            modeldfhandle = pandas_hdf5_handler(self.modeldfpath)
            self.model_df = modeldfhandle.read_df("data").sort_index()

    def select_df_columns(self,selected_columns):
        df = copy.deepcopy(self.model_df)
        for column in df.columns.tolist():
            if column not in selected_columns:
                df = df.drop(column, 1)
        self.model_widget = qgrid.show_grid(df)

    def inter_df_columns(self):
        column_list = self.model_df.columns.tolist()
        inter = ipyw.interactive(self.select_df_columns,{"manual":True},selected_columns=ipyw.SelectMultiple(options=column_list,description='Columns to Display:',disabled=False))
        display(inter)

    def handle_filter_changed(self,event,widget):
        df = widget.get_changed_df().sort_index()

        all_model_indices = self.train_df.index.get_level_values("Model #").unique().tolist()
        current_model_indices = df.index.get_level_values("Model #").unique().tolist()

        all_epochs = []
        all_loss = []
        for model_idx in all_model_indices:
            if model_idx in current_model_indices:
                filter_df = df.loc[model_idx]
                epochs,loss = (filter_df.index.get_level_values("Epoch").tolist(),filter_df[self.losskey].tolist())
                all_epochs += epochs
                all_loss += loss
                self.line_dict[model_idx].set_data(epochs,loss)
                self.line_dict[model_idx].set_label(str(model_idx))
            else:
                epochs_empty,loss_empty = ([],[])
                self.line_dict[model_idx].set_data(epochs_empty,loss_empty)
                self.line_dict[model_idx].set_label('_nolegend_')

        self.ax.set_xlim(min(all_epochs), max(all_epochs)+1)
        self.ax.set_ylim(0, max(all_loss)*1.1)
        self.ax.legend()
        self.fig.canvas.draw()

    def inter_plot_loss(self,losskey):
        self.losskey = losskey
        self.fig,self.ax = plt.subplots()
        self.grid_widget = qgrid.show_grid(self.train_df.sort_index())
        current_df = self.grid_widget.get_changed_df()

        self.line_dict = {}
        for model_idx in current_df.index.get_level_values("Model #").unique().tolist():
            filter_df = current_df.loc[model_idx]
            epochs,loss = (filter_df.index.get_level_values("Epoch").tolist(),filter_df[losskey].tolist())
            line, = self.ax.plot(epochs,loss,label=str(model_idx))
            self.line_dict[model_idx] = line

        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel(losskey)
        self.ax.legend()

    def export_models(self):
        writedir(self.modeldbpath,overwrite=False)
        writedir(self.modeldbpath+"/Parameters",overwrite=False)
        modeldbhandle = pandas_hdf5_handler(self.modeldbpath + "/Models.hdf5")
        if "Models.hdf5" in os.listdir(self.modeldbpath):
            old_df = modeldbhandle.read_df("data")
            current_df = self.models_widget.get_changed_df()
            current_df = pd.concat([old_df, current_df])
        else:
            current_df = self.models_widget.get_changed_df()
        modeldbhandle.write_df("data",current_df)

        indices = current_df.index.tolist()
        exp_names = [str(item[0]) for item in indices]
        model_numbers = [str(item[1]) for item in indices]
        dates = [item.replace(" ","_") for item in current_df["Date/Time"].tolist()]

        for i in range(len(model_numbers)):
            exp_name,model_number,date = (exp_names[i],model_numbers[i],dates[i])
            shutil.copyfile(self.modelpath + "/" + str(model_number) + ".pt",self.paramdbpath+"/"+exp_name+"_"+model_number+"_"+date+".pt")

class UNet_Segmenter:
    def __init__(self,headpath,modeldbpath):#,min_obj_size=20):#,cell_threshold_scaling=1.,border_threshold_scaling=1.,\
             #    layers=3,hidden_size=64,batch_size=100,gpuon=False):

        self.headpath = headpath
        self.kymopath = headpath + "/kymograph"
        self.nnoutputpath = headpath + "/phasesegmentation"
        self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        globaldf = self.meta_handle.read_df("global",read_metadata=True)
        self.all_channels = globaldf.metadata['channels']

        self.modeldbpath = modeldbpath
        self.update_dfs()

        if os.path.exists(self.modeldbpath):
            self.models_widget = qgrid.show_grid(self.models_df.sort_index())

#         self.min_obj_size = min_obj_size
#         self.border_threshold_scaling = border_threshold_scaling
#         self.gpuon = gpuon
#         self.layers = layers
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size

    def update_dfs(self):
        dfpath = self.modeldbpath + "/Models.hdf5"
        if os.path.exists(dfpath):
            df_handle = pandas_hdf5_handler(dfpath)
            self.models_df = df_handle.read_df("data")

    def select_df_columns(self,selected_columns):
        df = copy.deepcopy(self.models_df)
        for column in df.columns.tolist():
            if column not in selected_columns:
                df = df.drop(column, 1)
        self.model_widget = qgrid.show_grid(df)

    def inter_df_columns(self):
        column_list = self.models_df.columns.tolist()
        inter = ipyw.interactive(self.select_df_columns,{"manual":True},selected_columns=ipyw.SelectMultiple(options=column_list,description='Columns to Display:',disabled=False))
        display(inter)

    def select_model(self):
        current_df = self.model_widget.get_changed_df()
        print(current_df)

    def choose_seg_channel(self,seg_channel):
        self.seg_channel = seg_channel

#     def prepare_data(self,fov_num):
#         writedir(self.nnoutputpath,overwrite=False)
#         writepath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
#         img_path = self.kymopath + "/kymo_" + str(fov_num) + ".hdf5"
#         with h5py.File(writepath,"w") as outfile:
#             with h5py.File(img_path,"r") as infile:
#                 keys = list(infile.keys())
#                 ex_data = infile[keys[0]+"/"+self.seg_channel]
#                 out_shape = (len(keys)*ex_data.shape[2],1,ex_data.shape[0],ex_data.shape[1])
#                 chunk_shape = (1,1,out_shape[2],out_shape[3])
#                 img_handle = outfile.create_dataset("img",out_shape,chunks=chunk_shape,dtype=float)

#                 for i,trenchid in enumerate(keys):
#                     img_arr = infile[trenchid+"/"+self.seg_channel][:]
#                     img_arr = np.moveaxis(img_arr,(0,1,2),(1,2,0))
#                     img_arr = np.expand_dims(img_arr,1)
#                     img_arr = img_arr.astype(float)

#                     img_handle[i*ex_data.shape[2]:(i+1)*ex_data.shape[2]] = img_arr

#     def segment(self,fov_num):
#         torch.cuda.empty_cache()
#         self.model = UNet(1,3,layers=self.layers,hidden_size=self.hidden_size,withsoftmax=True)

#         if self.gpuon:
#             device = torch.device("cuda")
#             self.model.load_state_dict(torch.load(self.paramspath))
#             self.model.to(device)
#         else:
#             device = torch.device('cpu')
#             self.model.load_state_dict(torch.load(self.paramspath, map_location=device))
#         self.model.eval()

#         inputpath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
#         outputpath = self.nnoutputpath + "/nnoutput_" + str(fov_num) + ".hdf5"
#         with h5py.File(inputpath,"r") as infile:
#             out_shape = tuple((infile["img"].shape[0],3,infile["img"].shape[2],infile["img"].shape[3]))
#             chunk_shape = infile["img"].chunks
#         data = SegmentationDataset(inputpath,training=False)
#         data_iter = DataLoader(data,batch_size=self.batch_size,shuffle=False) #careful
#         with h5py.File(outputpath,"w") as outfile:
#             img_handle = outfile.create_dataset("img",out_shape,chunks=chunk_shape,dtype=float)
#             print(len(data_iter))
#             for i,b in enumerate(data_iter):
#                 print(i)
#                 x = torch.Tensor(b['img'].numpy())
#                 if self.gpuon:
#                     x = x.cuda()
#                 fx = self.model.forward(x) #N,3,y,x
#                 img_handle[i*self.batch_size:(i+1)*self.batch_size] = fx.cpu().data.numpy()

#     def postprocess(self,fov_num):
#         nninputpath = self.nnoutputpath + "/nninput_" + str(fov_num) + ".hdf5"
#         nnoutputpath = self.nnoutputpath + "/nnoutput_" + str(fov_num) + ".hdf5"
#         segpath = self.nnoutputpath + "/seg_" + str(fov_num) + ".hdf5"
#         kymopath = self.kymopath + "/kymo_" + str(fov_num) + ".hdf5"
#         with h5py.File(kymopath,"r") as kymofile:
#             trench_num = len(kymofile.keys())
#             trenchids = list(kymofile.keys())
#         with h5py.File(segpath,"w") as outfile:
#             with h5py.File(nnoutputpath,"r") as infile:
#                 num_img = infile["img"].shape[0]
#                 y_shape,x_shape = (infile["img"].shape[2],infile["img"].shape[3])
#                 timepoints = int(num_img/trench_num)
#                 for trench in range(trench_num):
#                     print(trench)
#                     trenchid = trenchids[trench]
#                     trench_data = infile["img"][trench*timepoints:(trench+1)*timepoints] #t,3,y,x
#                     trench_output = []
#                     for t in range(timepoints):
#                         cell_otsu = sk.filters.threshold_otsu(trench_data[t,1])*self.cell_threshold_scaling
#                         border_otsu = sk.filters.threshold_otsu(trench_data[t,2])*self.border_threshold_scaling
#                         cell_mask = trench_data[t,1]>cell_otsu
#                         border_mask = trench_data[t,2]>border_otsu
#                         trench_arr = cell_mask*(~border_mask)
#                         del cell_mask
#                         del border_mask
#                         trench_arr = sk.morphology.remove_small_objects(trench_arr,min_size=self.min_obj_size)
#                         trench_output.append(trench_arr)
#                         del trench_arr
#                     trench_output = np.array(trench_output)
#                     trench_output = np.moveaxis(trench_output,(0,1,2),(2,0,1))
#                     outdset = outfile.create_dataset(trenchid, data=trench_output, chunks=(y_shape,x_shape,1), dtype=bool)
# #         os.remove(nninputpath)
# #         os.remove(nnoutputpath)
