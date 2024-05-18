# fmt: off
import h5py

import numpy as np
import pandas as pd
import skimage as sk

import dask
import dask.dataframe as dd

from .daskutils import make_parquet_index, lookup_parquet_index, to_parquet_checkpoint


class projection_handler:
    def __init__(self, headpath, projection_fn=np.nanmean, projection_name="Mean"):
        self.headpath = headpath
        self.projection_fn = projection_fn
        self.projection_name = projection_name

        kymo_meta = dd.read_parquet(
            self.headpath + "/kymograph/metadata",
            engine="pyarrow",
            calculate_divisions=True,
        )
        self.file_indices = sorted(kymo_meta["File Index"].unique().compute().tolist())
        kymo_meta_parq_sorted = kymo_meta.reset_index().set_index("File Parquet Index",sorted=True)
        self.kymo_divisions = kymo_meta_parq_sorted.divisions

    def projection_values_to_df(self, projection_values, projection_name):
        ## Use list comprehension to create a list of tuples for the multi-index
        multi_index = [
            (i, j)
            for i in range(len(projection_values))
            for j in range(len(projection_values[i]))
        ]
        # Create a pandas DataFrame with the multi-index and data
        df = pd.DataFrame(
            data=[x for y in projection_values for x in y],
            index=multi_index,
            columns=[projection_name],
        )
        # Convert the tuples index to a multiindex
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["Cell Projection ID", "Projection Index"]
        )
        return df

    def compute_projections(
        self,
        seg_arr,
        img_arr,
        orientation,
        *projection_fn_args,
        projection_name="Mean",
        projection_fn=np.nanmean,
        **projection_fn_kwargs
    ):
        ## getting segment regionprops, consistent with tracking
        if orientation == 0:
            rps = sk.measure.regionprops(seg_arr, img_arr)
        else:
            rps = sk.measure.regionprops(seg_arr[::-1], img_arr[::-1])
        if len(rps) > 0:
            # crop out bbox, make them square using padding to keep rotation logic simple (since centroids can be fixed)
            bbox_seg_arrs = [rp.image for rp in rps]
            bbox_img_arrs = [rp.image_intensity for rp in rps]

            bbox_shapes = np.array([bbox_img.shape for bbox_img in bbox_img_arrs])
            max_bbox_dim = np.max(bbox_shapes, axis=1)
            padding_arr = max_bbox_dim[:, np.newaxis] - bbox_shapes
            padding_left = padding_arr // 2
            padding_right = padding_arr // 2 + padding_arr % 2
            padding_arr = np.stack([padding_left, padding_right], axis=1)
            padding_list = [
                [tuple(padding_arr[i, :, 0]), tuple(padding_arr[i, :, 1])]
                for i in range(padding_arr.shape[0])
            ]

            bbox_seg_arrs = [
                np.pad(bbox_seg_arr, padding_list[i])
                for i, bbox_seg_arr in enumerate(bbox_seg_arrs)
            ]
            bbox_img_arrs = [
                np.pad(bbox_img_arr, padding_list[i])
                for i, bbox_img_arr in enumerate(bbox_img_arrs)
            ]

            # # get the angle of the major axis
            radians = [rp.orientation for rp in rps]
            angles = [-(rp.orientation * 180) / np.pi for rp in rps]
            # # get centroids
            centroids = np.array([rp.centroid for rp in rps])
            major_axis_lengths = np.array([rp.major_axis_length for rp in rps])
            minor_axis_lengths = np.array([rp.minor_axis_length for rp in rps])
            # # rotate the image by the angle of the major axis
            rotated_bbox_img_arrs = [
                sk.transform.rotate(
                    bbox_img_arr, angles[i], resize=False, preserve_range=True
                )
                for i, bbox_img_arr in enumerate(bbox_img_arrs)
            ]
            rotated_bbox_seg_arrs = [
                sk.transform.rotate(
                    bbox_seg_arr, angles[i], resize=False, preserve_range=True
                )
                for i, bbox_seg_arr in enumerate(bbox_seg_arrs)
            ]

            # apply the boolean mask to the value array
            rotated_bbox_masked_img_arrs = [
                np.ma.masked_array(
                    rotated_bbox_img_arr,
                    mask=~rotated_bbox_seg_arrs[i],
                    fill_value=np.NaN,
                ).filled()
                for i, rotated_bbox_img_arr in enumerate(rotated_bbox_img_arrs)
            ]

            # filter out rows and columns with only NaN
            col_val_counts = [
                np.sum(~np.isnan(rotated_bbox_masked_img_arr), axis=0)
                for rotated_bbox_masked_img_arr in rotated_bbox_masked_img_arrs
            ]
            row_val_counts = [
                np.sum(~np.isnan(rotated_bbox_masked_img_arr), axis=1)
                for rotated_bbox_masked_img_arr in rotated_bbox_masked_img_arrs
            ]

            col_val_mask = [col_val_count > 1 for col_val_count in col_val_counts]
            row_val_mask = [row_val_count > 1 for row_val_count in row_val_counts]

            minor_projection_values = [
                projection_fn(
                    rotated_bbox_masked_img_arr[:, col_val_mask[i]],
                    *projection_fn_args,
                    axis=0,
                    **projection_fn_kwargs
                )
                for i, rotated_bbox_masked_img_arr in enumerate(
                    rotated_bbox_masked_img_arrs
                )
            ]
            major_projection_values = [
                projection_fn(
                    rotated_bbox_masked_img_arr[row_val_mask[i]],
                    *projection_fn_args,
                    axis=1,
                    **projection_fn_kwargs
                )
                for i, rotated_bbox_masked_img_arr in enumerate(
                    rotated_bbox_masked_img_arrs
                )
            ]

            major_projection_df, minor_projection_df = self.projection_values_to_df(
                major_projection_values, projection_name
            ), self.projection_values_to_df(minor_projection_values, projection_name)

            cell_length = major_projection_df.groupby("Cell Projection ID").size()
            cell_length.name = "Projection Length"
            cell_length = cell_length.to_frame()
            cell_length["Length"] = major_axis_lengths

            major_projection_df = major_projection_df.join(
                cell_length, on="Cell Projection ID"
            )

            cell_width = minor_projection_df.groupby("Cell Projection ID").size()
            cell_width.name = "Projection Length"
            cell_width = cell_width.to_frame()
            cell_width["Length"] = minor_axis_lengths

            minor_projection_df = minor_projection_df.join(
                cell_width, on="Cell Projection ID"
            )

            return major_projection_df, minor_projection_df

        else:
            return False

    def compute_file_projections_test(
        self,
        file_idx,
        headpath,
        intensity_channel,
        *projection_fn_args,
        seg_key="fluorsegmentation",
        projection_name="Mean",
        projection_fn=np.nanpercentile,
        **projection_fn_kwargs
    ):
        with h5py.File(headpath + "/kymograph/kymograph_" + str(file_idx) + ".hdf5", "r") as infile:
            intensity_data = infile[intensity_channel][:]
        with h5py.File(headpath + "/" + seg_key + "/segmentation_" + str(file_idx) + ".hdf5", "r") as infile:
            seg_data = infile["data"][:]

        orientation_conv_dict = {"top":0,"bottom":1}
        index_precisions=np.array([4,4,0])
        index_columns=["File Index","File Trench Index","File Trench Index"]
        
        index_lookup = {"File Index":file_idx}
        kymo_idx_i,kymo_idx_f = lookup_parquet_index(index_lookup,index_precisions,index_columns)
        kymo_df = dd.read_parquet(headpath + "/kymograph",scheduler="threads")
        kymo_df = kymo_df.reset_index().set_index("File Parquet Index",sorted=True,\
                                                  divisions=self.kymo_divisions,scheduler="threads")
        
        selected_kymo_df = kymo_df.loc[kymo_idx_i:kymo_idx_f].compute(scheudler="threads")
        orientation_df = selected_kymo_df.groupby("File Trench Index").first()[["lane orientation"]]
        orientation_dict = orientation_df["lane orientation"].apply(lambda x: orientation_conv_dict[x]).to_dict()
        del selected_kymo_df
        del orientation_df
        
        projection_output = []
        for k in range(seg_data.shape[0]):
            for t in range(seg_data.shape[1]):
                seg_arr = seg_data[k, t]
                img_arr = intensity_data[k, t]
                orientation = orientation_dict[k]

                compute_projections_output = self.compute_projections(
                    seg_arr,
                    img_arr,
                    orientation,
                    *projection_fn_args,
                    projection_name=projection_name,
                    projection_fn=projection_fn,
                    **projection_fn_kwargs
                )
                if compute_projections_output != False:
                    (
                        major_projection_df,
                        minor_projection_df,
                    ) = compute_projections_output

                    minor_projection_df["File Index"] = file_idx
                    minor_projection_df["File Trench Index"] = k
                    minor_projection_df["timepoints"] = t
                    minor_projection_df["Major Axis"] = 0

                    major_projection_df["File Index"] = file_idx
                    major_projection_df["File Trench Index"] = k
                    major_projection_df["timepoints"] = t
                    major_projection_df["Major Axis"] = 1

                    projection_output.append(major_projection_df)
                    projection_output.append(minor_projection_df)

                ### break here to reduce complexity for the test function
                break

        projection_output = pd.concat(projection_output, axis=0).reset_index()
        index_precisions = np.array([4, 4, 2, 2, 0], dtype="uint")
        index_columns = [
            "File Index",
            "File Trench Index",
            "timepoints",
            "Projection Length",
            "Major Axis",
        ]
        projection_output["Projection File Index"] = make_parquet_index(
            projection_output, index_columns, index_precisions
        )
        projection_output = projection_output.set_index(
            "Projection File Index"
        ).sort_index()

        return projection_output

    def compute_file_projections(
        self,
        file_idx,
        headpath,
        intensity_channel,
        *projection_fn_args,
        seg_key="fluorsegmentation",
        projection_name="Mean",
        projection_fn=np.nanpercentile,
        **projection_fn_kwargs
    ):
        with h5py.File(headpath + "/kymograph/kymograph_" + str(file_idx) + ".hdf5", "r") as infile:
            intensity_data = infile[intensity_channel][:]
        with h5py.File(headpath + "/" + seg_key + "/segmentation_" + str(file_idx) + ".hdf5", "r") as infile:
            seg_data = infile["data"][:]

        orientation_conv_dict = {"top":0,"bottom":1}
        index_precisions=np.array([4,4,0])
        index_columns=["File Index","File Trench Index","File Trench Index"]
        
        index_lookup = {"File Index":file_idx}
        kymo_idx_i,kymo_idx_f = lookup_parquet_index(index_lookup,index_precisions,index_columns)
        kymo_df = dd.read_parquet(headpath + "/kymograph",scheduler="threads")
        kymo_df = kymo_df.reset_index().set_index("File Parquet Index",sorted=True,\
                                                  divisions=self.kymo_divisions,scheduler="threads")
        
        selected_kymo_df = kymo_df.loc[kymo_idx_i:kymo_idx_f].compute(scheudler="threads")
        orientation_df = selected_kymo_df.groupby("File Trench Index").first()[["lane orientation"]]
        orientation_dict = orientation_df["lane orientation"].apply(lambda x: orientation_conv_dict[x]).to_dict()
        del selected_kymo_df
        del orientation_df
        
        projection_output = []
        for k in range(seg_data.shape[0]):
            for t in range(seg_data.shape[1]):
                seg_arr = seg_data[k, t]
                img_arr = intensity_data[k, t]
                orientation = orientation_dict[k]

                compute_projections_output = self.compute_projections(seg_arr,img_arr,orientation,*projection_fn_args,
                    projection_name=projection_name,projection_fn=projection_fn,**projection_fn_kwargs)
                if compute_projections_output != False:
                    (major_projection_df,minor_projection_df,) = compute_projections_output

                    minor_projection_df["File Index"] = file_idx
                    minor_projection_df["File Trench Index"] = k
                    minor_projection_df["timepoints"] = t
                    minor_projection_df["Major Axis"] = 0

                    major_projection_df["File Index"] = file_idx
                    major_projection_df["File Trench Index"] = k
                    major_projection_df["timepoints"] = t
                    major_projection_df["Major Axis"] = 1

                    projection_output.append(major_projection_df)
                    projection_output.append(minor_projection_df)

        projection_output = pd.concat(projection_output, axis=0).reset_index()
        index_precisions = np.array([4, 4, 2, 2, 0], dtype="uint")
        index_columns = [
            "File Index",
            "File Trench Index",
            "timepoints",
            "Projection Length",
            "Major Axis",
        ]
        projection_output["Projection File Index"] = make_parquet_index(
            projection_output, index_columns, index_precisions
        )
        projection_output = projection_output.set_index(
            "Projection File Index"
        ).sort_index()

        if len(projection_output) > 0:
            return projection_output
        else:
            return None

    def export_projection(
        self, dask_controller, seg_key, intensity_channel, output_path, overwrite=False
    ):
        projection_delayed_list = []
        for file_idx in self.file_indices:
            projection_delayed = dask.delayed(self.compute_file_projections)(
                file_idx,
                self.headpath,
                intensity_channel,
                seg_key=seg_key,
                projection_name=self.projection_name,
                projection_fn=self.projection_fn,
            )
            projection_delayed_list.append(projection_delayed)

        test_output = self.compute_file_projections_test(
            self.file_indices[0],
            self.headpath,
            intensity_channel,
            seg_key=seg_key,
            projection_name=self.projection_name,
            projection_fn=self.projection_fn,
        )
        projection_delayed_df = dd.from_delayed(
            projection_delayed_list, meta=test_output
        )

        to_parquet_checkpoint(
            dask_controller,
            projection_delayed_df,
            output_path,
            engine="pyarrow",
            overwrite=overwrite,
        )
