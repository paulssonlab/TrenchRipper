# fmt: off
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import pandas as pd
import h5py
import pickle
import copy

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, IntSlider, Dropdown, IntText, SelectMultiple, Select, IntRangeSlider, FloatRangeSlider
from skimage import filters,transform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from .kymograph import kymograph_multifov
from .segment import fluo_segmentation
from .utils import kymo_handle,pandas_hdf5_handler

class kymograph_interactive(kymograph_multifov):
    def __init__(self,headpath):
        """The kymograph class is used to generate and visualize kymographs.
        The central function of this class is the method 'generate_kymograph',
        which takes an hdf5 file of images from a single fov and outputs an
        hdf5 file containing kymographs from all detected trenches.

        NOTE: I need to revisit the row detection, must ensure there can be no overlap...

        Args:
        """
        #break all_channels,fov_list,t_subsample_step=t_subsample_step
        super(kymograph_interactive, self).__init__(headpath)

        self.channels = self.metadata["channels"]
        self.fov_list = self.metadata["fields_of_view"]
        self.timepoints_len = self.metadata["num_frames"]

        self.final_params = {}

    def view_image(self,fov_idx,t,channel,invert):
        img_entry = self.metadf.loc[fov_idx,t]
        file_idx = int(img_entry["File Index"])
        img_idx = int(img_entry["Image Index"])

        with h5py.File(self.headpath + "/hdf5/hdf5_" + str(file_idx) + ".hdf5", "r") as infile:
            img_arr = infile[channel][img_idx,:,:]
        if invert:
            img_arr = sk.util.invert(img_arr)
        plt.imshow(img_arr,cmap="Greys_r")

    def view_image_interactive(self):

        interact(self.view_image,fov_idx=Select(description='FOV number:',options=self.fov_list),\
             t=IntSlider(value=0, min=0, max=self.timepoints_len-1, step=1,continuous_update=False),
            channel=Dropdown(options=self.channels,value=self.channels[0],description='Channel:',disabled=False),\
            invert=Dropdown(options=[True,False],value=False))

    def import_hdf5_interactive(self):
        import_hdf5 = interactive(self.import_hdf5_files, {"manual":True}, all_channels=fixed(self.channels),\
                                  seg_channel=Dropdown(options=self.channels, value=self.channels[0]),invert=Dropdown(options=[True,False],\
                                  value=False), fov_list=SelectMultiple(options=self.fov_list),t_subsample_step=IntSlider(value=10,\
                                  min=0, max=200, step=1));
        display(import_hdf5)

    def preview_y_precentiles(self,imported_array_list, y_percentile, smoothing_kernel_y_dim_0,\
                          y_percentile_threshold):

        self.final_params['Y Percentile'] = y_percentile
        self.final_params['Y Smoothing Kernel'] = smoothing_kernel_y_dim_0
        self.final_params['Y Percentile Threshold'] = y_percentile_threshold

        y_percentiles_smoothed_list = self.map_to_fovs(self.get_smoothed_y_percentiles,imported_array_list,\
                                                       y_percentile,(smoothing_kernel_y_dim_0,1))

        self.plot_y_precentiles(y_percentiles_smoothed_list,self.fov_list,y_percentile_threshold)

        self.y_percentiles_smoothed_list = y_percentiles_smoothed_list

        return y_percentiles_smoothed_list

    def preview_y_precentiles_interactive(self):
        row_detection = interactive(self.preview_y_precentiles, {"manual":True},\
                        imported_array_list=fixed(self.imported_array_list), y_percentile=IntSlider(value=99,\
                        min=0, max=100, step=1), smoothing_kernel_y_dim_0=IntSlider(value=29, min=1,\
                        max=200, step=2), y_percentile_threshold=FloatSlider(value=0.2, min=0., max=1., step=0.01))
        display(row_detection)


    def plot_y_precentiles(self,y_percentiles_smoothed_list,fov_list,y_percentile_threshold):
        fig = plt.figure()

        ### Subplot dimensions of plot
        root_list_len = np.ceil(np.sqrt(len(y_percentiles_smoothed_list)))

        ### Looping through each fov
        idx=0
        for j,y_percentiles_smoothed in enumerate(y_percentiles_smoothed_list):
            ### Managing Subplots
            idx += 1
            ax = fig.add_subplot(root_list_len, root_list_len, idx, projection='3d')

            ### Making list of vertices (tuples) for use with PolyCollection
            vert_arr = np.array([np.add.accumulate(np.ones(y_percentiles_smoothed.shape,dtype=int),axis=0),y_percentiles_smoothed])
            verts = []
            for t in range(vert_arr.shape[2]):
                w_vert = vert_arr[:,:,t]
                verts.append([(w_vert[0,i],w_vert[1,i]) for i in range(0,w_vert.shape[1],10)])

            ### Making counting array for y position
            zs = np.add.accumulate(np.ones(len(verts)))

            ### Creating PolyCollection and add to plot
            poly = PolyCollection(verts,facecolors = ['b'])
            poly.set_alpha(0.5)
            ax.add_collection3d(poly,zs=zs, zdir='y')

            ### Depecting thresholds as straight lines
            x_len = y_percentiles_smoothed.shape[0]
            y_len = y_percentiles_smoothed.shape[1]
            thr_x = np.repeat(np.add.accumulate(np.ones(x_len,dtype=int))[:,np.newaxis],y_len,axis=1).T.flatten()
            thr_y = np.repeat(np.add.accumulate(np.ones(y_len,dtype=int)),x_len)
            thr_z = np.repeat(y_percentile_threshold,x_len*y_len)

            for i in range(0,x_len*y_len,x_len):
                ax.plot(thr_x[i:i+x_len],thr_y[i:i+x_len],thr_z[i:i+x_len],c='r')

            ### Plot lebels
            ax.set_title("FOV: " + str(fov_list[j]))
            ax.set_xlabel('y position')
            ax.set_xlim3d(0, vert_arr[0,-1,0])
            ax.set_ylabel('time (s)')
            ax.set_ylim3d(0, len(verts))
            ax.set_zlabel('intensity')
            ax.set_zlim3d(0, np.max(vert_arr[1]))

        plt.show()

    def preview_y_crop(self,y_percentiles_smoothed_list, imported_array_list,y_min_edge_dist, padding_y,\
                       trench_len_y,expected_num_rows,alternate_orientation,orientation_detection,orientation_on_fail,use_median_drift,\
                       images_per_row):

        self.final_params['Minimum Trench Length'] = y_min_edge_dist
        self.final_params['Y Padding'] = padding_y
        self.final_params['Trench Length'] = trench_len_y
        self.final_params['Orientation Detection Method'] = orientation_detection
        self.final_params['Expected Number of Rows (Manual Orientation Detection)'] = expected_num_rows
        self.final_params['Alternate Orientation'] = alternate_orientation
        self.final_params['Top Orientation when Row Drifts Out (Manual Orientation Detection)'] = orientation_on_fail
        self.final_params['Use Median Drift?'] = use_median_drift

        y_percentile_threshold = self.final_params['Y Percentile Threshold']

        get_trench_edges_y_output = self.map_to_fovs(self.get_trench_edges_y,y_percentiles_smoothed_list,y_percentile_threshold)
        trench_edges_y_lists = [item[0] for item in get_trench_edges_y_output]
        start_above_lists = [item[1] for item in get_trench_edges_y_output]
        end_above_lists = [item[2] for item in get_trench_edges_y_output]

        get_manual_orientations_output = self.map_to_fovs(self.get_manual_orientations,trench_edges_y_lists,start_above_lists,end_above_lists,\
                                                          alternate_orientation,expected_num_rows,orientation_detection,orientation_on_fail,y_min_edge_dist)

        orientations_list = [item[0] for item in get_manual_orientations_output]
        drop_first_row_list = [item[1] for item in get_manual_orientations_output]
        drop_last_row_list = [item[2] for item in get_manual_orientations_output]

        y_ends_lists = self.map_to_fovs(self.get_trench_ends,trench_edges_y_lists,start_above_lists,end_above_lists,orientations_list,drop_first_row_list,drop_last_row_list,y_min_edge_dist)
        y_drift_list = self.map_to_fovs(self.get_y_drift,y_ends_lists)
        if use_median_drift:
            median_drift = np.round(np.median(np.array(y_drift_list),axis=0)).astype(int)
            y_drift_list = [copy.copy(median_drift) for item in y_drift_list]

        keep_in_frame_kernels_output = self.map_to_fovs(self.keep_in_frame_kernels,y_ends_lists,y_drift_list,imported_array_list,orientations_list,padding_y,trench_len_y)
        valid_y_ends_list = [item[0] for item in keep_in_frame_kernels_output]
        valid_orientations_list = [item[1] for item in keep_in_frame_kernels_output]
        cropped_in_y_list = self.map_to_fovs(self.crop_y,imported_array_list,y_drift_list,valid_y_ends_list,valid_orientations_list,padding_y,trench_len_y)

        self.plot_y_crop(cropped_in_y_list,imported_array_list,self.fov_list,valid_orientations_list,images_per_row)

        self.cropped_in_y_list = cropped_in_y_list

        return cropped_in_y_list

    def preview_y_crop_interactive(self):

        y_cropping = interactive(self.preview_y_crop,{"manual":True},y_percentiles_smoothed_list=fixed(self.y_percentiles_smoothed_list),\
                imported_array_list=fixed(self.imported_array_list),\
                y_min_edge_dist=IntSlider(value=50, min=5, max=1000, step=5),\
                padding_y=IntSlider(value=20, min=0, max=500, step=5),\
                trench_len_y=IntSlider(value=270, min=0, max=1000, step=5),
                expected_num_rows=IntText(value=2,description='Number of Rows:',disabled=False),\
                alternate_orientation=Dropdown(options=[True,False],value=True,description='Alternate Orientation?:',disabled=False),\
               orientation_detection=Dropdown(options=[0, 1],value=0,description='Orientation:',disabled=False),\
                orientation_on_fail=Dropdown(options=[0, 1],value=0,description='Orientation when < expected rows:',disabled=False),\
                use_median_drift=Dropdown(options=[True,False],value=False,description='Use Median Drift?:',disabled=False),\
                    images_per_row=IntSlider(value=3, min=1, max=10, step=1))

        display(y_cropping)

    def plot_y_crop(self,cropped_in_y_list,imported_array_list,fov_list,valid_orientations_list,images_per_row):

        time_list = range(1,imported_array_list[0].shape[3]+1)
        time_per_img = len(time_list)
        ttl_lanes = np.sum([len(item) for item in valid_orientations_list])
        ttl_imgs = ttl_lanes*time_per_img

        remaining_imgs = time_per_img%images_per_row
        if remaining_imgs == 0:
            rows_per_lane = time_per_img//images_per_row
        else:
            rows_per_lane = (time_per_img//images_per_row) + 1

        nrows = rows_per_lane*ttl_lanes
        ncols = images_per_row

        fig, _ = plt.subplots(figsize=(20, 10))

        idx = 0
        for i,cropped_in_y in enumerate(cropped_in_y_list):
            num_rows = len(valid_orientations_list[i])
            for j in range(num_rows):
                for k,t in enumerate(time_list):
                    idx += 1
                    ax = plt.subplot(nrows,ncols,idx)
                    ax.axis("off")
                    ax.set_title("row=" + str(j) + ",fov=" + str(fov_list[i]) + ",t=" + str(t))
                    ax.imshow(cropped_in_y[j,0,:,:,k],cmap="Greys_r")
                if remaining_imgs != 0:
                    for t in range(0,(images_per_row-remaining_imgs)):
                        idx += 1

        fig.tight_layout()
        fig.show()

    def preview_x_percentiles(self,cropped_in_y_list, t, x_percentile, background_kernel_x,smoothing_kernel_x,\
                              otsu_scaling,min_threshold):

        self.final_params['X Percentile'] = x_percentile
        self.final_params['X Background Kernel'] = background_kernel_x
        self.final_params['X Smoothing Kernel'] = smoothing_kernel_x
        self.final_params['Otsu Threshold Scaling'] = otsu_scaling
        self.final_params['Minimum X Threshold'] = min_threshold

        smoothed_x_percentiles_list = self.map_to_fovs(self.get_smoothed_x_percentiles,cropped_in_y_list,x_percentile,\
                                                         (background_kernel_x,1),(smoothing_kernel_x,1))
        thresholds = []
        for smoothed_x_percentiles_row in smoothed_x_percentiles_list:
            for smoothed_x_percentiles in smoothed_x_percentiles_row:
                x_percentiles_t = smoothed_x_percentiles[:,t]
                thresholds.append(self.get_midpoints(x_percentiles_t,otsu_scaling,min_threshold)[1])
        self.plot_x_percentiles(smoothed_x_percentiles_list,self.fov_list, t, thresholds)

        self.smoothed_x_percentiles_list = smoothed_x_percentiles_list
        all_midpoints_list,x_drift_list = self.preview_midpoints(self.smoothed_x_percentiles_list)

        return smoothed_x_percentiles_list,all_midpoints_list,x_drift_list

    def preview_midpoints(self,smoothed_x_percentiles_list):
        otsu_scaling = self.final_params['Otsu Threshold Scaling']
        min_threshold = self.final_params['Minimum X Threshold']

        all_midpoints_list = self.map_to_fovs(self.get_all_midpoints,self.smoothed_x_percentiles_list,otsu_scaling,min_threshold)
        self.plot_midpoints(all_midpoints_list,self.fov_list)
        x_drift_list = self.map_to_fovs(self.get_x_drift,all_midpoints_list)

        self.all_midpoints_list,self.x_drift_list = (all_midpoints_list,x_drift_list)

        return all_midpoints_list,x_drift_list


    def preview_x_percentiles_interactive(self):
        trench_detection = interactive(self.preview_x_percentiles, {"manual":True}, cropped_in_y_list=fixed(self.cropped_in_y_list),t=IntSlider(value=0, min=0, max=self.cropped_in_y_list[0].shape[4]-1, step=1),\
                x_percentile=IntSlider(value=85, min=50, max=100, step=1),background_kernel_x=IntSlider(value=21, min=1, max=601, step=20), smoothing_kernel_x=IntSlider(value=9, min=1, max=31, step=2),\
               otsu_scaling=FloatSlider(value=0.25, min=0., max=2., step=0.01),min_threshold=IntSlider(value=0, min=0., max=65535, step=1));

        display(trench_detection)

    def plot_x_percentiles(self,smoothed_x_percentiles_list,fov_list,t,thresholds):
        fig = plt.figure()
        nrow = len(self.cropped_in_y_list) #fovs
        ncol = (sum([len(item) for item in self.cropped_in_y_list])//nrow)+1

        idx = 0
        for i,smoothed_x_percentiles_lanes in enumerate(smoothed_x_percentiles_list):
            for j,smoothed_x_percentiles in enumerate(smoothed_x_percentiles_lanes):
                idx += 1
                data = smoothed_x_percentiles[:,t]
                ax = fig.add_subplot(ncol, nrow, idx)
                ax.plot(data)

                current_threshold = thresholds[idx-1]
                threshold_data = np.repeat(current_threshold,len(data))
                ax.plot(threshold_data,c='r')
                ax.set_title("FOV: " + str(fov_list[i]) + " Lane: " + str(j))
                ax.set_xlabel('x position')
                ax.set_ylabel('intensity')

        plt.show()

    def plot_midpoints(self,all_midpoints_list,fov_list):
        fig = plt.figure()
        ax = fig.gca()

        nrows = 2*len(fov_list)
        ncols = 2

        idx = 0
        for i,top_bottom_list in enumerate(all_midpoints_list):
            for j,all_midpoints in enumerate(top_bottom_list):
                idx+=1
                ax = plt.subplot(nrows,ncols,idx)
                ax.set_title("row=" + str(j) + ",fov=" + str(fov_list[i]))
                data = np.concatenate([np.array([item,np.ones(item.shape,dtype=int)*k]).T for k,item in enumerate(all_midpoints)])
                ax.scatter(data[:,0],data[:,1],alpha=0.7)
                ax.set_xlabel('x position')
                ax.set_ylabel('time')

        plt.tight_layout()
        plt.show()

    def preview_kymographs(self,cropped_in_y_list,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr,use_median_drift):
        self.final_params['Trench Width'] = trench_width_x
        self.final_params['Trench Presence Threshold'] = trench_present_thr
        self.final_params['Use Median Drift?'] = use_median_drift
        if use_median_drift:
            if use_median_drift:
                median_drift = np.round(np.median(np.array([row_x_drift for fov_x_drift in x_drift_list for row_x_drift in fov_x_drift]),axis=0)).astype(int)
                for i in range(len(x_drift_list)):
                    for j in range(len(x_drift_list[i])):
                        x_drift_list[i][j] = copy.copy(median_drift)

        cropped_in_x_list = self.map_to_fovs(self.get_crop_in_x,cropped_in_y_list,all_midpoints_list,x_drift_list,\
                                             trench_width_x,trench_present_thr)

        corrected_midpoints_list = self.map_to_fovs(self.get_corrected_midpoints,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr)


        self.plot_kymographs(cropped_in_x_list,self.fov_list)
        self.plot_midpoints(corrected_midpoints_list,self.fov_list)

    def preview_kymographs_interactive(self):
            interact_manual(self.preview_kymographs,cropped_in_y_list=fixed(self.cropped_in_y_list),all_midpoints_list=fixed(self.all_midpoints_list),\
            x_drift_list=fixed(self.x_drift_list),trench_width_x=IntSlider(value=30, min=2, max=1000, step=2),\
            trench_present_thr=FloatSlider(value=0., min=0., max=1., step=0.05),\
            use_median_drift=Dropdown(options=[True,False],value=False,description='Use Median Drift?:',disabled=False))

    def plot_kymographs(self,cropped_in_x_list,fov_list,num_rows=2):
        plt.figure()
        idx = 0
        ncol = num_rows
        nrow = len(fov_list)*num_rows

        for i,row_list in enumerate(cropped_in_x_list):
            for j,channel in enumerate(row_list):
                seg_channel = channel[0]
                idx+=1
                rand_k = np.random.randint(0,seg_channel.shape[0])
                ax = plt.subplot(ncol,nrow,idx)
                ex_kymo = seg_channel[rand_k]
                self.plot_kymograph(ax,ex_kymo)
                ax.set_title("row=" + str(j) + ",fov=" + str(fov_list[i]) + ",trench=" + str(rand_k))

        plt.tight_layout()
        plt.show()


    def plot_kymograph(self,ax,kymograph):
        """Helper function for plotting kymographs. Takes a kymograph array of
        shape (y_dim,x_dim,t_dim).

        Args:
            kymograph (array): kymograph array of shape (y_dim,x_dim,t_dim).
        """
        list_in_t = [kymograph[:,:,t] for t in range(kymograph.shape[2])]
        img_arr = np.concatenate(list_in_t,axis=1)
        ax.imshow(img_arr,cmap="Greys_r")

    def process_results(self):
        self.final_params["All Channels"] = self.all_channels
        self.final_params["Invert"] = self.invert

        for key,value in self.final_params.items():
            print(key + " " + str(value))

    def write_param_file(self):
        with open(self.headpath + "/kymograph.par", "wb") as outfile:
            pickle.dump(self.final_params, outfile)

class fluo_segmentation_interactive(fluo_segmentation):

    def __init__(self,headpath,bit_max=0,scale_timepoints=False,scaling_percentile=0.9,img_scaling=1.,smooth_sigma=0.75,niblack_scaling=1.,\
                 hess_pad=6,global_threshold=25,cell_otsu_scaling=1.,local_otsu_r=15,min_obj_size=30,distance_threshold=2):

        fluo_segmentation.__init__(self,bit_max=bit_max,scale_timepoints=scale_timepoints,scaling_percentile=scaling_percentile,\
                                   img_scaling=img_scaling,smooth_sigma=smooth_sigma,niblack_scaling=niblack_scaling,\
                                  hess_pad=hess_pad,global_threshold=global_threshold,cell_otsu_scaling=cell_otsu_scaling,\
                                   local_otsu_r=local_otsu_r,min_obj_size=min_obj_size,distance_threshold=distance_threshold)

        self.headpath = headpath
        self.kymographpath = headpath + "/kymograph"
        self.metapath = headpath + "/kymograph/metadata"
#         self.metapath = headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(headpath + "/metadata.hdf5")
        self.kymodf = pd.read_parquet(self.metapath,columns=["timepoints","trenchid","File Index","File Trench Index"])
        
#         self.kymodf = self.meta_handle.read_df("kymograph",read_metadata=True)
        globaldf = self.meta_handle.read_df("global",read_metadata=True)
        self.all_channels = globaldf.metadata['channels']

        timepoint_num = len(self.kymodf["timepoints"].unique().tolist())
        self.t_range = (0,timepoint_num)
        self.trenchid_arr = self.kymodf["trenchid"].unique()
        self.kymodf = self.kymodf.set_index(["trenchid","timepoints"])

        self.final_params = {}

    def choose_seg_channel(self,seg_channel):
        self.seg_channel = seg_channel

    def choose_seg_channel_inter(self):
        choose_channel = interactive(self.choose_seg_channel,{"manual": True},\
        seg_channel=Dropdown(options=self.all_channels,value=self.all_channels[0]))
        display(choose_channel)

    def plot_img_list(self,img_list,cmap="Greys_r",interpolation=None):
        nrow = ((len(img_list)-1)//self.img_per_row)+1
        fig, axes = plt.subplots(nrows=nrow, ncols=self.img_per_row, figsize=self.fig_size)
        for i in range(len(img_list)):
            img = img_list[i]
            if nrow < 2:
                axes[i%self.img_per_row].imshow(img,cmap=cmap,interpolation=interpolation)
            else:
                axes[i//self.img_per_row,i%self.img_per_row].imshow(img,cmap=cmap,interpolation=interpolation)
        extra_slots = self.img_per_row - (len(img_list)%self.img_per_row)
        if extra_slots != 0:
            for slot in range(1,extra_slots+1):
                if nrow < 2:
                    axes[self.img_per_row-slot].axis('off')
                else:
                    axes[-1, self.img_per_row-slot].axis('off')
        plt.tight_layout()
        plt.show()

    def import_array(self,n_trenches,t_range=(0,None),t_subsample_step=1,fig_size_y=9,fig_size_x=6,img_per_row=2):
        self.fig_size = (fig_size_y,fig_size_x)
        self.img_per_row = img_per_row

        rand_trench_arr = np.random.choice(self.trenchid_arr,size=(n_trenches,),replace=False)
        self.selecteddf = self.kymodf.loc[list(zip(rand_trench_arr,np.zeros(len(rand_trench_arr)).astype(int)))]
        selectedlist = list(zip(self.selecteddf["File Index"].tolist(),self.selecteddf["File Trench Index"].tolist()))

        array_list = []
        for item in selectedlist:
            with h5py.File(self.kymographpath + "/kymograph_" + str(item[0]) + ".hdf5", "r") as hdf5_handle:
                if t_range[1] == None:
                    array = hdf5_handle[self.seg_channel][item[1],t_range[0]::t_subsample_step]
                else:
                    array = hdf5_handle[self.seg_channel][item[1],t_range[0]:t_range[1]+1:t_subsample_step]
            array_list.append(array)
        output_array = np.concatenate(np.expand_dims(array_list,axis=0),axis=0)
        self.t_tot = output_array.shape[1]
        self.plot_kymographs(output_array)
        self.output_array = output_array

        return output_array

    def import_array_inter(self):
        kymo_arr_int = interactive(self.import_array,{"manual": True},n_trenches=IntText(value=12,\
                       description="Number of trenches:", disabled=False),t_range=IntRangeSlider(value=[self.t_range[0],\
                       self.t_range[1] - 1],description="Time Range:",min=self.t_range[0],max=self.t_range[1] - 1,step=1,\
                       disabled=False),t_subsample_step=IntSlider(value=1, description="Time Subsampling Step:", min=1,\
                       max=20, step=1),fig_size_y=IntSlider(value=20, description="Figure Size (Y Dimension):", min=1,\
                       max=30, step=1),fig_size_x=IntSlider(value=12, description="Figure Size (X Dimension):", min=1,\
                       max=30, step=1),img_per_row=IntSlider(value=6, description="Images per Row:", min=1, max=30,\
                       step=1))
        display(kymo_arr_int)

    def plot_kymographs(self,kymo_arr):
        input_kymo = kymo_handle()
        img_list = []
        for k in range(kymo_arr.shape[0]):
            input_kymo.import_wrap(kymo_arr[k])
            img_list.append(input_kymo.return_unwrap())
        self.plot_img_list(img_list)
        return img_list

    def plot_processed(self,bit_max,scale_timepoints,scaling_percentile,img_scaling,smooth_sigma):
        
        self.final_params['8 Bit Maximum:'] = bit_max
        self.final_params['Scale Fluorescence?'] = scale_timepoints
        self.final_params["Scaling Percentile:"] = scaling_percentile
        self.final_params["Image Scaling Factor:"] = img_scaling
        self.final_params['Gaussian Kernel Sigma:'] = smooth_sigma
        
        output_array = copy.copy(self.output_array) #k,t,y,x
        
        percentile = int(np.percentile(output_array.flatten(), 99))
        print("99th percentile:" + str(percentile))
        fig, ax = plt.subplots(1, 1)
        ax.hist(output_array.flatten(),bins=50)
        ax.axvline(bit_max,c="r",linewidth=3,zorder=10)
        ax.set_title("Pixel Value Histogram w/ 8-bit Maximum",fontsize=20)
        ax.set_xlabel("Pixel Value",fontsize=15)
        fig.set_size_inches(9, 6)
        fig.show()
        
        output_array_list = []
        for k in range(output_array.shape[0]):
            scaled_output_array = self.to_8bit(output_array[k],bit_max)
            if scale_timepoints:
                scaled_output_array = self.scale_kymo(scaled_output_array,scaling_percentile)
            output_array_list.append(scaled_output_array)
        output_array = np.array(output_array_list)

#         proc_list = []
        unwrap_proc_list = []
        for k in range(output_array.shape[0]):
            t_tot = output_array[k].shape[0]
            output_array_unwrapped = kymo_handle()
            output_array_unwrapped.import_wrap(output_array[k])
            output_array_unwrapped = output_array_unwrapped.return_unwrap()
            rescaled_unwrapped = transform.rescale(output_array_unwrapped,img_scaling,anti_aliasing=False, preserve_range=True).astype("uint8")
            filtered_unwrapped = sk.filters.gaussian(rescaled_unwrapped,sigma=smooth_sigma,preserve_range=True,mode='reflect').astype("uint8")
#             filtered_wrapped = kymo_handle()
#             filtered_wrapped.import_unwrap(filtered_unwrapped,t_tot)
#             filtered_wrapped = filtered_wrapped.return_wrap()
#             proc_list.append(filtered_wrapped)
            unwrap_proc_list.append(filtered_unwrapped)
        self.proc_list = unwrap_proc_list
        del unwrap_proc_list
        self.eig_list = [self.get_eig_img(item,edge_padding=self.hess_pad) for item in self.proc_list]
        
        self.plot_img_list(self.proc_list)
        self.plot_img_list(self.eig_list)

    def plot_processed_inter(self):
        proc_list_int = interactive(
            self.plot_processed,
            {"manual": True},
            bit_max=IntSlider(
                value=1000,
                description="8-bit Maximum:",
                min=0,
                max=65535,
                step=250,
                disabled=False,
            ),
            scale_timepoints=Dropdown(
                options=[True, False],
                value=False,
                description="Scale Fluorescence?",
                disabled=False,
            ),
            scaling_percentile=IntSlider(
                value=90,
                description="Scaling Percentile:",
                min=0,
                max=100,
                step=1,
                disabled=False,
            ),
            img_scaling=FloatSlider(
                value=1.,
                description="Image Upsampling Factor:",
                min=1.,
                max=3.,
                step=0.25,
                disabled=False,
            ),
            smooth_sigma=FloatSlider(
                value=0.75,
                description="Gaussian Kernel Sigma:",
                min=0.0,
                max=3.0,
                step=0.25,
                disabled=False,
            ),
        )

        display(proc_list_int)
        
    def plot_cell_mask(self,global_threshold,cell_otsu_scaling,local_otsu_r,min_obj_size):
        self.final_params['Global Threshold:'] = global_threshold
        self.final_params['Cell Threshold Scaling:'] = cell_otsu_scaling
        self.final_params['Local Otsu Radius:'] = local_otsu_r
        self.final_params['Minimum Object Size:'] = min_obj_size
         
        proc_arr = np.array(self.proc_list)
        fig, ax = plt.subplots(1, 1)
        ax.hist(proc_arr.flatten(),bins=50)
        ax.axvline(global_threshold,c="r",linewidth=3,zorder=10)
        ax.set_title("Pixel Value Histogram w/ Global Threshold",fontsize=20)
        ax.set_xlabel("Pixel Value",fontsize=15)
        fig.set_size_inches(9, 6)
        fig.show()
        del proc_arr

        cell_mask_list = []
        for proc in self.proc_list:
            cell_mask = self.get_cell_mask(proc,global_threshold=global_threshold,cell_otsu_scaling=cell_otsu_scaling,\
                                           local_otsu_r=local_otsu_r,min_obj_size=min_obj_size)
            cell_mask_list.append(cell_mask)
        self.plot_img_list(self.proc_list)
        self.plot_img_list(cell_mask_list)
        self.cell_mask_list = cell_mask_list

    def plot_cell_mask_inter(self):
        cell_mask_list_int = interactive(
            self.plot_cell_mask,
            {"manual": True},
            global_threshold=IntSlider(
                value=50,
                description="Global Threshold:",
                min=0,
                max=255,
                step=1,
                disabled=False,
            ),
            cell_otsu_scaling=FloatSlider(
                value=1.,
                description="Cell Threshold Scaling:",
                min=0.0,
                max=2.0,
                step=0.01,
                disabled=False,
            ),
            local_otsu_r=IntSlider(
                value=15,
                description="Local Otsu Radius:",
                min=0,
                max=30,
                step=1,
                disabled=False,
            ),
            min_obj_size=IntSlider(
                value=30,
                description="Minimum Object Size:",
                min=0,
                max=100,
                step=2,
                disabled=False,
            ),
        )
        display(cell_mask_list_int)
        
    def plot_eig_mask(self,niblack_scaling):
        self.final_params['Niblack Scaling:'] = niblack_scaling
        
        eig_mask_list = []
        for eig in self.eig_list:
            eig_mask = self.get_eig_mask(eig,niblack_scaling=niblack_scaling)
            eig_mask_list.append(eig_mask)
            
        self.plot_img_list(self.eig_list)
        self.plot_img_list(eig_mask_list)
        
    def plot_eig_mask_inter(self):
        cell_eig_list_int = interactive(
            self.plot_eig_mask,
            {"manual": True},
            niblack_scaling=FloatSlider(
                value=1.,
                description="Edge Threshold Scaling:",
                min=0.0,
                max=2.0,
                step=0.01,
                disabled=False,
            ),
        )
        display(cell_eig_list_int)

    def plot_dist_mask(self,distance_threshold):
        self.final_params['Distance Threshold:'] = distance_threshold
        
        dist_img_list = []
        dist_mask_list = []
        for cell_mask in self.cell_mask_list:
            dist_img = ndi.distance_transform_edt(cell_mask).astype("uint8")
            dist_mask = dist_img>distance_threshold
            dist_img_list.append(dist_img)
            dist_mask_list.append(dist_mask)
            
        self.plot_img_list(dist_img_list)
        self.plot_img_list(dist_mask_list)
        self.dist_mask_list = dist_mask_list
        
    def plot_dist_mask_inter(self):
        dist_mask_int = interactive(
            self.plot_dist_mask,
            {"manual": True},
            distance_threshold=IntSlider(
                value=2,
                description="Distance Threshold:",
                min=0,
                max=20,
                step=1,
                disabled=False,
            ),
        )
        display(dist_mask_int)
        
    def plot_marker_mask(self,niblack_scaling,distance_threshold):      
        self.final_params['Niblack Scaling:'] = niblack_scaling
        self.final_params['Distance Threshold:'] = distance_threshold
        
        original_shape = (self.output_array.shape[2],self.output_array.shape[1]*self.output_array.shape[3]) #k,t,y,x
        segmentation_list = []
        for i in range(len(self.eig_list)):
            eig = self.eig_list[i]
            eig_mask = self.get_eig_mask(eig,niblack_scaling=niblack_scaling)
            
            cell_mask = self.cell_mask_list[i]
            dist_img = ndi.distance_transform_edt(cell_mask).astype("uint8")
            dist_mask = dist_img>distance_threshold
            
            marker_mask = dist_mask*eig_mask
            marker_mask = sk.measure.label(marker_mask)
            output_labels = watershed(-dist_img, markers=marker_mask, mask=cell_mask)
            
            output_labels = sk.transform.resize(output_labels,original_shape,order=0,anti_aliasing=False, preserve_range=True).astype("uint32")
            segmentation_list.append(output_labels)
            
        self.plot_kymographs(self.output_array)
        
        seg_plt_list = []
        for segmentation in segmentation_list:
            plt_mask = (segmentation == 0)
            plt_img = np.ma.array(segmentation, mask=plt_mask)
            seg_plt_list.append(plt_img)        
        self.plot_img_list(seg_plt_list,cmap="jet",interpolation='nearest')
        
    def plot_marker_mask_inter(self):
        if self.final_params['Niblack Scaling:'] is not None\
        and self.final_params['Distance Threshold:'] is not None:  
            marker_mask_int = interactive(
                self.plot_marker_mask,
                {"manual": True},
                niblack_scaling=FloatSlider(
                    value=self.final_params['Niblack Scaling:'],
                    description="Edge Threshold Scaling:",
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    disabled=False,
                ),
                distance_threshold=IntSlider(
                    value=self.final_params['Distance Threshold:'],
                    description="Distance Threshold:",
                    min=0,
                    max=20,
                    step=1,
                    disabled=False,
                ),
            )
        else:        
            marker_mask_int = interactive(
                self.plot_marker_mask,
                {"manual": True},
                niblack_scaling=FloatSlider(
                    value=1.,
                    description="Edge Threshold Scaling:",
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    disabled=False,
                ),
                distance_threshold=IntSlider(
                    value=2,
                    description="Distance Threshold:",
                    min=0,
                    max=20,
                    step=1,
                    disabled=False,
                ),
            )
        display(marker_mask_int)
        
    def process_results(self):
        self.final_params["Segmentation Channel:"] = self.seg_channel
        for key,value in self.final_params.items():
            print(key + " " + str(value))

    def write_param_file(self):
        with open(self.headpath + "/fluorescent_segmentation.par", "wb") as outfile:
            pickle.dump(self.final_params, outfile)