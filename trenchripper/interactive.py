import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import h5py
import pickle

from skimage import filters
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from .kymograph import kymograph_multifov
from .segment import fluo_segmentation
from .utils import kymo_handle,pandas_hdf5_handler

class kymograph_interactive(kymograph_multifov):
    def __init__(self,headpath):
        """The kymograph class is used to generate and visualize kymographs. The central function of this
        class is the method 'generate_kymograph', which takes an hdf5 file of images from a single fov and
        outputs an hdf5 file containing kymographs from all detected trenches.

        NOTE: I need to revisit the row detection, must ensure there can be no overlap...
            
        Args:

        """
        #break all_channels,fov_list,t_subsample_step=t_subsample_step
        super(kymograph_interactive, self).__init__(headpath)
        
        self.final_params = {}
        
    def get_image_params(self):
        channels = self.metadata["channels"]
        fov_list = self.metadata["fields_of_view"]
        timepoints_len = self.metadata["num_frames"]
        return channels,fov_list,timepoints_len
        
    def view_image(self,fov_idx,t,channel):
        img_entry = self.metadf.loc[fov_idx,t]
        file_idx = int(img_entry["File Index"])
        img_idx = int(img_entry["Image Index"])
        
        with h5py.File(self.headpath + "/hdf5/hdf5_" + str(file_idx) + ".hdf5", "r") as infile:
            img_arr = infile[channel][img_idx,:,:]
        plt.imshow(img_arr)

    def preview_y_precentiles(self,imported_array_list, y_percentile, smoothing_kernel_y_dim_0,\
                          triangle_nbins,triangle_scaling):
        
        self.final_params['Y Percentile'] = y_percentile
        self.final_params['Y Smoothing Kernel'] = smoothing_kernel_y_dim_0
        self.final_params['Triangle Threshold Bins'] = triangle_nbins
        self.final_params['Triangle Threshold Scaling'] = triangle_scaling
        
        y_percentiles_smoothed_list = self.map_to_fovs(self.get_smoothed_y_percentiles,imported_array_list,\
                                                       y_percentile,(smoothing_kernel_y_dim_0,1))
        thresholds = [self.triangle_threshold(y_percentiles_smoothed,triangle_nbins,triangle_scaling)[1] for y_percentiles_smoothed in y_percentiles_smoothed_list]
        self.plot_y_precentiles(y_percentiles_smoothed_list,self.fov_list,thresholds)
        return y_percentiles_smoothed_list
           
    def plot_y_precentiles(self,y_percentiles_smoothed_list,fov_list,thresholds):
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
#             
            thr_z = np.concatenate([np.repeat(threshold,x_len) for threshold in thresholds[j]],axis=0)
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
                       trench_len_y,vertical_spacing,expected_num_rows,orientation_detection,orientation_on_fail):
        
        self.final_params['Minimum Trench Length'] = y_min_edge_dist
        self.final_params['Y Padding'] = padding_y
        self.final_params['Trench Length'] = trench_len_y
        self.final_params['Orientation Detection Method'] = orientation_detection
        self.final_params['Expected Number of Rows (Manual Orientation Detection)'] = expected_num_rows
        self.final_params['Top Orientation when Row Drifts Out (Manual Orientation Detection)'] = orientation_on_fail
        
        triangle_nbins = self.final_params['Triangle Threshold Bins']
        triangle_scaling = self.final_params['Triangle Threshold Scaling']
        
        trench_edges_y_lists = self.map_to_fovs(self.get_trench_edges_y,y_percentiles_smoothed_list,triangle_nbins,\
                                               triangle_scaling,y_min_edge_dist)
        y_midpoints_list = self.map_to_fovs(self.get_y_midpoints,trench_edges_y_lists)
        y_drift_list = self.map_to_fovs(self.get_y_drift,y_midpoints_list)
                
        if orientation_detection == 'phase':
            valid_edges_y_output = self.map_to_fovs(self.keep_in_frame_kernels,trench_edges_y_lists,y_drift_list,imported_array_list,padding_y)
            valid_edges_y_lists = [item[0] for item in valid_edges_y_output]
            trench_orientations_list = self.map_to_fovs(self.get_phase_orientations,y_percentiles_smoothed_list,valid_edges_y_lists)
        
        elif orientation_detection == 0 or orientation_detection == 1:
            trench_orientations_list = self.map_to_fovs(self.get_manual_orientations,trench_edges_y_lists,expected_num_rows,orientation_detection,orientation_on_fail)
            valid_edges_y_output = self.map_to_fovs(self.keep_in_frame_kernels,trench_edges_y_lists,y_drift_list,imported_array_list,padding_y)
            valid_edges_y_lists = [item[0] for item in valid_edges_y_output]
            valid_orientation_lists = [item[1] for item in valid_edges_y_output]
            trench_orientations_list = [np.array(item)[valid_orientation_lists[i]].tolist() for i,item in enumerate(trench_orientations_list)]

        else:
            print("Orientation detection value invalid!")
        
        cropped_in_y_list = self.map_to_fovs(self.crop_y,imported_array_list,y_drift_list,valid_edges_y_lists,trench_orientations_list,padding_y,\
                                             trench_len_y)

        self.plot_y_crop(cropped_in_y_list,imported_array_list,self.fov_list,vertical_spacing,trench_orientations_list)
        return cropped_in_y_list
        
    def plot_y_crop(self,cropped_in_y_list,imported_array_list,fov_list,vertical_spacing,trench_orientations_list):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        time_list = range(1,imported_array_list[0].shape[3]+1)
        
        nrows = np.sum([len(item) for item in trench_orientations_list])
        ncols = len(time_list)
        
        idx = 0
        for i,cropped_in_y in enumerate(cropped_in_y_list):
            num_rows = len(trench_orientations_list[i])
            for j in range(num_rows):
                for k,t in enumerate(time_list):
                    idx += 1
                    ax = plt.subplot(nrows,ncols,idx)
                    ax.set_title("row=" + str(j) + ",fov=" + str(fov_list[i]) + ",t=" + str(t))
                    ax.imshow(cropped_in_y[j,0,:,:,k])
                    
        plt.tight_layout()
        plt.subplots_adjust(top=vertical_spacing)
        plt.show()
        
    def preview_x_percentiles(self,cropped_in_y_list, t, x_percentile, background_kernel_x,smoothing_kernel_x,\
                              otsu_nbins, otsu_scaling,vertical_spacing):
        
        self.final_params['X Percentile'] = x_percentile
        self.final_params['X Background Kernel'] = background_kernel_x
        self.final_params['X Smoothing Kernel'] = smoothing_kernel_x
        self.final_params['Otsu Threshold Bins'] = otsu_nbins
        self.final_params['Otsu Threshold Scaling'] = otsu_scaling
        
        smoothed_x_percentiles_list = self.map_to_fovs(self.get_smoothed_x_percentiles,cropped_in_y_list,x_percentile,\
                                                         (background_kernel_x,1),(smoothing_kernel_x,1))
        thresholds = []
        for smoothed_x_percentiles_row in smoothed_x_percentiles_list:
            for smoothed_x_percentiles in smoothed_x_percentiles_row:
                x_percentiles_t = smoothed_x_percentiles[:,t]
                thresholds.append(self.get_midpoints(x_percentiles_t,otsu_nbins,otsu_scaling)[1])
        self.plot_x_percentiles(smoothed_x_percentiles_list,self.fov_list, t, thresholds,vertical_spacing,num_rows=2)
        return smoothed_x_percentiles_list
        
        
    def plot_x_percentiles(self,smoothed_x_percentiles_list,fov_list,t,thresholds,vertical_spacing,num_rows=2):
        fig = plt.figure()
        
        ncol=num_rows
        nrow=len(smoothed_x_percentiles_list)
        
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
    
    
    def preview_midpoints(self,smoothed_x_percentiles_list,vertical_spacing):
        otsu_nbins = self.final_params['Otsu Threshold Bins']
        otsu_scaling = self.final_params['Otsu Threshold Scaling']
        
        all_midpoints_list = self.map_to_fovs(self.get_all_midpoints,smoothed_x_percentiles_list,otsu_nbins,otsu_scaling)
        self.plot_midpoints(all_midpoints_list,self.fov_list,vertical_spacing)
        x_drift_list = self.map_to_fovs(self.get_x_drift,all_midpoints_list)
        return all_midpoints_list,x_drift_list
        
#     def preview_corrected_midpoints(self,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr,vertical_spacing):
#         self.final_params['Trench Width'] = trench_width_x
#         self.final_params['Trench Presence Threshold'] = trench_present_thr
        
#         corrected_midpoints_list = self.map_to_fovs(self.get_corrected_midpoints,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr)
#         self.plot_midpoints(corrected_midpoints_list,self.fov_list,vertical_spacing)
                                                         
    def preview_kymographs(self,cropped_in_y_list,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr,vertical_spacing):
        self.final_params['Trench Width'] = trench_width_x
        self.final_params['Trench Presence Threshold'] = trench_present_thr
        
        cropped_in_x_list = self.map_to_fovs(self.get_crop_in_x,cropped_in_y_list,all_midpoints_list,x_drift_list,\
                                             trench_width_x,trench_present_thr)
        corrected_midpoints_list = self.map_to_fovs(self.get_corrected_midpoints,all_midpoints_list,x_drift_list,trench_width_x,trench_present_thr)
        
        self.plot_kymographs(cropped_in_x_list,self.fov_list,vertical_spacing)
        self.plot_midpoints(corrected_midpoints_list,self.fov_list,vertical_spacing)
        
    def plot_midpoints(self,all_midpoints_list,fov_list,vertical_spacing):
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
        plt.subplots_adjust(top=vertical_spacing)
        plt.show()
    
    def plot_kymographs(self,cropped_in_x_list,fov_list,vertical_spacing,num_rows=2):
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
        plt.subplots_adjust(top=vertical_spacing)
        plt.show()  
                

    def plot_kymograph(self,ax,kymograph):
        """Helper function for plotting kymographs. Takes a kymograph array of shape (y_dim,x_dim,t_dim).
        
        Args:
            kymograph (array): kymograph array of shape (y_dim,x_dim,t_dim).
        """
        list_in_t = [kymograph[:,:,t] for t in range(kymograph.shape[2])]
        img_arr = np.concatenate(list_in_t,axis=1)
        ax.imshow(img_arr)
        
    def process_results(self):
        self.final_params["All Channels"] = self.all_channels
        self.final_params["Time Range"] = self.t_range
        
        for key,value in self.final_params.items():
            print(key + " " + str(value))
    
    def write_param_file(self):
        with open(self.headpath + "/kymograph.par", "wb") as outfile:
            pickle.dump(self.final_params, outfile)

class fluo_segmentation_interactive(fluo_segmentation):
    def __init__(self,headpath,smooth_sigma=0.75,wrap_pad=0,hess_pad=4,min_obj_size=30,cell_mask_method='local',\
                 global_otsu_scaling=1.,cell_otsu_scaling=1.,local_otsu_r=15,edge_threshold_scaling=1.,threshold_step_perc=0.1,\
                 threshold_perc_num_steps=2,convex_threshold=0.8):
        
        fluo_segmentation.__init__(self,smooth_sigma=smooth_sigma,wrap_pad=wrap_pad,hess_pad=hess_pad,min_obj_size=min_obj_size,\
                                   cell_mask_method=cell_mask_method,global_otsu_scaling=global_otsu_scaling,cell_otsu_scaling=cell_otsu_scaling,\
                                   local_otsu_r=local_otsu_r,edge_threshold_scaling=edge_threshold_scaling,threshold_step_perc=threshold_step_perc,\
                                   threshold_perc_num_steps=threshold_perc_num_steps,convex_threshold=convex_threshold)

        self.headpath = headpath
        self.input_path = headpath + "/kymo"
        self.input_file_prefix = self.input_path + "/kymo_"
        self.metapath = headpath + "/metadata.hdf5"
        
        meta_handle = pandas_hdf5_handler(self.metapath)
        df_in = meta_handle.read_df("global",read_metadata=True)
        self.all_channels = df_in.metadata['channels']
        df_in = meta_handle.read_df("kymo")
        timepoint_num = len(df_in.index.get_level_values(3).unique().tolist())
        self.t_range = (0,timepoint_num)
        
        self.final_params = {}
        
    def choose_seg_channel(self,seg_channel):
        self.seg_channel = seg_channel
        
    def plot_img_list(self,img_list):
        nrow = (len(img_list)+1)//2
        fig, axes = plt.subplots(nrows=nrow, ncols=2, figsize=self.fig_size)
        for i in range(len(img_list)):
            img = img_list[i]
            if nrow < 2:
                axes[i%2].imshow(img)
            else:
                axes[i//2,i%2].imshow(img)
        if len(img_list)%2 == 1:
            if nrow < 2:
                axes[-1].axis('off')
            else: 
                axes[-1, -1].axis('off')
        plt.tight_layout()
        plt.show()         
        
    def import_array(self,fov_idx,n_trenches,t_range=(0,None),t_subsample_step=1,fig_size_y=9,fig_size_x=6):
        self.fig_size = (fig_size_y,fig_size_x)
        with h5py.File(self.input_file_prefix + str(fov_idx) + ".hdf5", "r") as hdf5_handle:
            ttl_trenches = len(hdf5_handle.keys())
            chosen_trenches = np.random.choice(np.array(list(range(0,ttl_trenches))),size=n_trenches,replace=False)
            chosen_trench_keys = np.array(list(hdf5_handle.keys()))[chosen_trenches].tolist()
            if t_range[1] == None:
                array_list = [hdf5_handle[trench_key + "/" + self.seg_channel][:,:,t_range[0]::t_subsample_step] for trench_key in chosen_trench_keys]
            else:
                array_list = [hdf5_handle[trench_key + "/" + self.seg_channel][:,:,t_range[0]:t_range[1]+1:t_subsample_step] for trench_key in chosen_trench_keys]
            output_array = np.concatenate(np.expand_dims(array_list,axis=0),axis=0)
            self.t_tot = output_array.shape[3]
        self.plot_kymographs(output_array)
        return output_array
    
    def plot_kymographs(self,kymo_arr):
        input_kymo = kymo_handle()
        img_list = []
        for k in range(kymo_arr.shape[0]):
            input_kymo.import_wrap(kymo_arr[k])
            img_list.append(input_kymo.return_unwrap(padding=self.wrap_pad))
        self.plot_img_list(img_list)
        return img_list
    
    def plot_scaled(self,kymo_arr,scale,scaling_percentile):
        self.final_params['Scale Fluorescence?'] = scale
        self.final_params["Scaling Percentile:"] = scaling_percentile
        input_kymo = kymo_handle()
        scaled_list = []
        for k in range(kymo_arr.shape[0]):
            input_kymo.import_wrap(kymo_arr[k],scale=scale,scale_perc=scaling_percentile)
            scaled_list.append(input_kymo.return_unwrap(padding=self.wrap_pad))
        self.plot_img_list(scaled_list)
        return scaled_list
    
    def plot_processed(self,scaled_list,smooth_sigma):
        self.final_params['Gaussian Kernel Sigma:'] = smooth_sigma
        proc_list = []
        for scaled in scaled_list:
            proc_img = self.preprocess_img(scaled,sigma=smooth_sigma)
            proc_list.append(proc_img)
        self.plot_img_list(proc_list)
        return proc_list
    
    def plot_eigval(self,proc_list):
        eigval_list = []
        for proc in proc_list:
            inverted = sk.util.invert(proc)
            min_eigvals = self.to_8bit(self.hessian_contrast_enc(inverted,self.hess_pad))
            eigval_list.append(min_eigvals)
        self.plot_img_list(eigval_list)
        return eigval_list
    
    def plot_cell_mask(self,proc_list,global_otsu_scaling,cell_mask_method,cell_otsu_scaling,local_otsu_r):
        self.final_params['Cell Mask Thresholding Method:'] = cell_mask_method
        self.final_params['Global Threshold Scaling:'] = global_otsu_scaling
        self.final_params['Cell Threshold Scaling:'] = cell_otsu_scaling
        self.final_params['Local Otsu Radius:'] = local_otsu_r
        
        cell_mask_list = []
        for proc in proc_list:
            cell_mask = self.cell_region_mask(proc,method=cell_mask_method,global_otsu_scaling=global_otsu_scaling,cell_otsu_scaling=cell_otsu_scaling,t_tot=self.t_tot,local_otsu_r=local_otsu_r)
            cell_mask_list.append(cell_mask)
        self.plot_img_list(cell_mask_list)
        return cell_mask_list
    
    def plot_threshold_result(self,eigval_list,cell_mask_list,edge_threshold_scaling,min_obj_size):
        composite_mask_list = []
        edge_mask_list = []
        for i,min_eigvals in enumerate(eigval_list):
            cell_mask = cell_mask_list[i]
            
            eig_kymo = kymo_handle()
            eig_kymo.import_unwrap(min_eigvals,self.t_tot,padding=self.wrap_pad)
            wrap_eig = eig_kymo.return_wrap()
            edge_threshold = self.get_mid_threshold_arr(wrap_eig,edge_threshold_scaling=edge_threshold_scaling,padding=self.wrap_pad)
            
            composite_mask = self.find_mask(cell_mask,min_eigvals,edge_threshold,min_obj_size=min_obj_size)
            composite_mask_list.append(composite_mask)
        
        self.plot_img_list(composite_mask_list)
        return composite_mask_list
            
        
    def plot_scores(self,eigval_list,cell_mask_list,edge_threshold_scaling,threshold_step_perc,threshold_perc_num_steps,min_obj_size):
        self.final_params['Edge Threshold Scaling:'] = edge_threshold_scaling
        self.final_params['Threshold Step Percent:'] = threshold_step_perc
        self.final_params['Number of Threshold Steps:'] = threshold_perc_num_steps
        self.final_params['Minimum Object Size:'] = min_obj_size
        
        conv_scores_list = []
        for i,min_eigvals in enumerate(eigval_list):
            cell_mask = cell_mask_list[i]
            
            eig_kymo = kymo_handle()
            eig_kymo.import_unwrap(min_eigvals,self.t_tot,padding=self.wrap_pad)
            wrap_eig = eig_kymo.return_wrap()
            mid_threshold_arr = self.get_mid_threshold_arr(wrap_eig,edge_threshold_scaling=edge_threshold_scaling,padding=self.wrap_pad)
            
            conv_scores = self.get_scores(cell_mask,min_eigvals,mid_threshold_arr,\
                                          threshold_step_perc=threshold_step_perc,threshold_perc_num_steps=threshold_perc_num_steps,min_obj_size=min_obj_size)
            conv_scores_list.append(conv_scores)
        self.plot_img_list(conv_scores_list)
        return conv_scores_list
    
    def plot_final_mask(self,conv_scores_list,convex_threshold):
        self.final_params['Convexity Threshold:'] = convex_threshold
        
        final_mask_list = []
        for conv_scores in conv_scores_list:
            final_mask = (conv_scores>convex_threshold)
            final_mask_list.append(final_mask)
        self.plot_img_list(final_mask_list)
        return final_mask_list
    
    def process_results(self):
        self.final_params["Segmentation Channel:"] = self.seg_channel
        for key,value in self.final_params.items():
            print(key + " " + str(value))
    
    def write_param_file(self):
        with open(self.headpath + "/fluorescent_segmentation.par", "wb") as outfile:
            pickle.dump(self.final_params, outfile)