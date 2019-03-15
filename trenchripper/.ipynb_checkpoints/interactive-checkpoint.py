import matplotlib.pyplot as plt
import numpy as np
import skimage as sk

from skimage import filters
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from .kymograph import kymograph_multifov

class kymograph_interactive(kymograph_multifov):
    def __init__(self,input_file_prefix,all_channels,fov_list,trench_len_y=270,padding_y=20,trench_width_x=30,\
                 t_subsample_step=1,t_range=(0,-1),y_percentile=85,y_min_edge_dist=50,smoothing_kernel_y=(9,1),\
                 triangle_nbins=50,triangle_scaling=1.,x_percentile=85,background_kernel_x=(301,1),smoothing_kernel_x=(9,1),\
                 otsu_nbins=50,otsu_scaling=1.):
        """The kymograph class is used to generate and visualize kymographs. The central function of this
        class is the method 'generate_kymograph', which takes an hdf5 file of images from a single fov and
        outputs an hdf5 file containing kymographs from all detected trenches.

        NOTE: I need to revisit the row detection, must ensure there can be no overlap...
            
        Args:
            input_file_prefix (string): File prefix for all input hdf5 files of the form
            [input_file_prefix][number].hdf5 
            all_channels (list): list of strings corresponding to the different image channels
            available in the input hdf5 file, with the channel used for segmenting trenches in
            the first position. NOTE: these names must match those of the input hdf5 file datasets.
            trench_len_y (int): Length from the end of the tenches to be used when cropping in the 
            y-dimension.
            padding_y (int): Padding to be used when cropping in the y-dimension.
            trench_width_x (int): Width to be used when cropping in the x-dimension.
            fov_list (list): List of ints corresponding to fovs of interest.

            t_subsample_step(int): Step size to be used for subsampling input files in time.

            y_percentile (int): Used for reducing signal in xyt to only the yt dimension when cropping
            in the y-dimension.
            y_min_edge_dist (int): Used when detecting present rows, filters for a minimum row size along the y dimension.
            smoothing_kernel_y (tuple): Two-entry tuple specifying a kernel size for smoothing out yt
            signal when cropping in the y-dimension.
            triangle_nbins (int): Number of bins to use when applying the triangle method to y-dimension signal.
            triangle_scaling (float): Threshold scaling factor for triangle method thresholding.

            x_percentile (int): Used for reducing signal in xyt to only the xt dimension when cropping
            in the x-dimension.
            background_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing background subtraction
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.
            smoothing_kernel_x (tuple): Two-entry tuple specifying a kernel size for performing smoothing
            on xt signal when cropping in the x-dimension. Dim_1 (time) should be set to 1.
            otsu_nbins (int): Number of bins to use when applying Otsu's method to x-dimension signal.
            otsu_scaling (float): Threshold scaling factor for Otsu's method thresholding.
        """
        super(kymograph_interactive, self).__init__(input_file_prefix,all_channels,fov_list,trench_len_y=trench_len_y,\
            padding_y=padding_y,trench_width_x=trench_width_x,t_subsample_step=t_subsample_step,t_range=t_range,y_percentile=y_percentile,\
            y_min_edge_dist=y_min_edge_dist,smoothing_kernel_y=smoothing_kernel_y,triangle_nbins=triangle_nbins,\
            triangle_scaling=triangle_scaling,x_percentile=x_percentile,background_kernel_x=background_kernel_x,\
            smoothing_kernel_x=smoothing_kernel_x,otsu_nbins=otsu_nbins,otsu_scaling=otsu_scaling)

    def preview_y_precentiles(self,imported_array_list, y_percentile, smoothing_kernel_y_dim_0,\
                          triangle_nbins,triangle_scaling):        
	    y_percentiles_smoothed_list = self.map_to_fovs(self.get_smoothed_y_percentiles,imported_array_list,\
	                                                   y_percentile,(smoothing_kernel_y_dim_0,1))
	    
	    thresholds = [sk.filters.threshold_triangle(y_percentiles_smoothed,nbins=triangle_nbins)*triangle_scaling for y_percentiles_smoothed in y_percentiles_smoothed_list]
	    self.plot_y_precentiles(y_percentiles_smoothed_list,self.fov_list,thresholds)
           
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
            thr_z = np.repeat(thresholds[j],x_len*y_len)
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
        
        
    def preview_y_crop(self,y_percentiles_smoothed_list, imported_array_list, triangle_nbins, triangle_scaling,\
                       y_min_edge_dist, padding_y, trench_len_y, top_orientation, vertical_spacing):
        
        trench_edges_y_lists = self.map_to_fovs(self.get_trench_edges_y,y_percentiles_smoothed_list,triangle_nbins,\
                                               triangle_scaling,y_min_edge_dist)
        row_num_list = self.map_to_fovs(self.get_row_numbers,trench_edges_y_lists)
        cropped_in_y_list = self.map_to_fovs(self.crop_y,trench_edges_y_lists,row_num_list,imported_array_list,padding_y,\
                                             trench_len_y,top_orientation)
        self.plot_y_crop(cropped_in_y_list,imported_array_list,self.fov_list,vertical_spacing,row_num_list)
        
    def plot_y_crop(self,cropped_in_y_list,imported_array_list,fov_list,vertical_spacing,row_num_list):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        time_list = range(1,imported_array_list[0].shape[3]+1)
        
        nrows = np.sum(row_num_list)
        ncols = len(time_list)
        
        idx = 0
        for i,cropped_in_y in enumerate(cropped_in_y_list):
            num_rows = row_num_list[i]
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
        
        smoothed_x_percentiles_list = self.map_to_fovs(self.get_smoothed_x_percentiles,cropped_in_y_list,x_percentile,\
                                                         (background_kernel_x,1),(smoothing_kernel_x,1))
        thresholds = []
        for smoothed_x_percentiles_row in smoothed_x_percentiles_list:
            for smoothed_x_percentiles in smoothed_x_percentiles_row:
                thresholds.append(sk.filters.threshold_otsu(smoothed_x_percentiles,nbins=otsu_nbins)*otsu_scaling)
        self.plot_x_percentiles(smoothed_x_percentiles_list,self.fov_list, t, thresholds,vertical_spacing,num_rows=2)
        
        
    def plot_x_percentiles(self,smoothed_x_percentiles_list,fov_list,t,thresholds,vertical_spacing,num_rows=2):
        fig = plt.figure()
        
        ncol=num_rows
        nrow=len(smoothed_x_percentiles_list)
        
        idx = 0
        for i,smoothed_x_percentiles_top_bottom in enumerate(smoothed_x_percentiles_list):
            for j,smoothed_x_percentiles in enumerate(smoothed_x_percentiles_top_bottom):
                idx += 1
                data = smoothed_x_percentiles[:,t]
                ax = fig.add_subplot(ncol, nrow, idx)
                ax.plot(data)
                
                current_threshold = thresholds[idx-1]
                threshold_data = np.repeat(current_threshold,len(data))
                ax.plot(threshold_data,c='r')
                ax.set_title("FOV: " + str(fov_list[j]))
                ax.set_xlabel('x position')
                ax.set_ylabel('intensity')

        plt.show()
    
    
    def preview_midpoints(self,smoothed_x_percentiles_list,otsu_nbins,otsu_scaling,vertical_spacing):
        all_midpoints_list = self.map_to_fovs(self.get_all_midpoints,smoothed_x_percentiles_list,otsu_nbins,otsu_scaling)
        self.plot_midpoints(all_midpoints_list,self.fov_list,vertical_spacing)
        
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
                                                 
           
    def preview_kymographs(self,cropped_in_y_list,all_midpoints_list,x_drift_list,trench_width_x,vertical_spacing):
        cropped_in_x_list = self.map_to_fovs(self.get_crop_in_x,cropped_in_y_list,all_midpoints_list,x_drift_list,\
                                             trench_width_x)
        self.plot_kymographs(cropped_in_x_list,self.fov_list,vertical_spacing)
    
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