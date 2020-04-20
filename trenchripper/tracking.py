import skimage as sk
import sklearn as skl
import numpy as np
import pandas as pd
import pickle as pkl

import copy
import h5py

from sklearn import cluster
from scipy.special import logsumexp
from matplotlib import pyplot as plt
from matplotlib import cm
from pulp import *
from ipywidgets import interactive, fixed, FloatSlider, IntSlider, IntRangeSlider, SelectMultiple

from .utils import pandas_hdf5_handler,writedir
from .cluster import dask_controller

def get_labeled_data(kymo_arr,orientation):
    labeled_data = []
    for t in range(kymo_arr.shape[0]):
        if orientation == 1:
            flipped_arr = kymo_arr[t,::-1]
            labeled = sk.morphology.label(flipped_arr,connectivity=1)
        else:
            labeled = sk.morphology.label(kymo_arr[t],connectivity=1)
        unique_idx = np.unique(labeled[-1])
        remove_indices = unique_idx[unique_idx!=0]
        for remove_idx in remove_indices:
            labeled[labeled==remove_idx] = 0
        labeled_data.append(labeled)
    labeled_data = np.array(labeled_data)
    return labeled_data

def get_segment_props(labeled_data,interpolate_empty_trenches=False):
    all_centroids = []
    all_sizes = []
    all_bboxes = []
    
    for t in range(labeled_data.shape[0]):
        rps = sk.measure.regionprops(labeled_data[t])
        centroids = [rp.centroid for rp in rps]
        sizes = np.array([rp.area for rp in rps])
        bboxes = np.array([rp.bbox for rp in rps])
        centroids = np.array([[centroid[1],centroid[0]] for centroid in centroids])
        all_centroids.append(centroids)
        all_sizes.append(sizes)
        all_bboxes.append(bboxes)
    
    if interpolate_empty_trenches:
        empty_trenches = []
        for t in range(labeled_data.shape[0]):
            if len(all_sizes[t]) == 0:
                all_centroids[t] = all_centroids[t-1]
                all_sizes[t] = all_sizes[t-1]
                all_bboxes[t] = all_bboxes[t-1]
                empty_trenches.append(t)
        return all_centroids,all_sizes,all_bboxes,empty_trenches
    else:
        return all_centroids,all_sizes,all_bboxes

class scorefn:
    def __init__(self,headpath,segfolder,u_pos=0.2,sig_pos=0.4,u_size=0.,sig_size=0.2):
        self.headpath = headpath
        self.segpath = headpath + "/" + segfolder
        self.u_pos,self.sig_pos = (u_pos,sig_pos)
        self.u_size,self.sig_size = (u_size,sig_size)
        self.viewpadding = 0
    
    def fpos(self,xT,xt,bbox_T,bbox_t,u_pos=0.2,sig_pos=0.4):
        min_edge_T,min_edge_t = (np.min(bbox_T[:,0]),np.min(bbox_t[:,0]))
        l_T = xT[:,1]-min_edge_T
        l_t = xt[:,1]-min_edge_t
        del_l = np.subtract.outer(l_T,l_t)
        norm_del_l = del_l/l_t
        f_pos = -((norm_del_l-u_pos)**2)/(2*(sig_pos**2))
        return f_pos,norm_del_l

    def del_pos(self,centroids,bboxes):
        posscores = []
        for t in range(len(centroids)-1):
            posscore,_ = self.fpos(centroids[t+1],centroids[t],bboxes[t+1],bboxes[t],u_pos=self.u_pos,sig_pos=self.sig_pos)
            posscores.append(posscore.T)
        posscores = np.array(posscores)
        return posscores
        
    def del_split_pos(self,centroids,union_centroids,bboxes):
        posscores = []
        for t in range(len(centroids)-1):
            if len(union_centroids[t+1]) > 0:
                posscore,_ = self.fpos(union_centroids[t+1],centroids[t],bboxes[t+1],bboxes[t],u_pos=self.u_pos,sig_pos=self.sig_pos)
            else:
                posscore = np.array([[]])
            posscores.append(posscore.T)   
        posscores = np.array(posscores)
        return posscores
    
    def del_merge_pos(self,centroids,union_centroids,bboxes):
        posscores = []
        for t in range(len(centroids)-1):
            if len(union_centroids[t]) > 0:
                posscore,_ = self.fpos(centroids[t+1],union_centroids[t],bboxes[t+1],bboxes[t],u_pos=self.u_pos,sig_pos=self.sig_pos)
            else:
                posscore = np.array([[]])
            posscores.append(posscore.T)   
        posscores = np.array(posscores)
        return posscores
    
    def fsize(self,sizeT,sizet,u_size=0.,sig_size=0.2):
        sizediff = np.subtract.outer(sizeT,sizet)
        normsizediff = sizediff/sizet
        f_size = -((normsizediff-u_size)**2)/(2*(sig_size**2))
        f_size[normsizediff<-0.1] = -np.inf

        return f_size,normsizediff
        
    def del_size(self,sizes):
        sizescores = []
        for t in range(len(sizes)-1):
            sizescore,_ = self.fsize(sizes[t+1],sizes[t],u_size=self.u_size,sig_size=self.sig_size)
            sizescores.append(sizescore.T)   
        sizescores = np.array(sizescores)
        return sizescores
    
    def del_half_size(self,sizes):
        sizescores = []
        for t in range(len(sizes)-1):
            size_ti = sizes[t]/2.
            size_tf = sizes[t+1]
            sizescore,_ = self.fsize(size_tf,size_ti,u_size=self.u_size,sig_size=self.sig_size)
            sizescores.append(sizescore.T)   
        sizescores = np.array(sizescores)
        return sizescores

    def del_split_size(self,sizes,union_sizes):
        sizescores = []
        for t in range(len(sizes)-1):
            if len(union_sizes[t+1]) > 0:
                sizescore,_ = self.fsize(union_sizes[t+1],sizes[t],u_size=self.u_size,sig_size=self.sig_size)
            else:
                sizescore = np.array([[]])
            sizescores.append(sizescore.T)   
        sizescores = np.array(sizescores)
        return sizescores
    
    def del_merge_size(self,sizes,union_sizes):
        sizescores = []
        for t in range(len(sizes)-1):
            if len(union_sizes[t]) > 0:
                sizescore,_ = self.fsize(sizes[t+1],union_sizes[t],u_size=self.u_size,sig_size=self.sig_size)
            else:
                sizescore = np.array([[]])
            sizescores.append(sizescore.T)   
        sizescores = np.array(sizescores)
        return sizescores
    
    def get_unions(self,centroids,sizes):
        union_sizes = []
        union_centroids = []
        for t in range(len(sizes)):
            union_sizes_t = []
            union_centroids_t = []
            for k in range(1,sizes[t].shape[0]):
                union_size = sizes[t][k-1] + sizes[t][k]
                union_centroid = (sizes[t][k-1]*centroids[t][k-1] + sizes[t][k]*centroids[t][k])/union_size
                union_sizes_t.append(union_size)
                union_centroids_t.append(union_centroid)
            union_sizes_t = np.array(union_sizes_t)
            union_centroids_t = np.array(union_centroids_t)
            union_sizes.append(union_sizes_t)
            union_centroids.append(union_centroids_t)
        size_diffs = [size[1:]-size[:-1] for size in sizes]
        return union_centroids,union_sizes,size_diffs
        
    
    def compute_score_arrays(self,centroids,sizes,bboxes):
        union_centroids,union_sizes,size_diffs = self.get_unions(centroids,sizes)
        
        sizeij = self.del_size(sizes)
        posij = self.del_pos(centroids,bboxes)
        
        posik = self.del_split_pos(centroids,union_centroids,bboxes)
        sizeik = self.del_split_size(sizes,union_sizes)
        
        poski = self.del_merge_pos(centroids,union_centroids,bboxes)
        sizeki = self.del_merge_size(sizes,union_sizes)
        
        half_sizeij = self.del_half_size(sizes)
        
        pos_e = [item[:,-1:] for item in posij]
        size_e = [item[:,-1:] for item in half_sizeij]
        
        
        Cij = []
        Cik = []
        Cki = []
        Ce = []
        
        for t in range(len(posij)):
            
            test_posij = np.isnan(np.sum(posij[t]))
            test_sizeij = np.isnan(np.sum(sizeij[t]))
            test_posik = np.isnan(np.sum(posik[t]))
            test_sizeik = np.isnan(np.sum(sizeik[t]))
            test_poski = np.isnan(np.sum(poski[t]))
            test_sizeki = np.isnan(np.sum(sizeki[t]))
            test_pos_e = np.isnan(np.sum(pos_e[t]))
            test_size_e = np.isnan(np.sum(size_e[t]))
            test_union_centroids = np.isnan(np.sum(union_centroids[t]))
            test_union_sizes = np.isnan(np.sum(union_sizes[t]))
            test_size_diffs = np.isnan(np.sum(size_diffs[t]))
            
            test_arr = [test_posij,test_sizeij,test_posik,test_sizeik,\
                       test_poski,test_sizeki,test_pos_e,test_size_e,\
                       test_union_centroids,test_union_sizes,test_size_diffs]
            
            arr_key = ["test_posij","test_sizeij","test_posik","test_sizeik",\
                       "test_poski","test_sizeki","test_pos_e","test_size_e",\
                       "test_union_centroids","test_union_sizes","test_size_diffs"]
            if np.any(test_arr):
                print(arr_key[test_arr])
            
            f_ij = posij[t]+sizeij[t]
            f_ik = posik[t]+sizeik[t]
            f_ki = poski[t]+sizeki[t]
            f_e = pos_e[t]+size_e[t]
            
            pad_f_ki = np.pad(f_ki,((1,1),(0,0)),"constant",constant_values=0.)

            if f_ik.shape[0] == 0:
                f_ik = np.zeros(f_ij.shape,dtype=float)
                
            
            log_sum = np.concatenate((f_ij,f_ik,pad_f_ki[1:],pad_f_ki[:-1]),axis=1)

            log_sum[:,0:1] += f_e
                    
            log_sum = logsumexp(log_sum,axis=1,keepdims=True)
            undefined_logsum = (log_sum==-np.inf)[:,0]

            f_ij[undefined_logsum] = 0.
            f_ik[undefined_logsum] = 0.
            f_ki[undefined_logsum[:-1]] = 0.
            f_e[undefined_logsum] = 0.
                        
            log_prob_ij = f_ij-log_sum
            log_prob_ik = f_ik-log_sum
            log_prob_ki = f_ki-log_sum[:-1]
            log_prob_e = f_e-log_sum
            
            ij_score = (-log_prob_ij)-100.
            ij_score[ij_score>0] = 0.
            ij_score[ij_score<-100.] = -100.
            Cij.append(ij_score)
        
            ik_score = ((-log_prob_ik)-100.)
            ik_score[ik_score>0] = 0.
            ik_score[ik_score<-100.] = -100.
            Cik.append(ik_score)    
        
            ki_score = 1.8*(-log_prob_ki-100.)
            ki_score[ki_score>0] = 0.
            ki_score[ki_score<-100.] = -100.
            Cki.append(ki_score)
        
            e_score = ((-log_prob_e)-100.)
            e_score[e_score>0.] = 0.
            e_score[e_score<-100.] = -100.
            Ce.append(e_score)
                                    
        return Cij,Cik,Cki,Ce,posij,posik,poski
    
    def compute_neighbor_stats(self,centroids,bboxes,sizes):
        neighbor_dists = []
        neighbor_sizes = []
        for t in range(len(centroids)-1):
            _,norm_del_l = self.fpos(centroids[t+1],centroids[t],bboxes[t+1],bboxes[t],u_pos=self.u_pos,sig_pos=self.sig_pos)
            _,normsizediff = self.fsize(sizes[t+1],sizes[t],u_size=self.u_size,sig_size=self.sig_size)
            
            pos_norm_del_l = copy.copy(norm_del_l)
            pos_norm_del_l[pos_norm_del_l<0.] = np.inf
            
            pos_norm_sizediff = copy.copy(normsizediff)
            pos_norm_sizediff[pos_norm_sizediff<0.] = np.inf
            
            neighbor_idx = np.argmin(pos_norm_del_l,axis=0)

            neighbor_dist = pos_norm_del_l[neighbor_idx,np.arange(pos_norm_del_l.shape[1])].flatten()
            neighbor_size = pos_norm_sizediff[neighbor_idx,np.arange(pos_norm_sizediff.shape[1])].flatten()
            
            neighbor_dist = neighbor_dist[neighbor_dist != np.inf].tolist()
            neighbor_size = neighbor_size[neighbor_size != np.inf].tolist()
            
            neighbor_dists += neighbor_dist
            neighbor_sizes += neighbor_size
        neighbor_dists = np.array(neighbor_dists)
        neighbor_sizes = np.array(neighbor_sizes)
        return neighbor_dists,neighbor_sizes
    
    def plot_neighbor_stats(self,ax_dist,ax_size,centroids,bboxes,sizes):
        neighbor_dists,neighbor_sizes = self.compute_neighbor_stats(centroids,bboxes,sizes)
        
        dist_avg = np.mean(neighbor_dists)
        dist_std = np.std(neighbor_dists)
        size_avg = np.mean(neighbor_sizes)
        size_std = np.std(neighbor_sizes)
        
        ax_dist.set_xlim(0.,dist_avg+(3.*dist_std))
        ax_size.set_xlim(0.,size_avg+(3.*size_std))
        
        ax_dist.hist(neighbor_dists,bins=30)
        ax_dist.set_title('Nearest-neighbor Normalized Centroid Distance')
        ax_dist.axvline(x=self.u_pos,c="r",linewidth=3,zorder=10)
        ax_dist.axvspan(self.u_pos-self.sig_pos, self.u_pos+self.sig_pos, alpha=0.3, color='r',zorder=10)
        
        ax_size.hist(neighbor_sizes,bins=30)
        ax_size.set_title('Nearest-neighbor Normalized Size Difference')
        ax_size.axvline(x=self.u_size,c="r",linewidth=3,zorder=10)
        ax_size.axvspan(self.u_size-self.sig_size, self.u_size+self.sig_size, alpha=0.3, color='r',zorder=10)
        
        print("==========================================================")
        print("Mean Normalized Centroid Distance: " + str(dist_avg))
        print("Standard Deviation of Normalized Centroid Distance: " + str(dist_std))
        print("")
        print("Mean Normalized Size Difference: " + str(size_avg))
        print("Standard Deviation of Normalized Size Difference: " + str(size_std))
        
        plt.show()
    
    def plot_segmentation(self,ax,kymo_arr,t0=0,tf=-1,cmap='coolwarm'):
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])
    
    def plot_ij_scores(self,ax,kymo_arr,centroids,Cij,t0=0,tf=-1,cmap='coolwarm'):
        cmap_fn = cm.get_cmap(cmap, 100)
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        if tf == -1:
            tf = len(centroids)-1
            
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])

            adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r",s=100,zorder=10)

            if t > t0:

                N_detect_i = len(centroids[t-1])
                N_detect_f = len(centroids[t])

                for i in range(N_detect_i):

                    for j in range(N_detect_f):

                        centroid_i = centroids[t-1][i]
                        centroid_f = centroids[t][j]

                        color_score = abs(Cij[t-1][i,j]/-100)

                        ax.plot([centroid_i[0] + (t-1)*x_dim + self.viewpadding, centroid_f[0] + t*x_dim + self.viewpadding],[centroid_i[1],centroid_f[1]],c=cmap_fn(color_score),linewidth=3)
        adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
        ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r")
        
    def plot_ik_scores(self,ax,kymo_arr,centroids,Cik,t0=0,tf=-1,cmap='coolwarm'):
        cmap_fn = cm.get_cmap(cmap, 100)
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        if tf == -1:
            tf = len(centroids)-1        
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])

            merged_centroids = (centroids[t][:-1,:]+centroids[t][1:,:])/2

            adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            adjusted_merged_centroids = merged_centroids + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])

            if t > t0:

                N_detect_i = len(centroids[t-1])
                N_detect_f = len(centroids[t])

                for i in range(N_detect_i):
                    for j in range(N_detect_f-1):

                        centroid_i = centroids[t-1][i]

                        centroid_k1 = centroids[t][j]
                        centroid_k2 = centroids[t][j+1]

                        color_score = abs(Cik[t-1][i,j]/-100)

                        ax.plot([centroid_i[0] + (t-1)*x_dim + self.viewpadding, centroid_k1[0] + t*x_dim + self.viewpadding],\
                                [centroid_i[1],(centroid_k1[1]+centroid_k2[1])/2],c=cmap_fn(color_score),linewidth=3)
            ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r",s=100,zorder=10)
            ax.scatter(adjusted_merged_centroids[:,0],adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)

        merged_centroids = (centroids[tf][:-1,:]+centroids[tf][1:,:])/2

        adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
        adjusted_merged_centroids = merged_centroids + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])

        ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r",s=100,zorder=10)
        ax.scatter(adjusted_merged_centroids[:,0],adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)
    
    def plot_ki_scores(self,ax,kymo_arr,centroids,Cki,t0=0,tf=-1,cmap='coolwarm'):
        cmap_fn = cm.get_cmap(cmap, 100)
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        if tf == -1:
            tf = len(centroids)-1        
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])

            merged_centroids = (centroids[t][:-1,:]+centroids[t][1:,:])/2

            adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            adjusted_merged_centroids = merged_centroids + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])

            if t > t0:

                N_detect_i = len(centroids[t-1])
                N_detect_f = len(centroids[t])

                for i in range(N_detect_i-1):
                    for j in range(N_detect_f):

                        centroid_k1 = centroids[t-1][i]
                        centroid_k2 = centroids[t-1][i+1]

                        centroid_i = centroids[t][j]

                        color_score = abs(Cki[t-1][i,j]/-100)

                        ax.plot([centroid_k1[0] + (t-1)*x_dim + self.viewpadding, centroid_i[0] + t*x_dim + self.viewpadding],\
                                [(centroid_k1[1]+centroid_k2[1])/2,centroid_i[1]],c=cmap_fn(color_score),linewidth=3)

            ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r",s=100,zorder=10)
            ax.scatter(adjusted_merged_centroids[:,0],adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)

        merged_centroids = (centroids[tf][:-1,:]+centroids[tf][1:,:])/2

        adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
        adjusted_merged_centroids = merged_centroids + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])

        ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r",s=100,zorder=10)
        ax.scatter(adjusted_merged_centroids[:,0],adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)
        
    def plot_score_metrics(self,kymo_df,trenchid,t_range=(0,-1),u_size=0.25,sig_size=0.05,u_pos=0.25,sig_pos=0.1,viewpadding=0):
        self.viewpadding = viewpadding
        
        trench = kymo_df.loc[trenchid]
        file_idx = trench["File Index"].unique().tolist()[0]
        trench_idx = trench["File Trench Index"].unique().tolist()[0]
        orientation_dict = {"top":0,"bottom":1}
        orientation = trench["lane orientation"].unique().tolist()[0]
        orientation = orientation_dict[orientation]
                
        with h5py.File(self.segpath + "/segmentation_" + str(file_idx) + ".hdf5") as infile:
            data = infile["data"][trench_idx]
        
        t0,tf = t_range
        self.u_pos,self.sig_pos = (u_pos,sig_pos)
        self.u_size,self.sig_size = (u_size,sig_size)
        labeled_data = get_labeled_data(data,orientation)
        
        centroids,sizes,bboxes,empty_trenches = get_segment_props(labeled_data,interpolate_empty_trenches=True)
        Cij,Cik,Cki,_,_,_,_ = self.compute_score_arrays(centroids,sizes,bboxes)
            
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        fig1.set_size_inches(18, 4)
        
        
        self.plot_neighbor_stats(ax1,ax2,centroids,bboxes,sizes)
                
        fig2, axs = plt.subplots(4)
        fig2.set_size_inches(18, 24)
        
        self.plot_segmentation(axs[0],labeled_data,t0=t0,tf=tf,cmap='coolwarm')
        axs[0].set_title('Segmentation')
        self.plot_ij_scores(axs[1],labeled_data,centroids,Cij,t0=t0,tf=tf,cmap='coolwarm')
        axs[1].set_title('Movement Transition Score')
        self.plot_ik_scores(axs[2],labeled_data,centroids,Cik,t0=t0,tf=tf,cmap='coolwarm')
        axs[2].set_title('Division Transition Score')
        self.plot_ki_scores(axs[3],labeled_data,centroids,Cki,t0=t0,tf=tf,cmap='coolwarm')
        axs[3].set_title('Merge Transition Score')
        
        plt.show()
        
        return data,orientation,empty_trenches
        
    def interactive_scorefn(self):
        meta_handle = pandas_hdf5_handler(self.headpath+"/metadata.hdf5")
        kymo_df = meta_handle.read_df("kymograph")
        num_trenches = len(kymo_df.index.get_level_values(0).unique())
        timepoints = len(kymo_df.index.get_level_values(1).unique())
  
        self.output = interactive(self.plot_score_metrics,{"manual":True},\
                 kymo_df = fixed(kymo_df),trenchid=IntSlider(value=0, min=0, max=num_trenches-1, step=1),\
                 t_range=IntRangeSlider(value=[0, timepoints-1],min=0,max=timepoints-1,\
                 step=1,disabled=False,continuous_update=False),\
                 u_size=FloatSlider(value=self.u_size, min=0., max=1., step=0.01),\
                 sig_size=FloatSlider(value=self.sig_size, min=0., max=0.5, step=0.01),\
                 u_pos=FloatSlider(value=self.u_pos, min=0., max=1., step=0.01),\
                 sig_pos=FloatSlider(value=self.sig_pos, min=0., max=0.5, step=0.01),\
                 viewpadding=IntSlider(value=0, min=0, max=20, step=1));
        
        display(self.output)
            
class tracking_solver:
    def __init__(self,headpath,segfolder,paramfile=False,ScoreFn=False,intensity_channel_list=None,props_list=['centroid','area'],\
                 pixel_scaling_fns=[(lambda x,p: (x[0]*p,x[1]*p)),(lambda x,p: x*(p**2))],\
                 intensity_props_list=['mean_intensity'],merge_per_iter=0,conv_tolerence=2,edge_limit=3,\
                 u_size=0.22,sig_size=0.08,u_pos=0.21,sig_pos=0.1):
        
        if paramfile:
            parampath = headpath + "/lineage_tracing.par"
            with open(parampath, 'rb') as infile:
                param_dict = pkl.load(infile)
            
            intensity_channel_list = param_dict["Channels:"]
            merge_per_iter = param_dict["Merges Detected Per Iteration:"]
            conv_tolerence = param_dict["# Unimproved Iterations Before Stopping:"]
            edge_limit = param_dict["Closest N Objects to Consider:"]
            u_size = param_dict["Mean Size Increase:"]
            sig_size = param_dict["Standard Deviation of Size Increase:"]
            u_pos = param_dict["Mean Position Increase:"]
            sig_pos = param_dict["Standard Deviation of Position Increase:"]
        
        self.headpath = headpath
        self.segpath = headpath + "/" + segfolder
        self.kymopath = headpath + "/kymograph"
        self.lineagepath = headpath + "/lineage"
        self.metapath = self.headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = fovdf.metadata
        self.intensity_channel_list = intensity_channel_list
        self.props_list = props_list
        
        
        self.pixel_scaling_fns = pixel_scaling_fns
        self.intensity_props_list = intensity_props_list
        
        if ScoreFn:
            self.ScoreFn = ScoreFn
        else:
            self.ScoreFn = scorefn(headpath,segfolder,u_size=u_size,sig_size=sig_size,u_pos=u_pos,sig_pos=sig_pos)
        self.merge_per_iter = merge_per_iter
        self.conv_tolerence = conv_tolerence
        self.edge_limit = edge_limit
        
        self.viewpadding = 0

    def define_tracking_problem(self,Cij,Cik,Cki,posij,posik,poski,sizes,merge_per_iter,edge_limit):

        prob = LpProblem("Lineage",LpMinimize)
        
        Aij = {}
        Aik = {}
        Aki = {}
        Ai = {}

        obj_fn_terms = []
        obj_fn_coeff = []
        
        valid_j_list = []
        
        for t in range(len(Cij)):
            cij = Cij[t]
            
            Aij[t] = {"ij":{}, "ji":{}}
            Aik[t] = {"ik":{}, "ki":{}}
            Aki[t] = {"ki":{}, "ik":{}}
            
            for i in range(cij.shape[0]):
                Aij[t]["ij"][i] = {}
                Aik[t]["ik"][i] = {}
                if i < (cij.shape[0]-1):
                    Aki[t]["ki"][i] = {}
                
            for j in range(cij.shape[1]):
                Aij[t]["ji"][j] = {}
                Aki[t]["ik"][j] = {}
                if j < (cij.shape[1]-1):
                    Aik[t]["ki"][j] = {}
                        
            working_posij = posij[t]    
            working_posik = posik[t]    
            working_poski = poski[t]
            
            valid_posij = np.argsort(-working_posij,axis=1)[:,:edge_limit]
            valid_posik = np.argsort(-working_posik,axis=1)[:,:edge_limit]
            valid_poski = np.argsort(-working_poski,axis=1)[:,:edge_limit]

            for i in range(cij.shape[0]):
                                
                valid_posij_slice = valid_posij[i]
                for j in range(cij.shape[1]):
                    if j >= i:
                        if j in valid_posij_slice:
                            var = LpVariable("ai="+str(i)+",j="+str(j)+",t="+str(t),0,1,cat='Continuous')
                            Aij[t]["ij"][i][j] = var
                            Aij[t]["ji"][j][i] = var                            
                            cijt = cij[i,j]
                            obj_fn_terms.append(var)
                            obj_fn_coeff.append(cijt)

                        if j < (cij.shape[1]-1):
                            valid_posik_slice = valid_posik[i]
                            if j in valid_posik_slice:
                                var = LpVariable("ai="+str(i)+",k="+str(j)+",t="+str(t),0,1,cat='Continuous')
                                Aik[t]["ik"][i][j] = var
                                Aik[t]["ki"][j][i] = var
                                cikt = Cik[t][i,j]
                                obj_fn_terms.append(var)
                                obj_fn_coeff.append(cikt)

                        if i < (cij.shape[0]-1):
                            valid_poski_slice = valid_poski[i]
                            if j in valid_poski_slice:                            
                                var = LpVariable("ak="+str(i)+",i="+str(j)+",t="+str(t),0,1,cat='Continuous')
                                Aki[t]["ki"][i][j] = var
                                Aki[t]["ik"][j][i] = var
                                ckit = Cki[t][i,j]
                                obj_fn_terms.append(var)
                                obj_fn_coeff.append(ckit)

                Ai[(i,t)] = LpVariable("ai="+str(i)+",t="+str(t),0,1,cat='Continuous')     
                obj_fn_terms.append(Ai[(i,t)])
                obj_fn_coeff.append(0.)

        prob += lpSum([obj_fn_coeff[n]*obj_fn_term for n,obj_fn_term in enumerate(obj_fn_terms)]), "Objective Function"
        
        for t in range(0,len(sizes)):
            if t == 0:
                N_detect = len(sizes[t])
                
                for j in range(N_detect):
                    neg_Aij = list(Aij[t]["ij"].get(j,{}).values())
                    neg_Aik = list(Aik[t]["ik"].get(j,{}).values())
                    neg_Aki = list(Aki[t]["ki"].get(j-1,{}).values()) + list(Aki[t]["ki"].get(j,{}).values())
                    neg_Ai = Ai[(j,t)]

                    prob += lpSum(neg_Aij) + lpSum(neg_Aik) + lpSum(neg_Aki) + neg_Ai <= 1,\
                    "Constraint 7 (" + str(t) + "," + str(j) + ")"

            elif t < len(sizes)-1:
                N_detect_i = len(sizes[t-1])
                N_detect_f = len(sizes[t])

                for j in range(N_detect_f):
                    pos_Aij = list(Aij[t-1]["ji"].get(j,{}).values())                    
                    pos_Aik = list(Aik[t-1]["ki"].get(j-1,{}).values()) + list(Aik[t-1]["ki"].get(j,{}).values())
                    pos_Aki = list(Aki[t-1]["ik"].get(j,{}).values())
                    
                    
                    neg_Aij = list(Aij[t]["ij"].get(j,{}).values())
                    neg_Aik = list(Aik[t]["ik"].get(j,{}).values())
                    neg_Aki = list(Aki[t]["ki"].get(j-1,{}).values()) + list(Aki[t]["ki"].get(j,{}).values())
                    neg_Ai = Ai[(j,t)]                
                    
                    prob += lpSum(pos_Aij) + lpSum(pos_Aik) + lpSum(pos_Aki) - \
                        lpSum(neg_Aij) - lpSum(neg_Aik) - lpSum(neg_Aki) - neg_Ai == 0,\
                        "Constraint 9 (" + str(t) + "," + str(j) + ")"

                    prob += lpSum(pos_Aij) + lpSum(pos_Aik) + lpSum(pos_Aki) <= 1,\
                    "Constraint 6 (" + str(t) + "," + str(j) + ")"

                    prob += lpSum(neg_Aij) + lpSum(neg_Aik) + lpSum(neg_Aki) + neg_Ai <= 1,\
                        "Constraint 7 (" + str(t) + "," + str(j) + ")"

                    if j != N_detect_f - 1:

                        i_above = list(range(j+1,N_detect_f))

                        above_Aij = []
                        above_Aik = []
                        above_Aki = list(Aki[t]["ki"].get(j,{}).values())
                        above_Ai = []
                        for i in i_above:                            
                            above_Aij += list(Aij[t]["ij"].get(i,{}).values())
                            above_Aik += list(Aik[t]["ik"].get(i,{}).values())
                            above_Aki += list(Aki[t]["ki"].get(i,{}).values())
                            above_Ai.append((i,t))

                        prob += N_detect_f*neg_Ai + lpSum(above_Aij) + \
                        lpSum(above_Aik) + lpSum(above_Aki) <= N_detect_f,\
                        "Contraint 8 (" + str(t) + "," + str(j) + ")"

            else:
                N_detect = len(sizes[t])

                for j in range(N_detect):
                    pos_Aij = list(Aij[t-1]["ji"].get(j,{}).values())                    
                    pos_Aki = list(Aki[t-1]["ik"].get(j,{}).values())
                    pos_Aik = list(Aik[t-1]["ki"].get(j-1,{}).values()) + list(Aik[t-1]["ki"].get(j,{}).values())

                    if j == N_detect_f - 1:

                        prob += lpSum(pos_Aij) + lpSum(pos_Aik) + \
                        lpSum(pos_Aki) <= 1, "Constraint 6 (" + str(t) + "," + str(j) + ")"

                    else:

                        prob += lpSum(pos_Aij) + lpSum(pos_Aik) + \
                        lpSum(pos_Aki) <= 1, "Constraint 6 (" + str(t) + "," + str(j) + ")"
        
        all_Aki_vars = []
        for _,tpt in Aki.items():
            for _,subdict in tpt["ki"].items():
                all_Aki_vars += list(subdict.values())
                                        
        prob += lpSum(all_Aki_vars) <= merge_per_iter, "Merge Limit"

        return prob,Aij,Aik,Aki,Ai

    def get_sln_params(self,Aij,Aik,Aki,Ai,posij,posik,poski,sizes,edge_limit):
        Aij_arr_list = [np.zeros((len(sizes[t]),len(sizes[t+1])),dtype=bool) for t in range(len(sizes)-1)]
        Aik_arr_list = [np.zeros((len(sizes[t]),len(sizes[t+1])-1),dtype=bool) for t in range(len(sizes)-1)]
        Aki_arr_list = [np.zeros((len(sizes[t])-1,len(sizes[t+1])),dtype=bool) for t in range(len(sizes)-1)]
        Ai_arr_list = [np.zeros(len(sizes[t]),dtype=bool) for t in range(len(sizes))]

        for t in range(len(sizes)-1):
            
            working_posij = posij[t]    
            working_posik = posik[t]    
            working_poski = poski[t]
            
            valid_posij = np.argsort(-working_posij,axis=1)[:,:edge_limit]
            valid_posik = np.argsort(-working_posik,axis=1)[:,:edge_limit]
            valid_poski = np.argsort(-working_poski,axis=1)[:,:edge_limit]
            
            num_i = Aij_arr_list[t].shape[0]
            num_j = Aij_arr_list[t].shape[1]
            
            for i in range(num_i):
                valid_posij_slice = valid_posij[i]
                for j in range(num_j):
                    if j >= i:
                        if j in valid_posij_slice:                            
                            Aij_arr_list[t][i,j] = bool(np.round(Aij[t]["ij"][i][j].varValue))

                        if j < num_j - 1:
                            valid_posik_slice = valid_posik[i]
                            if j in valid_posik_slice:
                                Aik_arr_list[t][i,j] = bool(np.round(Aik[t]["ik"][i][j].varValue))

                        if i < num_i - 1:
                            valid_poski_slice = valid_poski[i]
                            if j in valid_poski_slice:
                                Aki_arr_list[t][i,j] = bool(np.round(Aki[t]["ki"][i][j].varValue))

                Ai_arr_list[t][i] = bool(np.round(Ai[(i,t)].varValue))

        return Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list

    def split_all_merged(self,labeled_data,Aki_arr_list):
        new_labeled_data = copy.copy(labeled_data)
        for t in range(len(Aki_arr_list)):
            if np.any(Aki_arr_list[t]):
                merged_label_list = (np.where(np.any(Aki_arr_list[t],axis=1))[0]+1).tolist()
                output_arr = self.split_cells(labeled_data[t+1],merged_label_list)
                new_labeled_data[t+1] = output_arr
        return new_labeled_data

    def split_cells(self,labeled_arr,merged_label_list):
        output_arr = copy.copy(labeled_arr)
        current_idx = 1
        for idx in range(1,np.max(labeled_arr)+1):

            if idx in merged_label_list:

                bool_arr = (output_arr==idx)
                cell_indices = np.where(bool_arr)
                X = np.array(list(zip(cell_indices[0],cell_indices[1])))
                kmeans = skl.cluster.KMeans(n_clusters=2, random_state=0)

                C = kmeans.fit_predict(X)
                centroids = kmeans.cluster_centers_.astype(int)
                C0 = (C==0)
                zero_indices = (cell_indices[0][C0],cell_indices[1][C0])
                one_indices = (cell_indices[0][~C0],cell_indices[1][~C0])

                zero_first = centroids[0,0] < centroids[1,0]
                if zero_first:
                    output_arr[zero_indices] = current_idx
                    output_arr[one_indices] = current_idx+1
                else:
                    output_arr[zero_indices] = current_idx+1
                    output_arr[one_indices] = current_idx

                current_idx += 2

            else:
                output_arr[labeled_arr==idx] = current_idx

                current_idx += 1

        return output_arr
    
    def plot_tracking_sln(self,kymo_arr,centroids,Aij_arr_list,Aik_arr_list,Aki_arr_list,t0=0,tf=-1,x_size=18,y_size=6):
        
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        if tf == -1:
            tf = len(centroids)-1
        
        fig, ax = plt.subplots(1)
        fig.set_size_inches(x_size, y_size)
        
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(np.ma.array(padded_img, mask=(padded_img == 0)), extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]], cmap="Set1", vmin=0)
            
            adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r")

            Aij_cords = np.where(Aij_arr_list[t])
            for idx in range(len(Aij_cords[0])):
                i = Aij_cords[0][idx]
                j = Aij_cords[1][idx]
                centroid_i = centroids[t][i]
                centroid_j = centroids[t+1][j]

                ax.plot([centroid_i[0] + t*x_dim + self.viewpadding, centroid_j[0] + (t+1)*x_dim + self.viewpadding],[centroid_i[1],centroid_j[1]],c="c",linewidth=3)

            Aik_cords = np.where(Aik_arr_list[t])
            for idx in range(len(Aik_cords[0])):
                i = Aik_cords[0][idx]
                k = Aik_cords[1][idx]
                centroid_i = centroids[t][i]

                centroid_k1 = centroids[t+1][k]
                centroid_k2 = centroids[t+1][k+1]

                ax.plot([centroid_i[0] + t*x_dim + self.viewpadding, centroid_k1[0] + (t+1)*x_dim + self.viewpadding],[centroid_i[1],centroid_k1[1]],c="c",linewidth=3)
                ax.plot([centroid_i[0] + t*x_dim + self.viewpadding, centroid_k2[0] + (t+1)*x_dim + self.viewpadding],[centroid_i[1],centroid_k2[1]],c="c",linewidth=3)

            Aki_cords = np.where(Aki_arr_list[t])
            for idx in range(len(Aki_cords[0])):
                k = Aki_cords[0][idx]
                i = Aki_cords[1][idx]
                centroid_k1 = centroids[t][k]
                centroid_k2 = centroids[t][k+1]

                centroid_i = centroids[t+1][i]

                ax.plot([centroid_k1[0] + t*x_dim + self.viewpadding, centroid_i[0] + (t+1)*x_dim + self.viewpadding],[centroid_k1[1],centroid_i[1]],c="c",linewidth=3)
                ax.plot([centroid_k2[0] + t*x_dim + self.viewpadding, centroid_i[0] + (t+1)*x_dim + self.viewpadding],[centroid_k2[1],centroid_i[1]],c="c",linewidth=3)

        padded_img = np.pad(kymo_arr[tf][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
        ax.imshow(padded_img, extent=[tf*x_dim, (tf+1)*x_dim, 0, img_shape[1]])
        adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0])    
        ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r")
        
        plt.show()
        
    def plot_cleaned_sln(self,kymo_arr,centroids,Aij_arr_list,Aik_arr_list,Aki_arr_list,t0=0,tf=-1,x_size=18,y_size=6):
        
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        if tf == -1:
            tf = len(centroids)-1
        
        fig, ax = plt.subplots(1)
        fig.set_size_inches(x_size, y_size)
        
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(np.ma.array(padded_img, mask=(padded_img == 0)), extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]], cmap="Set1", vmin=0)
            
            adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r")
            
            Aik_cords = np.where(Aik_arr_list[t])
            div_is = Aik_cords[0]
            working_j = 0
            
            for i in range(len(centroids[t])):
                centroid_i = centroids[t][i]
                
                if working_j > len(centroids[t+1])-1:
                    break
                
                if i in div_is:
                    
                    if working_j > len(centroids[t+1])-2:
                        centroid_k1 = centroids[t+1][working_j]
                        ax.plot([centroid_i[0] + t*x_dim + self.viewpadding, centroid_k1[0] + (t+1)*x_dim + self.viewpadding],[centroid_i[1],centroid_k1[1]],c="c",linewidth=3)
                        working_j += 1
                        
                    else:
                        centroid_k1 = centroids[t+1][working_j]
                        centroid_k2 = centroids[t+1][working_j+1]
                        ax.plot([centroid_i[0] + t*x_dim + self.viewpadding, centroid_k1[0] + (t+1)*x_dim + self.viewpadding],[centroid_i[1],centroid_k1[1]],c="c",linewidth=3)
                        ax.plot([centroid_i[0] + t*x_dim + self.viewpadding, centroid_k2[0] + (t+1)*x_dim + self.viewpadding],[centroid_i[1],centroid_k2[1]],c="c",linewidth=3)
                        working_j += 2
                else:
                    centroid_j = centroids[t+1][working_j]
                    ax.plot([centroid_i[0] + t*x_dim + self.viewpadding, centroid_j[0] + (t+1)*x_dim + self.viewpadding],[centroid_i[1],centroid_j[1]],c="c",linewidth=3)
                    working_j += 1

        padded_img = np.pad(kymo_arr[tf][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
        ax.imshow(padded_img, extent=[tf*x_dim, (tf+1)*x_dim, 0, img_shape[1]])
        adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
        ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r")
        
        plt.show()
        
    def run_iteration(self,labeled_data,merge_per_iter,edge_limit):
        centroids,sizes,bboxes,empty_trenches = get_segment_props(labeled_data,interpolate_empty_trenches=True)
        Cij,Cik,Cki,_,posij,posik,poski = self.ScoreFn.compute_score_arrays(centroids,sizes,bboxes)
        prob,Aij,Aik,Aki,Ai = self.define_tracking_problem(Cij,Cik,Cki,posij,posik,poski,sizes,merge_per_iter,edge_limit)  
        prob.solve()
        Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list = self.get_sln_params(Aij,Aik,Aki,Ai,posij,posik,poski,sizes,edge_limit)

        return centroids,sizes,prob,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,empty_trenches
    
    def compute_lineage(self,data,orientation):
        labeled_data = get_labeled_data(data,orientation)
        working_labeled_data = copy.copy(labeled_data)

        iter_outputs = []
        iter_scores = []
        iter_labeled_data = []
        iter_labeled_data.append(labeled_data)

        iter_outputs.append(self.run_iteration(working_labeled_data,self.merge_per_iter,self.edge_limit))

        centroids,sizes,prob,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,empty_trenches = iter_outputs[-1]
        
        merged_cells = np.any([np.any(Aki_arr) for Aki_arr in Aki_arr_list])
        objective_val = value(prob.objective)
        active_edges = sum([v.varValue for v in prob.variables()])
        edge_normalized_objective = objective_val/active_edges

        print("Objective = ", objective_val)
        print("Number of Active Edges = ", active_edges)
        print("Edge Normalized Objective = ", edge_normalized_objective)
        
        if self.merge_per_iter == 0:
            
            return working_labeled_data,centroids,sizes,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,edge_normalized_objective,empty_trenches

        iter_scores.append(edge_normalized_objective)
        iter_since_incr = 0

        while merged_cells:
            working_labeled_data = self.split_all_merged(working_labeled_data,Aki_arr_list)
            iter_labeled_data.append(working_labeled_data)
            iter_outputs.append(self.run_iteration(working_labeled_data,self.merge_per_iter,self.edge_limit))
            _,_,prob,_,_,Aki_arr_list,_,_ = iter_outputs[-1]

            merged_cells = np.any([np.any(Aki_arr) for Aki_arr in Aki_arr_list])

            objective_val = value(prob.objective)
            active_edges = sum([v.varValue for v in prob.variables()])
            edge_normalized_objective = objective_val/active_edges

            print("Objective = ", objective_val)
            print("Number of Active Edges = ", active_edges)
            print("Edge Normalized Objective = ", edge_normalized_objective)

            iter_scores.append(edge_normalized_objective)
            min_iter_score = min(iter_scores)
            
            if iter_scores[-1]<=min_iter_score:
                iter_since_incr = 0
            else:
                iter_since_incr += 1
            
            if iter_since_incr>self.conv_tolerence:
                break

        best_iter_idx = np.argmin(iter_scores)
        working_labeled_data = iter_labeled_data[best_iter_idx]
        centroids,sizes,prob,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,empty_trenches = self.run_iteration(working_labeled_data,0,self.edge_limit)
        objective_val = value(prob.objective)
        active_edges = sum([v.varValue for v in prob.variables()])
        edge_normalized_objective = objective_val/active_edges

        return working_labeled_data,centroids,sizes,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,edge_normalized_objective,empty_trenches
    
    def get_nuclear_lineage(self,sizes,empty_trenches,Aik_arr_list):
        
        cell_ids = list(range(len(sizes[0])))
        current_max_id = np.max(cell_ids)
        cell_ids_list = [copy.copy(cell_ids)]
        mother_dict = {}
        daughter_dict = {}

        for t in range(0,len(Aik_arr_list)):
            if t in empty_trenches:
                cell_ids_list.append(copy.copy(cell_ids_list[-1]))
            else:
                Aik_cords = np.where(Aik_arr_list[t])
                ttl_added = 0
                for idx in range(len(Aik_cords[0])):
                    i = Aik_cords[0][idx]
                    
                    daughter_id1,daughter_id2 = (current_max_id+1,current_max_id+2)
                    
                    mother_dict[daughter_id1] = cell_ids[i+ttl_added]
                    mother_dict[daughter_id2] = cell_ids[i+ttl_added]
                    daughter_dict[cell_ids[i+ttl_added]] = (daughter_id1,daughter_id2)

                    del cell_ids[i+ttl_added]
                    cell_ids.insert(i+ttl_added,daughter_id1)
                    cell_ids.insert(i+1+ttl_added,daughter_id2)

                    current_max_id += 2
                    ttl_added += 1
                    
                max_cells = len(sizes[t+1])
                cell_ids = copy.copy(cell_ids[:max_cells])
                cell_ids_list.append(copy.copy(cell_ids[:max_cells]))
                
        return mother_dict,daughter_dict,cell_ids_list
    
    def get_cell_lineage(self,cell_id,mother_dict,daughter_dict):     
        if (cell_id in daughter_dict.keys()) and (cell_id in mother_dict.keys()):
            mother_id = mother_dict[cell_id]
            daughter_ids = daughter_dict[cell_id]
        elif cell_id in mother_dict.keys():
            mother_id = mother_dict[cell_id]
            daughter_ids = (-1,-1)
        elif cell_id in daughter_dict.keys():
            mother_id = -1
            daughter_ids = daughter_dict[cell_id]
        else:
            mother_id = -1
            daughter_ids = (-1,-1)
            
        output = [mother_id,daughter_ids]

        return output
    
    def get_lineage_df(self,file_idx,file_trench_idx,lineage_score,labeled_data,empty_trenches,mother_dict,daughter_dict,cell_ids_list):

        kymograph_file = self.kymopath + "/kymograph_" + str(file_idx) + ".hdf5"
        if self.intensity_channel_list is not None:
            kymo_arr_list = []
            with h5py.File(kymograph_file,"r") as kymofile:
                for intensity_channel in self.intensity_channel_list:
                    kymo_arr_list.append(kymofile[intensity_channel][:])
                    
        pixel_microns = self.metadata['pixel_microns']
        
        props_output = []
        for t in range(labeled_data.shape[0]):
            if t not in empty_trenches:
                cell_ids = cell_ids_list[t]
                ttl_ids = len(cell_ids)
                
                #non intensity info first
                non_intensity_rps = sk.measure.regionprops(labeled_data[t])
                if self.intensity_channel_list is not None:
                    intensity_rps_list = []
                    for i,intensity_channel in enumerate(self.intensity_channel_list):
                        intensity_rps = sk.measure.regionprops(labeled_data[t], kymo_arr_list[i][file_trench_idx,t])
                        intensity_rps_list.append(intensity_rps)
                
                for idx in range(ttl_ids):
                    cell_id = cell_ids[idx]
                    rp = non_intensity_rps[idx]

                    props_entry = [file_idx, file_trench_idx, t, cell_id, lineage_score]
                    props_entry += self.get_cell_lineage(cell_id,mother_dict,daughter_dict)

                    props_entry += [self.pixel_scaling_fns[k](getattr(rp, prop_key),pixel_microns) for k,prop_key in enumerate(self.props_list)]
                    
                    if self.intensity_channel_list is not None:
                        for i,intensity_channel in enumerate(self.intensity_channel_list):
                            intensity_rps=intensity_rps_list[i]
                            inten_rp = intensity_rps[idx]
                            props_entry += [getattr(inten_rp, prop_key) for prop_key in self.intensity_props_list]
                            
                    props_output.append(props_entry)
        
        base_list = ['File Index','File Trench Index','timepoints','CellID', 'Trench Score', 'Mother CellID', 'Daughter CellID']
        if self.intensity_channel_list is not None:
            channel_intensity_names = []
            for channel in self.intensity_channel_list:
                channel_intensity_names += [channel + " " + prop_key for prop_key in self.intensity_props_list]
            column_list = base_list + self.props_list + channel_intensity_names
            
        else:
            column_list = base_list + self.props_list
            
        df_out = pd.DataFrame(props_output, columns=column_list).reset_index()
        df_out = df_out.set_index(['File Index','File Trench Index','timepoints','CellID'], drop=True, append=False, inplace=False)

        df_out = df_out.sort_index()
        return df_out
    
    def lineage_trace(self,file_idx,file_trench_idx):
        meta_handle = pandas_hdf5_handler(self.headpath+"/metadata.hdf5")
        
        kymo_df = meta_handle.read_df("kymograph")
        fov_df = kymo_df.reset_index(inplace=False)
        fov_df = fov_df.set_index(["File Index","File Trench Index"], drop=True, append=False, inplace=False)
        fov_df = fov_df.sort_index()
        trench = fov_df.loc[file_idx,file_trench_idx]
        orientation_dict = {"top":0,"bottom":1}
        orientation = trench["lane orientation"].unique().tolist()[0]
        orientation = orientation_dict[orientation]
        
        with h5py.File(self.segpath + "/segmentation_" + str(file_idx) + ".hdf5") as infile:
            data = infile["data"][file_trench_idx]
        
        labeled_data,_,sizes,_,Aik_arr_list,_,_,lineage_score,empty_trenches = self.compute_lineage(data,orientation)
        
        mother_dict,daughter_dict,cell_ids_list = self.get_nuclear_lineage(sizes,empty_trenches,Aik_arr_list)
                
        df_out = self.get_lineage_df(file_idx,file_trench_idx,lineage_score,labeled_data,empty_trenches,mother_dict,daughter_dict,cell_ids_list)
        
        return df_out

    def lineage_trace_file(self,file_idx):
        writedir(self.lineagepath,overwrite=False)
        
        kymo_meta = self.meta_handle.read_df("kymograph")
        filedf = kymo_meta.reset_index(inplace=False)
        filedf = filedf.set_index(["File Index","File Trench Index"], drop=True, append=False, inplace=False)
        filedf = filedf.sort_index()
        filedf = filedf.loc[file_idx]
        trench_idx_list = filedf.index.get_level_values(0).unique().tolist()
        
        merged_df = []
        for file_trench_idx in trench_idx_list:
            try:
                df_out = self.lineage_trace(file_idx,file_trench_idx)
                merged_df.append(df_out)
            except:
                pass
        merged_df = pd.concat(merged_df)
        temp_df_path = self.lineagepath + "/temp_df_" + str(file_idx) + ".pkl"
        merged_df.to_pickle(temp_df_path)
        
        return file_idx
            
    def lineage_trace_all_files(self,dask_cont):
        kymo_meta = self.meta_handle.read_df("kymograph")
        file_list = kymo_meta["File Index"].unique().tolist()
        num_file_jobs = len(file_list)

        random_priorities = np.random.uniform(size=(num_file_jobs,))
        for k,file_idx in enumerate(file_list):
            priority = random_priorities[k]
            future = dask_cont.daskclient.submit(self.lineage_trace_file,file_idx,retries=1,priority=priority)
            dask_cont.futures["File Index: " + str(file_idx)] = future
            
    def compile_lineage_data(self,dask_cont):
        kymo_meta = self.meta_handle.read_df("kymograph")
        file_list = kymo_meta["File Index"].unique().tolist()
        num_file_jobs = len(file_list)
        file_idx_list = dask_cont.daskclient.gather([dask_cont.futures["File Index: " + str(file_idx)] for file_idx in file_list],errors="skip")
        
        ttl_indices = len(file_idx_list)
    
        df_out = []
        for file_idx in file_idx_list:
            temp_df_path = self.lineagepath + "/temp_df_" + str(file_idx) + ".pkl"
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
        mergeddf = mergeddf.set_index(["File Index","File Trench Index","timepoints","CellID"], drop=True, append=False, inplace=False)
        del mergeddf["index"]
        mergeddf = mergeddf.sort_index()
        
        mergeddf.to_pickle(self.lineagepath + "/lineage_data.pkl")
        
    def compute_all_lineages(self,dask_cont=None,n_workers=100,memory='4GB',walltime='04:00:00',local=False):        
        writedir(self.lineagepath,overwrite=True)
        
        if dask_cont is not None:
            dask_cont.daskclient.restart()
            dask_cont.futures = {}
            try:
                self.lineage_trace_all_files(dask_cont)
                self.compile_lineage_data(dask_cont)
            except:
                raise
        
        else:
            dask_cont = dask_controller(walltime=walltime,local=local,n_workers=n_workers,memory=memory,working_directory=self.headpath+"/dask")
            dask_cont.startdask()
            dask_cont.daskcluster.start_workers()
            dask_cont.displaydashboard()
            dask_cont.futures = {}
        
            try:
                self.lineage_trace_all_files(dask_cont)
                self.compile_lineage_data(dask_cont)
                dask_cont.shutdown()
            except:
                dask_cont.shutdown()
                raise

    def test_tracking(self,data,orientation,intensity_channel_list=None,t_range=(0,-1),merge_per_iter=0,edge_limit=3,conv_tolerence=2,viewpadding=0,x_size=18,y_size=6):
        if len(intensity_channel_list) == 0:
            self.intensity_channel_list = None
        else:
            self.intensity_channel_list = list(intensity_channel_list)
        self.viewpadding = viewpadding
        self.merge_per_iter = merge_per_iter
        self.edge_limit = edge_limit
        self.conv_tolerence = conv_tolerence
        
        labeled_data,centroids,sizes,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,lineage_score,empty_trenches = self.compute_lineage(data,orientation)

        ### don't delete this comment ###
#         self.plot_tracking_sln(labeled_data,centroids,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ae_arr_list,t0=t_range[0],tf=t_range[1],x_size=x_size,y_size=y_size)
        ###                           ###
    
        self.plot_cleaned_sln(labeled_data,centroids,Aij_arr_list,Aik_arr_list,Aki_arr_list,t0=t_range[0],tf=t_range[1],x_size=x_size,y_size=y_size)
    
    def interactive_tracking(self,data,orientation):
        
        available_channels = self.metadata["channels"]
        
        output = interactive(self.test_tracking,{"manual":True},data=fixed(data),orientation=fixed(orientation),\
                 intensity_channel_list = SelectMultiple(options=available_channels),\
                 t_range=IntRangeSlider(value=[0, data.shape[0]-1],min=0,max=data.shape[0]-1,\
                 step=1,disabled=False,continuous_update=False),\
                 merge_per_iter=IntSlider(value=1, min=0, max=10, step=1),\
                 edge_limit=IntSlider(value=3, min=1, max=6, step=1),\
                 conv_tolerence=IntSlider(value=2, min=1, max=4, step=1),\
                 viewpadding=IntSlider(value=0, min=0, max=20, step=1),\
                 x_size=IntSlider(value=18, min=0, max=30, step=1),\
                 y_size=IntSlider(value=8, min=0, max=20, step=1));
        
        display(output)
        
    def save_params(self):
        param_dict = {}
        
        param_dict["Channels:"] = self.intensity_channel_list
        param_dict["Merges Detected Per Iteration:"] = self.merge_per_iter
        param_dict["# Unimproved Iterations Before Stopping:"] = self.conv_tolerence
        param_dict["Closest N Objects to Consider:"] = self.edge_limit
        param_dict["Mean Size Increase:"] = self.ScoreFn.u_size
        param_dict["Standard Deviation of Size Increase:"] = self.ScoreFn.sig_size
        param_dict["Mean Position Increase:"] = self.ScoreFn.u_pos
        param_dict["Standard Deviation of Position Increase:"] = self.ScoreFn.sig_pos
        
        for key,value in param_dict.items():
            print(key + " " + str(value))
                        
        with open(self.headpath + "/lineage_tracing.par", "wb") as outfile:
            pkl.dump(param_dict, outfile)