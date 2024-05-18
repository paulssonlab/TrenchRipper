# fmt: off
import skimage as sk
import sklearn as skl
import numpy as np
import pandas as pd
import pickle as pkl
import dask.dataframe as dd
import dask.delayed as delayed
from time import sleep
from distributed.client import futures_of
from dask.distributed import as_completed

import os
import copy
import h5py
import shutil

from sklearn import cluster
from scipy.special import logsumexp
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from pulp import *
from ipywidgets import interactive, fixed, FloatSlider, IntSlider, IntRangeSlider, SelectMultiple, Dropdown

from .utils import pandas_hdf5_handler,writedir
from .trcluster import dask_controller
from .metrics import get_cell_dimensions_spherocylinder_ellipse,get_cell_dimensions_spherocylinder_perimeter_area,width_length_from_permeter_area

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

def get_segment_props(labeled_data,size_attr="axis_major_length",interpolate_empty_trenches=False):
    all_centroids = []
    all_sizes = []
    all_bboxes = []

    for t in range(labeled_data.shape[0]):
        rps = sk.measure.regionprops(labeled_data[t])
        centroids = [rp.centroid for rp in rps]
        sizes = np.array([getattr(rp, size_attr) for rp in rps])
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
    def __init__(self,headpath,segfolder,size_attr="axis_major_length",u_pos=0.2,sig_pos=0.4,u_size=0.,sig_size=0.2,w_pos=1.,w_size=1.,w_merge=0.8):
        self.headpath = headpath
        self.segpath = headpath + "/" + segfolder
        self.size_attr = size_attr
        self.u_pos,self.sig_pos = (u_pos,sig_pos)
        self.u_size,self.sig_size = (u_size,sig_size)
        self.w_pos,self.w_size,self.w_merge = w_pos,w_size,w_merge
        self.pos_coef,self.size_coef = (2*(w_pos/(w_pos+w_size)),2*(w_size/(w_pos+w_size)))

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
            if len(centroids[t]) == 0 or len(centroids[t+1])==0:
                posscore = np.array([[]])
            else:
                posscore,_ = self.fpos(centroids[t+1],centroids[t],bboxes[t+1],bboxes[t],u_pos=self.u_pos,sig_pos=self.sig_pos)
            posscores.append(posscore.T)
        # posscores = np.array(posscores)
        return posscores

    def del_split_pos(self,centroids,union_centroids,bboxes):
        posscores = []
        for t in range(len(centroids)-1):
            if len(union_centroids[t+1]) > 0:
                posscore,_ = self.fpos(union_centroids[t+1],centroids[t],bboxes[t+1],bboxes[t],u_pos=self.u_pos,sig_pos=self.sig_pos)
            else:
                posscore = np.array([[]])
            posscores.append(posscore.T)
        # posscores = np.array(posscores)
        return posscores

    def del_merge_pos(self,centroids,union_centroids,bboxes):
        posscores = []
        for t in range(len(centroids)-1):
            if len(union_centroids[t]) > 0 and len(centroids[t+1]) > 0:
                posscore,_ = self.fpos(centroids[t+1],union_centroids[t],bboxes[t+1],bboxes[t],u_pos=self.u_pos,sig_pos=self.sig_pos)
            else:
                posscore = np.array([[]])
            posscores.append(posscore.T)
        # posscores = np.array(posscores)
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
        # sizescores = np.array(sizescores)
        return sizescores

    def del_half_size(self,sizes):
        sizescores = []
        for t in range(len(sizes)-1):
            size_ti = sizes[t]/2.
            size_tf = sizes[t+1]
            sizescore,_ = self.fsize(size_tf,size_ti,u_size=self.u_size,sig_size=self.sig_size)
            sizescores.append(sizescore.T)
        # sizescores = np.array(sizescores)
        return sizescores

    def del_split_size(self,sizes,union_sizes):
        sizescores = []
        for t in range(len(sizes)-1):
            if len(union_sizes[t+1]) > 0:
                sizescore,_ = self.fsize(union_sizes[t+1],sizes[t],u_size=self.u_size,sig_size=self.sig_size)
            else:
                sizescore = np.array([[]])
            sizescores.append(sizescore.T)
        # sizescores = np.array(sizescores)
        return sizescores

    def del_merge_size(self,sizes,union_sizes):
        sizescores = []
        for t in range(len(sizes)-1):
            if len(union_sizes[t]) > 0:
                sizescore,_ = self.fsize(sizes[t+1],union_sizes[t],u_size=self.u_size,sig_size=self.sig_size)
            else:
                sizescore = np.array([[]])
            sizescores.append(sizescore.T)
        # sizescores = np.array(sizescores)
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
            if posij[t].size > 0:
                f_ij = self.pos_coef*posij[t]+self.size_coef*sizeij[t]
                f_ik = self.pos_coef*posik[t]+self.size_coef*sizeik[t]
                f_ki = self.pos_coef*poski[t]+self.size_coef*sizeki[t]
                f_e = self.pos_coef*pos_e[t]+self.size_coef*size_e[t]

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

                ki_score = (1.+self.w_merge)*(-log_prob_ki-100.)
                ki_score[ki_score>0] = 0.
                ki_score[ki_score<-100.] = -100.
                Cki.append(ki_score)

                e_score = ((-log_prob_e)-100.)
                e_score[e_score>0.] = 0.
                e_score[e_score<-100.] = -100.
                Ce.append(e_score)
            else:
                Cij.append(None)
                Cik.append(None)
                Cki.append(None)
                Ce.append(None)

        return Cij,Cik,Cki,Ce,posij,posik,poski

    def compute_neighbor_stats(self,centroids,bboxes,sizes):
        neighbor_dists = []
        neighbor_sizes = []
        for t in range(len(centroids)-1):
            if len(centroids[t]) > 0 and len(centroids[t+1]) > 0:
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

        ax_dist.hist(neighbor_dists,bins=30,range=(0.,dist_avg+(3.*dist_std)))
        ax_dist.set_title('Nearest-neighbor Normalized Centroid Distance')
        ax_dist.axvline(x=self.u_pos,c="r",linewidth=3,zorder=10)
        ax_dist.axvspan(self.u_pos-self.sig_pos, self.u_pos+self.sig_pos, alpha=0.3, color='r',zorder=10)

        ax_size.hist(neighbor_sizes,bins=30,range=(0.,size_avg+(3.*size_std)))
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
            padded_img = np.pad(kymo_arr[t],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])

    def plot_ij_scores(self,ax,kymo_arr,centroids,Cij,t0=0,tf=-1,cmap='coolwarm'):
        cmap_fn = cm.get_cmap(cmap, 100)
        img_shape = kymo_arr.shape[1:][::-1]

        x_dim = img_shape[0]+(self.viewpadding*2)
        y_len = img_shape[1]
        if tf == -1:
            tf = len(centroids)-1

        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])

            if len(centroids[t]) > 0:
                adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
                ax.scatter(adjusted_centroids[:,0],y_len-adjusted_centroids[:,1],c="r",s=100,zorder=10)

            if t > t0:
                if len(centroids[t]) > 0 and len(centroids[t-1]) > 0:

                    N_detect_i = len(centroids[t-1])
                    N_detect_f = len(centroids[t])

                    for i in range(N_detect_i):

                        for j in range(N_detect_f):

                            centroid_i = centroids[t-1][i]
                            centroid_f = centroids[t][j]

                            color_score = abs(Cij[t-1][i,j]/-100)

                            ax.plot([centroid_i[0] + (t-1)*x_dim + self.viewpadding, centroid_f[0] + t*x_dim + self.viewpadding],[y_len-centroid_i[1],y_len-centroid_f[1]],c=cmap_fn(color_score),linewidth=3)
        if len(centroids[tf]) > 0:
            adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            ax.scatter(adjusted_centroids[:,0],y_len-adjusted_centroids[:,1],c="r")

    def plot_ik_scores(self,ax,kymo_arr,centroids,Cik,t0=0,tf=-1,cmap='coolwarm'):
        cmap_fn = cm.get_cmap(cmap, 100)
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        y_len = img_shape[1]
        if tf == -1:
            tf = len(centroids)-1
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])

            if len(centroids[t]) > 0:
                merged_centroids = (centroids[t][:-1,:]+centroids[t][1:,:])/2

                adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
                adjusted_merged_centroids = merged_centroids + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])

                ax.scatter(adjusted_centroids[:,0],y_len-adjusted_centroids[:,1],c="r",s=100,zorder=10)
                ax.scatter(adjusted_merged_centroids[:,0],y_len-adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)

            if t > t0:
                if len(centroids[t]) > 0 and len(centroids[t-1]) > 0:
                    N_detect_i = len(centroids[t-1])
                    N_detect_f = len(centroids[t])

                    for i in range(N_detect_i):
                        for j in range(N_detect_f-1):

                            centroid_i = centroids[t-1][i]

                            centroid_k1 = centroids[t][j]
                            centroid_k2 = centroids[t][j+1]

                            color_score = abs(Cik[t-1][i,j]/-100)

                            ax.plot([centroid_i[0] + (t-1)*x_dim + self.viewpadding, centroid_k1[0] + t*x_dim + self.viewpadding],\
                                    [y_len-centroid_i[1],y_len-((centroid_k1[1]+centroid_k2[1])/2)],c=cmap_fn(color_score),linewidth=3)
        if len(centroids[tf]) > 0:
            merged_centroids = (centroids[tf][:-1,:]+centroids[tf][1:,:])/2

            adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            adjusted_merged_centroids = merged_centroids + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])

            ax.scatter(adjusted_centroids[:,0],y_len-adjusted_centroids[:,1],c="r",s=100,zorder=10)
            ax.scatter(adjusted_merged_centroids[:,0],y_len-adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)

    def plot_ki_scores(self,ax,kymo_arr,centroids,Cki,t0=0,tf=-1,cmap='coolwarm'):
        cmap_fn = cm.get_cmap(cmap, 100)
        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        y_len = img_shape[1]
        if tf == -1:
            tf = len(centroids)-1
        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])
        for t in range(t0,tf):
            padded_img = np.pad(kymo_arr[t],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            ax.imshow(padded_img, extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]])

            if len(centroids[t]) > 0:
                merged_centroids = (centroids[t][:-1,:]+centroids[t][1:,:])/2

                adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])
                adjusted_merged_centroids = merged_centroids + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])

                ax.scatter(adjusted_centroids[:,0],y_len-adjusted_centroids[:,1],c="r",s=100,zorder=10)
                ax.scatter(adjusted_merged_centroids[:,0],y_len-adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)

            if t > t0:
                if len(centroids[t]) > 0 and len(centroids[t-1]) > 0:
                    N_detect_i = len(centroids[t-1])
                    N_detect_f = len(centroids[t])

                    for i in range(N_detect_i-1):
                        for j in range(N_detect_f):

                            centroid_k1 = centroids[t-1][i]
                            centroid_k2 = centroids[t-1][i+1]

                            centroid_i = centroids[t][j]

                            color_score = abs(Cki[t-1][i,j]/-100)

                            ax.plot([centroid_k1[0] + (t-1)*x_dim + self.viewpadding, centroid_i[0] + t*x_dim + self.viewpadding],\
                                    [y_len-((centroid_k1[1]+centroid_k2[1])/2),y_len-centroid_i[1]],c=cmap_fn(color_score),linewidth=3)

        if len(centroids[tf]) > 0:
            merged_centroids = (centroids[tf][:-1,:]+centroids[tf][1:,:])/2

            adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            adjusted_merged_centroids = merged_centroids + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])

            ax.scatter(adjusted_centroids[:,0],y_len-adjusted_centroids[:,1],c="r",s=100,zorder=10)
            ax.scatter(adjusted_merged_centroids[:,0],y_len-adjusted_merged_centroids[:,1],c="white",s=100,zorder=10)

    def plot_score_metrics(self,kymo_df,trenchid,t_range=(0,-1),size_attr="axis_major_length",u_size=0.25,sig_size=0.05,u_pos=0.25,sig_pos=0.1,w_pos=1.,w_size=1.,w_merge=0.8,viewpadding=0):
        self.viewpadding = viewpadding

        trench = kymo_df.loc[trenchid]
        file_idx = trench["File Index"].unique().tolist()[0]
        trench_idx = trench["File Trench Index"].unique().tolist()[0]
        orientation_dict = {"top":0,"bottom":1}
        orientation = trench["lane orientation"].unique().tolist()[0]
        print(orientation)
        orientation = orientation_dict[orientation]
        print(orientation)

        with h5py.File(self.segpath + "/segmentation_" + str(file_idx) + ".hdf5", "r") as infile:
            data = infile["data"][trench_idx]
        t0,tf = t_range
        self.size_attr = size_attr
        self.u_pos,self.sig_pos = (u_pos,sig_pos)
        self.u_size,self.sig_size = (u_size,sig_size)
        self.w_merge = w_merge
        self.pos_coef,self.size_coef = (2*(w_pos/(w_pos+w_size)),2*(w_size/(w_pos+w_size)))
        labeled_data = get_labeled_data(data,orientation)

        centroids,sizes,bboxes = get_segment_props(labeled_data,size_attr=size_attr,interpolate_empty_trenches=False)
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

        return data,orientation

    def interactive_scorefn(self):
        kymo_df = pd.read_parquet(self.headpath+"/kymograph/metadata",columns=["trenchid","timepoints","File Index","File Trench Index","lane orientation"])
        num_trenches = len(kymo_df["trenchid"].unique())
        timepoints = len(kymo_df["timepoints"].unique())
        kymo_df = kymo_df.set_index("trenchid")

        self.output = interactive(self.plot_score_metrics,{"manual":True},\
                 kymo_df = fixed(kymo_df),trenchid=IntSlider(value=0, min=0, max=num_trenches-1, step=1),\
                 t_range=IntRangeSlider(value=[0, timepoints-1],min=0,max=timepoints-1,\
                 step=1,disabled=False,continuous_update=False),\
                 size_attr=Dropdown(options=["area","axis_major_length"],value="axis_major_length",\
                                    description='Size Measure:',disabled=False),\
                 u_size=FloatSlider(value=self.u_size, min=0., max=1., step=0.01),\
                 sig_size=FloatSlider(value=self.sig_size, min=0., max=0.5, step=0.01),\
                 u_pos=FloatSlider(value=self.u_pos, min=0., max=1., step=0.01),\
                 sig_pos=FloatSlider(value=self.sig_pos, min=0., max=0.5, step=0.01),\
                 w_pos=FloatSlider(value=self.w_pos, min=0., max=1., step=0.01),\
                 w_size=FloatSlider(value=self.w_size, min=0., max=1., step=0.01),\
                 w_merge=FloatSlider(value=self.w_merge, min=0., max=1., step=0.01),\
                 viewpadding=IntSlider(value=0, min=0, max=20, step=1));

        display(self.output)

class tracking_solver:
    def __init__(self,headpath,segfolder,paramfile=False,ScoreFn=False,intensity_channel_list=None,\
                 size_estimation=True,size_estimation_method="Perimeter/Area",props_list=['area'],\
                 custom_props_list=[],props_to_unpack={},pixel_scaling_factors={'area': 2},\
                 intensity_props_list=['mean_intensity'],custom_intensity_props_list=[],\
                 edge_limit=3,size_attr="axis_major_length",u_size=0.22,sig_size=0.08,u_pos=0.21,sig_pos=0.1,w_pos=1.,w_size=1.,w_merge=0.8):

        if paramfile:
            parampath = headpath + "/lineage_tracing.par"
            with open(parampath, 'rb') as infile:
                param_dict = pkl.load(infile)

            intensity_channel_list = param_dict["Channels:"]
#             merge_per_iter = param_dict["Merges Detected Per Iteration:"]
#             conv_tolerence = param_dict["# Unimproved Iterations Before Stopping:"]
            edge_limit = param_dict["Closest N Objects to Consider:"]
            size_attr = param_dict["Size Measurement:"]
            u_size = param_dict["Mean Size Increase:"]
            sig_size = param_dict["Standard Deviation of Size Increase:"]
            u_pos = param_dict["Mean Position Increase:"]
            sig_pos = param_dict["Standard Deviation of Position Increase:"]
            w_pos = param_dict["Cell Position Weight:"]
            w_size = param_dict["Cell Size Weight:"]
            w_merge = param_dict["Cell Merging Weight:"]

        self.headpath = headpath
        self.segpath = headpath + "/" + segfolder
        self.kymopath = headpath + "/kymograph"
        self.lineagepath = headpath + "/lineage"
        self.metapath = self.headpath + "/metadata.hdf5"
        self.meta_handle = pandas_hdf5_handler(self.metapath)
        fovdf = self.meta_handle.read_df("global",read_metadata=True)
        self.metadata = fovdf.metadata
        self.intensity_channel_list = intensity_channel_list
        self.size_estimation = size_estimation
        self.size_estimation_method = size_estimation_method
        self.props_list = props_list
        self.custom_props_list = custom_props_list
        self.custom_props_str_list = [item.__name__ for item in self.custom_props_list]

        self.pixel_scaling_factors = pixel_scaling_factors
        self.props_to_unpack = props_to_unpack
        self.intensity_props_list = intensity_props_list
        self.custom_intensity_props_list = custom_intensity_props_list
        self.custom_intensity_props_str_list = [item.__name__ for item in self.custom_intensity_props_list]

        if ScoreFn:
            self.ScoreFn = ScoreFn
        else:
            self.ScoreFn = scorefn(headpath,segfolder,size_attr=size_attr,u_size=u_size,sig_size=sig_size,u_pos=u_pos,sig_pos=sig_pos,w_pos=w_pos,w_size=w_size,w_merge=w_merge)
#         self.merge_per_iter = merge_per_iter
#         self.conv_tolerence = conv_tolerence
        self.edge_limit = edge_limit

        self.viewpadding = 0

        if self.size_estimation:
            est_pixel_scaling_factors = {"Length":1,"Width":1,"Volume":3,"Surface Area":2}
            self.pixel_scaling_factors = {**self.pixel_scaling_factors, **est_pixel_scaling_factors}

    def solve_tracking_problem(self,Cij,Cik,Cki,posij,posik,poski,sizes,edge_limit):

        Aij = {}
        Aik = {}
        Aki = {}
        Ai = {}

        obj_fn_terms = []
        obj_fn_coeff = []

        valid_j_list = []

        prob = LpProblem("Lineage",LpMinimize)

        for t in range(len(Cij)):
            cij = Cij[t]

            Aij[t] = {"ij":{}, "ji":{}}
            Aik[t] = {"ik":{}, "ki":{}}
            Aki[t] = {"ki":{}, "ik":{}}
            if cij is not None:

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
                                if i == j:
                                    var.setInitialValue(1.)
                                else:
                                    var.setInitialValue(0.)
                                Aij[t]["ij"][i][j] = var
                                Aij[t]["ji"][j][i] = var
                                cijt = cij[i,j]
                                obj_fn_terms.append(var)
                                obj_fn_coeff.append(cijt)

                            if j < (cij.shape[1]-1):
                                valid_posik_slice = valid_posik[i]
                                if j in valid_posik_slice:
                                    var = LpVariable("ai="+str(i)+",k="+str(j)+",t="+str(t),0,1,cat='Continuous')
                                    var.setInitialValue(0.)
                                    Aik[t]["ik"][i][j] = var
                                    Aik[t]["ki"][j][i] = var
                                    cikt = Cik[t][i,j]
                                    obj_fn_terms.append(var)
                                    obj_fn_coeff.append(cikt)

                            if i < (cij.shape[0]-1):
                                valid_poski_slice = valid_poski[i]
                                if j in valid_poski_slice:
                                    var = LpVariable("ak="+str(i)+",i="+str(j)+",t="+str(t),0,1,cat='Continuous')
                                    var.setInitialValue(0.)
                                    Aki[t]["ki"][i][j] = var
                                    Aki[t]["ik"][j][i] = var
                                    ckit = Cki[t][i,j]
                                    obj_fn_terms.append(var)
                                    obj_fn_coeff.append(ckit)

            for i in range(len(sizes[t])):
                var = LpVariable("ai="+str(i)+",t="+str(t),0,1,cat='Continuous')
                var.setInitialValue(0.)
                Ai[(i,t)] = var
                obj_fn_terms.append(Ai[(i,t)])
                obj_fn_coeff.append(0.)

        prob += lpSum([obj_fn_coeff[n]*obj_fn_term for n,obj_fn_term in enumerate(obj_fn_terms)]), "Objective Function"

        for t in range(len(Cij)):
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

                    prob += lpSum(pos_Aij) + lpSum(pos_Aik) + \
                    lpSum(pos_Aki) <= 1, "Constraint 6 (" + str(t) + "," + str(j) + ")"

#             all_Aki_vars = []
#             for t in range(init_t,init_t+working_t_chunk):
#                 tpt = Aki[t]
#                 for _,subdict in tpt["ki"].items():
#                     all_Aki_vars += list(subdict.values())
#             prob += lpSum(all_Aki_vars) <= merge_per_iter, "Merge Limit"
        prob.solve(PULP_CBC_CMD(warmStart=True))

        return prob,Aij,Aik,Aki,Ai

    def get_sln_params(self,Aij,Aik,Aki,Ai,posij,posik,poski,sizes,edge_limit):
        Aij_arr_list = []
        Aik_arr_list = []
        Aki_arr_list = []
        Ai_arr_list = []
        for t in range(len(sizes)-1):
            if len(sizes[t]) > 0 and len(sizes[t+1]) > 0:
                Aij_arr_list.append(np.zeros((len(sizes[t]),len(sizes[t+1])),dtype=bool))
            else:
                Aij_arr_list.append(None)
            if len(sizes[t]) > 0 and len(sizes[t+1]) > 1:
                Aik_arr_list.append(np.zeros((len(sizes[t]),len(sizes[t+1])-1),dtype=bool))
            else:
                Aik_arr_list.append(None)
            if len(sizes[t]) > 1 and len(sizes[t+1]) > 0:
                Aki_arr_list.append(np.zeros((len(sizes[t])-1,len(sizes[t+1])),dtype=bool))
            else:
                Aki_arr_list.append(None)

        for t in range(len(sizes)):
            if len(sizes[t]) > 0:
                Ai_arr_list.append(np.zeros(len(sizes[t]),dtype=bool))
            else:
                Ai_arr_list.append(None)

        for t in range(len(sizes)-1):

            working_posij = posij[t]
            working_posik = posik[t]
            working_poski = poski[t]

            valid_posij = np.argsort(-working_posij,axis=1)[:,:edge_limit]
            valid_posik = np.argsort(-working_posik,axis=1)[:,:edge_limit]
            valid_poski = np.argsort(-working_poski,axis=1)[:,:edge_limit]

            num_i = len(sizes[t])
            num_j = len(sizes[t+1])

            if num_i > 0 and num_j > 0:
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

            for i in range(num_i):
                Ai_arr_list[t][i] = bool(np.round(Ai[(i,t)].varValue))

        return Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list

    def plot_cleaned_sln(self,kymo_arr,centroids,mother_dict,cell_ids_list,t0=0,tf=-1,x_size=18,y_size=6,dot_size=50):

        img_shape = kymo_arr.shape[1:][::-1]
        x_dim = img_shape[0]+(self.viewpadding*2)
        y_len = img_shape[1]
        if tf == -1:
            tf = len(centroids)-1

        fig, ax = plt.subplots(1)
        fig.set_size_inches(x_size, y_size)

        ax.set_xlim(x_dim*t0,x_dim*tf)
        ax.set_ylim(0,img_shape[1])

        cmap = cm.get_cmap('Set3')
        bounds=np.linspace(0.5,12.5,num=13)
        norm = colors.BoundaryNorm(bounds, cmap.N)

        unused_colors = list(range(1,13)) + list(range(1,13))
        current_cell_ids = copy.copy(cell_ids_list[t0])
        color_conversion = {}
        for t in range(t0,tf):
            cell_ids = cell_ids_list[t]
#             print("t=" +str(t) + ": " + str(len(cell_ids)))
            new_cell_ids = []

            for cell_id in current_cell_ids:
                if cell_id not in cell_ids:
                    free_color = color_conversion[cell_id]
                    unused_colors.append(free_color)

            current_cell_ids = copy.copy(cell_ids)

            for cell_id in cell_ids:
                if cell_id not in color_conversion.keys():
                    color_conversion[cell_id] = unused_colors[0]
                    del unused_colors[0]
                    new_cell_ids.append(cell_id)

            padded_img = np.pad(kymo_arr[t],((0,0),(self.viewpadding,self.viewpadding)),'constant')
            relabeled_img = copy.copy(padded_img)

            max_label = np.max(padded_img)
            for l in range(1,np.max(padded_img)+1):
                if l < (len(cell_ids)+1):
                    relabeled_img[padded_img==l] = color_conversion[cell_ids[l-1]]
                else:
                    relabeled_img[padded_img==l] = 0 #TEMP

            ax.imshow(np.ma.array(relabeled_img, mask=(relabeled_img == 0)), extent=[t*x_dim, (t+1)*x_dim, 0, img_shape[1]], cmap=cmap, norm=norm)#, vmin=0)
            if len(centroids[t]) > 0:
                adjusted_centroids = centroids[t] + t*np.array([x_dim,0]) + np.array([self.viewpadding,0])

                ax.scatter(adjusted_centroids[:,0],y_len-adjusted_centroids[:,1],c="r",s=dot_size)

            if t > t0:
                if len(centroids[t-1]) > 0 and len(centroids[t]):
                    for cell_id in new_cell_ids:
                        mother_id = mother_dict[cell_id]
                        if mother_id != -1:
                            last_cell_ids = cell_ids_list[t-1]
                            i = last_cell_ids.index(mother_id)
                            j = cell_ids.index(cell_id)

                            centroid_i = centroids[t-1][i]
                            centroid_j = centroids[t][j]
                            ax.plot([centroid_i[0] + (t-1)*x_dim + self.viewpadding, centroid_j[0] + t*x_dim + self.viewpadding],[y_len-centroid_i[1],y_len-centroid_j[1]],c="c",linewidth=3)

        cell_ids = cell_ids_list[tf]
        padded_img = np.pad(kymo_arr[tf][::-1],((0,0),(self.viewpadding,self.viewpadding)),'constant')
        relabeled_img = copy.copy(padded_img)
        for l in range(1,np.max(padded_img)):
            if l <= len(cell_ids):
                relabeled_img[padded_img==l] = (cell_ids[l-1]+1)
            else:
                relabeled_img[padded_img==l] = 0
        ax.imshow(relabeled_img, extent=[tf*x_dim, (tf+1)*x_dim, 0, img_shape[1]])
        if len(centroids[tf]) > 0:
            adjusted_centroids = centroids[tf] + tf*np.array([x_dim,0]) + np.array([self.viewpadding,0])
            ax.scatter(adjusted_centroids[:,0],adjusted_centroids[:,1],c="r")

        plt.show()

    def run_iteration(self,labeled_data,edge_limit,size_attr):
        centroids,sizes,bboxes = get_segment_props(labeled_data,size_attr=size_attr,interpolate_empty_trenches=False)
        Cij,Cik,Cki,_,posij,posik,poski = self.ScoreFn.compute_score_arrays(centroids,sizes,bboxes)
        prob,Aij,Aik,Aki,Ai = self.solve_tracking_problem(Cij,Cik,Cki,posij,posik,poski,sizes,edge_limit)
        Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list = self.get_sln_params(Aij,Aik,Aki,Ai,posij,posik,poski,sizes,edge_limit)

        return centroids,sizes,prob,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list

    def compute_lineage(self,data,orientation):
        labeled_data = get_labeled_data(data,orientation)
        working_labeled_data = copy.copy(labeled_data)

        iter_outputs = []
        iter_scores = []
        iter_labeled_data = []
        iter_labeled_data.append(labeled_data)

        iter_outputs.append(self.run_iteration(working_labeled_data,self.edge_limit,self.ScoreFn.size_attr))

        centroids,sizes,prob,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list = iter_outputs[-1]

        merged_cells = np.any([np.any(Aki_arr) for Aki_arr in Aki_arr_list])
        objective_val = value(prob.objective)
        active_edges = sum([v.varValue for v in prob.variables()])
        edge_normalized_objective = objective_val/active_edges

        print("Objective = ", objective_val)
        print("Number of Active Edges = ", active_edges)
        print("Edge Normalized Objective = ", edge_normalized_objective)

        return working_labeled_data,centroids,sizes,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,edge_normalized_objective

    def get_nuclear_lineage(self,sizes,Aik_arr_list): ##need to include orientation here
        # if orientation == 1:
        #     sizes = [size[::-1] for size in sizes]
        #     Aik_arr_list = [Aik[::-1,::-1] for Aik in Aik_arr_list] #just flipping wont work, since the algo is greedy from one side

        cell_ids = list(range(len(sizes[0])))
        current_max_id = np.max(cell_ids)
        cell_ids_list = [copy.copy(cell_ids)]
        mother_dict = {}
        daughter_dict = {}
        sister_dict = {}

        for t in range(0,len(Aik_arr_list)):
            max_cells = len(sizes[t+1])
            Aik_cords = np.where(Aik_arr_list[t])
            ttl_added = 0
            for idx in range(len(Aik_cords[0])):
                i = Aik_cords[0][idx]
                current_idx = i+ttl_added
                if current_idx < len(cell_ids):
                    if current_idx == (max_cells-1):

                        daughter_id1,daughter_id2 = (current_max_id+1,-1)

                        mother_dict[daughter_id1] = cell_ids[current_idx]
                        sister_dict[daughter_id1] = daughter_id2

                        daughter_dict[cell_ids[current_idx]] = (daughter_id1,daughter_id2)

                        del cell_ids[current_idx]
                        cell_ids.insert(current_idx,daughter_id1)

                        current_max_id += 1
                        break

                    elif current_idx>(max_cells-1):
                        daughter_id1,daughter_id2 = (-1,-1)
                        daughter_dict[cell_ids[current_idx]] = (daughter_id1,daughter_id2)
                        del cell_ids[current_idx]
                        break

                    else:

                        daughter_id1,daughter_id2 = (current_max_id+1,current_max_id+2)

                        mother_dict[daughter_id1] = cell_ids[current_idx]
                        sister_dict[daughter_id1] = daughter_id2

                        mother_dict[daughter_id2] = cell_ids[current_idx]
                        sister_dict[daughter_id2] = daughter_id1

                        daughter_dict[cell_ids[current_idx]] = (daughter_id1,daughter_id2)

                        del cell_ids[current_idx]
                        cell_ids.insert(current_idx,daughter_id1)
                        cell_ids.insert(current_idx+1,daughter_id2)

                        current_max_id += 2
                        ttl_added += 1

            if max_cells > len(cell_ids):
                num_to_fill = max_cells - len(cell_ids)
                for i in range(num_to_fill):
                    mother_dict[current_max_id+1] = -1
                    sister_dict[current_max_id+1] = -1
                    cell_ids.append(current_max_id+1)
                    current_max_id += 1

            else:
                cell_ids = cell_ids[:max_cells]
            cell_ids_list.append(copy.copy(cell_ids))

        return mother_dict,daughter_dict,sister_dict,cell_ids_list

    def get_cell_lineage(self,file_idx,file_trench_idx,cell_id,mother_dict,daughter_dict,sister_dict):

        if cell_id in daughter_dict.keys():
            daughter_id_1 = daughter_dict[cell_id][0]
            daughter_id_2 = daughter_dict[cell_id][1]
        else:
            daughter_id_1 = -1
            daughter_id_2 = -1
        if cell_id in mother_dict.keys():
            mother_id = mother_dict[cell_id]
        else:
            mother_id = -1
        if cell_id in sister_dict.keys():
            sister_id = sister_dict[cell_id]
        else:
            sister_id = -1

        output_list = [mother_id,daughter_id_1,daughter_id_2,sister_id]
        output = []

        for item in output_list:
            if item == -1:
                output.append(-1)
            else:
                global_id = int(f'{file_idx:08n}{file_trench_idx:04n}{item:04n}')
                output.append(global_id)

        return output

    def get_lineage_df(self,file_idx,file_trench_idx,fov,row,trench_idx,trenchid,lineage_score,\
                       labeled_data,orientation,y_local,x_local,mother_dict,daughter_dict,sister_dict,centroids,cell_ids_list):

        kymograph_file = self.kymopath + "/kymograph_" + str(file_idx) + ".hdf5"
        if self.intensity_channel_list is not None:
            kymo_arr_list = []
            with h5py.File(kymograph_file,"r") as kymofile:
                for intensity_channel in self.intensity_channel_list:
                    intensity_data = kymofile[intensity_channel][:]
                    if orientation == 1:
                        intensity_data = intensity_data[:,:,::-1]
                    kymo_arr_list.append(intensity_data)

        pixel_microns = self.metadata['pixel_microns']
        len_y = labeled_data.shape[1]

        props_output = []
        for t in range(labeled_data.shape[0]):
            cell_ids = cell_ids_list[t]
            ttl_ids = len(cell_ids)

            #non intensity info first
            if orientation == 0:
                non_intensity_rps = sk.measure.regionprops(labeled_data[t],extra_properties=self.custom_props_list)
            else:
                non_intensity_rps = sk.measure.regionprops(labeled_data[t][::-1],extra_properties=self.custom_props_list)
            if self.size_estimation:
                if self.size_estimation_method == "Ellipse":
                    cell_dimensions = get_cell_dimensions_spherocylinder_ellipse(non_intensity_rps)
                elif self.size_estimation_method == "Perimeter/Area":
                    cell_dimensions = get_cell_dimensions_spherocylinder_perimeter_area(non_intensity_rps)
##                 coord_list = center_and_rotate(labeled_data[t])
#                 cell_dimensions = get_cell_dimensions(coord_list, min_dist_integral_start = 0.3)

            if self.intensity_channel_list is not None:
                intensity_rps_list = []
                for i,intensity_channel in enumerate(self.intensity_channel_list):
                    if orientation == 0:
                        intensity_rps = sk.measure.regionprops(labeled_data[t], kymo_arr_list[i][file_trench_idx,t],extra_properties=self.custom_intensity_props_list)
                    else:
                        intensity_rps = sk.measure.regionprops(labeled_data[t][::-1], kymo_arr_list[i][file_trench_idx,t][::-1],extra_properties=self.custom_intensity_props_list)
                    intensity_rps_list.append(intensity_rps)

            for idx in range(ttl_ids):
                cell_id = cell_ids[idx]
                centroid = centroids[t][idx]
                global_cell_id = int(f'{file_idx:08n}{file_trench_idx:04n}{cell_id:04n}')
                rp = non_intensity_rps[idx]

                props_entry = [file_idx, file_trench_idx, fov, row, trench_idx, trenchid, t, idx, cell_id, global_cell_id, orientation, lineage_score]
                props_entry += self.get_cell_lineage(file_idx,file_trench_idx,cell_id,mother_dict,daughter_dict,sister_dict)

                if orientation==0:
                    centroid_x,centroid_y = (centroid[0]*pixel_microns,centroid[1]*pixel_microns)
                    cell_x_local,cell_y_local = (x_local+centroid_x,y_local+(centroid[1]*pixel_microns))
                else:
                    centroid_x,centroid_y = (centroid[0]*pixel_microns,(len_y-centroid[1])*pixel_microns)
                    cell_x_local,cell_y_local = (x_local+centroid_x,y_local+centroid_y)

                props_entry += [centroid_x,centroid_y,cell_x_local,cell_y_local]

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
                        inten_rp=intensity_rps[idx]

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

        base_list = ['File Index','File Trench Index','fov','row','trench','trenchid','timepoints','Segment Index','CellID','Global CellID','lane orientation','Trench Score','Mother CellID','Daughter CellID 1','Daughter CellID 2','Sister CellID','Centroid X','Centroid Y','Cell X (local)', 'Cell Y (local)']

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
        ## Tag mother cells

        df_out["Mother"] = False
        mother_indices = df_out.groupby(['File Index','File Trench Index','timepoints']).apply(lambda x: x['Centroid Y'].idxmin() if x.iloc[0]["lane orientation"]==0 else x['Centroid Y'].idxmax()).tolist()
        df_out.loc[mother_indices, "Mother"] = True

        df_out = df_out.set_index(['File Index','File Trench Index','timepoints','CellID'], drop=True, append=False, inplace=False)
        df_out = df_out.sort_index()

        return df_out

    def lineage_trace(self,kymo_meta,file_idx,file_trench_idx):
        file_trench_idx_i = int(f'{file_idx:08n}{file_trench_idx:04n}{0:04n}')
        file_trench_idx_f = int(f'{file_idx:08n}{file_trench_idx+1:04n}{0:04n}')-1
        trench = kymo_meta.loc[file_trench_idx_i:file_trench_idx_f]
        fov = trench["fov"].iloc[0]
        row = trench["row"].iloc[0]
        trench_idx = trench["trench"].iloc[0]
        trenchid = trench["trenchid"].iloc[0]

        orientation_dict = {"top":0,"bottom":1}
        orientation = trench["lane orientation"].unique().tolist()[0]
        orientation = orientation_dict[orientation]

        y_local = trench["y (local)"].iloc[0]
        x_local = trench["x (local)"].iloc[0]

        with h5py.File(self.segpath + "/segmentation_" + str(file_idx) + ".hdf5", "r") as infile:
            data = infile["data"][file_trench_idx]

        labeled_data,centroids,sizes,_,Aik_arr_list,_,_,lineage_score = self.compute_lineage(data,orientation)

        mother_dict,daughter_dict,sister_dict,cell_ids_list = self.get_nuclear_lineage(sizes,Aik_arr_list)

        df_out = self.get_lineage_df(file_idx,file_trench_idx,fov,row,trench_idx,trenchid,lineage_score,labeled_data,orientation,y_local,x_local,mother_dict,daughter_dict,sister_dict,centroids,cell_ids_list)

        return df_out

    def lineage_trace_file(self,file_idx,n_partitions,divisions):
        writedir(self.lineagepath,overwrite=False)

        kymo_df = dd.read_parquet(self.headpath+"/kymograph/metadata",calculate_divisions=True)
        kymo_df = kymo_df.reset_index(drop=False).set_index("File Parquet Index",sorted=True,npartitions=n_partitions,divisions=divisions)

        start_idx = int(str(file_idx) + "00000000")
        end_idx = int(str(file_idx) + "99999999")

        kymo_df = kymo_df.loc[start_idx:end_idx].compute(scheduler='threads')
        trench_idx_list = kymo_df["File Trench Index"].unique().tolist()

        mergeddf = []
        for file_trench_idx in trench_idx_list:
            try:
                df_out = self.lineage_trace(kymo_df,file_idx,file_trench_idx)
                mergeddf.append(df_out)
            except:
                pass
        if len(mergeddf) > 0:
            mergeddf = pd.concat(mergeddf).reset_index()
            kymo_parq_file_idx = mergeddf.apply(lambda x: int(f'{int(x["File Index"]):08n}{int(x["File Trench Index"]):04n}{int(x["timepoints"]):04n}'), axis=1)
            kymo_parq_file_idx.index = mergeddf.index
            kymo_parq_fov_idx = mergeddf.apply(lambda x: int(f'{int(x["fov"]):08n}{int(x["row"]):04n}{int(x["trench"]):04n}{int(x["timepoints"]):04n}'), axis=1)
            kymo_parq_fov_idx.index = mergeddf.index
            kymo_parq_trenchid_idx = mergeddf.apply(lambda x: int(f'{int(x["trenchid"]):08n}{int(x["timepoints"]):04n}'), axis=1)
            kymo_parq_trenchid_idx.index = mergeddf.index
            mergeddf["Kymograph File Parquet Index"] = kymo_parq_file_idx
            mergeddf["Kymograph FOV Parquet Index"] = kymo_parq_fov_idx
            mergeddf["Trenchid Timepoint Index"] = kymo_parq_trenchid_idx
            del mergeddf["index"]

            mergeddf = mergeddf.dropna(axis=0,subset=["Mother CellID","Daughter CellID 1","Daughter CellID 2","Sister CellID"])

    ##         kymo_df = kymo_df.join(mergeddf.set_index("File Parquet Index")).dropna(axis=0,subset=["Mother CellID","Daughter CellID 1",\
    ##                                                                                        "Daughter CellID 2","Sister CellID"]) #removed to make smaller output without kymograph merge

            mergeddf["Mother CellID"] = mergeddf["Mother CellID"].astype(int)
            mergeddf["Daughter CellID 1"] = mergeddf["Daughter CellID 1"].astype(int)
            mergeddf["Daughter CellID 2"] = mergeddf["Daughter CellID 2"].astype(int)
            mergeddf["Sister CellID"] = mergeddf["Sister CellID"].astype(int)
            mergeddf["Segment Index"] = mergeddf["Segment Index"].astype(int)
            mergeddf["CellID"] = mergeddf["CellID"].astype(int)
            mergeddf["Global CellID"] = mergeddf["Global CellID"].astype(int)

            #remove old indices
    #         del mergeddf["FOV Parquet Index"]

            parq_file_idx = mergeddf.apply(lambda x: int(f'{int(x["File Index"]):08n}{int(x["File Trench Index"]):04n}{int(x["timepoints"]):04n}{int(x["CellID"]):04n}'), axis=1)
            parq_fov_idx = mergeddf.apply(lambda x: int(f'{int(x["fov"]):04n}{int(x["row"]):04n}{int(x["trench"]):04n}{int(x["timepoints"]):04n}{int(x["CellID"]):04n}'), axis=1)

            mergeddf["File Parquet Index"] = parq_file_idx
            mergeddf["FOV Parquet Index"] = parq_fov_idx

            mergeddf = mergeddf.set_index("File Parquet Index")

            print("Done.")
            mergeddf.to_hdf(self.lineagepath + "/temp_output/temp_output." + str(file_idx) + ".hdf", "data",mode="w",format="table")

            return (True,file_idx)
        else:
            return (False,file_idx)

    def lineage_trace_all_files(self,dask_cont,entries_per_partition = 100000, overwrite=False):


        writedir(self.lineagepath+ "/temp_output",overwrite=overwrite)

        # Check for existing files in case of in progress run
        if os.path.exists(self.lineagepath + "/progress.pkl"):
            with open(self.lineagepath + "/progress.pkl", 'rb') as infile:
                finished_files = pkl.load(infile)
        if os.path.exists(self.lineagepath + "/failed.pkl"):
            with open(self.lineagepath + "/failed.pkl", 'rb') as infile:
                failed_files = pkl.load(infile)
        else:
            finished_files = []
            failed_files = []

        kymo_meta = dd.read_parquet(self.headpath+"/kymograph/metadata",calculate_divisions=True)
        kymo_meta = kymo_meta.set_index("File Parquet Index",sorted=True)
        n_partitions,divisions = (kymo_meta.npartitions,kymo_meta.divisions)
        n_entries = len(kymo_meta)

        original_file_list = kymo_meta["File Index"].unique().compute().tolist()
        ttl_files = len(original_file_list)
        entries_per_file = n_entries//ttl_files

        file_list = list(set(original_file_list) - set(finished_files))
        num_file_jobs = len(file_list)

        kymograph_files_per_partition = (entries_per_partition//entries_per_file) + 1

        print("Kymograph files per partition: " + str(kymograph_files_per_partition))

        random_priorities = np.random.uniform(size=(num_file_jobs,))
        lineage_futures = []

        ## check which files have been completed (skipping for now)

        for k,file_idx in enumerate(file_list):
            future = dask_cont.daskclient.submit(self.lineage_trace_file,file_idx,n_partitions,divisions,retries=1)
            lineage_futures.append(future)

            ## df_delayed = delayed(self.lineage_trace_file)(file_idx,n_partitions,divisions)
            ## delayed_list.append(df_delayed)

        for future in as_completed(lineage_futures):
            result = future.result()
            if result[0]:
                finished_files = finished_files + [result[1]]
            else:
                finished_files = finished_files + [result[1]]
                failed_files = failed_files + [result[1]]
            with open(self.lineagepath + "/progress.pkl", 'wb') as outfile:
                pkl.dump(finished_files,outfile)
            with open(self.lineagepath + "/failed.pkl", 'wb') as outfile:
                pkl.dump(failed_files,outfile)
            future.cancel()

        dask_cont.reset_worker_memory()

        unfailed_file_list = sorted(list(set(original_file_list) - set(failed_files)))
        temp_output_file_list = [self.lineagepath + "/temp_output/temp_output." + str(file_idx) + ".hdf" for file_idx in unfailed_file_list]
        temp_df = dd.read_hdf(temp_output_file_list,"data",mode="r",sorted_index=True)

        div_tail = "".join(["0" for i in range(12)])

        if kymograph_files_per_partition != 1:
            repartition_divisions = [int(str(div)+div_tail) for div in range(0,len(unfailed_file_list),kymograph_files_per_partition)] + [temp_df.divisions[-1]]
            temp_df = temp_df.repartition(divisions=repartition_divisions,force=True)
        dd.to_parquet(temp_df, self.lineagepath + "/output",engine='fastparquet',compression='gzip',write_metadata_file=True,overwrite=True)

        dask_cont.reset_worker_memory()

        shutil.rmtree(self.lineagepath + "/temp_output")
        # os.remove(self.lineagepath + "/progress.pkl")
        # os.remove(self.lineagepath + "/failed.pkl")

        return n_partitions,divisions

    def get_stats(self,n_partitions,divisions):

        kymo_df = dd.read_parquet(self.headpath+"/kymograph/metadata",calculate_divisions=True)
        kymo_df = kymo_df.reset_index(drop=False).set_index("File Parquet Index",sorted=True,npartitions=n_partitions,divisions=divisions)

        lineage_df = dd.read_parquet(self.lineagepath + "/output/",calculate_divisions=True)

        n_indices = len(lineage_df["Kymograph File Parquet Index"])
        n_splits = (n_indices//30000000)+1

        initial_trenches = kymo_df["trenchid"].compute().unique().tolist()
        final_trenches = lineage_df.set_index("trenchid",sorted=True).groupby("trenchid").first()["Kymograph File Parquet Index"].compute().index.tolist()

        num_initial_trenches = len(initial_trenches)
        num_final_trenches = len(final_trenches)

        print("# Trenches Processed / # Trenches = " + str(num_final_trenches) + "/" + str(num_initial_trenches))

####     def reorg_parquet(self,dask_cont,futures_list):
#         df_futures = dask_cont.daskclient.gather(futures_list)
#         df_out = dd.concat(df_futures)
#         dd.to_parquet(df_out, self.lineagepath + "/output/",engine='fastparquet',compression='gzip',write_metadata_file=True)

#         df = dd.read_parquet(self.lineagepath + "/block_*.parquet",engine='fastparquet',index=False).set_index("index",drop=False).repartition(partition_size="500MB")
#         dd.to_parquet(df, self.lineagepath + "/output/",engine='fastparquet',compression='gzip',write_metadata_file=True)
#         files = os.listdir(self.lineagepath)
#         for file in files:
#             if "block" in file:
#                 os.remove(file)

#     def wait_for_completion(self,dask_cont):
#         kymo_meta = self.meta_handle.read_df("kymograph")
#         file_list = kymo_meta["File Index"].unique().tolist()
#         num_file_jobs = len(file_list)
#         file_idx_list = dask_cont.daskclient.gather([dask_cont.futures["File Index: " + str(file_idx)] for file_idx in file_list],errors="skip")

    def compute_all_lineages(self,dask_cont,entries_per_partition=100000,overwrite=False):
        writedir(self.lineagepath,overwrite=overwrite)
        dask_cont.futures = {}
        try:
            n_partitions,divisions = self.lineage_trace_all_files(dask_cont,entries_per_partition=entries_per_partition,overwrite=overwrite)
            self.get_stats(n_partitions,divisions)
        except:
            raise


    def test_tracking(self,data,orientation,intensity_channel_list=None,t_range=(0,-1),edge_limit=3,viewpadding=0,x_size=18,y_size=6,dot_size=50):
        if len(intensity_channel_list) == 0:
            self.intensity_channel_list = None
        else:
            self.intensity_channel_list = list(intensity_channel_list)
        self.viewpadding = viewpadding
#         self.merge_per_iter = merge_per_iter
        self.edge_limit = edge_limit
#         self.conv_tolerence = conv_tolerence

        labeled_data,centroids,sizes,Aij_arr_list,Aik_arr_list,Aki_arr_list,Ai_arr_list,lineage_score = self.compute_lineage(data,orientation)
        mother_dict,daughter_dict,sister_dict,cell_ids_list = self.get_nuclear_lineage(sizes,Aik_arr_list)

        self.plot_cleaned_sln(labeled_data,centroids,mother_dict,cell_ids_list,t0=t_range[0],tf=t_range[1],x_size=x_size,y_size=y_size,dot_size=dot_size)

    def interactive_tracking(self,data,orientation):

        available_channels = self.metadata["channels"]

        output = interactive(self.test_tracking,{"manual":True},data=fixed(data),orientation=fixed(orientation),\
                 intensity_channel_list = SelectMultiple(options=available_channels),\
                 t_range=IntRangeSlider(value=[0, data.shape[0]-1],min=0,max=data.shape[0]-1,\
                 step=1,disabled=False,continuous_update=False),\
                 edge_limit=IntSlider(value=self.edge_limit, min=1, max=6, step=1),\
                 viewpadding=IntSlider(value=0, min=0, max=20, step=1),\
                 x_size=IntSlider(value=18, min=0, max=30, step=1),\
                 y_size=IntSlider(value=8, min=0, max=20, step=1),\
                 dot_size=IntSlider(value=50, min=0, max=200, step=5));

        display(output)

    def save_params(self):
        param_dict = {}

        param_dict["Channels:"] = self.intensity_channel_list
##         param_dict["Merges Detected Per Iteration:"] = self.merge_per_iter
##         param_dict["# Unimproved Iterations Before Stopping:"] = self.conv_tolerence
        param_dict["Closest N Objects to Consider:"] = self.edge_limit
        param_dict["Size Measurement:"] = self.ScoreFn.size_attr
        param_dict["Mean Size Increase:"] = self.ScoreFn.u_size
        param_dict["Standard Deviation of Size Increase:"] = self.ScoreFn.sig_size
        param_dict["Mean Position Increase:"] = self.ScoreFn.u_pos
        param_dict["Standard Deviation of Position Increase:"] = self.ScoreFn.sig_pos
        param_dict["Cell Position Weight:"] = self.ScoreFn.w_pos
        param_dict["Cell Size Weight:"] = self.ScoreFn.w_size
        param_dict["Cell Merging Weight:"] = self.ScoreFn.w_merge

        for key,value in param_dict.items():
            print(key + " " + str(value))

        with open(self.headpath + "/lineage_tracing.par", "wb") as outfile:
            pkl.dump(param_dict, outfile)
