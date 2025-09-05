import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
import pickle as pkl
import dask.dataframe as dd

import shutil
import dask
import os
import statsmodels

import scipy.stats

### from recombinator.block_bootstrap import stationary_bootstrap
from dask.distributed import wait
from arch.bootstrap import StationaryBootstrap
# from recombinator.tapered_block_bootstrap import tapered_block_bootstrap
from statsmodels.robust.scale import qn_scale

from .daskutils import to_parquet_checkpoint
#### some helper functions for the various estimators

def coeff_var(x,axis=None):
    return np.std(x,axis=axis)/np.mean(x,axis=axis)

def nancoeff_var(x,axis=None):
    return np.nanstd(x,axis=axis)/np.nanmean(x,axis=axis)

def nancoeff_var_mad(x,axis=None,coeff=1/sp.stats.norm().ppf(3/4)):
    # median uses quickselect
    median = np.nanmedian(x,axis=axis,keepdims=True)
    abs_deviation = np.abs(x-median) #broadcast the median subtraction
    mad = np.nanmedian(abs_deviation,axis=axis,keepdims=True)
    mad_cv = coeff*(mad/median)
    mad_cv = np.squeeze(mad_cv, axis=axis)
    return mad_cv

def nanpearsonr(x):
    nanmask = np.any(np.isnan(x),axis=0)
    x_nonan = x[:,~nanmask]
    r = np.corrcoef(x_nonan)[0,1]
    return r

def dn_fn(n):
    dn_list = np.array([np.NaN,np.NaN,0.399,0.994,0.512,0.844,0.611,0.857,0.669,0.872])
    dn = np.zeros(n.shape)
    under_10_mask = (n<=9)
    even_mask = (n%2==0)
    get_even_over_9_mask = (~under_10_mask)&(even_mask)
    get_odd_over_9_mask = (~under_10_mask)&(~even_mask)
    
    dn[under_10_mask] = dn_list[n[under_10_mask]]
    dn[get_even_over_9_mask] = n[get_even_over_9_mask]/(n[get_even_over_9_mask]+3.8)
    dn[get_odd_over_9_mask] = n[get_odd_over_9_mask]/(n[get_odd_over_9_mask]+1.4)

    return dn
        
## this is as optimized as I can manage at the moment
## if this is really difficult, reconsider numba
def vectorized_qn_scale(x, c=2.219144465985076,stable_eps=10**-12):
    # takes a (N,k) array of observations and computes a Qn estimate for each set
    
    n = x.shape[1]
    if n < 2:
        return np.array([np.NaN for i in range(x.shape[0])])
    nan_count = np.sum(np.isnan(x),axis=1)
    n_effective = n-nan_count
    
    undersampled_mask = (n_effective<2)
    n_well_sampled = n_effective[~undersampled_mask]    
    x_well_sampled = x[~undersampled_mask]
    Qn_output = np.zeros((x.shape[0],))
    Qn_output[undersampled_mask] = np.NaN
    
    residual_nan_count = np.sum(np.isnan(x_well_sampled),axis=1)
    has_nans = (np.sum(residual_nan_count)>0)
    
    dn = dn_fn(n_well_sampled)
    
    vec_xi_minus_xj_mat = np.abs(x_well_sampled[:,:,None]-x_well_sampled[:,None,:])+stable_eps
    vec_xi_minus_xj_mat_diag = np.triu(vec_xi_minus_xj_mat,k=1)
    upper_right = vec_xi_minus_xj_mat_diag[vec_xi_minus_xj_mat_diag!=0]

    n_comparisons = ((x_well_sampled.shape[1]-1)*x_well_sampled.shape[1])//2 
    pair_arr = upper_right.reshape(vec_xi_minus_xj_mat.shape[0],n_comparisons)
    
    h = np.floor(n_effective/2).astype(int) + 1
    q = (h*(h-1))//2
    q = q-1 # to account for base 0
    q[q>pair_arr.shape[1]] = pair_arr.shape[1]-1
    
    if has_nans:
        k = np.array([np.partition(pair_arr[i],q[i])[q[i]] for i in range(x_well_sampled.shape[0])])
    else:
        q_single_val = q[0]
        k = np.partition(pair_arr,q_single_val,axis=1)[:,q_single_val] ## NaNs are considered inf
        
    Qn = dn*c*k
    Qn_output[~undersampled_mask] = Qn
    
    return Qn_output

## partner function for vectorized_qn_scale that uses the statsmodels implemenation
## of the Rousseeuw and Croux algorithm for finding Qn
## it has been checked for consistency with vectorized_qn_scale so it should
## produce the same results. Flip to using this when the sample size exceeds 200

def RandC_algo_qn_scale(x, c=2.219144465985076):
    n = x.shape[1]        
    nan_count = np.sum(np.isnan(x),axis=1)
    n_effective = n-nan_count
    
    undersampled_mask = (n_effective<2)
    n_well_sampled = n_effective[~undersampled_mask]    
    x_well_sampled = x[~undersampled_mask]
    Qn_output = np.zeros((x.shape[0],))
    Qn_output[undersampled_mask] = np.NaN
    
    residual_nan_count = np.sum(np.isnan(x_well_sampled),axis=1)
    has_nans = (np.sum(residual_nan_count)>0)
    
    dn = dn_fn(n_well_sampled)
    if has_nans:
        Qn = []
        for i in range(x_well_sampled.shape[0]):
            x_well_sampled_nonan = x_well_sampled[i][~np.isnan(x_well_sampled[i])]
            Qn.append(dn[i]*qn_scale(x_well_sampled_nonan,c=c,axis=0))
        Qn = np.array(Qn)
    else:
        Qn = dn*qn_scale(x_well_sampled,c=c,axis=1)
    
    Qn_output[~undersampled_mask] = Qn
    
    return Qn_output

def qn_scale_fastest_algo(x, c=2.219144465985076, switch_thr=200, axis=None):
    # takes a (N,k) array of observations and computes a Qn estimate for each set
    # picks the best algorithm for computing this by switching to the better scaling one
    # at a sample size of 200
    # axis included for consistency
    
    if x.shape[1] > switch_thr:
        Qn_output = RandC_algo_qn_scale(x,c=c)
    else:
        Qn_output = vectorized_qn_scale(x,c=c)
    
    return Qn_output

def internal_CV_Qn(x,axis=None):
    # takes a (N,k) array of observations and computes an internal Cv (Qn/mean) estimate for each set
    # axis included for consistency
    Qn = qn_scale_fastest_algo(x)
    mu = np.nanmean(x,axis=1)
    int_cv_qn = Qn/mu
    return int_cv_qn

def Qn_pearson_estimate(x,ddof=None):
    # takes (N,t,2) values and computes N Qn pearson r estimates
    # ddof included for comparability with pearson_bivariate_estimator

    Qn_estimate_x = np.stack([qn_scale_fastest_algo(x[:,:,i]) for i in range(x.shape[2])],axis=1)# N,2 estimates of the individual variable scale =
    x_std = x/Qn_estimate_x[:,None,:] # N,t,2 standardized values
    
    add_var_estimate = qn_scale_fastest_algo(x_std[:,:,0] + x_std[:,:,1])**2 # N additive estimates
    sub_var_estimate = qn_scale_fastest_algo(x_std[:,:,0] - x_std[:,:,1])**2 # N subtractive estimates
    r = (add_var_estimate-sub_var_estimate)/(add_var_estimate+sub_var_estimate)
    return r

def Qn_pearson_estimate_same_scale(x,ddof=None):
    # takes (N,t,2) values and computes N Qn pearson r estimates
    # ddof included for comparability with pearson_bivariate_estimator
    # same scale meaning that the variances of the input variables are
    # assumed to be the same (as in the case of an acf)    
    add_var_estimate = qn_scale_fastest_algo(x[:,:,0] + x[:,:,1])**2 # N estimates
    sub_var_estimate = qn_scale_fastest_algo(x[:,:,0] - x[:,:,1])**2 # N estimates
    r = (add_var_estimate-sub_var_estimate)/(add_var_estimate+sub_var_estimate)
    return r

def vec_cov_axis_1(x,ddof=1):
    #https://stackoverflow.com/questions/40394775/vectorizing-numpy-covariance-for-3d-array
    #computes (N,k,k) covariance matrices for array (N,t,k)
    #can handle NaNs
    x_copy = x.copy()
    N = x_copy.shape[1] #fixed
    nan_count = np.sum(np.any(np.isnan(x_copy),axis=2,keepdims=True),axis=1,keepdims=True)
    N_without_nan = N-nan_count
    x_copy[np.isnan(x_copy)] = 0.
    m1 = x - np.sum(x_copy,axis=1,keepdims=True)/N_without_nan
    m1[np.isnan(m1)] = 0.
    cov = np.einsum('ikj,ikl->ijl',m1,m1)  / (N_without_nan - ddof)
    return cov

def vec_pearson_r_axis_1(x,ddof=1):
    # takes a (N,t,k) array and computes an (N,k,k) pearson correlation matrix
    cov = vec_cov_axis_1(x,ddof=ddof)
    cov_diag = np.diagonal(cov,axis1=1,axis2=2)
    root_cov_diag = np.sqrt(cov_diag)
    pearson_matrix = cov/root_cov_diag[:,:,np.newaxis]
    pearson_matrix = pearson_matrix/root_cov_diag[:,np.newaxis,:]
    return pearson_matrix

def pearson_bivariate_estimator(x,ddof=1):
    # takes a (N,t,2) array and computes an (N,) bivariate pearson r array
    pearson_matrix = vec_pearson_r_axis_1(x,ddof=ddof)
    pearson_r = pearson_matrix[:,1,0]
    return pearson_r

def ccf_estimator_over_variants(x,max_ccf_lag=5,ddof=1,pearson_estimator=Qn_pearson_estimate,**estkwargs):
    # can be used for acfs
    # takes a list of k (N,t,2) arrays, with t not necessarily being the same for each set
    # Converts it into a set of lagged (N,sum_k(t-lag),2) arrays and computes the lagged ccfs
    # then concatenates the k (N,) outputs into n (N,k) output
    ccf_output = []
    for lag in range(0,max_ccf_lag+1):
        lag_arr = []
        for trench_arr in x:
            if trench_arr.shape[1] >= lag:
                trench_lag_arr = np.stack([trench_arr[:,:trench_arr.shape[1]-lag,0],trench_arr[:,lag:,1]],axis=2) # a lagged (N,t-lag,2) array
                lag_arr.append(trench_lag_arr)
        if len(lag_arr)>0:
            lag_arr = np.concatenate(lag_arr,axis=1)
            pearson_r_output = pearson_estimator(lag_arr,ddof=ddof,**estkwargs)
        else:
            pearson_r_output = np.repeat(np.NaN,x[0].shape[0])
        ccf_output.append(pearson_r_output)
    ccf_output = np.stack(ccf_output,axis=1)
    return ccf_output

def bootstrap_density(trench_df,estimator,param_groups,return_kde=False,n_bootstraps_trench_density=1000,**estkwargs):
    params = list(set.union(*[set(param_group) for param_group in param_groups]))
    param_group_positions = [[params.index(param) for param in param_group] for param_group in param_groups]
    n_samples = len(trench_df)
    sample_list = list(range(n_samples))
    sample_array = np.random.choice(sample_list,size=(n_bootstraps_trench_density,n_samples))
    val_arr = trench_df[params].values
    val_sample = val_arr[sample_array]
    ## check if single parameter estimator (assume it can set the axis), else assume it has the same format as the pearson estimator
    if len(param_groups[0]) == 1:
        estimate_sample = np.stack([estimator(val_sample[:,:,param_group_position[0]],axis=1,**estkwargs) for param_group_position in param_group_positions],axis=1)
    else:
        estimate_sample = np.stack([estimator(val_sample[:,:,param_group_position],**estkwargs) for param_group_position in param_group_positions],axis=1)
    if return_kde:
        try:
            estimate_kde = [sp.stats.gaussian_kde(estimate_sample[:,i][~np.isnan(estimate_sample[:,i])]) for i in range(estimate_sample.shape[1])]
            return estimate_kde
        except:
            return None
    else:
        return estimate_sample

# ### the tapered block bootstrap is broken because recombinator is broken
# ### only the stationary bootstrap and standard bootstrap are usable
# def tapered_block_bootstrap_density(trench_df,estimator,param_groups,return_kde=False,n_bootstraps_trench_density=1000,bootstrap_block_len=5,**estkwargs):
#     ## currently estimators must accept NaN values
#     ## param_groups must be a list of lists of grouped parameters for grouped statistics
#     ## Can only take mother values
    
#     params = list(set.union(*[set(param_group) for param_group in param_groups]))
#     param_group_positions = [[params.index(param) for param in param_group] for param_group in param_groups]
    
#     if np.sum(trench_df["Mother"]) == 0:
#         return None
    
#     mother_df = trench_df[trench_df["Mother"]]
#     cell_cycle_timeseries = mother_df[params].values
#     while bootstrap_block_len>0:
#         if cell_cycle_timeseries.shape[0] > bootstrap_block_len:
#             tapered_bootstrap_sample = tapered_block_bootstrap(cell_cycle_timeseries,bootstrap_block_len,n_bootstraps_trench_density)
#             if len(params) == 1:
#                 tapered_bootstrap_sample = tapered_bootstrap_sample[:,:,np.newaxis]
                
#             if len(param_groups[0]) == 1:
#                 estimate_sample = np.stack([estimator(tapered_bootstrap_sample[:,:,param_group_position[0]],axis=1,**estkwargs) for param_group_position in param_group_positions],axis=1)
                
#             else:
#                 estimate_sample = np.stack([estimator(tapered_bootstrap_sample[:,:,param_group_position],**estkwargs) for param_group_position in param_group_positions],axis=1)
            
#             if return_kde:            
#                 try:
#                     estimate_kde = [sp.stats.gaussian_kde(estimate_sample[:,i][~np.isnan(estimate_sample[:,i])]) for i in range(estimate_sample.shape[1])]
#                     return estimate_kde
#                 except:
#                     return None
#             else:
#                 return estimate_sample
#         else:
#             bootstrap_block_len -= 1
#     return None

def get_trajectory_array(trench_df,trench_df_cellid_indexed):
    # get first and second generation cellids
    # trench_df_cellid_indexed = trench_df.set_index("Global CellID")
    mother_df = trench_df[trench_df["Mother"]]
    mother_cellids = mother_df["Global CellID"].tolist()
    daughter_2_gen_1_cellids = mother_df["Daughter CellID 2"].tolist()[:-1]
    all_cellids = trench_df["Global CellID"].tolist()
    all_cellids = [idx for idx in all_cellids if idx != -1]
    daughter_2_gen_1_cellids = [gen_1_cellid if gen_1_cellid in all_cellids else -1 for gen_1_cellid in daughter_2_gen_1_cellids]
    
    cellid_to_daughter_1_dict = trench_df_cellid_indexed["Daughter CellID 1"].to_dict()
    cellid_to_daughter_1_dict[-1] = -1
    
    daughter_2_gen_2_cellids = [cellid_to_daughter_1_dict[cellid] for cellid in daughter_2_gen_1_cellids[:-1]]
    daughter_2_gen_2_cellids = [gen_2_cellid if gen_2_cellid in all_cellids else -1 for gen_2_cellid in daughter_2_gen_2_cellids]
        
    # construct trajectory array with mother trajectory in the bottom row
    # nonexistant cellids take value -1
    trajectory_arr = np.repeat(np.array(mother_cellids)[np.newaxis,:],len(mother_cellids),axis=0)
    np.fill_diagonal(trajectory_arr[:-1,1:],daughter_2_gen_1_cellids)
    np.fill_diagonal(trajectory_arr[:-2,2:],daughter_2_gen_2_cellids)
    trajectory_arr[np.triu_indices(trajectory_arr.shape[0],k=3)] = -1
    trajectory_arr[:-1][np.tril_indices(trajectory_arr.shape[0]-1,k=-1)] = -1
    
    return trajectory_arr

## signature is a super useful argument! See following for usen_bootstraps_trench_density
## Make this faster by reindexing the cellids
def get_trajectory_timeseries(trench_df,params):
    ## lower integer values of global cellids
    trench_df_cellid_indexed = trench_df.set_index("Global CellID")
    trajectory_arr = get_trajectory_array(trench_df,trench_df_cellid_indexed)
    cellid_params_dict = dict(zip(trench_df_cellid_indexed.index.tolist(),trench_df_cellid_indexed[params].values))
    cellid_params_dict[-1] = np.array([np.NaN for param in params])
    cellid_params_vec = np.vectorize(cellid_params_dict.__getitem__,signature='()->(n)')
    trajectory_timeseries = cellid_params_vec(trajectory_arr)
    return trajectory_timeseries

def stationary_bootstrap_density(trench_df,estimator,param_groups,return_kde=False,n_bootstraps_trench_density=1000,bootstrap_block_len=5,**estkwargs):
    ## currently estimators must accept NaN values
    ## param_groups must be a list of lists of grouped parameters for grouped statistics
    ## can also return kdes for each trench if necessary for resampling
    ## the stationary bootstrap uses the data from both the mother cell and the first daughter lineage
    ## other parts of the lineage are omitted for simplicity since they are highly correlated with these two
    ## components anyways. The stationary bootstrap should NOT be used to estimate the ACF since
    ## the discontinuities of values at the block joints will create a bias in the ACF estimator
    ## For this purpose, I favor using the tapered block bootstrap, which has the limitation of 
    ## only using the mother cell data, but that should be fine for the ACF which only uses
    ## the mother cell anyways.
    
    params = list(set.union(*[set(param_group) for param_group in param_groups]))
    param_group_positions = [[params.index(param) for param in param_group] for param_group in param_groups]
    
    if np.sum(trench_df["Mother"]) == 0:
        return None
    
    trajectory_timeseries = get_trajectory_timeseries(trench_df,params)
    while bootstrap_block_len>0:
        if trajectory_timeseries.shape[1] > bootstrap_block_len:
            flat_trajectory_timeseries = trajectory_timeseries.swapaxes(0,1).reshape(trajectory_timeseries.shape[0],-1)
            stationary_bootstrap_sample = np.stack([boot[0][0] for boot in StationaryBootstrap(bootstrap_block_len, flat_trajectory_timeseries).bootstrap(n_bootstraps_trench_density)])
            stationary_bootstrap_sample = stationary_bootstrap_sample.reshape(stationary_bootstrap_sample.shape[0],\
                                            stationary_bootstrap_sample.shape[1],trajectory_timeseries.shape[1],trajectory_timeseries.shape[2]).swapaxes(1,2)
            #note that the following steps eliminate the order of the samples, should not be used for ACFs!
            stationary_bootstrap_sample = stationary_bootstrap_sample.reshape(stationary_bootstrap_sample.shape[0],-1,stationary_bootstrap_sample.shape[3])
            stationary_bootstrap_sample_nancount = np.sum(np.isnan(stationary_bootstrap_sample[:,:,0]),axis=1)
            stationary_bootstrap_sample_minnan = np.min(stationary_bootstrap_sample_nancount)
            
            if stationary_bootstrap_sample_minnan > 0:
                stationary_bootstrap_sample_trim_argsort = np.argsort(stationary_bootstrap_sample[:,:,0],axis=1)
                stationary_bootstrap_sample_trimmed = np.take_along_axis(stationary_bootstrap_sample, stationary_bootstrap_sample_trim_argsort[:,:,None], axis=1)[:,:-stationary_bootstrap_sample_minnan]
            else:
                stationary_bootstrap_sample_trimmed = stationary_bootstrap_sample
                        
            if len(param_groups[0]) == 1:
                estimate_sample = np.stack([estimator(stationary_bootstrap_sample_trimmed[:,:,param_group_position[0]],axis=1,**estkwargs) for param_group_position in param_group_positions],axis=1)
            else:
                estimate_sample = np.stack([estimator(stationary_bootstrap_sample_trimmed[:,:,param_group_position],**estkwargs) for param_group_position in param_group_positions],axis=1)
            if return_kde:
                try:
                    estimate_kde = [sp.stats.gaussian_kde(estimate_sample[:,i][~np.isnan(estimate_sample[:,i])]) for i in range(estimate_sample.shape[1])]
                    return estimate_kde
                except:
                    return None
            else:
                return estimate_sample
        else:
            bootstrap_block_len -= 1
    return None

# def stationary_bootstrap_density(trench_df,estimator,param_groups,return_kde=False,n_bootstraps_trench_density=1000,bootstrap_block_len=5,**estkwargs):
#     ## currently estimators must accept NaN values
#     ## param_groups must be a list of lists of grouped parameters for grouped statistics
#     ## can also return kdes for each trench if necessary for resampling
#     ## the stationary bootstrap uses the data from both the mother cell and the first daughter lineage
#     ## other parts of the lineage are omitted for simplicity since they are highly correlated with these two
#     ## components anyways. The stationary bootstrap should NOT be used to estimate the ACF since
#     ## the discontinuities of values at the block joints will create a bias in the ACF estimator
#     ## For this purpose, I favor using the tapered block bootstrap, which has the limitation of 
#     ## only using the mother cell data, but that should be fine for the ACF which only uses
#     ## the mother cell anyways.
    
#     params = list(set.union(*[set(param_group) for param_group in param_groups]))
#     param_group_positions = [[params.index(param) for param in param_group] for param_group in param_groups]
    
#     if np.sum(trench_df["Mother"]) == 0:
#         return None
    
#     trajectory_timeseries = get_trajectory_timeseries(trench_df,params)
#     while bootstrap_block_len>0:
#         if trajectory_timeseries.shape[1] > bootstrap_block_len:
#             flat_trajectory_timeseries = trajectory_timeseries.swapaxes(0,1).reshape(trajectory_timeseries.shape[0],-1)
#             stationary_bootstrap_sample = stationary_bootstrap(flat_trajectory_timeseries,bootstrap_block_len,n_bootstraps_trench_density)
#             stationary_bootstrap_sample = stationary_bootstrap_sample.reshape(stationary_bootstrap_sample.shape[0],\
#                                             stationary_bootstrap_sample.shape[1],trajectory_timeseries.shape[1],trajectory_timeseries.shape[2]).swapaxes(1,2)
#             #note that the following steps eliminate the order of the samples, should not be used for ACFs!
#             stationary_bootstrap_sample = stationary_bootstrap_sample.reshape(stationary_bootstrap_sample.shape[0],-1,stationary_bootstrap_sample.shape[3])
#             stationary_bootstrap_sample_nancount = np.sum(np.isnan(stationary_bootstrap_sample[:,:,0]),axis=1)
#             stationary_bootstrap_sample_minnan = np.min(stationary_bootstrap_sample_nancount)
            
#             if stationary_bootstrap_sample_minnan > 0:
#                 stationary_bootstrap_sample_trim_argsort = np.argsort(stationary_bootstrap_sample[:,:,0],axis=1)
#                 stationary_bootstrap_sample_trimmed = np.take_along_axis(stationary_bootstrap_sample, stationary_bootstrap_sample_trim_argsort[:,:,None], axis=1)[:,:-stationary_bootstrap_sample_minnan]
#             else:
#                 stationary_bootstrap_sample_trimmed = stationary_bootstrap_sample
                        
#             if len(param_groups[0]) == 1:
#                 estimate_sample = np.stack([estimator(stationary_bootstrap_sample_trimmed[:,:,param_group_position[0]],axis=1,**estkwargs) for param_group_position in param_group_positions],axis=1)
#             else:
#                 estimate_sample = np.stack([estimator(stationary_bootstrap_sample_trimmed[:,:,param_group_position],**estkwargs) for param_group_position in param_group_positions],axis=1)
#             if return_kde:
#                 try:
#                     estimate_kde = [sp.stats.gaussian_kde(estimate_sample[:,i][~np.isnan(estimate_sample[:,i])]) for i in range(estimate_sample.shape[1])]
#                     return estimate_kde
#                 except:
#                     return None
#             else:
#                 return estimate_sample
#         else:
#             bootstrap_block_len -= 1
#     return None

def tapered_block_bootstrap_ccf_density(mother_df,pearson_estimator,param_groups,return_kde=False,max_ccf_lag=3,n_bootstraps=1000,bootstrap_block_len=6,**estkwargs):
    ## A tapered block bootstrap defined across a variant df
    ## variant df must be sorted by timepoint for each trench, this can be done by sorting global cellids
    ## currently estimators must accept NaN values
    ## param_groups must be a list of lists of grouped parameters for grouped statistics
    ## Can only take mother values
    ## bootstrap blocks should be larger than acf lag (good rule of thumb is 4x the length, but 2x also fine)
    ## note that this bootstrap is over the entire variant set since the sampling per variant is poor
    ## still, I have added resampling of the trenches on top of this
    
    params = list(set.union(*[set(param_group) for param_group in param_groups]))

    cell_cycle_timeseries_df = mother_df[params]
    timepoint_counts = cell_cycle_timeseries_df.groupby("Multi-Experiment Phenotype Trenchid").size() #note that this counts NaNs
    
    ## filtering for examples at least twice the length of the chosen block size
    ## just picked this because it seems to control the variance outliers, perhaps a more principled choice is possible
    trenches_longer_than_bootstrap = timepoint_counts>(2*bootstrap_block_len)
    cell_cycle_timeseries_df_filtered = cell_cycle_timeseries_df.loc[trenches_longer_than_bootstrap[trenches_longer_than_bootstrap].index.tolist()]
    trench_idx_list = cell_cycle_timeseries_df_filtered.index.get_level_values(0).unique()
    n_trenches = len(trench_idx_list)
    
    tapered_bootstrap_sample_all_variants = []
    for i,trench_idx in enumerate(trench_idx_list):
        cell_cycle_timeseries = cell_cycle_timeseries_df_filtered.loc[trench_idx][params].values
        tapered_bootstrap_sample = tapered_block_bootstrap(cell_cycle_timeseries,bootstrap_block_len,n_bootstraps)
        if len(params) == 1:
            tapered_bootstrap_sample = tapered_bootstrap_sample[:,:,np.newaxis] #N, t, #param bootstrap
        tapered_bootstrap_sample = xr.DataArray(tapered_bootstrap_sample,dims=["Bootstrap","Timepoint","Parameter"],\
                                            coords=[range(n_bootstraps),range(tapered_bootstrap_sample.shape[1]),params])
        tapered_bootstrap_sample = tapered_bootstrap_sample.to_dataframe(name="Value").reset_index()
        tapered_bootstrap_sample["Bootstrap"] = tapered_bootstrap_sample["Bootstrap"]+(i*n_bootstraps)
        tapered_bootstrap_sample_all_variants.append(tapered_bootstrap_sample)
    
    if len(tapered_bootstrap_sample_all_variants) == 0:
        return None
    
    tapered_bootstrap_sample_all_variants = pd.concat(tapered_bootstrap_sample_all_variants).set_index(["Bootstrap","Timepoint","Parameter"])
    tapered_bootstrap_sample_all_variants = tapered_bootstrap_sample_all_variants.to_xarray().to_array()[0]
    
    ccf_estimate_sample = []
    for param_group in param_groups:
        working_sample = tapered_bootstrap_sample_all_variants.loc[:,:,param_group].to_numpy()
        bootstrap_roll = np.random.choice(np.arange(0,working_sample.shape[0]),size=(n_bootstraps*n_trenches))
        bootstrap_roll = working_sample[bootstrap_roll]
        del working_sample
        bootstrap_roll = bootstrap_roll.reshape(n_trenches,n_bootstraps,bootstrap_roll.shape[1],2)
        bootstrap_roll = [bootstrap_roll[k] for k in range(bootstrap_roll.shape[0])]
        
        ccf_output = ccf_estimator_over_variants(bootstrap_roll,max_ccf_lag=max_ccf_lag,pearson_estimator=pearson_estimator,**estkwargs)
        ccf_estimate_sample.append(ccf_output)
    ccf_estimate_sample = np.stack(ccf_estimate_sample,axis=1)

    if return_kde:            
        try:
            estimate_kde = []
            for i in range(ccf_estimate_sample.shape[1]):
                estimate_kde_lag_list = []
                for j in range(1,ccf_estimate_sample.shape[2]):
                    estimate_kde_lag = sp.stats.gaussian_kde(ccf_estimate_sample[:,i,j][~np.isnan(ccf_estimate_sample[:,i,j])]) 
                    estimate_kde_lag_list.append(estimate_kde_lag)
                estimate_kde.append(estimate_kde_lag_list)
            return estimate_kde
        except:
            return None
    
    else:
        return ccf_estimate_sample

def all_trench_bootstrap(trench_df,bootstrap_density_function,param_groups_list,return_kde=False,estimators=[np.nanmean],estimator_names=["Mean"],\
                         n_bootstraps_trench_density=1000,variant_index="oDEPool7_id",trench_index="Multi-Experiment Phenotype Trenchid",\
                         cell_index="Multi-Experiment Global CellID",**kwargs):
    
    estimate_samples_df = []
    for i,estimator in enumerate(estimators):
        param_groups = param_groups_list[i]
        param_groups_joined = ["-".join(param_group) for param_group in param_groups]
        
        #sort again since later sgRNA sort messes up order
        trench_df = trench_df.reset_index().set_index(cell_index).sort_index().reset_index().set_index(trench_index)
        estimate_sample = bootstrap_density_function(trench_df,estimator,param_groups,return_kde=return_kde,n_bootstraps_trench_density=n_bootstraps_trench_density,**kwargs) #N x K output
        
        if estimate_sample is None:
            pass
        else:
            if return_kde:
                estimate_sample_df = xr.DataArray(estimate_sample,coords=[param_groups_joined],\
                                                  dims=["Variable(s)"],name="KDE").to_dataframe()
            else:
                estimate_sample_df = xr.DataArray(estimate_sample,coords=[range(0,n_bootstraps_trench_density),param_groups_joined],\
                                                  dims=["Trench Bootstrap Sample","Variable(s)"],name="Value").to_dataframe()
            estimate_sample_df["Estimator"] = estimator_names[i]
            estimate_samples_df.append(estimate_sample_df)
            
    if len(estimate_samples_df)>0:
        estimate_samples_df = pd.concat(estimate_samples_df).reset_index()
        estimate_samples_df.index.name="Bootstrap Index"
        if len(trench_df) > 0:
            estimate_samples_df[variant_index] = trench_df.iloc[0][variant_index]
        else:
            estimate_samples_df[variant_index] = -1
        return estimate_samples_df
    else:
        return None
    
def export_estimator_df(steady_state_df_path,preinduction_df_path,single_variable_list,bivariate_variable_list,\
                        estimators = [np.nanmean],estimator_names = ["Mean"],bivariate_estimator_list = [False],\
                        estimator_names_to_agg = ["Mean","Mean"],unpaired_aggregators=[np.nanmedian,np.nanmean],\
                        bivariate_aggregator_list=[False,False],paired_aggregators=[],\
                        agg_names=["Mean (Robust)","Mean (True)"],filter_proliferating=False,variant_index='oDEPool7_id',\
                        control_categories=['OnlyPlasmid', 'NoTarget'],\
                        final_columns = ['sgRNA','EcoWG1_id', 'Gene', 'N Mismatch', 'Category', 'TargetID']):
    final_columns = [variant_index] + final_columns
    wrapped_single_variable_list = [[variable] for variable in single_variable_list]
    param_groups_list = [bivariate_variable_list if item else wrapped_single_variable_list for item in bivariate_estimator_list]
    param_groups_to_agg_list = [bivariate_variable_list if item else wrapped_single_variable_list for item in bivariate_aggregator_list]
    aggregators = unpaired_aggregators + paired_aggregators
    is_paired_aggregator_list = [False for i in range(len(unpaired_aggregators))]+[True for i in range(len(paired_aggregators))]

    trench_estimator_path = steady_state_df_path + "_Trench_Estimators.pkl"
    estimator_path = steady_state_df_path + "_Estimators.pkl"

    final_df_ss = dd.read_parquet(steady_state_df_path, engine="pyarrow",calculate_divisions=True)
    final_df_preinduction = dd.read_parquet(preinduction_df_path, engine="pyarrow",calculate_divisions=True)
    if filter_proliferating:
        final_df_ss = final_df_ss[final_df_ss["Proliferating"]]
        final_df_preinduction = final_df_preinduction[final_df_preinduction["Proliferating"]]

    trench_estimator_ss_df_output = get_estimator_df_all_trenches(final_df_ss,estimator_names,estimators,param_groups_list,final_columns,\
                                                                  trench_index="Multi-Experiment Phenotype Trenchid")
    trench_estimator_preinduction_df_output = get_estimator_df_all_trenches(final_df_preinduction,estimator_names,estimators,param_groups_list,\
                                                                            final_columns,trench_index="Multi-Experiment Phenotype Trenchid")
    trench_estimator_ss_df_output["Induction"] = "Post"
    trench_estimator_preinduction_df_output["Induction"] = "Pre"
    trench_estimator_ss_df_output = trench_estimator_ss_df_output.reset_index().set_index(["Induction","Estimator","Multi-Experiment Phenotype Trenchid"])
    trench_estimator_preinduction_df_output = trench_estimator_preinduction_df_output.reset_index().set_index(["Induction","Estimator","Multi-Experiment Phenotype Trenchid"])
    
    trench_estimator_df = pd.concat([trench_estimator_ss_df_output,trench_estimator_preinduction_df_output])
    del trench_estimator_ss_df_output
    del trench_estimator_preinduction_df_output
    trench_estimator_df.to_pickle(trench_estimator_path)
    del trench_estimator_df
    trench_estimator_df = pd.read_pickle(trench_estimator_path)
    trench_estimator_df = trench_estimator_df.reset_index()
    trench_estimator_dd = dd.from_pandas(trench_estimator_df,100).persist();
    trench_estimator_dd = trench_estimator_dd.set_index("Multi-Experiment Phenotype Trenchid").persist()
    wait(trench_estimator_dd);
    del trench_estimator_df

    estimator_df = get_estimator_df(trench_estimator_dd,final_columns,param_groups_to_agg_list,\
                    estimator_names_to_agg,aggregators,is_paired_aggregator_list,agg_names,variant_index=variant_index)

    if ("Mean (Robust)" in agg_names) and ("Variance (Extrinsic)" in agg_names):
        ###Adding derived values
        ### note that CV (external) is normalized by wt median
        control_estimator_df = estimator_df[estimator_df["Category"].isin(control_categories)]
        control_medians = control_estimator_df.loc["Mean (Robust)"].groupby("Variable(s)")["Value"].median()
        
        CV_extrinsic_df = estimator_df.loc["Variance (Extrinsic)"].copy()
        CV_extrinsic_df["Value"] = np.sqrt(estimator_df.loc["Variance (Extrinsic)"]["Value"])/control_medians
        CV_extrinsic_df["Estimator"] = "CV (Extrinsic)"
        CV_extrinsic_df = CV_extrinsic_df.reset_index().set_index(["Estimator","Variable(s)",variant_index])
        
        estimator_df = pd.concat([estimator_df,CV_extrinsic_df]).sort_index()
    
    estimator_df.to_pickle(estimator_path)

def EVE_intrinsic_variance_agg(Qn_arr,axis=0):
    ## x is (t,N) where t is trench, N is bootstraps
    intrinsic_variance = np.nanmedian(Qn_arr**2,axis=axis)
    return intrinsic_variance

def EVE_extrinsic_variance_agg(mean_arr,axis=0):
    ## x is (t,N) where t is trench, N is bootstraps
    extrinsic_variance = np.nanvar(mean_arr,axis=axis)
    return extrinsic_variance

def EVE_total_variance_agg(mean_Qn_arr,axis=0):
    mean_arr,Qn_arr = mean_Qn_arr[:,:,0],mean_Qn_arr[:,:,1]
    intrinsic_variance = np.nanmedian(Qn_arr**2,axis=axis)
    extrinsic_variance = np.nanvar(mean_arr,axis=axis)
    ttl_variance = intrinsic_variance+extrinsic_variance
    ttl_variance = ttl_variance[:,None]
    return ttl_variance

def Frac_EVE_ext_over_int_agg(mean_Qn_arr,axis=0):
    mean_arr,Qn_arr = mean_Qn_arr[:,:,0],mean_Qn_arr[:,:,1]
    intrinsic_variance = np.nanmedian(Qn_arr**2,axis=axis)
    extrinsic_variance = np.nanvar(mean_arr,axis=axis)
    frac_extrinsic = extrinsic_variance/intrinsic_variance
    frac_extrinsic = frac_extrinsic[:,None]
    return frac_extrinsic

def CV_total_agg(mean_Qn_arr,axis=0):
    mean_arr,Qn_arr = mean_Qn_arr[:,:,0],mean_Qn_arr[:,:,1]
    intrinsic_variance = np.nanmedian(Qn_arr**2,axis=axis)
    extrinsic_variance = np.nanvar(mean_arr,axis=axis)
    ttl_variance = intrinsic_variance+extrinsic_variance
    expected_value = np.nanmedian(mean_arr,axis=axis)
    CV_ttl = np.sqrt(ttl_variance)/expected_value
    CV_ttl = CV_ttl[:,None]
    return CV_ttl

def all_variant_ccf_bootstrap(variant_df,bootstrap_density_function,param_groups_list,pearson_estimators=[pearson_bivariate_estimator],estimator_names=["Pearson R"],\
                         n_bootstraps=1000,max_ccf_lag=3,bootstrap_block_len=6,variant_index="oDEPool7_id",trench_index="Multi-Experiment Phenotype Trenchid",\
                              timepoint_index="initial timepoints",**kwargs):
    
    #sort again since later sgRNA sort messes up order
    mother_df = variant_df[variant_df["Mother"]]
    mother_df = mother_df.reset_index().set_index([trench_index,timepoint_index]).sort_index()
        
    estimate_samples_df = []
    for i,pearson_estimator in enumerate(pearson_estimators):
        param_groups = param_groups_list[i]
        param_groups_joined = ["-".join(param_group) for param_group in param_groups]
        
        estimate_sample = bootstrap_density_function(mother_df,pearson_estimator,param_groups,n_bootstraps=n_bootstraps,max_ccf_lag=max_ccf_lag,\
                                                     bootstrap_block_len=bootstrap_block_len,**kwargs) #N x K output
        estimate_sample_df = xr.DataArray(estimate_sample,coords=[range(0,n_bootstraps),param_groups_joined,range(0,max_ccf_lag+1)],\
                                          dims=["Variant Bootstrap Sample","Variable(s)","Lag"],name="Value").to_dataframe()
        estimate_sample_df["Estimator"] = estimator_names[i]
        estimate_samples_df.append(estimate_sample_df)

    estimate_samples_df = pd.concat(estimate_samples_df).reset_index()
    estimate_samples_df.index.name="Bootstrap Index"
    estimate_samples_df[variant_index] = variant_df.index[0]
    estimate_samples_df = estimate_samples_df.drop(columns=[variant_index])
    
    return estimate_samples_df

##note that this may be broken for the control bootstrapping, have not tested this with the .sort_index call on the second line
def get_extrinsic_CV(partition,control_medians,value_col_name="Value"):
    input_columns = partition.columns
    partition = partition.set_index("Variable(s)").sort_index()
    index_list = partition.index.unique().tolist()
    partition[value_col_name] = np.sqrt(partition[value_col_name])/control_medians[index_list]
    partition["Estimator"] = "CV (Extrinsic)"
    partition = partition.reset_index()
    partition = partition.reindex(columns=input_columns)
    return partition

def cull_empty_partitions(df):
    ll = list(df.map_partitions(len).compute())
    df_delayed = df.to_delayed()
    df_delayed_new = list()
    pempty = None
    for ix, n in enumerate(ll):
        if 0 == n:
            pempty = df.get_partition(ix)
        else:
            df_delayed_new.append(df_delayed[ix])
    if pempty is not None:
        df = dd.from_delayed(df_delayed_new, meta=pempty)
    return df

def get_estimator_df_single_trench(trench_df,param_groups,estimator,final_columns,estimator_name,trench_index="Multi-Experiment Phenotype Trenchid"):
    param_groups_joined = ["-".join(param_group) for param_group in param_groups]
    estimator_output_list = []
    for param_group in param_groups:
        if len(param_group) == 1:
            estimator_output = estimator(trench_df[param_group].values.T, axis=1)[0]
        else:
            estimator_output = estimator(trench_df[param_group].values[None,:,:])[0]
        estimator_output_list.append(estimator_output)
    estimator_output = pd.DataFrame([estimator_output_list]).rename(columns={i:param_groups_joined[i] for i in range(len(estimator_output_list))})
    if len(trench_df) == 0:
        return None
    trenchid_out = trench_df.index[0]
    estimator_output[trench_index] = trenchid_out
    estimator_output = estimator_output.set_index(trench_index)

    for final_column in final_columns:
        estimator_output[final_column] = trench_df[final_column].iloc[0]    
    estimator_output["Estimator"] = estimator_name
    
    return estimator_output

def get_estimator_df_all_trenches(df,estimator_names,estimators,param_groups_list,final_columns,trench_index="Multi-Experiment Phenotype Trenchid"):
    trench_groupby = df.groupby(trench_index)
    trench_estimator_df_output = []
    for i,estimator_name in enumerate(estimator_names):
        estimator = estimators[i]
        param_groups = param_groups_list[i]
        trench_estimator_df = trench_groupby.apply(lambda x: get_estimator_df_single_trench(x,param_groups,estimator,\
                                                final_columns,estimator_name,trench_index=trench_index)).compute()
        trench_estimator_df_output.append(trench_estimator_df)
    trench_estimator_df_output = pd.concat(trench_estimator_df_output).droplevel(1).reset_index().set_index(["Estimator",trench_index])
    return trench_estimator_df_output

def get_estimator_df_single_variant(variant_df,param_groups,estimator_name,final_columns,aggregator,is_paired_aggregator,agg_name,trench_index="Multi-Experiment Phenotype Trenchid"):
    variant_df = variant_df.reset_index().set_index(["Induction","Estimator",trench_index]).sort_index()
    post_in_index = ("Post",estimator_name) in variant_df.index
    pre_in_index = ("Pre",estimator_name) in variant_df.index
    
    if post_in_index:
        if not pre_in_index and is_paired_aggregator:
            return None

        post_estimator_variant_df = variant_df.loc["Post",estimator_name]
        
        if pre_in_index:
            pre_estimator_variant_df = variant_df.loc["Pre",estimator_name]
        param_groups_joined = ["-".join(param_group) for param_group in param_groups]

        estimator_series_output = {}
        for i,param_group in enumerate(param_groups_joined):
            post_param_arr = np.array(post_estimator_variant_df[param_group].tolist())[:,None,None]
            if is_paired_aggregator:
                pre_param_arr = np.array(pre_estimator_variant_df[param_group].tolist())[:,None,None]
                estimator_output = aggregator(pre_param_arr,post_param_arr,axis=0)[0,0]
            else:
                estimator_output = aggregator(post_param_arr,axis=0)[0,0]
            estimator_series_output[param_group] = estimator_output
        estimator_output = pd.Series(estimator_series_output)
        del estimator_series_output

        estimator_output = pd.DataFrame(estimator_output).reset_index().rename(columns={"index":"Variable(s)",0:"Value"})
        estimator_output["Estimator"] = agg_name

        for final_column in final_columns:
            estimator_output[final_column] = variant_df.iloc[0][final_column]
    
        return estimator_output
    
    else:
        return None

def get_estimator_df(df,final_columns,param_groups_to_agg_list,estimator_names_to_agg,aggregators,is_paired_aggregator_list,agg_names,\
                     variant_index='oDEPool7_id',trench_key='Multi-Experiment Phenotype Trenchid'):
    
    ## filter for trenches present pre and post treatment
    pre_post_mask = df.groupby(trench_key).apply(lambda x: sorted(x["Induction"].unique())==["Post","Pre"]).compute()
    trenchids_prepost = pre_post_mask[pre_post_mask].index.tolist()
    pre_post_df = df.loc[trenchids_prepost]
    
    sgrna_sorted = df.reset_index().set_index(variant_index).persist()
    sgrna_sorted_pre_post = pre_post_df.reset_index().set_index(variant_index).persist()
    
    sgrna_groupby = sgrna_sorted.groupby(variant_index)
    sgrna_groupby_pre_post = sgrna_sorted_pre_post.groupby(variant_index)
    
    first_idx = sgrna_sorted.index.unique().compute()[0]
    test_input = sgrna_sorted.loc[first_idx].compute()
    
    first_idx_pre_post = sgrna_sorted_pre_post.index.unique().compute()[0]
    test_input_pre_post = sgrna_sorted_pre_post.loc[first_idx_pre_post].compute()
    
    estimator_df_output = []
    for i,agg_name in enumerate(agg_names):
        param_groups = param_groups_to_agg_list[i]
        estimator_name = estimator_names_to_agg[i]
        aggregator = aggregators[i]
        is_paired_aggregator = is_paired_aggregator_list[i]
        
        if is_paired_aggregator:
            working_test_input = test_input_pre_post
            working_sgrna_groupby = sgrna_groupby_pre_post
        else:
            working_test_input = test_input
            working_sgrna_groupby = sgrna_groupby
        
        test_output = get_estimator_df_single_variant(test_input,param_groups,estimator_name,\
                        final_columns,aggregator,is_paired_aggregator,agg_name)
        estimator_df = working_sgrna_groupby.apply(lambda x: get_estimator_df_single_variant(x,param_groups,estimator_name,\
                                                final_columns,aggregator,is_paired_aggregator,agg_name),meta=test_output).compute()
        
        estimator_df_output.append(estimator_df)
    estimator_df_output = pd.concat(estimator_df_output).drop(columns=variant_index).reset_index().drop(columns=["level_1"]).set_index(["Estimator","Variable(s)",variant_index]).sort_index()
    return estimator_df_output

def median_difference(pre_array,post_array,axis=0):
    return np.nanmedian(post_array-pre_array,axis=axis)
def mean_difference(pre_array,post_array,axis=0):
    return np.nanmean(post_array-pre_array,axis=axis)
def median_difference_variance(pre_array,post_array,axis=0):
    return np.nanmedian((post_array**2)-(pre_array**2),axis=axis)
def morgan_pitman_statistic(pre_array,post_array,spearman=False,axis=0):
    #spearman takes too long, but precomputing the rank before bootstrap could be a fix
    #moving on without implementing since this is already taking too long and pearson r
    #is more effecient anyways
    U = post_array+pre_array
    V = post_array-pre_array
    U_flat,V_flat = U.reshape(U.shape[0],-1),V.reshape(V.shape[0],-1)
    UV_flat = np.stack([U_flat,V_flat],axis=2)
    UV_flat = np.swapaxes(UV_flat,0,1)
    if spearman:
        UV_flat = sp.stats.rankdata(UV_flat, method='average', axis=1)
    r = pearson_bivariate_estimator(UV_flat)
    r = r.reshape(U.shape[1],U.shape[2])
    return r


def trenchwise_bootstrap_main(dask_controller,steady_state_df_path,preinduction_df_path,bootstrap_density_function,estimators,estimator_names,\
                              bivariate_variable_list,single_variable_list,pearsonr_variable_list,variant_index="oDEPool7_id",filter_proliferating=True,\
                              n_bootstraps_trench_density=200,n_variants_per_block=50,overwrite=False,**density_kwargs):
        
    wrapped_single_variable_list = [[variable] for variable in single_variable_list]
    wrapped_pearsonr_variable_list = [pearsonr_variable_list]
    param_groups_list = [wrapped_pearsonr_variable_list if bivariate_variable else wrapped_single_variable_list for bivariate_variable in bivariate_variable_list]
    
    filtered_steady_state_df_path = steady_state_df_path + "_temp_filtered"
    filtered_preinduction_df_path = preinduction_df_path + "_temp_filtered"

    trench_bootstrap_steady_state_temp_path = steady_state_df_path + "_Trench_Estimator_Bootstrap_temp"
    trench_bootstrap_preinduction_temp_path = preinduction_df_path + "_Trench_Estimator_Bootstrap_temp"
    trench_bootstrap_path_temp = steady_state_df_path + "_Trench_Estimator_Bootstrap_temp_2"
    
    trench_bootstrap_path = steady_state_df_path + "_Trench_Estimator_Bootstrap"
    progress_path = steady_state_df_path + "_trenchwise_bootstrap_checkpoints.pkl"
    
    if overwrite:
        progress_state = 0
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
    else:
        if os.path.exists(progress_path):
            with open(progress_path, 'rb') as progfile:
                progress_state = pkl.load(progfile)
        else:
            progress_state = 0
            with open(progress_path, 'wb') as progfile:
                pkl.dump(progress_state, progfile)
    
    ## part 1 ##    
    
    if progress_state == 0:
        print(progress_state)
        
        ##filtering for proliferating cells and minimum observations

        final_df_ss = dd.read_parquet(steady_state_df_path, engine="pyarrow",calculate_divisions=True)
        final_df_preinduction = dd.read_parquet(preinduction_df_path, engine="pyarrow",calculate_divisions=True)

        if filter_proliferating:

            final_df_ss = final_df_ss[final_df_ss["Proliferating"]]
            final_df_preinduction = final_df_preinduction[final_df_preinduction["Proliferating"]]

        ss_sgrna_obs_count = final_df_ss.groupby(variant_index)[final_df_ss.columns[0]].size().compute()
        preinduction_sgrna_obs_count = final_df_preinduction.groupby(variant_index)[final_df_ss.columns[0]].size().compute()
        ss_sgrnas_to_remove = set(ss_sgrna_obs_count[ss_sgrna_obs_count<2].index)
        preinduction_sgrnas_to_remove = set(preinduction_sgrna_obs_count[preinduction_sgrna_obs_count<2].index)
        sgrnas_to_remove = list(ss_sgrnas_to_remove or preinduction_sgrnas_to_remove)

        final_df_ss = final_df_ss[~final_df_ss[variant_index].isin(sgrnas_to_remove)]
        final_df_preinduction = final_df_preinduction[~final_df_preinduction[variant_index].isin(sgrnas_to_remove)]

        to_parquet_checkpoint(dask_controller,final_df_ss,filtered_steady_state_df_path,engine='pyarrow',overwrite=overwrite)
        to_parquet_checkpoint(dask_controller,final_df_preinduction,filtered_preinduction_df_path,engine='pyarrow',overwrite=overwrite)

        dask_controller.daskclient.cancel(final_df_ss)
        dask_controller.daskclient.cancel(final_df_preinduction)
        
        progress_state = 1
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
        
        overwrite = True
            
    ## part 2 ##
    
    if progress_state == 1:
        print(progress_state)
        final_df_ss = dd.read_parquet(filtered_steady_state_df_path, engine="pyarrow",calculate_divisions=True)

        input_partition = final_df_ss.get_partition(0).compute()
        output_partition = input_partition.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x: \
                            all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,estimators=estimators,\
                            estimator_names=estimator_names,n_bootstraps_trench_density=n_bootstraps_trench_density,\
                            variant_index=variant_index,**density_kwargs))
        trench_bootstrap_dd = final_df_ss.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x: \
                            all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,estimators=estimators,\
                            estimator_names=estimator_names,n_bootstraps_trench_density=n_bootstraps_trench_density,\
                            variant_index=variant_index,**density_kwargs),meta=output_partition)
        
        to_parquet_checkpoint(dask_controller,trench_bootstrap_dd,trench_bootstrap_steady_state_temp_path,engine='pyarrow',overwrite=overwrite)
        dask_controller.daskclient.cancel(trench_bootstrap_dd)
        
        progress_state = 2
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
            
        overwrite = True
    ## part 3 ##
    
    if progress_state == 2:
        print(progress_state)
        final_df_preinduction = dd.read_parquet(filtered_preinduction_df_path, engine="pyarrow",calculate_divisions=True)

        input_partition = final_df_preinduction.get_partition(0).compute()
        output_partition = input_partition.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x: \
                            all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,estimators=estimators,\
                            estimator_names=estimator_names,n_bootstraps_trench_density=n_bootstraps_trench_density,\
                            variant_index=variant_index,**density_kwargs))

        trench_preinduction_bootstrap_dd = final_df_preinduction.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x: \
                            all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,estimators=estimators,\
                            estimator_names=estimator_names,n_bootstraps_trench_density=n_bootstraps_trench_density,\
                            variant_index=variant_index,**density_kwargs),meta=output_partition)
        
        to_parquet_checkpoint(dask_controller,trench_preinduction_bootstrap_dd,trench_bootstrap_preinduction_temp_path,engine='pyarrow',overwrite=overwrite)
        dask_controller.daskclient.cancel(trench_preinduction_bootstrap_dd)
        
        progress_state = 3
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
            
        overwrite = True
            
    if progress_state == 3:
        print(progress_state)
    
        ### part 4 ###

        trench_bootstrap_dd = dd.read_parquet(trench_bootstrap_steady_state_temp_path, engine="pyarrow")
        trench_preinduction_bootstrap_dd = dd.read_parquet(trench_bootstrap_preinduction_temp_path, engine="pyarrow")
        
        trench_bootstrap_dd = trench_bootstrap_dd.reset_index().set_index("Multi-Experiment Phenotype Trenchid")
        trench_preinduction_bootstrap_dd = trench_preinduction_bootstrap_dd.reset_index().set_index("Multi-Experiment Phenotype Trenchid")

        trench_bootstrap_dd["Induction"] = "Post"
        trench_preinduction_bootstrap_dd["Induction"] = "Pre"

        trench_bootstrap_dd_joined = dd.concat([trench_bootstrap_dd,trench_preinduction_bootstrap_dd],interleave_partitions=True,join='inner')
        trench_bootstrap_dd_joined = trench_bootstrap_dd_joined.reset_index()
        trench_bootstrap_dd_joined = trench_bootstrap_dd_joined.repartition(partition_size="1GB",force=False)
        to_parquet_checkpoint(dask_controller,trench_bootstrap_dd_joined,trench_bootstrap_path,engine='pyarrow',overwrite=overwrite)
        ### trench_bootstrap_dd_joined.to_parquet(trench_bootstrap_path_temp,engine="pyarrow",overwrite=overwrite)

        # trench_bootstrap_dd_joined = dd.read_parquet(trench_bootstrap_path_temp, engine="pyarrow",calculate_divisions=True)
        # trench_bootstrap_dd_joined = trench_bootstrap_dd_joined.reset_index()
        # trench_bootstrap_dd_joined = trench_bootstrap_dd_joined.repartition(partition_size="1GB")
        # to_parquet_checkpoint(dask_controller,trench_bootstrap_dd_joined,trench_bootstrap_path,engine='pyarrow',overwrite=overwrite)
        # # trench_bootstrap_dd_joined.to_parquet(trench_bootstrap_path,engine="pyarrow",overwrite=overwrite)    

        dask_controller.daskclient.cancel(trench_bootstrap_dd)
        dask_controller.daskclient.cancel(trench_preinduction_bootstrap_dd)
        dask_controller.daskclient.cancel(trench_bootstrap_dd_joined)
        
        # put this back when this runs successfully
        # shutil.rmtree(trench_bootstrap_preinduction_temp_path)
        # shutil.rmtree(trench_bootstrap_steady_state_temp_path)
        # shutil.rmtree(trench_bootstrap_path_temp)
        
        progress_state = 4
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
            
    if progress_state == 4:
        print(progress_state)
    
        ## part 5 ##
        trench_bootstrap_dd_joined = dd.read_parquet(trench_bootstrap_path, engine="pyarrow",calculate_divisions=True)
        trench_bootstrap_dd_joined = trench_bootstrap_dd_joined.set_index(variant_index,sorted=False)
        ## set this force to false (2/6/2023)
        # trench_bootstrap_dd_joined = trench_bootstrap_dd_joined.repartition(partition_size="500MB",force=False)
        variant_min = trench_bootstrap_dd_joined.index.min().compute()
        variant_max = trench_bootstrap_dd_joined.index.max().compute()
        trench_bootstrap_dd_joined = trench_bootstrap_dd_joined.repartition(divisions=list(range(int(variant_min),int(variant_max),n_variants_per_block)) + [int(variant_max)], force=False)
        to_parquet_checkpoint(dask_controller,trench_bootstrap_dd_joined,trench_bootstrap_path_temp,engine='pyarrow',overwrite=overwrite)
        
        dask_controller.daskclient.cancel(trench_bootstrap_dd_joined)
        shutil.rmtree(trench_bootstrap_path)
        os.rename(trench_bootstrap_path_temp,trench_bootstrap_path)
        
        progress_state = 5
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
        
    if progress_state == 5:
        print("Done!")
    
def get_trench_sample_df(trench_df,n_bootstraps_trench_density,n_bootstraps_trench_aggregate,param_groups_joined,variant_index="oDEPool7_id"):
    
    sampled_trench_df = trench_df.reset_index().set_index(["Trench Bootstrap Sample","Variable(s)"]).sort_index().reindex(labels=param_groups_joined,level="Variable(s)")
    sampled_trench_df = sampled_trench_df.reset_index().set_index("Trench Bootstrap Sample")
    bootstrap_samples_list = sorted(sampled_trench_df.index.get_level_values(0).unique()) #may be different than density in cases with NaNs, hacky
    
    sampled_trench_df = sampled_trench_df.loc[np.random.choice(bootstrap_samples_list,size=(n_bootstraps_trench_aggregate,))].reset_index()
    sampled_trench_df["Aggregate Trench Bootstrap Sample"] = np.repeat(range(n_bootstraps_trench_aggregate),len(param_groups_joined))
    sampled_trench_df = sampled_trench_df.set_index(["Aggregate Trench Bootstrap Sample","Variable(s)"])
    sampled_trench_df = sampled_trench_df.drop(columns=["Trench Bootstrap Sample","Bootstrap Index",variant_index,"Multi-Experiment Phenotype Trenchid"])
    
    return sampled_trench_df

def trench_aggregate_bootstrap(variant_df,param_groups_to_agg_list,estimator_names_to_agg=["Mean"],aggregators=[np.nanmedian],\
                               agg_names=["Mean"],is_paired_aggregator_list=[False],n_bootstraps_trench_aggregate=1000,\
                               trench_key="Multi-Experiment Phenotype Trenchid",variant_index="oDEPool7_id"):
    
    n_bootstraps_trench_density = variant_df["Trench Bootstrap Sample"].max()+1
    variant_df = variant_df.reset_index().set_index("Estimator").sort_index()
    variant_df = variant_df.dropna(subset=["Value"]) ##gets rid of NA inherited from earlier join
        
    aggregate_samples_df = []
    for i,estimator_name in enumerate(estimator_names_to_agg):
        
        param_groups = param_groups_to_agg_list[i]
        is_paired_aggregator = is_paired_aggregator_list[i]
        aggregator = aggregators[i]
        agg_name = agg_names[i]
        
        param_groups_joined = ["-".join(param_group) for param_group in param_groups]
        variant_estimator_df = variant_df.loc[estimator_name]
        
        if is_paired_aggregator:
            variant_estimator_df = variant_estimator_df.reset_index().set_index(trench_key).sort_index()
            pre_post_mask = variant_estimator_df.groupby(trench_key).apply(lambda x: sorted(x["Induction"].unique())==["Post","Pre"])
            trenchids_prepost = pre_post_mask[pre_post_mask].index.tolist()
            variant_estimator_df = variant_estimator_df.loc[trenchids_prepost]
            variant_estimator_df = variant_estimator_df.reset_index().set_index(["Induction",trench_key]).sort_index()
        
            if len(variant_estimator_df)>0:
                #sort again since later sgRNA sort messes up order
                variant_estimator_df_pre = variant_estimator_df.loc["Pre"]
                variant_estimator_df_post = variant_estimator_df.loc["Post"]

                trench_index_groupby_pre = variant_estimator_df_pre.groupby(trench_key)
                trench_index_groupby_post = variant_estimator_df_post.groupby(trench_key)
                sampled_variant_df_pre = trench_index_groupby_pre.apply(lambda x: get_trench_sample_df(x,n_bootstraps_trench_density,n_bootstraps_trench_aggregate,param_groups_joined,variant_index=variant_index))
                sampled_variant_df_post = trench_index_groupby_post.apply(lambda x: get_trench_sample_df(x,n_bootstraps_trench_density,n_bootstraps_trench_aggregate,param_groups_joined,variant_index=variant_index))

                pre_arr = sampled_variant_df_pre.to_xarray().reindex(indexers={"Variable(s)":param_groups_joined}).to_array().values[1].astype(float)
                post_arr = sampled_variant_df_post.to_xarray().reindex(indexers={"Variable(s)":param_groups_joined}).to_array().values[1].astype(float)
                
                aggregate_sample = aggregator(pre_arr,post_arr,axis=0)
                
                aggregate_sample_df = xr.DataArray(aggregate_sample,coords=[range(0,n_bootstraps_trench_aggregate),param_groups_joined],dims=["Bootstrap Sample","Variable(s)"],name="Value").to_dataframe()
                aggregate_sample_df["Estimator"] = agg_name
                aggregate_samples_df.append(aggregate_sample_df)

                aggregate_sample = aggregator(pre_arr,post_arr,axis=0)

        else:
            variant_estimator_df = variant_estimator_df.reset_index().set_index(["Induction",trench_key]).sort_index()
            if "Post" in variant_estimator_df.index.get_level_values(0).unique().tolist():
                variant_estimator_df_post = variant_estimator_df.loc["Post"]
                trench_index_groupby_post = variant_estimator_df_post.groupby(trench_key)
                sampled_variant_df_post = trench_index_groupby_post.apply(lambda x: get_trench_sample_df(x,n_bootstraps_trench_density,n_bootstraps_trench_aggregate,param_groups_joined,variant_index=variant_index))
                post_arr = sampled_variant_df_post.to_xarray().reindex(indexers={"Variable(s)":param_groups_joined}).to_array().values[1].astype(float)
                aggregate_sample = aggregator(post_arr,axis=0)

                aggregate_sample_df = xr.DataArray(aggregate_sample,coords=[range(0,n_bootstraps_trench_aggregate),param_groups_joined],dims=["Bootstrap Sample","Variable(s)"],name="Value").to_dataframe()
                aggregate_sample_df["Estimator"] = agg_name
                aggregate_samples_df.append(aggregate_sample_df)
    if len(aggregate_samples_df)>0:
        aggregate_samples_df = pd.concat(aggregate_samples_df).reset_index()
        aggregate_samples_df.index.name="Bootstrap Index"
        return aggregate_samples_df
    else:
        return None
    
def trench_aggregate_bootstrap_main(dask_controller,steady_state_df_path,estimator_names,bivariate_variable_list,estimator_names_to_agg,\
                                    unpaired_aggregators,paired_aggregators,agg_names,single_variable_list,pearsonr_variable_list,\
                                    n_bootstraps_trench_aggregate=1000,overwrite=False,variant_index="oDEPool7_id",\
                                    control_categories=['OnlyPlasmid', 'NoTarget']):

    trench_bootstrap_path = steady_state_df_path + "_Trench_Estimator_Bootstrap"
    aggregators = unpaired_aggregators + paired_aggregators
    is_paired_aggregator_list=[False for i in range(len(unpaired_aggregators))]+[True for i in range(len(paired_aggregators))]
    temp_variant_bootstrap_path = steady_state_df_path + "_Variant_Estimator_Bootstrap_temp"
    variant_bootstrap_path = steady_state_df_path + "_Variant_Estimator_Bootstrap"
    estimator_path = steady_state_df_path + "_Estimators.pkl"
    progress_path = steady_state_df_path + "_variant_bootstrap_checkpoints.pkl"
    
    wrapped_single_variable_list = [[variable] for variable in single_variable_list]
    wrapped_pearsonr_variable_list = [pearsonr_variable_list]
    bivariate_variable_list_to_agg = [bivariate_variable_list[estimator_names.index(estimator_name)] for estimator_name in estimator_names_to_agg]
    param_groups_to_agg_list=[wrapped_pearsonr_variable_list if bivariate_variable else wrapped_single_variable_list for bivariate_variable in bivariate_variable_list_to_agg]
    
    if overwrite:
        progress_state = 0
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
    else:
        if os.path.exists(progress_path):
            with open(progress_path, 'rb') as progfile:
                progress_state = pkl.load(progfile)
        else:
            progress_state = 0
            with open(progress_path, 'wb') as progfile:
                pkl.dump(progress_state, progfile)
                    
    ### part 1 ###
    
    if progress_state == 0:
        print(progress_state)
    
        trench_bootstrap_dd = dd.read_parquet(trench_bootstrap_path, engine="pyarrow",calculate_divisions=True)
        variant_bootstrap_input = trench_bootstrap_dd.get_partition(0).compute()
        variant_bootstrap_output = variant_bootstrap_input.groupby(variant_index).apply(lambda x: trench_aggregate_bootstrap(x,param_groups_to_agg_list,\
                                                        estimator_names_to_agg=estimator_names_to_agg,aggregators=aggregators,agg_names=agg_names,\
                                                        is_paired_aggregator_list=is_paired_aggregator_list,\
                                                        n_bootstraps_trench_aggregate=n_bootstraps_trench_aggregate,variant_index=variant_index))

        variant_bootstrap_dd = trench_bootstrap_dd.groupby(variant_index).apply(lambda x: trench_aggregate_bootstrap(x,param_groups_to_agg_list,\
                                                        estimator_names_to_agg=estimator_names_to_agg,aggregators=aggregators,agg_names=agg_names,\
                                                        is_paired_aggregator_list=is_paired_aggregator_list,\
                                                        n_bootstraps_trench_aggregate=n_bootstraps_trench_aggregate,variant_index=variant_index),\
                                                        meta=variant_bootstrap_output)
        variant_bootstrap_dd = variant_bootstrap_dd.reset_index()
        ## write output
        to_parquet_checkpoint(dask_controller,variant_bootstrap_dd,temp_variant_bootstrap_path,engine='pyarrow',overwrite=overwrite)
        dask_controller.daskclient.cancel(variant_bootstrap_dd)
        
        progress_state = 1
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
            
        overwrite = True
        
    ## part 2 ##
        
    if progress_state == 1:
        print(progress_state)

        variant_bootstrap_dd = dd.read_parquet(temp_variant_bootstrap_path, engine="pyarrow",calculate_divisions=True)

        variant_bootstrap_dd["Global Bootstrap Index"] = variant_bootstrap_dd.apply(lambda x: int(f'{int(x[variant_index]):08n}{int(x["Bootstrap Index"]):09n}'), axis=1, meta=("Global Bootstrap Index",int)).persist()
        wait(variant_bootstrap_dd);

        variant_bootstrap_dd = variant_bootstrap_dd.set_index("Global Bootstrap Index",sorted=False)
        # variant_bootstrap_dd.to_parquet(variant_bootstrap_path,engine="pyarrow",overwrite=overwrite)
        to_parquet_checkpoint(dask_controller,variant_bootstrap_dd,variant_bootstrap_path,engine='pyarrow',overwrite=overwrite)
        dask_controller.daskclient.cancel(variant_bootstrap_dd)
        
        progress_state = 2
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
            
        overwrite = True
        
    ## part 3 ##
        
    if progress_state == 2:
        print(progress_state)
        if ("Mean (Robust)" in agg_names) and ("Variance (Extrinsic)" in agg_names):
            ## Adding Extrinsic CV
            estimator_df_output = pd.read_pickle(estimator_path)
            ### note that CV (external) is normalized by wt median
            control_estimator_df_output = estimator_df_output[estimator_df_output["Category"].isin(control_categories)]
            control_medians = control_estimator_df_output.loc["Mean (Robust)"].groupby("Variable(s)")["Value"].median()
    
            variant_bootstrap_dd = dd.read_parquet(variant_bootstrap_path,calculate_divisions=True)
            CV_extrinsic_variant_dd = variant_bootstrap_dd[variant_bootstrap_dd["Estimator"]=="Variance (Extrinsic)"].persist()
            CV_extrinsic_variant_dd = cull_empty_partitions(CV_extrinsic_variant_dd).persist()
            wait(CV_extrinsic_variant_dd);
            CV_extrinsic_variant_dd = dd.map_partitions(get_extrinsic_CV,CV_extrinsic_variant_dd,control_medians,meta=CV_extrinsic_variant_dd,align_dataframes=False).persist()
            wait(CV_extrinsic_variant_dd);
    
            max_bootstrap_idx = variant_bootstrap_dd.groupby(variant_index)["Bootstrap Index"].max().compute().iloc[0]
            n_additional_indices = (n_bootstraps_trench_aggregate*len(single_variable_list))
            additional_indices = list(range(max_bootstrap_idx+1,max_bootstrap_idx+n_additional_indices+1))
            old_indices = sorted(list(CV_extrinsic_variant_dd["Bootstrap Index"].unique()))
            new_index_map = dict(zip(old_indices,additional_indices))
            CV_extrinsic_variant_dd["Bootstrap Index"] = CV_extrinsic_variant_dd["Bootstrap Index"].apply(lambda x: new_index_map[x],meta=(None,int)).persist()
            wait(CV_extrinsic_variant_dd);
            CV_extrinsic_variant_dd["New Global Bootstrap Index"] = CV_extrinsic_variant_dd.apply(lambda x: int(f'{int(x[variant_index]):08n}{int(x["Bootstrap Index"]):09n}'),\
                                                                                              axis=1, meta=("New Global Bootstrap Index",int)).persist()
            wait(CV_extrinsic_variant_dd);
            CV_extrinsic_variant_dd = CV_extrinsic_variant_dd.set_index("New Global Bootstrap Index",divisions=variant_bootstrap_dd.divisions).persist()
            wait(CV_extrinsic_variant_dd);
            CV_extrinsic_variant_dd.index = CV_extrinsic_variant_dd.index.rename("Global Bootstrap Index")
            variant_bootstrap_dd.divisions = variant_bootstrap_dd.clear_divisions().compute_current_divisions()
            CV_extrinsic_variant_dd.divisions = CV_extrinsic_variant_dd.clear_divisions().compute_current_divisions()
    
            # the merge drops the last variant somehow, play with this
            variant_bootstrap_dd = dd.concat([variant_bootstrap_dd,CV_extrinsic_variant_dd],interleave_partitions=True)
            variant_bootstrap_dd = variant_bootstrap_dd.reset_index().set_index("Global Bootstrap Index").persist()
            wait(variant_bootstrap_dd);
            variant_bootstrap_dd.to_parquet(temp_variant_bootstrap_path,engine="pyarrow",overwrite=overwrite)
            # to_parquet_checkpoint(dask_controller,variant_bootstrap_dd,temp_variant_bootstrap_path,engine='pyarrow',overwrite=overwrite)
            dask_controller.daskclient.cancel(variant_bootstrap_dd)
            shutil.rmtree(variant_bootstrap_path)
            os.rename(temp_variant_bootstrap_path,variant_bootstrap_path)
            
        progress_state = 3
        with open(progress_path, 'wb') as progfile:
            pkl.dump(progress_state, progfile)
            
    if progress_state == 3:
        print("Done!")

def bootstrap_null_model(control_bootstrap_df_path,estimator_param_name,aggregator,is_paired_aggregator,agg_name,m_trenches_sampled_list,n_bootstraps_from_kdes = 100000,\
                         n_bootstrap_per_chunk=5000,trench_key="Multi-Experiment Phenotype Trenchid",min_obs_per_trench=3):

    control_trench_kde_df = dd.read_parquet(control_bootstrap_df_path, engine="pyarrow",calculate_divisions=True)
    trench_kde_df = control_trench_kde_df.loc[estimator_param_name].compute(scheduler="threads").reset_index()

    if is_paired_aggregator:

        trench_kde_df = trench_kde_df.set_index(trench_key).sort_index()
        pre_post_mask = trench_kde_df.groupby(trench_key).apply(lambda x: sorted(x["Induction"].unique())==["Post","Pre"])
        trenchids_prepost = pre_post_mask[pre_post_mask].index.tolist()
        trench_kde_df = trench_kde_df.loc[trenchids_prepost]
        trench_kde_df = trench_kde_df.reset_index().set_index(["Induction",trench_key]).sort_index()

        trench_kde_df_pre = trench_kde_df.loc["Pre"]
        trench_kde_df_post = trench_kde_df.loc["Post"]

        trench_kde_pre_values_series = trench_kde_df_pre.groupby(trench_key).apply(lambda x: np.array(x["Value"]))
        trench_kde_post_values_series = trench_kde_df_post.groupby(trench_key).apply(lambda x: np.array(x["Value"]))
        
        trench_pre_fitted_kde_series = trench_kde_pre_values_series.apply(lambda x: sp.stats.gaussian_kde(x[~np.isnan(x)]) if \
                                        len(np.unique(x[~np.isnan(x)]))>=min_obs_per_trench else None)
        trench_post_fitted_kde_series = trench_kde_post_values_series.apply(lambda x: sp.stats.gaussian_kde(x[~np.isnan(x)]) if \
                                        len(np.unique(x[~np.isnan(x)]))>=min_obs_per_trench else None)
        
        notnone_mask = (trench_pre_fitted_kde_series!=None)&(trench_post_fitted_kde_series!=None)
        trench_pre_fitted_kde_series = trench_pre_fitted_kde_series[notnone_mask]
        trench_post_fitted_kde_series = trench_post_fitted_kde_series[notnone_mask]
        
        trench_pre_kde_list = trench_pre_fitted_kde_series.tolist()
        trench_post_kde_list = trench_post_fitted_kde_series.tolist()

        trench_pre_kde_list = [item for item in trench_pre_kde_list if item is not None]
        trench_post_kde_list = [item for item in trench_post_kde_list if item is not None]

    else:
        trench_kde_df = trench_kde_df.set_index(["Induction",trench_key]).sort_index()
        trench_kde_df_post = trench_kde_df.loc["Post"]
        trench_kde_post_values_series = trench_kde_df_post.groupby(trench_key).apply(lambda x: np.array(x["Value"]))
        trench_post_fitted_kde_series = trench_kde_post_values_series.apply(lambda x: sp.stats.gaussian_kde(x[~np.isnan(x)]) if \
                                        len(np.unique(x[~np.isnan(x)]))>=min_obs_per_trench else None)
        trench_post_kde_list = trench_post_fitted_kde_series.tolist()
        trench_post_kde_list = [item for item in trench_post_kde_list if item is not None]

    n_trenches = len(trench_post_kde_list)
    m_max = max(m_trenches_sampled_list)
    if n_trenches == 0:
        return None

    aggregated_vals_output = []
    n_chunks = (n_bootstraps_from_kdes//n_bootstrap_per_chunk)+1

    for working_bootstrap_idx in range(n_chunks):
        n_bootstraps_working_chunk_start = (working_bootstrap_idx*n_bootstrap_per_chunk)
        if (n_bootstraps_from_kdes-n_bootstraps_working_chunk_start)//n_bootstrap_per_chunk == 0:
            n_bootstraps_working_chunk = n_bootstraps_from_kdes-n_bootstraps_working_chunk_start
        else:
            n_bootstraps_working_chunk = n_bootstrap_per_chunk

        multinomial_rv = sp.stats.multinomial(n_bootstraps_working_chunk*m_max,np.repeat(1/n_trenches,n_trenches))
        multinomial_roll = multinomial_rv.rvs(size=1)[0]
        post_resample = []
        pre_resample = []
        for i,post_kde in enumerate(trench_post_kde_list):
            sampled_post_values = post_kde.resample(multinomial_roll[i]) #d x N
            post_resample.append(sampled_post_values)

            if is_paired_aggregator:
                pre_kde = trench_pre_kde_list[i]
                sampled_pre_values = pre_kde.resample(multinomial_roll[i]) #d x N
                pre_resample.append(sampled_pre_values)

        post_resample = np.concatenate(post_resample,axis=1).T #N x d
        np.random.shuffle(post_resample)
        post_resample = post_resample.reshape((m_max,n_bootstraps_working_chunk))        
        
        if is_paired_aggregator:
            pre_resample = np.concatenate(pre_resample,axis=1).T #N x d
            np.random.shuffle(pre_resample)
            pre_resample = pre_resample.reshape((m_max,n_bootstraps_working_chunk))
            
        aggregated_vals_over_m = []
        for m in m_trenches_sampled_list:
            if is_paired_aggregator:
                aggregated_vals = aggregator(pre_resample[:m,:,None],post_resample[:m,:,None],axis=0)[:,0]
            else:
                aggregated_vals = aggregator(post_resample[:m,:,None],axis=0)[:,0]
            aggregated_vals_over_m.append(aggregated_vals)
        aggregated_vals_over_m = np.array(aggregated_vals_over_m).T
        aggregated_vals_output.append(aggregated_vals_over_m)
    aggregated_vals_output = np.concatenate(aggregated_vals_output,axis=0) #bootstraps x m
    aggregated_vals_output = xr.DataArray(aggregated_vals_output,dims=["Bootstrap #","N Trenches Bootstrapped"],\
                                          coords=[range(n_bootstraps_from_kdes),m_trenches_sampled_list])
    aggregated_vals_output = aggregated_vals_output.to_dataframe("Aggregate Values")
    aggregated_vals_output["Estimator"] = agg_name
    aggregated_vals_output["Variable(s)"] = trench_kde_df.iloc[0]["Variable(s)"]
    aggregated_vals_output = aggregated_vals_output.reset_index()

    return aggregated_vals_output

def trench_bootstrap_null_model_main(dask_controller,steady_state_df_path,preinduction_df_path,bootstrap_density_function,estimators,estimator_names,bivariate_variable_list,\
                              single_variable_list,pearsonr_variable_list,filter_proliferating=True,n_bootstraps_trench_density = 200,sample_minimum = 3,\
                            variant_index="oDEPool7_id",control_categories=['OnlyPlasmid', 'NoTarget'],**density_kwargs):
    
    wrapped_single_variable_list = [[variable] for variable in single_variable_list]
    wrapped_pearsonr_variable_list = [pearsonr_variable_list]    
    param_groups_list = [wrapped_pearsonr_variable_list if bivariate_variable else wrapped_single_variable_list for bivariate_variable in bivariate_variable_list]
    
    repartitioned_steady_state_df_path = steady_state_df_path + "_temp_repartitioned"
    repartitioned_preinduction_df_path = preinduction_df_path + "_temp_repartitioned"

    control_trench_steady_state_bootstrap_path = steady_state_df_path + "_Steady_State_Control_Bootstrap"
    control_trench_preinduction_df_path_bootstrap_path = steady_state_df_path + "_Preinduction_Control_Bootstrap"
    control_trench_bootstrap_path = steady_state_df_path + "_Control_Bootstrap"
    control_trench_bootstrap_path_temp = steady_state_df_path + "_Control_Bootstrap_temp"
        
    ##Filter and repartition
    
    final_df_ss = dd.read_parquet(steady_state_df_path, engine="pyarrow",calculate_divisions=True)
    final_df_preinduction = dd.read_parquet(preinduction_df_path, engine="pyarrow",calculate_divisions=True)
    
    if filter_proliferating:
        final_df_ss = final_df_ss[final_df_ss["Proliferating"]]
        final_df_preinduction = final_df_preinduction[final_df_preinduction["Proliferating"]]
        
    final_df_ss = final_df_ss.reset_index().set_index("Multi-Experiment Global CellID",sorted=True).repartition(partition_size="10MB").reset_index().set_index("Multi-Experiment Phenotype Trenchid",sorted=True)
    final_df_ss.to_parquet(repartitioned_steady_state_df_path,engine="pyarrow",overwrite=True)
    dask_controller.daskclient.cancel(final_df_ss)

    final_df_preinduction = final_df_preinduction.reset_index().set_index("Multi-Experiment Global CellID",sorted=True).repartition(partition_size="10MB").reset_index().set_index("Multi-Experiment Phenotype Trenchid",sorted=True)
    final_df_preinduction.to_parquet(repartitioned_preinduction_df_path,engine="pyarrow",overwrite=True)
    dask_controller.daskclient.cancel(final_df_preinduction)
        
    ## next section
    final_cell_cycle_df_ss_repartitioned = dd.read_parquet(repartitioned_steady_state_df_path, engine="pyarrow",calculate_divisions=True)
    final_cell_cycle_df_ss_controls = final_cell_cycle_df_ss_repartitioned[final_cell_cycle_df_ss_repartitioned['Category'].isin(control_categories)].persist()
    final_cell_cycle_df_ss_controls = final_cell_cycle_df_ss_controls.repartition(partition_size="1MB").reset_index().set_index("Multi-Experiment Phenotype Trenchid",sorted=True).persist()
    wait(final_cell_cycle_df_ss_controls);
    first_idx = final_cell_cycle_df_ss_controls.index.divisions[0]
    input_partition = final_cell_cycle_df_ss_controls.loc[first_idx:first_idx].compute()
    output_partition = input_partition.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x:\
                    all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,return_kde=False,\
                    estimators=estimators,estimator_names=estimator_names,\
                    n_bootstraps_trench_density=n_bootstraps_trench_density,variant_index=variant_index,**density_kwargs))
    control_trench_kde_df = final_cell_cycle_df_ss_controls.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x:\
                            all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,\
                            return_kde=False,estimators=estimators,estimator_names=estimator_names,\
                            n_bootstraps_trench_density=n_bootstraps_trench_density,variant_index=variant_index,\
                            **density_kwargs),meta=output_partition).persist()
    control_trench_kde_df["Estimator-Variable Index"] = control_trench_kde_df.apply(lambda x: "-".join([x["Estimator"],x["Variable(s)"]]), axis=1, meta=str).persist()
    wait(control_trench_kde_df);
    estimator_variable_list = sorted(list(control_trench_kde_df["Estimator-Variable Index"].unique().compute()))
    control_trench_kde_df = control_trench_kde_df.reset_index().set_index("Estimator-Variable Index",sorted=False,divisions=estimator_variable_list+[estimator_variable_list[-1]]).persist()
    wait(control_trench_kde_df);
    control_trench_kde_df.to_parquet(control_trench_steady_state_bootstrap_path,engine="pyarrow",overwrite=True)
    dask_controller.daskclient.cancel(control_trench_kde_df)
    
    final_cell_cycle_df_preinduction_repartitioned = dd.read_parquet(repartitioned_preinduction_df_path, engine="pyarrow",calculate_divisions=True)
    final_cell_cycle_df_preinduction_controls = final_cell_cycle_df_preinduction_repartitioned[final_cell_cycle_df_preinduction_repartitioned['Category'].isin(control_categories)].persist()
    final_cell_cycle_df_preinduction_controls = final_cell_cycle_df_preinduction_controls.repartition(partition_size="1MB").reset_index().set_index("Multi-Experiment Phenotype Trenchid",sorted=True).persist()
    wait(final_cell_cycle_df_preinduction_controls);
    first_idx = final_cell_cycle_df_preinduction_controls.index.divisions[0]
    input_partition = final_cell_cycle_df_preinduction_controls.loc[first_idx:first_idx].compute()
    output_partition = input_partition.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x: all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,return_kde=False,\
                                                                                                estimators=estimators,estimator_names=estimator_names,\
                                                                                                n_bootstraps_trench_density=n_bootstraps_trench_density,variant_index=variant_index,**density_kwargs))
    control_trench_preinduction_kde_df = final_cell_cycle_df_preinduction_controls.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x: all_trench_bootstrap(x,bootstrap_density_function,param_groups_list,\
                                                                                                return_kde=False,estimators=estimators,estimator_names=estimator_names,\
                                                                                                n_bootstraps_trench_density=n_bootstraps_trench_density,variant_index=variant_index,**density_kwargs),\
                                                                                                meta=output_partition).persist()
    control_trench_preinduction_kde_df["Estimator-Variable Index"] = control_trench_preinduction_kde_df.apply(lambda x: "-".join([x["Estimator"],x["Variable(s)"]]), axis=1, meta=str).persist()
    wait(control_trench_preinduction_kde_df);
    control_trench_preinduction_kde_df = control_trench_preinduction_kde_df.reset_index().set_index("Estimator-Variable Index",sorted=False,divisions=estimator_variable_list+[estimator_variable_list[-1]]).persist()
    wait(control_trench_preinduction_kde_df);
    control_trench_preinduction_kde_df.to_parquet(control_trench_preinduction_df_path_bootstrap_path,engine="pyarrow",overwrite=True)
    dask_controller.daskclient.cancel(control_trench_preinduction_kde_df)
    
    control_trench_bootstrap_dd = dd.read_parquet(control_trench_steady_state_bootstrap_path, engine="pyarrow",calculate_divisions=True)
    control_trench_preinduction_bootstrap_dd = dd.read_parquet(control_trench_preinduction_df_path_bootstrap_path, engine="pyarrow",calculate_divisions=True)

    control_trench_bootstrap_dd["Induction"] = "Post"
    control_trench_preinduction_bootstrap_dd["Induction"] = "Pre"

    control_trench_bootstrap_dd_joined = dd.concat([control_trench_bootstrap_dd,control_trench_preinduction_bootstrap_dd],interleave_partitions=True,join='inner')
    
    #hack to fix partition issue
    control_trench_bootstrap_dd_joined = control_trench_bootstrap_dd_joined.repartition(divisions=estimator_variable_list+[estimator_variable_list[-1]])
    control_trench_bootstrap_dd_joined.to_parquet(control_trench_bootstrap_path,engine="pyarrow",overwrite=True)

    dask_controller.daskclient.cancel(control_trench_bootstrap_dd)
    dask_controller.daskclient.cancel(control_trench_preinduction_bootstrap_dd)
    dask_controller.daskclient.cancel(control_trench_bootstrap_dd_joined)

    shutil.rmtree(control_trench_steady_state_bootstrap_path)
    shutil.rmtree(control_trench_preinduction_df_path_bootstrap_path)
    
    #hack: filter to remove trenches with NaNs
    control_trench_kde_df = dd.read_parquet(control_trench_bootstrap_path, engine="pyarrow",calculate_divisions=True)
    
    trench_num_nans = control_trench_kde_df.groupby("Multi-Experiment Phenotype Trenchid")["Value"].apply(lambda x: np.sum(np.isnan(x))).compute()
    trenchids_to_drop = list(trench_num_nans[trench_num_nans>0].index)
    control_trench_kde_df = control_trench_kde_df[~control_trench_kde_df["Multi-Experiment Phenotype Trenchid"].isin(trenchids_to_drop)]
    
    control_trench_kde_df.to_parquet(control_trench_bootstrap_path_temp,engine="pyarrow",overwrite=True)
    dask_controller.daskclient.cancel(control_trench_kde_df)
    shutil.rmtree(control_trench_bootstrap_path)
    os.rename(control_trench_bootstrap_path_temp,control_trench_bootstrap_path)
    
        
def trench_aggregate_bootstrap_null_model_main(dask_controller,steady_state_df_path,estimator_names,bivariate_variable_list,estimator_names_to_agg,unpaired_aggregators,paired_aggregators,\
                           agg_names,single_variable_list,pearsonr_variable_list,n_bootstraps_trench_aggregate=1000,sample_minimum=3,variant_index="oDEPool7_id",control_categories=['OnlyPlasmid', 'NoTarget']):
    
    control_trench_bootstrap_path = steady_state_df_path + "_Control_Bootstrap"
    control_variant_bootstrap_path_temp = steady_state_df_path + "_Control_Variant_Bootstrap_Temp"
    control_variant_bootstrap_path = steady_state_df_path + "_Control_Variant_Bootstrap"
    estimator_path = steady_state_df_path + "_Estimators.pkl"
    
    aggregators = unpaired_aggregators + paired_aggregators
    is_paired_aggregator_list=[False for i in range(len(unpaired_aggregators))]+[True for i in range(len(paired_aggregators))]
    
    wrapped_single_variable_list = [[variable] for variable in single_variable_list]
    wrapped_pearsonr_variable_list = [pearsonr_variable_list]
    bivariate_variable_list_to_agg = [bivariate_variable_list[estimator_names.index(estimator_name)] for estimator_name in estimator_names_to_agg]
    param_groups_to_agg_list=[wrapped_pearsonr_variable_list if bivariate_variable else wrapped_single_variable_list for bivariate_variable in bivariate_variable_list_to_agg]

    final_cell_cycle_df_ss = dd.read_parquet(steady_state_df_path, engine="pyarrow",calculate_divisions=True)
    control_trench_kde_df = dd.read_parquet(control_trench_bootstrap_path, engine="pyarrow",calculate_divisions=True)
    final_cell_cycle_df_ss_one_trench = final_cell_cycle_df_ss.groupby("Multi-Experiment Phenotype Trenchid").apply(lambda x: x.iloc[0]).compute()
    num_each_variant = final_cell_cycle_df_ss_one_trench.groupby(variant_index).size()
    num_each_variant = num_each_variant[num_each_variant>=sample_minimum]
    m_trenches_sampled_list = sorted(list(num_each_variant.unique()))

    variant_bootstrap_delayed_list = []
    for agg_idx,estimator_name in enumerate(estimator_names_to_agg):
        estimator_param_names = ['-'.join([estimator_name]+param_group) for param_group in param_groups_to_agg_list[agg_idx]]
        aggregator = aggregators[agg_idx]
        agg_name = agg_names[agg_idx]
        is_paired_aggregator = is_paired_aggregator_list[agg_idx]
        for param_idx,estimator_param_name in enumerate(estimator_param_names):
            variant_bootstrap_delayed = dask.delayed(bootstrap_null_model)(control_trench_bootstrap_path,estimator_param_name,aggregator,is_paired_aggregator,agg_name,m_trenches_sampled_list,n_bootstraps_trench_aggregate)
            variant_bootstrap_delayed_list.append(variant_bootstrap_delayed)

    test_output = bootstrap_null_model(control_trench_bootstrap_path,estimator_param_name,aggregator,is_paired_aggregator,agg_name,[m_trenches_sampled_list[10]],100)

    variant_control_bootstrap_output = dd.from_delayed(variant_bootstrap_delayed_list,meta=test_output)
    variant_control_bootstrap_output.to_parquet(control_variant_bootstrap_path,engine="pyarrow",overwrite=True)
    dask_controller.daskclient.cancel(variant_control_bootstrap_output)
    
    control_variant_bootstrap_df = dd.read_parquet(control_variant_bootstrap_path, engine="pyarrow",calculate_divisions=True).persist()
    wait(control_variant_bootstrap_df);
    control_variant_bootstrap_df = control_variant_bootstrap_df.repartition(partition_size="500MB").persist()
    wait(control_variant_bootstrap_df);
    
    if ("Mean (Robust)" in agg_names) and ("Variance (Extrinsic)" in agg_names):
        #### note that CV (external) is normalized by wt median
        estimator_df_output = pd.read_pickle(estimator_path)
        control_estimator_df_output = estimator_df_output[estimator_df_output["Category"].isin(control_categories)]
        control_medians = control_estimator_df_output.loc["Mean (Robust)"].groupby("Variable(s)")["Value"].median()
        
        CV_extrinsic_df = control_variant_bootstrap_df[control_variant_bootstrap_df["Estimator"]=="Variance (Extrinsic)"].persist()
        CV_extrinsic_df = cull_empty_partitions(CV_extrinsic_df).persist()
        wait(CV_extrinsic_df);
        CV_extrinsic_df = dd.map_partitions(get_extrinsic_CV,CV_extrinsic_df,control_medians,meta=CV_extrinsic_df,align_dataframes=False,value_col_name="Aggregate Values").persist()
        wait(CV_extrinsic_df);
        control_variant_bootstrap_df = dd.concat([control_variant_bootstrap_df,CV_extrinsic_df]).persist()
        wait(control_variant_bootstrap_df);
        
    control_variant_bootstrap_df["Estimator-Variable(s)-NTrenches"] = control_variant_bootstrap_df.apply(lambda x: \
                            "-".join([x["Estimator"],x["Variable(s)"],str(x["N Trenches Bootstrapped"])]), axis=1, meta=str).persist()
    wait(control_variant_bootstrap_df);
    control_variant_bootstrap_df = control_variant_bootstrap_df.set_index("Estimator-Variable(s)-NTrenches").persist()
    wait(control_variant_bootstrap_df);

    control_variant_bootstrap_df.to_parquet(control_variant_bootstrap_path_temp,engine="pyarrow",overwrite=True)
    dask_controller.daskclient.cancel(control_variant_bootstrap_df)
    if ("Mean (Robust)" in agg_names) and ("Variance (Extrinsic)" in agg_names):
        dask_controller.daskclient.cancel(CV_extrinsic_df)
    shutil.rmtree(control_variant_bootstrap_path)
    os.rename(control_variant_bootstrap_path_temp,control_variant_bootstrap_path)

def compute_estimator_statistic(variant_bootstrap_df,statistic_name="Variance",statistic=np.nanvar,grouping_indices=["Estimator","Variable(s)"]):
    statistic_bootstrap_df = variant_bootstrap_df.set_index(grouping_indices).sort_index()
    statistic_df = statistic_bootstrap_df.groupby(grouping_indices).apply(lambda x: statistic(x["Value"])).reset_index()
    statistic_df = statistic_df.rename(columns={0:statistic_name})
    statistic_df.index.name="Statistic Index"
    return statistic_df

def percentile_ci(values,ci_alpha=0.05,aggregate=np.nanmedian):
    sorted_values = np.sort(values)
    n_values = len(values)
    if n_values > 2:
        ci_low = sorted_values[int(np.floor(n_values*ci_alpha))]
        ci_high = sorted_values[int(np.ceil(n_values*(1.-ci_alpha)))]
        ci_width = ci_high-ci_low
    else:
        ci_width = np.NaN
    return ci_width

def get_bootstrap_p(x):
    if len(x)>0:
        bootstrap_val_arr = np.array(eval(x["Aggregate Value List"]))
        estimate_dict = eval(x["Estimate Dict"])
        estimator_arr = np.array([[key,val] for key,val in estimate_dict.items()])
        keys_out = list(estimator_arr[:,0].astype(int))
        value_arr = estimator_arr[:,1]
        
        upper_pval = (np.sum(bootstrap_val_arr[:,None]<value_arr[None,:],axis=0)+1)/(bootstrap_val_arr.shape[0]+1)
        lower_pval = (np.sum(bootstrap_val_arr[:,None]>value_arr[None,:],axis=0)+1)/(bootstrap_val_arr.shape[0]+1)
        pval = np.min(np.stack([upper_pval,lower_pval]),axis=0)*2
        pval_dict = {key:pval[i] for i,key in enumerate(keys_out)}
    else:
        pval_dict = {}

    return pval_dict

def get_final_bootstrap_output(steady_state_df_path,include_ccf=True,include_pvals=True,filter_proliferating=True,variant_index="oDEPool7_id"):

    variant_bootstrap_path = steady_state_df_path + "_Variant_Estimator_Bootstrap"
    estimator_path = steady_state_df_path + "_Estimators.pkl"
    control_variant_bootstrap_path = steady_state_df_path + "_Control_Variant_Bootstrap"
    estimator_stats_path = steady_state_df_path + "_Estimators_wStats.pkl"
    
    if include_ccf:
        ccf_bootstrap_path = steady_state_df_path + "_CCF_Estimator_Bootstrap"
        ccf_estimator_path = steady_state_df_path + "_CCF_Estimators.pkl"
        ccf_estimator_stats_path = steady_state_df_path + "_CCF_Estimators_wStats.pkl"

    variant_bootstrap_dd = dd.read_parquet(variant_bootstrap_path, engine="pyarrow",calculate_divisions=True)
    estimator_df_output = pd.read_pickle(estimator_path)
    
    if include_ccf:
        ccf_variant_bootstrap_dd = dd.read_parquet(ccf_bootstrap_path, engine="pyarrow",calculate_divisions=True)
        ccf_estimator_df_output = pd.read_pickle(ccf_estimator_path)
    
    ## Computing the estimator variance
    variant_bootstrap_dd = variant_bootstrap_dd.set_index(variant_index,sorted=True)
    estimator_variance_df = variant_bootstrap_dd.groupby(variant_index).apply(lambda x: compute_estimator_statistic(x,statistic_name="Estimator Variance",statistic=np.nanvar),\
                                                                         meta=pd.DataFrame(data=[],columns=["Estimator","Variable(s)","Estimator Variance"]).astype(\
                                                                             {"Estimator":str,"Variable(s)":str,"Estimator Variance":float})).compute()
    estimator_variance_df = estimator_variance_df.reset_index().set_index(["Estimator","Variable(s)",variant_index]).drop(columns=["Statistic Index"]).sort_index()
    
    if include_ccf:
        ccf_estimator_variance_df = ccf_variant_bootstrap_dd.groupby(variant_index).apply(lambda x: compute_estimator_statistic(x,statistic_name="Estimator Variance",\
                                                                        statistic=np.nanvar,grouping_indices=["Estimator","Variable(s)","Lag"]),\
                                                                meta=pd.DataFrame(data=[],columns=["Estimator","Variable(s)","Lag","Estimator Variance"]).astype(\
                                                                             {"Estimator":str,"Variable(s)":str,"Lag":int,"Estimator Variance":float})).compute()
        ccf_estimator_variance_df = ccf_estimator_variance_df.reset_index().set_index(["Estimator","Variable(s)","Lag",variant_index]).drop(columns=["Statistic Index"]).sort_index()

    ## Computing the estimator 90% confidence interval width
    estimator_ci_df = variant_bootstrap_dd.groupby(variant_index).apply(lambda x: compute_estimator_statistic(x,statistic_name="CI Width",statistic=percentile_ci),\
                                                                       meta=pd.DataFrame(data=[],columns=["Estimator","Variable(s)","CI Width"]).astype(\
                                                                             {"Estimator":str,"Variable(s)":str,"CI Width":float})).compute()
    estimator_ci_df = estimator_ci_df.reset_index().set_index(["Estimator","Variable(s)",variant_index]).drop(columns=["Statistic Index"]).sort_index()
    
    if include_ccf:
        ccf_estimator_ci_df = ccf_variant_bootstrap_dd.groupby(variant_index).apply(lambda x: compute_estimator_statistic(x,statistic_name="CI Width",\
                                                                    statistic=percentile_ci,grouping_indices=["Estimator","Variable(s)","Lag"]),\
                                                                    meta=pd.DataFrame(data=[],columns=["Estimator","Variable(s)","Lag","CI Width"]).astype(\
                                                                             {"Estimator":str,"Variable(s)":str,"Lag":int,"CI Width":float})).compute()
        ccf_estimator_ci_df = ccf_estimator_ci_df.reset_index().set_index(["Estimator","Variable(s)","Lag",variant_index]).drop(columns=["Statistic Index"]).sort_index()

    estimator_df_output = pd.read_pickle(estimator_path)

    if include_pvals:
        control_variant_bootstrap_dd = dd.read_parquet(control_variant_bootstrap_path, engine="pyarrow",calculate_divisions=True)
        if filter_proliferating:        
            estimator_df_output_new_idx = estimator_df_output.reset_index().apply(lambda x: "-".join([x["Estimator"],x["Variable(s)"],str(int(x["N Observations Proliferating"]))]), axis=1)
        else:
            estimator_df_output_new_idx = estimator_df_output.reset_index().apply(lambda x: "-".join([x["Estimator"],x["Variable(s)"],str(int(x["N Observations"]))]), axis=1)
        estimator_df_output_new_idx.index = estimator_df_output.index
        estimator_df_output["Estimator-Variable(s)-NTrenches"] = estimator_df_output_new_idx
    
        estimator_df_output_reindexed = estimator_df_output.reset_index().set_index("Estimator-Variable(s)-NTrenches").sort_index()
        estimator_df_output_reindexed = estimator_df_output_reindexed.groupby("Estimator-Variable(s)-NTrenches").apply(lambda x: x[[variant_index,"Value"]].set_index(variant_index).to_dict()["Value"])
    
        control_variant_bootstrap_pval_calc_df = control_variant_bootstrap_dd.groupby("Estimator-Variable(s)-NTrenches").apply(lambda x: x["Aggregate Values"].tolist()).persist()
        wait(control_variant_bootstrap_pval_calc_df);
        control_variant_bootstrap_pval_calc_df = control_variant_bootstrap_pval_calc_df.to_frame().rename(columns={0:"Aggregate Value List"})
        control_variant_bootstrap_pval_calc_df = control_variant_bootstrap_pval_calc_df.join(estimator_df_output_reindexed,how='inner').rename(columns={0:"Estimate Dict"}).persist()
        wait(control_variant_bootstrap_pval_calc_df);
    
        control_variant_bootstrap_pval_df = pd.DataFrame(control_variant_bootstrap_pval_calc_df.apply(lambda x: get_bootstrap_p(x), axis=1, meta=(None,float)).compute()).reset_index()
        control_variant_bootstrap_pval_df["Estimator-Variable(s)"] = control_variant_bootstrap_pval_df.apply(lambda x: "-".join(x["Estimator-Variable(s)-NTrenches"].split("-")[:-1]), axis=1)
        final_pval_df = control_variant_bootstrap_pval_df.groupby("Estimator-Variable(s)").apply(lambda x: {key:val for dict_entry in x[0].tolist() for key,val in dict_entry.items()})
        final_pval_df = final_pval_df.apply(lambda x: pd.Series(x)).reset_index().melt(id_vars=['Estimator-Variable(s)']).rename(columns={"variable":variant_index,"value":"P-Value"})
        final_pval_df = final_pval_df.set_index(["Estimator-Variable(s)",variant_index]).sort_index()
    
        ## Correction
        corrected_pvals = final_pval_df.groupby("Estimator-Variable(s)").apply(lambda x: statsmodels.stats.multitest.fdrcorrection(x["P-Value"])[1]).explode()
        corrected_pvals = corrected_pvals.astype(float)
        corrected_pvals.index = final_pval_df.index
        final_pval_df["Corrected P-Value"] = corrected_pvals
        final_pval_df = final_pval_df.reset_index()
        final_pval_df["Estimator"] = final_pval_df.apply(lambda x: x["Estimator-Variable(s)"].split("-")[0], axis=1)
        final_pval_df["Variable(s)"] = final_pval_df.apply(lambda x: "-".join(x["Estimator-Variable(s)"].split("-")[1:]), axis=1)
        final_pval_df = final_pval_df.set_index(["Estimator","Variable(s)",variant_index]).sort_index().drop(columns=["Estimator-Variable(s)"])
        
        estimator_df_output = pd.concat([estimator_df_output,final_pval_df,estimator_variance_df,estimator_ci_df],axis=1,join="inner")
        
    else:

        estimator_df_output = pd.concat([estimator_df_output,estimator_variance_df,estimator_ci_df],axis=1,join="inner")
        
    estimator_df_output.to_pickle(estimator_stats_path)
    if include_ccf:
        ccf_estimator_df_output = pd.read_pickle(ccf_estimator_path)
        ccf_estimator_df_output = pd.concat([ccf_estimator_df_output,ccf_estimator_variance_df,ccf_estimator_ci_df],axis=1,join="inner")
        ccf_estimator_df_output.to_pickle(ccf_estimator_stats_path)
