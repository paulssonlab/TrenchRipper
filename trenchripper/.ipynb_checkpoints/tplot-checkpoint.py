## fmt: off
import numpy as np
from matplotlib import pyplot as plt
import skimage as sk

def plot_kymograph(kymograph,cmap="Greys_r",vmin=None,vmax=None):
    """Helper function for plotting kymographs. Takes a kymograph array of
    shape (t_dim,y_dim,x_dim).

    Args:
        kymograph (array): kymograph array of shape (y_dim,x_dim,t_dim).
    """
    list_in_t = [kymograph[t,:,:] for t in range(kymograph.shape[0])]
    img_arr = np.concatenate(list_in_t,axis=1)
    plt.imshow(img_arr,cmap=cmap,vmin=vmin,vmax=vmax)

def get_magenta_green_overlay(green_arr,magenta_arr,green_arr_min,green_arr_max,magenta_arr_min,\
                              magenta_arr_max,green_weight=1.,magenta_weight=1.):
    
    # Expand dimensions to make them 3D arrays
    image1 = np.expand_dims(green_arr, axis=-1)
    image2 = np.expand_dims(magenta_arr, axis=-1)
    
    # Make image1 green
    img_green = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint16)
    img_green[:, :, 1] = image1[:,:,0]
    
    # Make image2 magenta
    img_magenta = np.zeros((image2.shape[0], image2.shape[1], 3), dtype=np.uint16)
    img_magenta[:, :, 0] = image2[:,:,0]
    img_magenta[:, :, 2] = image2[:,:,0]
    
    # # Apply contrast to the green channel
    img_green[:, :, 1] = sk.exposure.rescale_intensity(np.clip(img_green[:, :, 1],green_arr_min,green_arr_max)).astype('uint16')
    
    # # Apply contrast to the magenta channel
    img_magenta[:, :, 0] = sk.exposure.rescale_intensity(np.clip(img_magenta[:, :, 0],magenta_arr_min,magenta_arr_max)).astype('uint16')
    img_magenta[:, :, 2] = sk.exposure.rescale_intensity(np.clip(img_magenta[:, :, 2],magenta_arr_min,magenta_arr_max)).astype('uint16')
    
    merged_image = ((img_green*green_weight)+(img_magenta*magenta_weight)).astype('uint16')
    return merged_image


def plot_kymograph_2channel(channel_1_kymo,channel_2_kymo,vmin_channel_1,vmax_channel_1,vmin_channel_2,vmax_channel_2,channel_1_weight=1,channel_2_weight=1,seg_kymo=None,seg_sigma=3,seg_thickness=0.75):
    """Helper function for plotting 2 channel kymographs. Takes a kymograph array of
    shape (channel,t_dim,y_dim,x_dim).

    Args:
        kymograph (array): kymograph array of shape (channel,t_dim,y_dim,x_dim).
    """
    
    list_in_t = [channel_1_kymo[t,:,:] for t in range(channel_1_kymo.shape[0])]
    channel_1_arr = np.concatenate(list_in_t,axis=1)
    list_in_t = [channel_2_kymo[t,:,:] for t in range(channel_2_kymo.shape[0])]
    channel_2_arr = np.concatenate(list_in_t,axis=1)

    merged_image = get_magenta_green_overlay(channel_1_arr,channel_2_arr,vmin_channel_1,vmax_channel_1,\
                                       vmin_channel_2,vmax_channel_2,green_weight=channel_1_weight,\
                                        magenta_weight=channel_2_weight)
    merged_image = sk.util.img_as_ubyte(merged_image)
    plt.imshow(merged_image)

    if seg_kymo is not None:
        seg_output = []
        for t in range(seg_kymo.shape[0]):
            working_seg_kymo = seg_kymo[t]
            working_seg_kymo[working_seg_kymo>0] = working_seg_kymo[working_seg_kymo>0]+(100*t)
            seg_output.append(working_seg_kymo)
            
        seg_output = np.concatenate(seg_output,axis=1)
        seg_output, _, _ = sk.segmentation.relabel_sequential(seg_output)
        
        seg_borders_list = []

        for i in sorted(list(set(np.unique(seg_output))-set([0]))):
            contours = sk.measure.find_contours(sk.filters.gaussian(seg_output == i,sigma=seg_sigma), level=0.5)
            if len(contours)>0:
                seg_borders = contours[0]            
                seg_borders_list.append(seg_borders)
            
        for seg_border in seg_borders_list:
            plt.plot(seg_border[:,1],seg_border[:,0],color="white",linestyle="--",linewidth=seg_thickness)
        