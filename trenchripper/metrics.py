# fmt: off
import copy

import skimage.measure

import numpy as np
import skimage as sk
import scipy as sp

def object_f_scores(true_labels,pred_labels,eps = (10**-5)):
    any_true,any_pred = (np.any(true_labels>0),np.any(pred_labels>0))
    if any_true and any_pred:
        true_label_projection = np.array([true_labels==i for i in range(1,np.max(true_labels)+1)])
        pred_label_projection = np.array([pred_labels==i for i in range(1,np.max(pred_labels)+1)])
        true_label_flatten = true_label_projection.reshape(true_label_projection.shape[0],-1).astype(int)
        pred_label_flatten = pred_label_projection.reshape(pred_label_projection.shape[0],-1).astype(int)

        match_arr = true_label_flatten@pred_label_flatten.T

        matched_pred_label_projection = pred_label_projection[np.argmax(match_arr,axis=1)]
        matched_pred_label_reshape = matched_pred_label_projection.reshape(matched_pred_label_projection.shape[0],-1)
        true_label_flatten = true_label_flatten.astype(bool)

        TP = np.sum(true_label_flatten&matched_pred_label_reshape,axis=1) + eps
        FP = np.sum((~true_label_flatten)&matched_pred_label_reshape,axis=1) + eps
        FN = np.sum(true_label_flatten&(~matched_pred_label_reshape),axis=1) + eps

        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        f_score = 2*((Precision*Recall)/(Precision+Recall))

        return Precision,Recall,f_score
    elif any_true:
        num_cells = np.max(true_labels)
        Precision = np.array([0. for i in range(num_cells)])
        Recall = np.array([0. for i in range(num_cells)])
        f_score = np.array([0. for i in range(num_cells)])

        return Precision,Recall,f_score
    else:
        return np.array([np.NaN]),np.array([np.NaN]),np.array([np.NaN])

## For cell dimension qualtification:

def get_rot_mat(theta):
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return R

# def center_and_rotate(label_arr):
#     rps = sk.measure.regionprops(label_arr)
#     coord_list = []

#     for i,rp in enumerate(rps):

#         R = get_rot_mat(rp.orientation)
#         centroid = np.array(list(rp.centroid))
#         img = label_arr==(i+1)
#         coords = sk.measure.find_contours(img, level=0)[0]
#         centered_coords = coords-centroid
#         rotated_coords = (R@centered_coords.T).T
#         reflected_coords = copy.copy(rotated_coords)
#         reflected_coords[:,1] = -reflected_coords[:,1]
#         all_coords = np.concatenate([rotated_coords,reflected_coords], axis=0)
#         half_coords = all_coords[all_coords[:,1]>0.]
#         half_coords = half_coords[np.argsort(half_coords[:,0])]
#         coord_list.append(half_coords)

#     return coord_list

def width_length_from_permeter_area(perimeter,area):
    width = (perimeter-np.sqrt(perimeter**2 - 4*np.pi*area))/np.pi
    length = ((perimeter - (np.pi*width))/2) + width
    return width,length

def get_cell_dimensions_spherocylinder_perimeter_area(rps):
    cell_dimensions = []

    for rp in rps:
        perimeter,area = rp.perimeter_crofton,rp.area
        width,length = width_length_from_permeter_area(perimeter,area)

        volume = np.pi * ((1/4)*(width**2)*(length-width) + (1/6)*(width**3))
        surface_area = np.pi*width*length
        cell_dimension_dict = {"Length":length,"Width":width,"Volume": volume,"Surface Area": surface_area}

        cell_dimensions.append(cell_dimension_dict)

    return cell_dimensions

def get_cell_dimensions_spherocylinder_ellipse(rps):
    cell_dimensions = []

    for rp in rps:
        width,length = rp.minor_axis_length,rp.major_axis_length
        volume = np.pi * ((1/4)*(width**2)*(length-width) + (1/6)*(width**3))
        surface_area = np.pi*width*length
        cell_dimension_dict = {"Length":length,"Width":width,"Volume": volume,"Surface Area": surface_area}

        cell_dimensions.append(cell_dimension_dict)

    return cell_dimensions



def get_cell_dimensions(label_arr, min_dist_integral_start = 0.3):

    # This method is not very robust; recommend an improved version before using at scale

    rps = sk.measure.regionprops(label_arr)

    cell_dimensions = []

    for i,rp in enumerate(rps):

        try:

            cell_dimension_dict = {}

            R = get_rot_mat(rp.orientation)
            centroid = np.array(list(rp.centroid))
            img = label_arr==(i+1)
            coords = sk.measure.find_contours(img, level=0)[0]
            centered_coords = coords-centroid
            rotated_coords = (R@centered_coords.T).T
            reflected_coords = copy.copy(rotated_coords)
            reflected_coords[:,1] = -reflected_coords[:,1]
            all_coords = np.concatenate([rotated_coords,reflected_coords], axis=0)
            half_coords = all_coords[all_coords[:,1]>0.]
            coords = half_coords[np.argsort(half_coords[:,0])]

            y_coords = coords[:,0]
            x_coords = coords[:,1]

            no_cap_mask = x_coords>min_dist_integral_start
            filtered_y_coords = y_coords[no_cap_mask]
            filtered_x_coords = x_coords[no_cap_mask]

            spline_fit = sp.interpolate.UnivariateSpline(y_coords, x_coords, k=4)
            a,b = np.min(filtered_y_coords),np.max(filtered_y_coords)
            x = np.linspace(a,b,num=int((b-a)))
            f_x = spline_fit(x)

            vol = (np.pi*sp.integrate.simpson(f_x**2, x))
            SA = 2*np.pi*spline_fit.integral(a,b)
            width = 2*np.max(f_x)
            height = np.max(y_coords) - np.min(y_coords)

            cell_dimension_dict = {"Volume": vol,"Surface Area": SA,"Width": width,"Length": height}

        except:

            cell_dimension_dict = {"Volume": np.NaN,"Surface Area": np.NaN,"Width": np.NaN,"Length": np.NaN}

        cell_dimensions.append(cell_dimension_dict)

    return cell_dimensions
