# fmt: off
import numpy as np
import skimage as sk

def object_f_scores(true_mask,pred_mask,eps = (10**-5)):
    any_true,any_pred = (np.any(true_mask),np.any(pred_mask))
    if any_true and any_pred:
        true_labels = sk.measure.label(true_mask)
        pred_labels = sk.measure.label(pred_mask)
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
        num_cells = np.max(sk.measure.label(true_mask))
        Precision = np.array([0. for i in range(num_cells)])
        Recall = np.array([0. for i in range(num_cells)])
        f_score = np.array([0. for i in range(num_cells)])

        return Precision,Recall,f_score
    else:
        return np.array([np.NaN]),np.array([np.NaN]),np.array([np.NaN])
