# fmt: off
import numpy as np
from matplotlib import pyplot as plt

def plot_kymograph(kymograph):
    """Helper function for plotting kymographs. Takes a kymograph array of shape (y_dim,x_dim,t_dim).
    
    Args:
        kymograph (array): kymograph array of shape (y_dim,x_dim,t_dim).
    """
    list_in_t = [kymograph[t,:,:] for t in range(kymograph.shape[0])]
    img_arr = np.concatenate(list_in_t,axis=1)
    plt.imshow(img_arr)