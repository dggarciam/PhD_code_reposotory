#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Mateo Tobón Henao <mtobonh@unal.edu.co>
Created on Fri  29/04/2022
"""

from itertools import compress
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

# =============================================================================
# List Indexing
# =============================================================================

def custom_indexing_list(list_, index):
    return [list_[i] for i in index]

def boolean_indexing_list(list_, index):
    return list(compress(list_, index))


# =============================================================================
# Min-Max Normalization
# =============================================================================

def min_max_normalization(X, axis=-1):
    X -= X.min(axis=axis, keepdims=True)
    X /= X.max(axis=axis, keepdims=True)
    return X

# =============================================================================
# Read metrics of gridsearch
# =============================================================================

def grid_search_info(cv_results_, metrics):
    """
    Return the important information of the grid_search dictionary results.
    INPUT
    -----
    1. cv_results_: (dict) grid_search.cv_results_ attribute.
    2. metrics: (list of srt) list of metrics names.
    OUTPUT
    ------
    2. (tuple) 
        1. best_params 
        2.(mean_test, std_test): (list) list with values of mean and std for every metric.
        3.(mean_train, std_train): (list) list with values of mean and std for every metric.
        4.(mean_test_time, std_test_time)
        5.(mean_fit_time, std_fit_time )
    """
    mean_test = []
    std_test = []
    mean_train = []
    std_train = []
    best_index = np.argwhere(cv_results_['rank_test_'+metrics[0]]==1)[0][0]
    best_params  = cv_results_['params'][best_index]
    for met in metrics:
        mean_test.append(np.round(cv_results_['mean_test_'+met][best_index], 4))
        std_test.append(np.round(cv_results_['std_test_'+met][best_index], 4))
        mean_train.append(np.round(cv_results_['mean_train_'+met][best_index], 4))
        std_train.append(np.round(cv_results_['std_train_'+met][best_index], 4))
        mean_test_time = np.round(cv_results_['mean_score_time'][best_index], 4)
        std_test_time = np.round(cv_results_['std_score_time'][best_index], 4)
        mean_fit_time = np.round(cv_results_['mean_fit_time'][best_index], 4)
        std_fit_time = np.round(cv_results_['std_fit_time'][best_index], 4)
    return best_params, (mean_test, std_test), (mean_train, std_train), (mean_test_time, std_test_time), (mean_fit_time, std_fit_time )

# =============================================================================
# Plot Connectivities in matrix form
# =============================================================================

def plot_connectivity_matrix(X, ch_names, mode='functional', num_w=None, num_fb=None, mean_by_time_windows=False, mean_by_frequencies=False, title='', wlabel=None, fblabel=None, cmap='hot', fig_size=(30,30), show=True, save=False, path='', format='png'):
    """
    Plot functional or effective connectivities in matrix form.

    INPUT
    ----------
    1. X : (1D array) (0.5*ch*(ch-1)*windows*frequency_bands) if mode='functional'
                    (ch*ch*windows*frequency_bands)         if mode='effective'
    2. ch_names : (list of str)
        channels names used in the montage.
    3. mode : (str), default= 'functional'
        type of connectivity to plot.
    4. num_w : (int) default=1
        number of windows.
    5. num_fb : (int) default=1
        number of frequency bands.
    6. mean_by_time_windows : (bool), default=False
        mean by time windows before to plot.
    7. mean_by_frequencies : (bool), default=False
        mean by frequencies before to plot.
    8. title : (str), default=''
        figure title.
    9. wlabel : (list of str), default=None
        list of names for time windows.
    10. fblabl : (list of str), default=None
        list of names for frequency bands.
    11. figsize : (tuple), default=(30,30)
        figure size.
    12. cmap : (str), default='hot'
        color map for connectivities.
    13. show : (bool), default=True
        show connectivities matrix plot.
    14. save : (bool), default=False
        Save connectivities matrix plot.
    15. path : (str), default=''
        path to save connectivities matrix plot.
    16. format : (str), default='png'
        format to save connectivities matrix plot.
    OUTPUT
    -------
    1. None.

    """
    ch = len(ch_names) #Number of channels
    #Number of connectivities 
    if mode == 'functional':
        con = int(0.5*ch*(ch-1))
    elif mode == 'effective':
        con = int(ch*ch)
    else:
        raise ValueError('No valid mode')

    X = X.reshape(con, num_w, num_fb)

  #Mean by windows and/or frequency bands.
    if mean_by_time_windows and mean_by_frequencies:
        X = X.mean(axis=(1,2), keepdims=True)
    else:
        if mean_by_time_windows:
            X = X.mean(axis=1, keepdims=True)
        else:
            if mean_by_frequencies:
                X = X.mean(axis=2, keepdims=True)
            else:
                pass

    X = min_max_normalization(X, axis=None)
    fig, axs = plt.subplots(nrows=X.shape[1], ncols=X.shape[2], squeeze=False)
    fig.set_size_inches(fig_size[0], fig_size[1])
    fig.suptitle(title, fontsize=20, fontweight='bold')
    if mode == 'functional':
        for w in range(X.shape[1]):
            for f in range(X.shape[2]):
                axs[w,f].imshow(squareform(X[:,w,f]), cmap=cmap) #Matrix form
                axs[w,f].set_xticks(np.arange(len(ch_names)))
                axs[w,f].set_xticklabels(ch_names, rotation=90)
                axs[w,f].set_yticks(np.arange(len(ch_names)))
                axs[w,f].set_yticklabels(ch_names)
    else:
        for w in range(X.shape[1]):
            for f in range(X.shape[2]):
                axs[w,f].imshow(X[:,w,f].reshape(ch, ch), cmap=cmap) #Matrix form
                axs[w,f].set_xticks(np.arange(len(ch_names)))
                axs[w,f].set_xticklabels(ch_names, rotation=90)
                axs[w,f].set_yticks(np.arange(len(ch_names)))
                axs[w,f].set_yticklabels(ch_names)  

    if wlabel is not None:
        for w in range(X.shape[1]):
            axs[w,0].set_ylabel(wlabel[w])

    if fblabel is not None:
        for fb in range(X.shape[2]):
            axs[0,fb].set_title(fblabel[fb])

    cax = fig.add_axes([axs[-1,-1].get_position().x1 + 0.05,axs[-1,-1].get_position().y0,0.02,axs[0,-1].get_position().y1-axs[-1,-1].get_position().y0])
    #Mappeable objects for  colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(X.ravel())
    fig.colorbar(sm, cax=cax)
    #Save plot
    if save == True:
        plt.savefig(path+'.'+format, format=format, bbox_inches='tight')
    #Show plot
    if show == True:
        plt.show()
    else:
        plt.close()

    return

# =============================================================================
# Normalize Cxx_w_f
# =============================================================================

def normalize(X, min_max_normalization_type=None):
    """
    Normalization
    INPUT
    -----
    1. X: (3D array) shape (features, windows, frequency_bands)
    2. min_max_normalization_type: (str) {None, 'all', 'windows', 'frequency', 'individual'}. Default=None
    OUTPUT:
    1. X: (3D array) Normalized X, shape (features, windows, frequency_bands) 
    """
    if min_max_normalization_type is None:
        pass
    elif min_max_normalization_type == 'all':
        X = min_max_normalization(X, axis=None)
    elif min_max_normalization_type == 'windows':
        X = min_max_normalization(X, axis=(0,-1))
    elif min_max_normalization_type == 'frequency':
        X = min_max_normalization(X, axis=(0,1))
    elif min_max_normalization_type == 'individual':
        X = min_max_normalization(X, axis=0)
    else:
        ValueError('Invalid min_max_normalization_type')
    return X