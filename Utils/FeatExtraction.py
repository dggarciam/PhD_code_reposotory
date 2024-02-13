#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Mateo Tobón Henao <mtobonh@unal.edu.co>
Created on Fri  29/04/2022
"""

import numpy as np
from MI_EEG_ClassMeth.Preprocessing import butterworth_digital_filter
from abc import ABCMeta, abstractmethod
from mne.channels.layout import _find_topomap_coords
from MI_EEG_ClassMeth.utils import normalize
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr, kendalltau
from scipy.signal import hilbert
from itertools import permutations
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import os
import shutil
from tqdm import tqdm
from MI_EEG_ClassMeth.utils import min_max_normalization

# =============================================================================
# Functions
# =============================================================================

def takens_delay_embedding(x, *, tau, dim):
  """
  takens delay embedding of a time serie.
  INPUT
  -----
    1. x: (1D array) unidimensional time serie in R^{T}
    2. tau: (int) time delay embedding in N
    3. dim: (int) embedding dimension in N
  OUTPUT
  ------
    1. x_emb: (2D array) embedding time serie in R^{row_emb,dim}
  """
  len_    = x.shape[0]
  row_emb = len_ - (dim-1)*tau
  if row_emb <= 0:
    raise ValueError("The embeddings dimension and time delay embedding doesn't fit the time serie")
  x_emb = np.zeros((row_emb,dim))
  for i,j in enumerate(np.arange(0,tau*dim,tau)):
    x_emb[:,i] = x[np.arange(j,j + row_emb)]

  return x_emb

# =============================================================================
# Time Frequency 
# =============================================================================

class TimeFrequencyRpr(BaseEstimator, TransformerMixin):
  """
  Time frequency representation of EEG signals.

  Parameters
  ----------
    1. sfreq:  (float) Sampling frequency in Hz. 
    2. f_bank: (2D array) Filter banks Frequencies. Default=None
    3. vwt:    (2D array) Interest time windows. Default=None
  Methods
  -------
    1. fit(X, y=None)
    2. transform(X, y=None)
  """
  def __init__(self, sfreq, f_bank=None, vwt=None):
    self.sfreq = sfreq
    self.f_bank = f_bank
    self.vwt = vwt
# ------------------------------------------------------------------------------

  def _validation_param(self):
    """
    Validate Time-Frequency characterization parameters.
    INPUT
    -----
      1. self
    ------
      2. None
    """
    if self.sfreq <= 0:
      raise ValueError('Non negative sampling frequency is accepted')
    

    if self.f_bank is None:
      self.flag_f_bank = False
    elif self.f_bank.ndim != 2:
      raise ValueError('Band frequencies have to be a 2D array')
    else:
      self.flag_f_bank = True

    if self.vwt is None:
      self.flag_vwt = False
    elif self.vwt.ndim != 2:
      raise ValueError('Time windows have to be a 2D array')
    else:
      self.flag_vwt = True

# ------------------------------------------------------------------------------
  def _filter_bank(self, X):
    """
    Filter bank Characterization.
    INPUT
    -----
      1. X: (3D array) set of EEG signals, shape (trials, channels, time_samples)
    OUTPUT
    ------
      1. X_f: (4D array) set of filtered EEG signals, shape (trials, channels, time_samples, frequency_bands)
    """
    X_f = np.zeros((X.shape[0], X.shape[1], X.shape[2], self.f_bank.shape[0])) #epochs, Ch, Time, bands
    for f in np.arange(self.f_bank.shape[0]):
      X_f[:,:,:,f] = butterworth_digital_filter(X, N=5, Wn=self.f_bank[f], btype='bandpass', fs=self.sfreq)
    return X_f

# ------------------------------------------------------------------------------
  def _sliding_windows(self, X):
    """
    Sliding Windows Characterization.
    INPUT
    -----
      1. X: (3D array) set of EEG signals, shape (trials, channels, time_samples)
    OUTPUT
    ------
      1. X_w: (4D array) shape (trials, channels, window_time_samples, number_of_windows)
    """
    window_lenght = int(self.sfreq*self.vwt[0,1] - self.sfreq*self.vwt[0,0])
    X_w = np.zeros((X.shape[0], X.shape[1], window_lenght, self.vwt.shape[0]))
    for w in np.arange(self.vwt.shape[0]):
        X_w[:,:,:,w] = X[:,:,int(self.sfreq*self.vwt[w,0]):int(self.sfreq*self.vwt[w,1])]
    return X_w

# ------------------------------------------------------------------------------
  def fit(self, X, y=None):
    """
    fit.
    INPUT
    -----
      1. X: (3D array) set of EEG signals, shape (trials, channels, time_samples)
      2. y: (1D array) target labels. Default=None
    OUTPUT
    ------
      1. None
    """
    pass

# ------------------------------------------------------------------------------
  def transform(self, X, y=None):
    """
    Time frequency representation of EEG signals.
    INPUT
    -----
      1. X: (3D array) set of EEG signals, shape (trials, channels, times)
    OUTPUT
    ------
      1. X_wf: (5D array) Time-frequency representation of EEG signals, shape (trials, channels, window_time_samples, number_of_windows, frequency_bands)
    """
    self._validation_param()     #Validate sfreq, f_freq, vwt

    #Avoid edge effects of digital filter, 1st:fbk, 2th:vwt
    if self.flag_f_bank:
        X_f = self._filter_bank(X)
    else:
        X_f = X[:,:,:,np.newaxis]

    if self.flag_vwt:
      X_wf = []
      for f in range(X_f.shape[3]):
        X_wf.append(self._sliding_windows(X_f[:,:,:,f]))
      X_wf = np.stack(X_wf, axis=-1)
    else:
      X_wf = X_f[:,:,:,np.newaxis,:]

    return X_wf

# =============================================================================
# Connectivities base class 
# =============================================================================

class Connectivities(metaclass=ABCMeta):
  """
  Base Class for Connectivites

  Parameters
  ----------

  Methods
  -------
    1. flow_of_connectivities(X, mode)
    2. plot_connectivities(c_xx, info, channels_names, used_channels_names=None, mode='functional',
                          n_vwt = None, n_f_bank=None, mean_by_time_windows=False, mean_by_frequencies=False,
                          min_max_normalization_type=None,
                          thr=99,
                          fig_title='',  w_label=None, fb_label=None, labels_params = {'ylabel_fonfamily':'serif', 'ylabel_fontsize':18, 'ylabel_weight':1000, 'xlabel_fonfamily':'serif', 'xlabel_fontsize':18, 'xlabel_weight':500, 'rotation':0, 'cl_size':16},
                          plot_channels=True, relevant_channels=True, channel_importance=True, plot_channels_names=True, show_connectivity_colorbar=True, show_topomap_colorbar=True,
                          figsize=(30,30), cmap_connectivities='hot', cmap_tplt='bone',
                          show=True, save=False, path='', format='png')
  """

  def __init__(self):
      pass
 
 # ------------------------------------------------------------------------------
  def flow_of_connectivities(self, X, mode='functional'):
    """
    Calculate flow of connectivites.
    INPUT
    -----
    1. X: (4D array) shape (trials, 0.5*channels*(channels-1), number_of_windows, frequency_bands) if mode='functional'
                           (trials, channels*channels, number_of_windows, frequency_bands)         if mode='effective'
    2. mode: (str) Kind of connectivity {'functional', 'effective'}, Default='functional'
    OUTPUT
    ------
    1. flow_con: (4D array) flow of connectivities, shape (trials, channels, number_of_windows, frequency_bands).
    """
    n_trial = X.shape[0]
    n_vwt = X.shape[2]
    n_f_bank = X.shape[3]
    if mode == 'functional':
      ch = int((1 + np.sqrt(1 + 8*X.shape[1]))/2)
      flow_con = np.zeros((n_trial, ch, n_vwt, n_f_bank))
      for n in range(n_trial):
          for w in range(n_vwt):
              for fb in range(n_f_bank):
                  flow_con[n,:,w,fb] = squareform(X[n,:,w,fb]).sum(axis=1)
    elif mode == 'effective':
      ch = int(np.sqrt(X.shape[1]))
      for n in range(n_trial):
          for w in range(n_vwt):
              for fb in range(n_f_bank):
                flow_con[n,:,w,fb] = (X[n,:,w,fb].reshape(ch,ch)).sum(axis=1)
    else:
      raise ValueError('No valid kind of connectivity')

    return flow_con
  
# ------------------------------------------------------------------------------
  def plot_connectivities(self, c_xx, info, channels_names, used_channels_names=None, mode='functional',
                          n_vwt = None, n_f_bank=None, mean_by_time_windows=False, mean_by_frequencies=False,
                          min_max_normalization_type=None,
                          thr=99,
                          fig_title='',  w_label=None, fb_label=None, labels_params = {'ylabel_fonfamily':'serif', 'ylabel_fontsize':18, 'ylabel_weight':1000, 'xlabel_fonfamily':'serif', 'xlabel_fontsize':18, 'xlabel_weight':500, 'rotation':0, 'cl_size':16},
                          plot_channels=True, relevant_channels=True, channel_importance=True, plot_channels_names=True, show_connectivity_colorbar=True, show_topomap_colorbar=True,
                          figsize=(30,30), cmap_connectivities='hot', cmap_tplt='bone',
                          show=True, save=False, path='', format='png'):

      """
      
      Plot functional or effective connectivities.

      INPUT
      ----------
      1. c_xx: (1D array) (0.5*channels*(channels-1)*windows*frequency_bands) if mode='functional'
                           (channels*channels*windows*frequency_bands)         if mode='effective'
      2. info:(mne info)
      3. channels_names : (list of strings)
          channels names used in the montage.
      4. used_channels_names : (list of strings), default=None
          channels names used to calculate connectivities.
      5. mode: (str), default= 'functional'
          type of connectivity to plot.
      6. n_vwt: (int), default=None
          number of time windows
      7. n_f_bank: (int), default=None
          number of frequency bands
      8. mean_by_time_windows: (bool), default=False
          mean by time windows before to plot.
      9. mean_by_frequencies: (bool), default=False
          mean by frequencies before to plot.
      10. min_max_normalization_type: (str) {None, 'all', 'windows', 'frequency', 'individual'}. Default=None
          min-max normalization of connectivities:
          - None: not apply min-max normalization
          - 'all': maximum and minimun value of  the connectivities across time windows and frequencies.
          - 'windows':maximum and minimun value of the connectivities in each time window.
          - 'frequency':maximum and minimun value of the connectivities in each frequecy band. 
          - 'individual':maximum and minimun value of the connectivities in each time window and each frequency band.
      11. thr: (float), default=99
          plot connectivities greater than the percentile thr.
      12. fig_title: (str), default=''
          figure title.
      13. w_label: (list of str), default=None
          labels for each time window.
      14. fb_label: (list of str), default=None
          labels for each frequency band. 
      15. labels_params: (dict), default={'ylabel_fonfamily':'serif', 'ylabel_fontsize':18, 'ylabel_weight':1000, 'xlabel_fonfamily':'serif', 'xlabel_fontsize':18, 'xlabel_weight':500, 'rotation':0, 'cl_size':16}
          dict with font style parameters for w_label and y_label.
      16. plot_channels: (bool), default=True
          plot channels positions.
      17. relevant_channels: (bool), default=True
          plot only channels positions where connectivity values are greater than thr.
      18. channel_importance: (bool), default=True
          plot spatial relevance in channel positions (Only used if relevant_channels=True).
      19. plot_channels_names: (bool), default=True
          plot channels names
      20. show_connectivity_colorbar: (bool), default=True
          show colorbar for connectivities.
      21. show_topomap_colorbar: (bool), default=True
          show colorbar for topomaps.
      22. figsize: (tuple), default=(30,30)
          figure size.
      23. cmap_connectivities: (str), default='hot'
          color map for connectivities.
      24. cmap_tplt: (str), default='bone'
          colormaps for topomap.
      25. show: (bool), default=True
          show connectivities plot.
      26. save: (bool), default=False
          Save connectivities plot.
      27. path: (str), default=''
          path to save connectivities plot.
      28. format: (str), default='png'
          format to save connectivities plot.


      OUTPUT
      -------
      1. None.

      """      
      ch = len(channels_names) #Number of channels

      #Number of connectivities 
      if mode == 'functional':
         con = int(0.5*ch*(ch-1))
      elif mode == 'effective':
         con = int(ch*ch)
      else:
         raise ValueError('No valid mode')

      n_vwt = 1 if n_vwt is None else n_vwt
      n_f_bank = 1 if n_f_bank is None else n_f_bank

      def convert_con_of_partial_channels_to_full_channels(c_xx, used_channels_names, channels_names, mode):
        """
        Fit connectivity matrix of all channels used in the montage with connectivity matrix 
        of partial channels used to calculate connectivities.
        INPUT
        -----
        1. c_xx: (1D array) (0.5*channels*(channels-1)*windows*frequency_bands) if mode='functional'
                            (channels*channels*windows*frequency_bands)         if mode='effective'
        2. channels_names: (list of strings)
            channels names used in the montage.
        3. used_channels_names: (list of strings)
            channels names used to calculate connectivities.
        4. mode: (str)
            type of connectivity to plot.
        OUTPUT
        ------
        1. c_xx: (1D array) (0.5*channels*(channels-1)*windows*frequency_bands) if mode='functional'
                            (channels*channels*windows*frequency_bands)         if mode='effective'
        """
        if mode == 'functional':
          n_used_channels = len(used_channels_names)
          n_con_used_channels = int(0.5*n_used_channels*(n_used_channels-1))
          tmp_used_con = c_xx.reshape(n_con_used_channels, -1)
          used_con = np.asarray([squareform(tmp_used_con[:,i]) for i in range(tmp_used_con.shape[-1])]).T
          idx_used_channels = np.isin(channels_names, used_channels_names)
          all_con = np.zeros((len(channels_names), len(channels_names), used_con.shape[-1]))
        else:
          n_used_channels = len(used_channels_names)
          n_con_used_channels = int(n_used_channels*n_used_channels)
          tmp_used_con = c_xx.reshape(n_con_used_channels, -1)
          used_con = np.asarray([tmp_used_con[:,i].reshape(n_used_channels, n_used_channels) for i in range(tmp_used_con.shape[-1])]).T
          idx_used_channels = np.isin(channels_names, used_channels_names)
          all_con = np.zeros((len(channels_names), len(channels_names), used_con.shape[-1]))
        for wf in range(used_con.shape[-1]):
            idx_ch1_used = 0
            for ch1 in range(len(idx_used_channels)):
                if idx_used_channels[ch1]==True:
                    idx_ch2_used=0
                    for ch2 in range(len(idx_used_channels)):
                        if idx_used_channels[ch2]==True:
                            all_con[ch1,ch2,wf] = used_con[idx_ch1_used,idx_ch2_used,wf]
                            idx_ch2_used+=1
                        else:
                            all_con[ch1,ch2,wf] = 0
                    idx_ch1_used+=1
                else:
                    all_con[ch1,:,wf]=0
        if mode == 'functional':
          vectorized_connectivity = np.asarray([squareform(all_con[:,:,i]) for i in range(all_con.shape[-1])]).T
        else:
          vectorized_connectivity = np.asarray([all_con[:,:,i].reshape(-1) for i in range(all_con.shape[-1])]).T
        return vectorized_connectivity.ravel()

      if used_channels_names is not None:
        c_xx = convert_con_of_partial_channels_to_full_channels(c_xx, used_channels_names, channels_names, mode)
      
      c_xx_w_f = c_xx.reshape(con, n_vwt, n_f_bank)  #Connectivity matrix (ch(ch-1)/2, windows, frequency bands) or (ch*ch, windows, frequency bands)
      
      #Mean by windows and/or frequency bands.
      if mean_by_time_windows and mean_by_frequencies:
        c_xx_w_f = c_xx_w_f.mean(axis=(1,2), keepdims=True)
      else:
        if mean_by_time_windows:
          c_xx_w_f = c_xx_w_f.mean(axis=1, keepdims=True)
        else:
          if mean_by_frequencies:
            c_xx_w_f = c_xx_w_f.mean(axis=2, keepdims=True)
          else:
            pass
            

      #pick channels
      pos = _find_topomap_coords(info, picks=None)
      pos = pos[:, :2]
      pos_x, pos_y = pos.T
                            
      c_xx_w_f= normalize(c_xx_w_f, min_max_normalization_type=min_max_normalization_type)
      cNorm  = colors.Normalize(vmin=c_xx_w_f.min(), vmax=c_xx_w_f.max())
      scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap_connectivities)
      flow_of_connectivities = np.squeeze(self.flow_of_connectivities(c_xx_w_f[np.newaxis, ...], mode=mode), axis=0)
      flow_of_connectivities = normalize(flow_of_connectivities, min_max_normalization_type=min_max_normalization_type)
               
      fig, axs = plt.subplots(n_vwt, n_f_bank, squeeze=False)
      fig.set_size_inches(figsize[0], figsize[1])
      fig.suptitle(fig_title, fontsize=labels_params['xlabel_fontsize']*2, fontweight='bold')
      area_channel_marker = int((figsize[0]/c_xx_w_f.shape[1])*(figsize[1]/c_xx_w_f.shape[2])*2)
      
      if w_label is not None:
        for w in range(n_vwt):
          axs[w,0].set_ylabel(w_label[w], family = labels_params['ylabel_fonfamily'], size=labels_params['ylabel_fontsize'], weight=labels_params['ylabel_weight'], rotation=labels_params['rotation'], ha='right')

      if fb_label is not None:
        for fb in range(n_f_bank):
          axs[0,fb].set_title(fb_label[fb], family = labels_params['xlabel_fonfamily'], size=labels_params['xlabel_fontsize'], weight=labels_params['xlabel_weight'])

      if mode == 'functional':
         for frec in np.arange(n_f_bank):
             for time in np.arange(n_vwt):
                ax = axs[time,frec]
                #Plot topomap
                plot_topomap(flow_of_connectivities[:,time,frec], pos, axes=ax, cmap=cmap_tplt, show=False, contours=0, sensors=False, sphere=None, outlines='head', vmin=flow_of_connectivities.min(), vmax=flow_of_connectivities.max())
                
                connectivity = squareform(c_xx_w_f[:,time,frec]) #Get connectivity matrix
                #Plot connectivities with values greater than thr
                if thr >= 1:
                  if min_max_normalization_type is None or min_max_normalization_type == 'all':
                    indx_pct  = np.where(np.triu(connectivity)>np.percentile(c_xx_w_f.ravel()[c_xx_w_f.ravel()>1e-8], thr))
                  elif min_max_normalization_type == 'windows':
                    indx_pct  = np.where(np.triu(connectivity)>np.percentile(c_xx_w_f[:,time,:].ravel()[c_xx_w_f[:,time,:].ravel()>1e-8], thr))
                  elif min_max_normalization_type == 'frequency':
                    indx_pct  = np.where(np.triu(connectivity)>np.percentile(c_xx_w_f[:,:,frec].ravel()[c_xx_w_f[:,:,frec].ravel()>1e-8], thr))
                  else:
                    indx_pct  = np.where(np.triu(connectivity)>np.percentile(c_xx_w_f[:,time,frec][c_xx_w_f[:,time,frec]>1e-8], thr))
                else:
                    indx_pct = np.where(np.triu(connectivity)>thr) 
                ax.set(xticks=[], yticks=[], aspect='equal')

                #Plot arrows for connectivities
                for i in np.arange(np.shape(indx_pct)[-1]):
                    ch1=indx_pct[0][i]
                    ch2=indx_pct[1][i]     
                    ax.arrow(pos_x[ch1], pos_y[ch1], pos_x[ch2]-pos_x[ch1], pos_y[ch2]-pos_y[ch1],
                             width=0.002, length_includes_head=False, head_width=0.0,
                             color=scalarMap.to_rgba(connectivity[ch1,ch2]))
                    
                
                if plot_channels:
                    if relevant_channels:
                        if channel_importance:
                            #Plot channels of relevant connections and importance of every channel(# times channels appears in connecions)
                            for ch1 in  np.unique(np.ravel(indx_pct)):
                                ax.scatter(pos_x[ch1], pos_y[ch1], s=area_channel_marker*np.sum(np.ravel(indx_pct)==ch1), color='k', edgecolors='w')
                        else:
                            #Plot channels of relevant connections
                            for ch1 in  np.unique(np.ravel(indx_pct)):
                                ax.scatter(pos_x[ch1], pos_y[ch1], s=area_channel_marker, color='k', edgecolors='w')
                    else:
                    #Plot all channels
                        for ch1 in  np.arange(ch):
                            ax.scatter(pos_x[ch1], pos_y[ch1], s=area_channel_marker, color='k', edgecolors='w')
                            
                if plot_channels_names:
                    if relevant_channels:
                        #Plot channels names  of relevant connections
                        for ch1 in  np.unique(np.ravel(indx_pct)):
                                ax.annotate(info['ch_names'][ch1], xy=pos[ch1,:], size=15)
                    else:
                    #Plot all channels names
                        for ch1 in  np.arange(ch):
                            ax.annotate(info['ch_names'][ch1], xy=pos[ch1,:], size=15)
                
                
      else:
          for frec in np.arange(n_f_bank):
            for time in np.arange(n_vwt):
                ax = axs[time, frec]
                #Plot topomap
                plot_topomap(flow_of_connectivities[:,time,frec], pos, axes=ax, cmap=cmap_tplt, show=False, contours=0, sensors=False, sphere=None, outlines='head', vmin=flow_of_connectivities.min(), vmax=flow_of_connectivities.max())
                
                connectivity = c_xx_w_f[:,time,frec].reshape(ch,ch) #Get connectivity matrix
                #Plot connectivities with values greater than thr
                if thr >=1:
                  if min_max_normalization_type is None or min_max_normalization_type == 'all':
                    indx_pct  = np.where(connectivity>np.percentile(c_xx_w_f.ravel()[c_xx_w_f.ravel()>1e-8], thr))
                  elif min_max_normalization_type == 'windows':
                    indx_pct  = np.where(connectivity>np.percentile(c_xx_w_f[:,time,:].ravel()[c_xx_w_f[:,time,:].ravel()>1e-8], thr))
                  elif min_max_normalization_type == 'frequency':
                    indx_pct  = np.where(connectivity>np.percentile(c_xx_w_f[:,:,frec].ravel()[c_xx_w_f[:,:,frec].ravel()>1e-8], thr))
                  else:
                    indx_pct  = np.where(connectivity>np.percentile(c_xx_w_f[:,time,frec][c_xx_w_f[:,time,frec]>1e-8], thr))
                else:
                    indx_pct = np.where(connectivity>thr) 
                ax.set(xticks=[], yticks=[], aspect='equal')

                #Plot arrows for connectivities
                for i in np.arange(np.shape(indx_pct)[-1]):
                    ch1=indx_pct[0][i]
                    ch2=indx_pct[1][i]     
                    ax.arrow(pos_x[ch1],pos_y[ch1],pos_x[ch2]-pos_x[ch1],
                            pos_y[ch2]-pos_y[ch1],width=0.002,length_includes_head=True,
                            head_width=0.03, color=scalarMap.to_rgba(connectivity[ch1,ch2]))
                    
                if plot_channels:
                    if relevant_channels:
                        if channel_importance:
                            #Plot channels of relevant connections and importance of every channel(# times channels appears in connecions)
                            for ch1 in  np.unique(np.ravel(indx_pct)):
                                ax.scatter(pos_x[ch1], pos_y[ch1], s=area_channel_marker*np.sum(np.ravel(indx_pct)==ch1), color='k', edgecolors='w')
                        else:
                            #Plot channels of relevant connections
                            for ch1 in  np.unique(np.ravel(indx_pct)):
                                ax.scatter(pos_x[ch1], pos_y[ch1], s=area_channel_marker, color='k', edgecolors='w')
                    else:
                    #Plot all channels
                        for ch1 in  np.arange(ch):
                            ax.scatter(pos_x[ch1], pos_y[ch1], s=area_channel_marker, color='k', edgecolors='w')
                            
                if plot_channels_names:
                    if relevant_channels:
                        #Plot channels names  of relevant connections
                        for ch1 in  np.unique(np.ravel(indx_pct)):
                                ax.annotate(info['ch_names'][ch1], xy=pos[ch1,:], size=15)
                    else:
                    #Plot all channels names
                        for ch1 in  np.arange(ch):
                            ax.annotate(info['ch_names'][ch1], xy=pos[ch1,:], size=15)

      fig.tight_layout(rect=[0, 0, 1, 0.95])
      
      alpha1 = int(figsize[0]/figsize[1])
      alpha2 = int(figsize[1]/figsize[0])
      if show_connectivity_colorbar:
          wide = alpha2*0.01 if figsize[0] < figsize[1] else 0.01
          pad = 0.05 if figsize[0] < figsize[1] else 0.05/alpha1
          #axes for connectivities colorbar
          cax_con=fig.add_axes([axs[-1,-1].get_position().x1 + pad, axs[-1,-1].get_position().y0, wide, axs[0,-1].get_position().y1-axs[-1,-1].get_position().y0])
          #Mappeable objects for connectivities colorbar
          sm1 = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap_connectivities)
          sm1.set_array(c_xx_w_f.ravel())
          cbar_con = fig.colorbar(sm1, cax=cax_con)
          for t in cbar_con.ax.get_yticklabels():
              t.set_fontsize(labels_params['cl_size'])
          
      if show_topomap_colorbar:
          wide = alpha1*0.01 if figsize[1] < figsize[0] else 0.01
          pad = 0.05 if figsize[1] < figsize[0] else 0.05/alpha2
          #axes for topomaps colorbar
          cax_tplt=fig.add_axes([axs[-1,0].get_position().x0, axs[-1,0].get_position().y0 - pad, axs[-1,-1].get_position().x1-axs[-1,0].get_position().x0, wide])
          norm  = colors.Normalize(vmin=flow_of_connectivities.min(), vmax=flow_of_connectivities.max())
          #Mappeable objects for topomaps colorbar
          sm2 = plt.cm.ScalarMappable(norm=norm , cmap=cmap_tplt)
          sm2.set_array(flow_of_connectivities.ravel())
          cbar_tplt = fig.colorbar(sm2, cax=cax_tplt, orientation='horizontal')
          for t in cbar_tplt.ax.get_xticklabels():
              t.set_fontsize(labels_params['cl_size'])
          
      #Save plot
      if save == True:
          plt.savefig(path+'.'+format,format=format, bbox_inches='tight')


      #Show plot
      if show == True:
         plt.show()
      else:
          plt.close()

          
      return

# ------------------------------------------------------------------------------
  @abstractmethod
  def fit(self, X, y=None, **kargs):
    """
    Calculate Connectivity paramas
    INPUT
    ------
      1. X: (5D array) shape (trials, channels, window_time_samples, number_of_windows, frequency_bands)
      2. y: None
      3. **kargs
    OUTPUT
    ------
      1. C_xx_w_f: (2D array) (trials, channels*(channels-1)*0.5, number_of_windows, frequency_bands)
    """
    pass

# ------------------------------------------------------------------------------
  @abstractmethod
  def transform(self, X, y=None, **kargs):
    """
    Calculate Connectivity
    INPUT
    ------
      1. X: (5D array) shape (trials, channels, window_time_samples, number_of_windows, frequency_bands)
      2. y: None
      3. **kargs
    OUTPUT
    ------
      1. C_xx_w_f: (2D array) (trials, channels*(channels-1)*0.5, number_of_windows, frequency_bands)
    """
    pass

# =============================================================================
# Power-based Connectivities
# =============================================================================

class Power_based_Connectivities(BaseEstimator, TransformerMixin, Connectivities):
  """
  Power based Connectivities

  Parameters
  ----------

  Methods
  -------
    1. flow_of_connectivities(X, mode)
    2. plot_connectivities(c_xx, info, channels_names, used_channels_names=None, mode='functional',
                          n_vwt = None, n_f_bank=None, mean_by_time_windows=False, mean_by_frequencies=False,
                          min_max_normalization_type=None,
                          thr=99,
                          fig_title='',  w_label=None, fb_label=None, labels_params = {'ylabel_fonfamily':'serif', 'ylabel_fontsize':18, 'ylabel_weight':1000, 'xlabel_fonfamily':'serif', 'xlabel_fontsize':18, 'xlabel_weight':500, 'rotation':0, 'cl_size':16},
                          plot_channels=True, relevant_channels=True, channel_importance=True, plot_channels_names=True, show_connectivity_colorbar=True, show_topomap_colorbar=True,
                          figsize=(30,30), cmap_connectivities='hot', cmap_tplt='bone',
                          show=True, save=False, path='', format='png')
    3. fit(X, y=None, **kargs)
    4. transform(X, y=None, **kargs)
  """
  def __init__(self):
    Connectivities.__init__(self)
  
# ------------------------------------------------------------------------------
  def _pearson(self, X):
    """
    Calculate pearson correlation coefficient among envelope of EEG channels.
    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1.  C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    C_xy = np.abs(1 - pdist(X, metric='correlation'))
    return C_xy

# ------------------------------------------------------------------------------
  def _spearman(self, X):
    """
    Calculate spearman correlation coefficient among envelope of EEG channels.
    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    utri_ind =  np.triu_indices(X.shape[0], 1)
    C_xy = np.abs(spearmanr(X, axis=1)[0][utri_ind])
    return C_xy

# ------------------------------------------------------------------------------
  def _kendall(self, X):
    """
    Calculate kendall correlation coefficient among envelope of EEG channels.
    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """

    def kendall_tau_b(x, y):
        return kendalltau(x, y)[0]
    
    C_xy = np.abs(pdist(X, metric=kendall_tau_b))
    return C_xy
# ------------------------------------------------------------------------------
  def _gfc(self, X, gammad):
    """
    Calculate kernel dot among envelope of EEG channels.
    gfc = e^{-(||x - x'||_{2}^{2})/(2*gammad*sigma^2)}

    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
      2. gammad: (float+) band width factor. default=1
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    dist = pdist(X, metric='euclidean')
    sigma = np.median(dist)
    C_xy = np.exp(-1*(dist**2)/(2*gammad*sigma**2))
    return C_xy

# ------------------------------------------------------------------------------
  def fit(self, X, y=None, **kargs):
    ''''''''
    pass

# ------------------------------------------------------------------------------
  def transform(self, X, y=None, **kargs):
    """
    **kargs:  
        1. type_con: (str) type of power based connectivity
                       {'pearson', 'spearman', 'kendall', 'gfc'}
        2. gammad: (float+) for gfc band width factor. default=1
    """
    if kargs['type_con'] == 'pearson':
      fn = self._pearson
    elif kargs['type_con'] == 'spearman':
      fn = self._spearman
    elif kargs['type_con'] == 'kendall':
      fn = self._kendall
    elif kargs['type_con'] == 'gfc':
      if kargs['gammad'] <= 0:
        raise ValueError("band width factor has to be greater than zero")
      fn = lambda X: self._gfc(X, gammad=kargs['gammad'])
    else:
      raise ValueError('No valid connectivity type')
    
    C_xx_w_f = np.zeros((X.shape[0], int(0.5*X.shape[1]*(X.shape[1]-1)), X.shape[3], X.shape[4]))

    for w in range(X.shape[3]):
      for f in range(X.shape[4]):
        C_xx_w_f[:,:,w,f] = np.array(Parallel(n_jobs=-1, verbose=0)(delayed(fn)(X[n,:,:,w,f]) for n in range(X.shape[0])))
    return C_xx_w_f

# =============================================================================
# Phase-based Connectivities
# =============================================================================

class Phase_based_Connectivities(BaseEstimator, TransformerMixin, Connectivities):
  """
  Phase based Connectivities

  Parameters
  ----------

  Methods
  -------
    1. flow_of_connectivities(X, mode)
    2. plot_connectivities(c_xx, info, channels_names, used_channels_names=None, mode='functional',
                          n_vwt = None, n_f_bank=None, mean_by_time_windows=False, mean_by_frequencies=False,
                          min_max_normalization_type=None,
                          thr=99,
                          fig_title='',  w_label=None, fb_label=None, labels_params = {'ylabel_fonfamily':'serif', 'ylabel_fontsize':18, 'ylabel_weight':1000, 'xlabel_fonfamily':'serif', 'xlabel_fontsize':18, 'xlabel_weight':500, 'rotation':0, 'cl_size':16},
                          plot_channels=True, relevant_channels=True, channel_importance=True, plot_channels_names=True, show_connectivity_colorbar=True, show_topomap_colorbar=True,
                          figsize=(30,30), cmap_connectivities='hot', cmap_tplt='bone',
                          show=True, save=False, path='', format='png')
    3. fit(X, y=None, **kargs)
    4. transform(X, y=None, **kargs)
  """
  def __init__(self):
    Connectivities.__init__(self)

# ------------------------------------------------------------------------------
  def _coherence(self, X):
    """
    Calculate coherence among channels of an EEG signal.
             | E[Sxy] |
    coh = ---------------------
        sqrt(E[Sxx] * E[Syy])

    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    def coh(x, y):
      s_xy = x*np.conj(y) #chapter 26 page 345 Analyzing Neural Time Series Data Mike X Cohen
      s_xx, s_yy = np.abs(x)**2, np.abs(y)**2
      num = np.abs(np.mean(s_xy))
      den = np.sqrt(np.mean(s_xx)*np.mean(s_yy))
      c_xy = num/den
      return c_xy

    C_xy = pdist(X, metric=coh)
    return C_xy

# ------------------------------------------------------------------------------
  def _imaginary_coherence(self, X):
    """
    Calculate imaginary coherence among channels of an EEG signal.
              Im(E[Sxy])
    icoh = ----------------------
        sqrt(E[Sxx] * E[Syy])
        
    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    def icoh(x, y):
      s_xy = x*np.conj(y)
      s_xx, s_yy = np.abs(x)**2, np.abs(y)**2
      num = np.imag(np.mean(s_xy))
      den = np.sqrt(np.mean(s_xx)*np.mean(s_yy))
      c_xy = num/den
      return c_xy

    C_xy = np.abs(pdist(X, metric=icoh))
    return C_xy

# ------------------------------------------------------------------------------
  def _phase_locking_value(self, X):
    """
    Calculate phase locking value among channels of an EEG signal.
    plv = |E[Sxy/|Sxy|]|
        
    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    def plv(x, y):
      s_xy = x*np.conj(y)
      c_xy = np.abs(np.mean(s_xy/np.abs(s_xy)))
      return c_xy

    C_xy = pdist(X, metric=plv)
    return C_xy

# ------------------------------------------------------------------------------
  def _corrected_imaginary_plv(self, X):
    """
    Calculate corrected imaginary plv among channels of an EEG signal.
                    |E[Im(Sxy/|Sxy|)]|
    ciplv = ------------------------------------
            sqrt(1 - |E[real(Sxy/|Sxy|)]| ** 2)

    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    def ciplv(x, y):
      s_xy = x*np.conj(y)
      num = np.abs(np.mean(np.imag(s_xy/np.abs(s_xy))))
      den = np.sqrt(1 - np.abs(np.mean(np.real(s_xy/np.abs(s_xy))))**2)
      c_xy = num/den
      return c_xy

    C_xy = pdist(X, metric=ciplv)
    return C_xy

# ------------------------------------------------------------------------------
  def _phase_lag_index(self, X):
    """
    Calculate phase_ lag index among channels of an EEG signal.
    pli = |E[sign(Im(Sxy))]|

    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    def pli(x, y):
      s_xy = x*np.conj(y)
      c_xy = np.abs(np.mean(np.sign(np.imag(s_xy))))
      return c_xy

    C_xy = pdist(X, metric=pli)
    return C_xy

# ------------------------------------------------------------------------------
  def _weighted_phase_lag_index(self, X):
    """
    Calculate weighted phase lag index among channels of an EEG signal.
              |E[Im(Sxy)]|
    wpli = ------------------
              E[|Im(Sxy)|]

    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    def wpli(x, y):
      s_xy = x*np.conj(y)
      num = np.abs(np.mean(np.imag(s_xy)))
      den = np.mean(np.abs(np.imag(s_xy)))
      c_xy = num/den
      return c_xy

    C_xy = pdist(X, metric=wpli)
    return C_xy

# ------------------------------------------------------------------------------
  def fit(self, X, y=None, **kargs):
    ''''''''
    pass

# ------------------------------------------------------------------------------
  def transform(self, X, y=None, **kargs):
    """
    **kargs:  
        1. type_con: (str) type of phase based connectivity
                       {'coh', 'icoh', 'plv', 'ciplv', 'pli', 'wpli'}
    """
    if kargs['type_con'] == 'coh':
      fn = self._coherence
    elif kargs['type_con'] == 'icoh':
      fn = self._imaginary_coherence
    elif kargs['type_con'] == 'plv':
      fn = self._phase_locking_value
    elif kargs['type_con'] == 'ciplv':
      fn = self._corrected_imaginary_plv
    elif kargs['type_con'] == 'pli':
      fn = self._phase_lag_index
    elif kargs['type_con'] == 'wpli':
      fn = self._weighted_phase_lag_index
    else:
      raise ValueError('No valid connectivity type')
    
    X = hilbert(X, axis=2) #Analytical time series
    C_xx_w_f = np.zeros((X.shape[0], int(0.5*X.shape[1]*(X.shape[1]-1)), X.shape[3], X.shape[4]))

    for w in range(X.shape[3]):
      for f in range(X.shape[4]):
        C_xx_w_f[:,:,w,f] = np.array(Parallel(n_jobs=-1, verbose=0)(delayed(fn)(X[n,:,:,w,f]) for n in range(X.shape[0])))
    return C_xx_w_f

# =============================================================================
# Information-based Connectivities
# =============================================================================

class MotifSynchronization(BaseEstimator, TransformerMixin, Connectivities):
  """
  Calculate Motif synchronization Connectivity of EEG signals.

  Parameters
  ----------
    1. tau: (int) time delay embedding
    2. dim: (int) optimal embedding dimension
  Methods
  -------
    1. flow_of_connectivities(X, mode)
    2. plot_connectivities(c_xx, info, channels_names, used_channels_names=None, mode='functional',
                          n_vwt = None, n_f_bank=None, mean_by_time_windows=False, mean_by_frequencies=False,
                          min_max_normalization_type=None,
                          thr=99,
                          fig_title='',  w_label=None, fb_label=None, labels_params = {'ylabel_fonfamily':'serif', 'ylabel_fontsize':18, 'ylabel_weight':1000, 'xlabel_fonfamily':'serif', 'xlabel_fontsize':18, 'xlabel_weight':500, 'rotation':0, 'cl_size':16},
                          plot_channels=True, relevant_channels=True, channel_importance=True, plot_channels_names=True, show_connectivity_colorbar=True, show_topomap_colorbar=True,
                          figsize=(30,30), cmap_connectivities='hot', cmap_tplt='bone',
                          show=True, save=False, path='', format='png')
    3. fit(X, y=None, **kargs)
    4. transform(X, y=None, **kargs)
  """
  def __init__(self, tau=1, dim=3):
    Connectivities.__init__(self)
    self.tau = tau
    self.dim = dim

  def _motif_representation(self, X):
    """
    Motif Representation of EEG signals.
    INPUT
    -----
      1. X: (3D array) set of EEG signals, shape (trials, channels, time_samples)
    OUTPUT
    ------
      1. X_motif: (3D array) set of motif EEG signals, shape (trials, channels, (time_samples - (dim-1)*tau))
    """
    motifs = np.array(list(permutations(np.arange(self.dim, dtype=np.ushort), int(self.dim))), dtype=np.ushort)
    motifs_order = np.argsort(motifs, axis=1)
    X_emb = np.zeros((X.shape[0], X.shape[1], (X.shape[-1] - (self.dim-1)*self.tau), self.dim))
    for n in np.arange(X.shape[0]):
      for ch in np.arange(X.shape[1]):
        X_emb[n, ch, :, :] = takens_delay_embedding(X[n, ch, :], tau=self.tau, dim=self.dim)
    X_emb_order = np.argsort(X_emb.reshape(-1, self.dim), axis=1)
    X_motif = np.zeros(X_emb_order.shape[0])
    for motif in np.arange(motifs_order.shape[0]):
      X_motif[np.array((X_emb_order == motifs_order[motif]).prod(axis=1),dtype=np.bool)] = motif
    return X_motif.reshape(X.shape[0], X.shape[1], (X.shape[-1] - (self.dim-1)*self.tau))
  
  def _mtconnectivity(self, X):
    """
    Calculate coincidence probability among channels of an EEG signal based in motif representation.
    INPUT
    -----
      1. X: (2D array) EEG signal, shape (channels, time_samples)
    OUTPUT
    ------
      1. C_xy: (1D array) of shape (0.5*channels*(channels-1))
    """
    C_xy = 1 - pdist(X, metric='hamming')
    return C_xy

  def fit(self, X, y=None, **kargs):
    ''''''''
    pass


  def transform(self, X, y=None, **kargs):
    ''''''''
    C_xx_w_f = np.zeros((X.shape[0], int(0.5*X.shape[1]*(X.shape[1]-1)), X.shape[3], X.shape[4]))
    for w in range(X.shape[3]):
      for f in range(X.shape[4]):
        X_mt = self._motif_representation(X[:,:,:,w,f])
        C_xx_w_f[:,:,w,f] = np.array(Parallel(n_jobs=-1, verbose=0)(delayed(self._mtconnectivity)(X_mt[n,:,:]) for n in np.arange(X.shape[0])))
    return C_xx_w_f

# =============================================================================
# Topoplots
# =============================================================================

class Topoplot():
  """
  Class for getting topoplots.
  Parameters
  ----------

  Methods
  ---------
    1. get_topoplot(X, info, axis=1, cmap_tplt='gray',
                    resolution=40, interpolation_method=cv2.INTER_LINEAR, format='png',
                    show=False, save_orig_tplt=False)
    2. save_topoplot(X, info, axis=1, cmap_tplt='gray',
                    path='./', format='png', show=False)
  """
  def __init__(self):
      pass

# ------------------------------------------------------------------------------
  def _crop_image(self, filename, pixel_value):
    """
    crop rows and columns with the same value in each of it's elements.
    INPUT
    -----
    1. filename: (str) file path of the image.
    2. pixel_value: (int) element to search in row or column to remove.
    OUTPUT
    ------
    1. cropped_image (2D array) cropped image in gray scale.
    """
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    crop_rows = img_gray[~np.all(img_gray == pixel_value, axis=1), :] #remove rows which elements are all zero
    cropped_image = crop_rows[:, ~np.all(crop_rows == pixel_value, axis=0)] #remove columns which elements are all zero

    return cropped_image

# ------------------------------------------------------------------------------
  def _resize_img(self, img, size, interpolation_method):
    """
    Resize image.
    INPUT
    -----
    1. img: (2D array) Input image.
    2. size: (tuple (x,y)) desired size for the output image.
    3. interpolation_method: flag that takes one of the following methods. INTER_NEAREST – a nearest-neighbor interpolation INTER_LINEAR – 
                            a bilinear interpolation (used by default) INTER_AREA – resampling using pixel area relation. 
                            It may be a preferred method for image decimation, as it gives moire’-free results.
                            But when the image is zoomed, it is similar to the INTER_NEAREST method. 
                            INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood
    OUTPUT
    ------
    1. resize_img: (2D array) resized image.
    """
    return cv2.resize(img, size, interpolation=interpolation_method)

# ------------------------------------------------------------------------------
  def get_topoplot(self, X, info, axis=1, cmap_tplt='gray', resolution=40, interpolation_method=cv2.INTER_LINEAR, format='png', show=False, save_orig_tplt=False):
    """
    get topoplots in a numpy array for a given representation.
    INPUT
    ------
    1. X: (4D array) shape (trials, channels, number_of_windows, frequency_bands)
    2. info: (oject) 
        mne info object.
    3. axis: (None or int or ist), defualt=1
        axis over which to compute min-max normalization
    4. cmap_tplt: (str), default=gray
        color map for topoplots.
    5. resolution: (int), default=40
        desired resolution of topoplots.
    6. interpolation_method (), default=cv2.INTER_LINEAR
        interpolation method to use for rezising the image to the
        desired resolution.
    7. format: (str), default='png'
        format to save  topoplots images.
    8. show: (bool), default=False
        show topolots images.
    9. save_orig_tplt: (bool), default=False
          save topoplots like images with the specified format.
    OUTPUT
    1. X_tplt: (6D array) shape (trials, resolution, resolution, 1, number_of_windows, frequency_band)
    """
    X = min_max_normalization(X, axis=axis)
    
    cmap = plt.get_cmap(cmap_tplt)

    parent_dir_temp = 'original_tplts'
    try:
      os.mkdir(parent_dir_temp)
    except FileExistsError:
      shutil.rmtree(parent_dir_temp)
      os.mkdir(parent_dir_temp)

    X_tplt = np.zeros((X.shape[0], resolution, resolution, 1, X.shape[2], X.shape[3]), dtype=np.int)
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    
    for w in tqdm(range(X.shape[2])):
      window_dir_temp = os.path.join(parent_dir_temp, 'W'+str(w+1)+'/')
      os.mkdir(window_dir_temp)
      for fb in range(X.shape[3]):
        f_bank_dir_temp = os.path.join(window_dir_temp,'FB'+str(fb+1)+'/')
        os.mkdir(f_bank_dir_temp)
        for n in range(X.shape[0]):          
          plot_topomap(X[n,:,w,fb], info, vmin=0, vmax=1, cmap=cmap, contours=0, sensors=False, axes=axs,
                      show=show)
          fig.savefig(f_bank_dir_temp+'trial_'+str(n)+'.'+format, format=format, facecolor='w', bbox_inches = 'tight') 
          if not show:
            plt.close()
          axs.cla()
          
          crop_tplt = self._crop_image(f_bank_dir_temp+'trial_'+str(n)+'.'+format, pixel_value=255)
          rez_tplt = self._resize_img(crop_tplt, size=(resolution, resolution), interpolation_method=interpolation_method)
          X_tplt[n,:,:,:,w,fb] = rez_tplt[..., np.newaxis]
          
    if not save_orig_tplt: 
      shutil.rmtree(parent_dir_temp)

    return X_tplt

# ------------------------------------------------------------------------------
  def save_topoplot(self, X, info, axis=1, cmap_tplt='gray',  path='./', format='png', show=False):
    """
    Save topoplots like images for a given representation.
    INPUT
    ------
    1. X: (4D array) shape (trials, channels, number_of_windows, frequency_bands)
    2. info: (oject) 
        mne info object.
    3. axis: (None or int or ist), defualt=1
        axis over which to compute min-max normalization
    4. cmap_tplt: (str), default=gray
        color map for topoplots.
    5. path: (str), default='./sbj'
        path to save  topoplots.
    6. format: (str), default='png'
        format to save  topoplots.
    7. show: (bool), default=False
        show topolots.
    OUTPUT
    1. directory: (dir) directory with topoplots images in the specified format.   
    """
    X = min_max_normalization(X, axis=axis)
    
    cmap = plt.get_cmap(cmap_tplt)
    
    parent_dir_temp = 'original_tplts'
    try:
      os.mkdir(path)
    except FileExistsError:
      shutil.rmtree(path)
      os.mkdir(path)
      
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    for w in tqdm(range(X.shape[2])):
      window_dir_temp = os.path.join(parent_dir_temp, 'W'+str(w+1)+'/')
      os.mkdir(window_dir_temp)
      for fb in range(X.shape[3]):
        f_bank_dir_temp = os.path.join(window_dir_temp,'FB'+str(fb+1)+'/')
        os.mkdir(f_bank_dir_temp)
        for n in range(X.shape[0]):          
          plot_topomap(X[n,:,w,fb], info, vmin=0, vmax=1, cmap=cmap, contours=0, sensors=False, axes=axs,
                      show=show)
          fig.savefig(f_bank_dir_temp+'trial_'+str(n)+'.'+format, format=format, facecolor='w', bbox_inches = 'tight')  
          if not show:
            plt.close()
          axs.cla()
