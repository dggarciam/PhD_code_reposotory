#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Mateo Tobón Henao <mtobonh@unal.edu.co>
Created on Fri  29/04/2022
"""

from scipy.signal import butter, filtfilt
import numpy as np
from scipy.stats import  pearsonr, zscore
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import FastICA
from mne import EpochsArray
from mne.preprocessing import compute_current_source_density
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# Digital Filters
# =============================================================================

def butterworth_digital_filter(X, N, Wn, btype, fs, axis=-1, padtype=None, padlen=0, method='pad', irlen=None):
  """
  Apply digital butterworth filter
  INPUT
  ------
  1. X: (D array)
    array with signals.
  2. N: (int+)
    The order of the filter.
  3. Wn: (float+ or 1D array)
    The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for bandpass and bandstop filters, Wn is a length-2 vector.
    For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
    If fs is not specified, Wn units are normalized from 0 to 1, where 1 is the Nyquist frequency (Wn is thus in half cycles / sample and defined as 2*critical frequencies / fs). If fs is specified, Wn is in the same units as fs.
  4. btype: (str) {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    The type of filter
  5. fs: (float+)
    The sampling frequency of the digital system.
  6. axis: (int), Default=1.
    The axis of x to which the filter is applied.
  7. padtype: (str) or None, {'odd', 'even', 'constant'}
    This determines the type of extension to use for the padded signal to which the filter is applied. If padtype is None, no padding is used. The default is ‘odd’.
  8. padlen: (int+) or None, Default=0
    The number of elements by which to extend x at both ends of axis before applying the filter. This value must be less than x.shape[axis] - 1. padlen=0 implies no padding.
  9. method: (str), {'pad', 'gust'}
    Determines the method for handling the edges of the signal, either “pad” or “gust”. When method is “pad”, the signal is padded; the type of padding is determined by padtype 
    and padlen, and irlen is ignored. When method is “gust”, Gustafsson’s method is used, and padtype and padlen are ignored.
  10. irlen: (int) or None, Default=nONE
    When method is “gust”, irlen specifies the length of the impulse response of the filter. If irlen is None, no part of the impulse response is ignored.
    For a long signal, specifying irlen can significantly improve the performance of the filter.
  OUTPUT
  ------
  X_fil: (D array)
    array with filtered signals.
  """
  b, a = butter(N, Wn, btype, analog=False, output='ba', fs=fs)
  return filtfilt(b, a, X, axis=axis, padtype=padtype, padlen=padlen, method=method, irlen=irlen)

# =============================================================================
# ICA-based Artifact Removal
# =============================================================================

# Authors: Denis Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)
def find_outliers(X, threshold=3.0, max_iter=2, tail=0):
  """Find outliers based on iterated Z-scoring.
  This procedure compares the absolute z-score against the threshold.
  After excluding local outliers, the comparison is repeated until no
  local outlier is present any more.
  Parameters
  ----------
  X : np.ndarray of float, shape (n_elemenets,)
      The scores for which to find outliers.
  threshold : float
      The value above which a feature is classified as outlier.
  max_iter : int
      The maximum number of iterations.
  tail : {0, 1, -1}
      Whether to search for outliers on both extremes of the z-scores (0),
      or on just the positive (1) or negative (-1) side.
  Returns
  -------
  bad_idx : np.ndarray of int, shape (n_features)
      The outlier indices.
  """
  my_mask = np.zeros(len(X), dtype=bool)
  for _ in range(max_iter):
      X = np.ma.masked_array(X, my_mask)
      if tail == 0:
          this_z = np.abs(zscore(X))
      elif tail == 1:
          this_z = zscore(X)
      elif tail == -1:
          this_z = -zscore(X)
      else:
          raise ValueError("Tail parameter %s not recognised." % tail)
      local_bad = this_z > threshold
      my_mask = np.max([my_mask, local_bad], 0)
      if not np.any(local_bad):
          break

  bad_idx = np.where(my_mask)[0]
  return bad_idx


@ignore_warnings(category=ConvergenceWarning)
def ica_artiact_removal(X, A, seed=23):
  """
  INPUT
  -------
  1. X: (3D array) shape (trials, channels, times)
    set of EEG signals
  2. A: (3D array) shape (trials, artifactual_channels, times)
    set of artifactual channels
  3. seed: (int+)
    seed for FastICA algorithm
  OUTPUT
  ------
  1. X_clean: (3D array) shape (trials, channels, times)
    set of clean EEG signals
  2. idx_noise_trials: (1D Array)
    index of artifactual EEG signals
  3. W_unmixing: list, len=trials
     Unmixing matrix for clean EEG signals, coefficients of noise sources for noise trials.
  4. rho: (3D array) (trials, channels, artifactual_channels)
     pearson correlation coefficeint among sources and artifactual channels
  """
  n_trials, n_sources = X.shape[:2]
  n_artifacts  = A.shape[1]
  rho = np.zeros((n_trials, n_sources, n_artifacts))
  X_clean = X.copy()
  idx_noise_trials = []
  W_unmixing = []

  for n in range(n_trials):
    #Sources by FastICA algorithm
    ica = FastICA(n_components=n_sources, algorithm='parallel', whiten=True, fun='exp', fun_args=None, max_iter=200, tol=1e-3, w_init=None, random_state=seed)
    S = ica.fit_transform(X[n].T)
    S = S.T

    #pearson correlation coefficient among sources and artifacts
    for s_idx in range(n_sources):
      for a_idx in range(n_artifacts):
        rho[n, s_idx, a_idx] = pearsonr(S[s_idx], A[n][a_idx])[0]

    #Find outliers based on iterated Z-scoring
    idx_art_S = np.unique(np.concatenate([find_outliers(rho[n,:,a_idx], threshold=3.0, max_iter=2, tail=0)
                                        for a_idx in range(n_artifacts)]))

    if idx_art_S.size != 0:
      idx_noise_trials.append(n)
      w_noise_S = ica._unmixing[idx_art_S] 
      w_noise_S = w_noise_S[np.newaxis, ...] if w_noise_S.ndim == 1 else w_noise_S
      W_unmixing.append(w_noise_S)  
      S[idx_art_S, :] = 0 #Remove sources related to artifacts
      X_clean[n] = ica.mixing_.dot(S) #EEG signal free of artifacts and zero-mean
    else:
      X_clean[n] = X[n]
      W_unmixing.append(ica._unmixing)

  return X_clean, np.array(idx_noise_trials), W_unmixing, rho

# =============================================================================
# Spherical Spline Surface Laplacian
# =============================================================================

def spherical_spline_surface_laplacian(X, info, sphere='auto', lambda2=1e-05, stiffness=4, n_legendre_terms=50): #only applies to eeg channels, select them from epochs object
  """
  References:
      - Perrin, F., Pernier, J., Bertrand, O. & Echallier, J.F. (1989). Spherical splines for scalp 
          potential and current density mapping. Electroencephalography and clinical Neurophysiology, 72, 
          184-187.
      - Cohen, M.X. (2014). Surface Laplacian In Analyzing neural time series data: theory and practice 
          (pp. 275-290). London, England: The MIT Press.
  instinstance of Raw, Epochs or Evoked
  INPUT
  -------
  1. X: (3D array) shape (trials, channels, times)
    set of EEG signals.
  2. info: (mne info)
  3. sphere: (str) or (1D array) shape (4,)
    The sphere, head-model of the form (x, y, z, r) where x, y, z is the center of the sphere and r is the radius in meters. Can also be “auto” to use a digitization-based fit.
  4. lambda2: (float+)
    Regularization parameter, produces smoothness. Defaults to 1e-5.
  5. stiffness: (float+)
    Stiffness of the spline.
  6. n_legendre_terms: (int+)
    Number of Legendre terms to evaluate.
  OUTPUT
  ------
  1. X_sl: (3D array)
    Filtered set of EEG signals.
  """
  EpochsX = EpochsArray(X, info)
  EpochsX_sl = compute_current_source_density(EpochsX, sphere=sphere, lambda2=lambda2, stiffness=stiffness, n_legendre_terms=n_legendre_terms, copy=True)
  return EpochsX_sl.get_data()

# =============================================================================
# Z-score
# =============================================================================

class Standarization(BaseEstimator, TransformerMixin):
  """
  Compute the z score of each value in the sample, relative to the sample mean and standard deviation
  Parameters
  ----------
    1. axis: (int or list or None), Default: None
        axis along which to operate. If None, compute over the whole array.
  Methods
  -------
    1. fit(X, y=None)
    2. transform(X, y=None)
  """
  def __init__(self, axis=None):
    self.axis=axis

# ------------------------------------------------------------------------------
  def fit(self, X, y=None):
    ''''''''
    pass

# ------------------------------------------------------------------------------
  def transform(self, X, y=None):
    """
    INPUT
    -----
    X: (array)
    OUTPUT
    ------
    Z_x: (array)
    """
    Z_x = zscore(X, axis=self.axis)
    return Z_x

# =============================================================================
# Flatten
# =============================================================================
class Flatten(BaseEstimator, TransformerMixin):
  """
  Flatten tensor
  """
  def __init__(self):
    pass

# ------------------------------------------------------------------------------
  def fit(self, X, y=None):
    pass

# ------------------------------------------------------------------------------
  def transform(self, X, y=None):
    return X.reshape(X.shape[0], -1)