#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Mateo Tobón Henao <mtobonh@unal.edu.co> and
Daniel Guillermo García Murillo dggarciam@unal.edu.co
Created on Fri  29/04/2022
"""

from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tempfile
from sklearn.kernel_approximation import  RBFSampler

# =============================================================================
# Lasso
# =============================================================================

class Lasso_feats(TransformerMixin,BaseEstimator):
  def __init__(self, l1=0):
    self.l1=l1
    
  def fit(self, X, y, *_):
    self.lasso = Lasso(l1=self.l1)
    self.lasso.fit(X,y)
    
  def transform(self, X, *_):
    Xr = X[:,np.abs(self.lasso.coef_)> 0]
    return Xr

  def fit_transform(self,X,y,*_):
    self.fit(X,y)
    return self.transform(X)

# =============================================================================
# Elasticnet
# =============================================================================    

class Elastic_net_feats(TransformerMixin,BaseEstimator):
  def __init__(self, l1=0, l2=0):
    self.l1=l1
    self.l2=l2
    
  def fit(self, X, y, *_):
    self.alpha = self.l1 + 2*self.l2
    self.l1_ratio = self.l1/self.alpha
    self.elatic = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=0)
    mlb = OneHotEncoder()
    y = mlb.fit_transform(y.reshape((-1,1))).toarray()
    self.elatic.fit(X,y)

  def transform(self, X, *_):
    if len(self.elatic.coef_.shape)>1:
        Xr = np.dot(X,self.elatic.coef_.T)
    else:
        Xr = np.dot(X,self.elatic.coef_)
        Xr = Xr.reshape((-1,1))
    return Xr

  def fit_transform(self, X, y, *_):
    self.fit(X,y)
    return self.transform(X)

# =============================================================================
# CKA
# =============================================================================
tf.keras.backend.clear_session()
tf.random.set_seed(42)

class Keras_CKA(BaseEstimator, TransformerMixin):
    
  def __init__(self, epochs=200, batch_size=30, Q=0.9, learning_rate=1e-3 ,optimizer='Adam',
      l1_l2=0 ,validation_split=0.2, gamma=1.0, verbose=1, init_='def_keras', gamma_y=1e-13):
    self.epochs = epochs
    self.gamma = gamma
    self.gamma_y = gamma_y
    self.batch_size = batch_size
    self.learning_rate = learning_rate 
    self.l1_l2 = l1_l2
    self.validation_split = validation_split
    self.verbose = verbose
    self.optimizer = optimizer
    self.Q=Q
    self.init_=init_
    
    # Define custom loss
  def custom_cka_loss(self,y_true,y_pred): #ytrue labels, ypred  = Xw
    ####gradiente##########################################
    scalar_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1, length_scale=self.gamma)
    scalar_kernely = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1, length_scale=self.gamma_y)
    k = scalar_kernel.matrix(y_pred, y_pred)
    l = scalar_kernely.matrix(y_true, y_true)
    ######################################################
    N = tf.shape(l)[0]
    N2 = tf.cast(tf.shape(l)[0],dtype=tf.float32)
    h = tf.eye(N) - (1.0/N2)*tf.ones([N,1])*tf.ones([1,N]) #matrix for centered kernel
    trkl = tf.linalg.trace(tf.matmul(tf.matmul(k,h),tf.matmul(l,h)))
    trkk = tf.linalg.trace(tf.matmul(tf.matmul(k,h),tf.matmul(k,h)))
    trll = tf.linalg.trace(tf.matmul(tf.matmul(l,h),tf.matmul(l,h)))
    #####funcion de costo############################################3
    f     = -trkl/tf.sqrt(trkk*trll)# negative cka cost function (minimizing) f \in [-1,0]
    return f
    
  def fit(self,X,Y):
    #input X numpy array first dimension Trials x features1 x features2.
    #input Y numpy array vector len = Trials.
    #if self.alpha_1 is not None and self.l1_ratio is not None:
    #	self.alpha = self.alpha_1*self.l1_ratio
    #	self.l1_ratio = 0.5*self.alpha_1*(1-self.l1_param)
    #self.alpha = self.l1 + 2*self.l2
    #self.l1_ratio = self.l1/self.alpha
    Y = np.array(Y, dtype=np.float)
    if self.optimizer == "Adam":
      opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    elif self.optimizer == "SGD":
      opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
    else:
      opt=self.optimizer
    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.validation_split, random_state=0,stratify=Y)
    scal=StandardScaler()
    if self.init_ == 'pca':
      pca_2 = PCA(n_components=self.Q, random_state = 0,svd_solver = 'arpack')
      pca_2.fit(X_train.reshape((X_train.shape[0],-1)))
      self.init_ = pca_2.components_.T
      self.Q = pca_2.n_components_
    else:
      if self.Q<1:
        xxx = scal.fit_transform(X_train)
        if xxx.shape[0]>xxx.shape[1]:
          Cov = xxx.T@xxx 
        else:
          Cov = xxx@xxx.T
        vr = np.linalg.eigvals(Cov)
        csr = np.cumsum(vr)/sum(vr)
        if csr.min()>self.Q:
          self.Q = 1
        else: 
          self.Q = np.where(csr <= self.Q)[0][-1]+1
    if len(tf.shape(X_train))>2:
      input_layer = tf.keras.layers.Input(shape=[X_train.shape[1],X_train.shape[2]])
      flatten_layer = tf.keras.layers.Flatten(input_shape=(X_train.shape[1],X_train.shape[2]))(input_layer)
      if self.init_ == 'def_keras':
        output_layer = tf.keras.layers.Dense(self.Q, activation="linear", 
          kernel_regularizer=regularizers.l1_l2(l1=self.l1_l2, l2=self.l1_l2), 
          use_bias=False, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0), name= 'Proj')(flatten_layer)
      else:
        output_layer = tf.keras.layers.Dense(self.Q, activation="linear", 
          kernel_regularizer=regularizers.l1_l2(l1=self.l1_l2, l2=self.l1_l2),
          use_bias=False, weights=[self.init_], name= 'Proj')(flatten_layer)
    else:
      input_layer = tf.keras.layers.Input(shape = [X_train.shape[1]])
      if self.init_ == 'def_keras':
        output_layer = tf.keras.layers.Dense(self.Q, activation="linear", 
          kernel_regularizer=regularizers.l1_l2(l1=self.l1_l2, l2=self.l1_l2), 
          use_bias=False, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0), name= 'Proj')(input_layer)
      else:
        output_layer = tf.keras.layers.Dense(self.Q, activation="linear", 
          kernel_regularizer=regularizers.l1_l2(l1=self.l1_l2, l2=self.l1_l2),
          use_bias=False, weights=[self.init_], name= 'Proj')(input_layer)
    self.model = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    self.model.compile(loss=self.custom_cka_loss, optimizer=opt)
    keys = [weight.name for layer in self.model.layers for weight in layer.weights]
    weights_in = self.model.get_weights()
    self.A_in = {}
    for key, weight in zip(keys, weights_in):
      self.A_in[key] = weight 
    self.history = self.model.fit(X_train, y_train, epochs=self.epochs,
          validation_data=(X_test, y_test),batch_size=self.batch_size,
                    verbose=self.verbose)
    weights_out = self.model.get_weights()
    self.A_out = {}
    for key, weight in zip(keys, weights_out):
      self.A_out[key] = weight
    return self
    
  def transform(self, X, *_):
    Xr = self.model.predict(X)
    return  Xr
    
  def fit_transform(self,X,y):
    self.fit(X,y)
    return  self.transform(X)
    
  def plot_history(self):
    plt.plot(self.history.history['loss'],label='loss')
    plt.plot(self.history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.savefig("losscka.eps")
    return
    
  def __getstate__(self):
    model_str = ""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as fd:
      try:
        self.model.save(fd.name,save_format='h5')
        model_str = fd.read()
      except:
        model_str='model no trained'
        self.A_out='model no trained'
        self.A_in='model no trained'
    d = { 'model_str': model_str,
    'A_in':self.A_in,
    'A_out':self.A_out,
    'gamma':self.gamma,
    'gamma_y':self.gamma_y,
    'epochs':self.epochs,
    'batch_size':self.batch_size,
    'learning_rate':self.learning_rate,
    'l1_l2':self.l1_l2,
    'validation_split':self.validation_split,
    'verbose':self.verbose,
    'optimizer':self.optimizer,
    'Q':self.Q,
    'init_':self.init_}
    return d
    
  def __setstate__(self, state):
    self.gamma=state['gamma']
    self.gamma_y=state['gamma_y']
    self.epochs=state['epochs']
    self.batch_size=state['batch_size']
    self.learning_rate=state['learning_rate']
    self.l1_l2=state['l1_l2']
    self.validation_split=state['validation_split']
    self.verbose=state['verbose']
    self.optimizer=state['optimizer']
    self.Q=state['Q']
    self.init_=state['init_']
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as fd:
      try:
        fd.write(state['model_str'])
        fd.flush()
        self.model = tf.keras.models.load_model(fd.name,custom_objects={'custom_cka_loss': self.custom_cka_loss})	
        self.A_in=state['A_in']
        self.A_out=state['A_out']
      except:
        self.mode=state['model_str']
        #print(state['model_str'])
        
# =============================================================================
# Random Feature Fourier
# =============================================================================

class RFF(BaseEstimator, TransformerMixin):
  """
  Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform.
  Parameters
  ----------
  1. f_components: (float) n_features_feature_map = n_original_features*f_components. Default=1.0
  2. gamma: (float) Parameter of RBF kernel: exp(-gamma * x^2). Default=1.0
  Methods
  -------
  1. fit(X, y=None, *_)
  2. transform(X)
  3. fit_transfom(X, y=None, *_)
  """
  def __init__(self, f_components=1, gamma=1):
    self.f_components = f_components
    self.gamma = gamma

  def fit(self, X, y=None, *_):
    self.rff = RBFSampler(n_components=int(X.shape[-1]*self.f_components), gamma=self.gamma, random_state=0).fit(X)
    return self
  
  def transform(self, X):
    return self.rff.transform(X)

  def fit_transfom(self, X, y=None, *_):
    self.fit(X,y)
    return self.transform(X)