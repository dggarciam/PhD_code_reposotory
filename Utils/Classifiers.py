#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Mateo Tobón Henao <mtobonh@unal.edu.co>
Created on Fri  29/04/2022
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.random import set_seed
from keras.backend import clear_session
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Flatten, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# WDCNN
# =============================================================================

class WDCNN(BaseEstimator, ClassifierMixin):
  def __init__(self, hidden_units=100, l1_l2=0, rate=0, output_units=2, act_func_output='softmax',
               learning_rate=None, batch_size=None, epochs=None, verbose=0, validation_split=0, sample=1):
    """
    Wide and Deep Convolutional Neuronal Networ (WDCNN) Classifier.
    PARAMETERS
    ----------
      1. hidden_units: (int) Number of units in hidden layer, default=100.
      2. l1_l2: (float) lasso and ridge regularizer, default=0.
      3. rate: (float) Float between 0 and 1. Fraction of the input units to drop, default=0.
      4. output_units: (int) Number of output units, default=2.
      5. act_func_output: (str) Activation function output layer, default='softmax',
      6. learning_rate: (float or tensorflow.python.keras.optimizer_v2.learning_rate_schedule object) learning rate, default=tensorflow.python.keras.optimizer_v2.learning_rate_schedule.PolynomialDecay object.
      7. batch_size: (int) batch size, default=Number of samples.
      8. epochs: (int) epochs, default=200.
      9. verbose: (int) verbose, default=0.
      10. validation_split: (float) Float between 0 and 1. Percent for validation set, default=0.
      11. sample: (int) Integer greater than zero. Number of times to predict in the model for the monte carlo dropout, default=1.

    ATTRIBUTES
    ----------
      1. model: (keras model) WDCNN model.
      2. history: (history object) Training model data.
  
    METHODS
    -------
      1. fit(X, y)
      2. evaluate(X, y, verbose=True)
      3. predict_proba(X)
      4. predict(X)
      5. get_model(path)
      6. set_model(path)
      7. get_params(deep=True)
      8. set_params()
      9. plot_training(save=False, path='', format='png')
      10. plot_model_diagram(save=False, path='', format='png')
      INPUTS
      ------
        1. X (4D array): Input Data of shape (Trials, resolution, resolution, 1, features).
        2. y (1D array): Labels for Input Data of shape (Trials)

      Returns
      -------
        1. self
    """
    self.hidden_units = hidden_units
    self.l1_l2 = l1_l2
    self.rate = rate
    self.output_units = output_units
    self.act_func_output = act_func_output
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.epochs = epochs
    self.verbose = verbose
    self.validation_split = validation_split
    self.sample = sample
      
  def _data_to_wdcnn_format(self, X, y):
    """
    Transform input data to the wdcnn input format
    """
    X_wdcnn = []
    for i in np.arange(X.shape[-1]):
      X_wdcnn.append(X[:,:,:,:,i] - self.mean[:,:,:,i])
    return X_wdcnn,y

  def fit(self, X, y):
    if self.validation_split == 0:
      self.mean = X.mean(axis=0)
      X_train, y_train = self._data_to_wdcnn_format(X, y)
    elif self.validation_split > 0 and self.validation_split < 1:
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, random_state=0, stratify=y)
      self.mean = X_train.mean(axis=0)
      X_train, y_train = self._data_to_wdcnn_format(X_train, y_train)
      X_val, y_val = self._data_to_wdcnn_format(X_val, y_val)
    else:
      raise ValueError("Validation split value has to be greater than zero and less than one")

    if self.batch_size is None:
      self.batch_size = X_train[0].shape[0]

    if self.epochs is None:
      self.epochs=200

    if self.learning_rate is None:
      self.learning_rate = PolynomialDecay(
      initial_learning_rate=0.5e-2,
      decay_steps=self.epochs,
      end_learning_rate=0.5e-5,
      power=1.0,
      cycle=False,
      name=None,
      )
    
    if self.rate == 0 and self.sample != 1:
      raise ValueError('Monte Carlo Dropout requires a rate in dropout layers')

    clear_session()
    np.random.seed(0)
    set_seed(0)

    input_ = [Input(shape=X_train[0].shape[1:]) for i in np.arange(len(X_train))]
    conv1_ =  [Conv2D(filters=2, kernel_size=5, strides=1, padding='same', activation='relu', kernel_initializer='GlorotNormal')(i) for i in input_]
    batchconv1_ = [BatchNormalization()(i) for i in conv1_]
    pool1_ =  [MaxPooling2D(pool_size=2)(i) for i in batchconv1_]
    conv2_ =  [Conv2D(filters=2, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='GlorotNormal')(i) for i in pool1_]
    batchconv2_ = [BatchNormalization()(i) for i in conv2_]
    pool2_ =  [MaxPooling2D(pool_size=2)(i) for i in batchconv2_]
    concat_ = concatenate(inputs=pool2_)
    flat_ = Flatten()(concat_)
    drop1_ = Dropout(rate=self.rate)(flat_)
    batchn1_ = BatchNormalization()(drop1_)
    hidden1_ = Dense(units=self.hidden_units, activation='relu', kernel_initializer='GlorotNormal', kernel_regularizer=l1_l2(l1=self.l1_l2, l2=self.l1_l2), kernel_constraint=max_norm(max_value=1))(batchn1_)
    drop2_ = Dropout(rate=self.rate)(hidden1_)
    batchn2_ = BatchNormalization()(drop2_)
    output_ = Dense(units=self.output_units, activation=self.act_func_output, kernel_initializer='GlorotNormal', kernel_regularizer=l1_l2(l1=self.l1_l2, l2=self.l1_l2), kernel_constraint=max_norm(max_value=1))(batchn2_)
    self.model = Model(inputs=input_, outputs=output_)

    optimizer = Adam(learning_rate=self.learning_rate)
    self.model.compile(loss=sparse_categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    if self.validation_split == 0:
      self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
    else:
      self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=(X_val, y_val))

    return self

  def evaluate(self, X, y, verbose=True):
    X,y = self._data_to_wdcnn_format(X,y)
    loss, acc = self.model.evaluate(X, y, verbose=0)
    if verbose:
      print('Loss = ',np.round(loss, 2))
      print('Accuracy = ',np.round(acc*100, 1))
    return loss, acc

  def predict_proba(self, X):
    X,_ = self._data_to_wdcnn_format(X,None)
    if self.sample == 1:
      y_pred = self.model.predict(X)
    else:
      y_pred = np.stack([self.model(X, training=True) for i in range(self.sample)])
      y_pred = y_pred.mean(axis=0)
    return y_pred

  def predict(self, X):
    return np.argmax(self.predict_proba(X), axis=1)

  def get_model(self, path):
    self.model.save(path)
    return

  def set_model(self, path):
    self.model = load_model(path)
    return
    
  def get_params(self, deep=True):
    return {'hidden_units':self.hidden_units, 'l1_l2':self.l1_l2, 'rate':self.rate,
            'output_units':self.output_units, 'act_func_output':self.act_func_output,
            'learning_rate':self.learning_rate, 'batch_size':self.batch_size,
            'epochs':self.epochs, 'verbose':self.verbose, 'validation_split':self.validation_split,
            'sample':self.sample}

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)

    return self

  def plot_training(self, save=False, path='', format='png'):
    """
    Plot training loss and accuracy.
    INPUT:
      1. save:(bool) save or not like an image, default=False.
      2. path:(str) path to save the image, default=''.
      3. format: (str) format of the image, default='png'.
    OUTPUT:
    1. (None)
    """
    fig, ax1 = plt.subplots(1,1, figsize=(6,4))
    ax1.plot(np.array(self.history.history['loss']), 'g', linewidth=2 )
    ax1.set_title('Training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Categorical Crossentropy')
    ax2 = ax1.twinx()
    ax2.plot(np.array(self.history.history['accuracy'])*100, 'r', linewidth=2)
    ax2.set_ylabel('Accuracy')
    if self.validation_split != 0:
      ax1.plot(np.array(self.history.history['val_loss']), 'y--', linewidth=2 )
      ax2.plot(np.array(self.history.history['val_accuracy'])*100, 'b--', linewidth=2)
      ax1.legend(['loss', 'val_loss'], loc=2)
      ax2.legend(['acc', 'val_acc'], loc=1)
    fig.show()
    if save:
      fig.savefig(path+'.'+format,format=format, bbox_inches='tight')

    return

  def plot_model_diagram(self, save=False, path='', format='png'):
    """
    Plot model architecture.
    INPUT
    -----
      1. save:(bool) save or not the model architecture like an image, default=False.
      2. path:(str) path to save the image, default=''.
      3. format: (str) format of the image, default='png'.
    OUTPUT
    ------
      1. (None)
    """
    if save:
      plot_model(self.model, to_file=path+'.'+format, show_shapes=True, show_layer_names=True, rankdir='LR')
    else:
      plot_model(self.model, show_shapes=True, show_layer_names=True, rankdir='LR')

    return