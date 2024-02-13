# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:39:21 2019

@author:Usage
Here you can get help of any object by pressing Ctrl+I in front of it, either on the Editor or the Console.
 andre
"""
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt #, freqz
import numpy as np
from mne.io import read_raw_edf
#from mne.decoding import CSP
from MI_EEG_ClassMeth.modCSP import CSP
#from modCSP import CSP
import matplotlib.pyplot as plt
import pandas as pd
import json as  js #conda install -c jmcmurray json
import warnings
import seaborn as sns
import mne
from numpy import matlib
import matplotlib
import os
from matplotlib.animation import FuncAnimation
from ipywidgets import interact
warnings.filterwarnings("ignore")
Sujetos_giga = {}


#%%
def Read_GIGA_data_fCloud(path_filename,ch,vt,sbj_id):
    #--- info ----------------
    # 2  ---> sample rate 
    # 7  ---> imaginary_left 
    # 8  ---> imaginary_right 
    # 11 ---> imaginary_event 
    # 14 ---> bad_trials 
    # class1: left 
    # class2: right
    #------------------------
    raw     = sio.loadmat(path_filename)
    eeg_raw = raw['eeg']
    sfreq   = np.float(eeg_raw[0][0][2])
    id_MI   = np.where(eeg_raw[0][0][11]==1)
    id_MI   = id_MI[1]
    raw_c1  = []
    raw_c2  = []
    y_c1    = []
    y_c2    = []
    for i in range(len(id_MI)):
        l_thr = id_MI[i]-(sfreq*2-1) 
        h_thr = id_MI[i]+(sfreq*5)
        tmp_c1 = eeg_raw[0][0][7][ch,np.int(l_thr):np.int(h_thr)]
        tmp_c2 = eeg_raw[0][0][8][ch,np.int(l_thr):np.int(h_thr)]
        raw_c1.append(tmp_c1[:,np.int(vt[0]*sfreq):np.int(vt[1]*sfreq)])
        raw_c2.append(tmp_c2[:,np.int(vt[0]*sfreq):np.int(vt[1]*sfreq)])
        y_c1.append(1.0)
        y_c2.append(2.0)    
    # remove bad trials
    id_bad_tr_voltage_c1 = eeg_raw[0][0][14][0][0][0][0][0]
    id_bad_tr_voltage_c2 = eeg_raw[0][0][14][0][0][0][0][1]   
    id_bad_tr_mi_c1      = eeg_raw[0][0][14][0][0][1][0][0]
    id_bad_tr_mi_c2      = eeg_raw[0][0][14][0][0][1][0][1]
    ref_axis_c1          = 1
    ref_axis_c2          = 1    
    if id_bad_tr_mi_c1.shape[0]>id_bad_tr_mi_c1.shape[1]:
        id_bad_tr_mi_c1 = id_bad_tr_mi_c1.T
    if id_bad_tr_mi_c2.shape[0]>id_bad_tr_mi_c2.shape[1]:
        id_bad_tr_mi_c2 = id_bad_tr_mi_c2.T
    if id_bad_tr_voltage_c1.shape[1] == 0:
        id_bad_tr_voltage_c1 = np.reshape(id_bad_tr_voltage_c1, (id_bad_tr_voltage_c1.shape[0], id_bad_tr_mi_c1.shape[1]))
    if id_bad_tr_voltage_c2.shape[1] == 0:
        id_bad_tr_voltage_c2 = np.reshape(id_bad_tr_voltage_c2, (id_bad_tr_voltage_c2.shape[0], id_bad_tr_mi_c2.shape[1])) 
    if (id_bad_tr_voltage_c1.shape[1] > id_bad_tr_mi_c1.shape[1]):
        if id_bad_tr_mi_c1.shape[0] == 0:
            id_bad_tr_mi_c1 = np.reshape(id_bad_tr_mi_c1, (id_bad_tr_mi_c1.shape[0],id_bad_tr_voltage_c1.shape[1]))
            ref_axis_c1     = 0
    if (id_bad_tr_voltage_c2.shape[1] > id_bad_tr_mi_c2.shape[1]):
        if id_bad_tr_mi_c2.shape[0] == 0:
            id_bad_tr_mi_c2 = np.reshape(id_bad_tr_mi_c2, (id_bad_tr_mi_c2.shape[0],id_bad_tr_voltage_c2.shape[1]))
            ref_axis_c2     = 0
    if (id_bad_tr_mi_c1.shape[0] > id_bad_tr_voltage_c1.shape[0]):
        ref_axis_c1 = 0
    if (id_bad_tr_mi_c2.shape[0] > id_bad_tr_voltage_c2.shape[0]):
        ref_axis_c2 = 0
    if (id_bad_tr_voltage_c1.shape[0] > id_bad_tr_mi_c1.shape[0]):
        ref_axis_c1 = 0
    if (id_bad_tr_voltage_c2.shape[0] > id_bad_tr_mi_c2.shape[0]):
        ref_axis_c2 = 0    
    id_bad_tr_c1 = np.concatenate((id_bad_tr_voltage_c1,id_bad_tr_mi_c1),axis=ref_axis_c1)
    id_bad_tr_c1 = id_bad_tr_c1.ravel()-1
    for ele in sorted(id_bad_tr_c1, reverse = True):  
        del raw_c1[ele]
        del y_c1[ele]
    id_bad_tr_c2 = np.concatenate((id_bad_tr_voltage_c2,id_bad_tr_mi_c2),axis=ref_axis_c2)
    id_bad_tr_c2= id_bad_tr_c2.ravel()-1
    for ele in sorted(id_bad_tr_c2, reverse = True):  
        del raw_c2[ele]
        del y_c2[ele]     
    Xraw = np.array(raw_c1 + raw_c2)
    y    = np.array(y_c1 + y_c2)  
    return Xraw, y, sfreq
#%%
def leer_bci42a_train_full(path_filename,clases,Ch,vt):
    
    raw = read_raw_edf(path_filename,preload=False)
    sfreq=raw.info['sfreq']
    
    #raw.save('tempraw.fif',overwrite=True)#, tmin=3, tmax=5,overwrite = True)
    #rawo = mne.io.read_raw_fif('tempraw.fif', preload=True)  # load data  
    # depurar canales
    #rawo.plot()
    
    #clases_b = [769,770,771,772] #codigo clases
    i_muestras_   = raw._raw_extras[0]['events'][1]           # Indices de las actividades.
    i_clases_ = raw._raw_extras[0]['events'][2]           # Marcadores de las actividades.
    
    remov   = np.ndarray.tolist(i_clases_)                 # Quitar artefactos.
    Trials_eli = 1023                                   # Elimina los trials con artefactos.
    m       = np.array([i for i,x in enumerate(remov) if x==Trials_eli])   # Identifica en donde se encuentra los artefactos.
    m_      = m+1
    tt      = np.array(raw._raw_extras[0]['events'][0]*[1],dtype=bool)
    tt[m]   = False
    tt[m_]  = False
    i_muestras = i_muestras_[tt] # indices en muestra del inicio estimulo -> tomar 2 seg antes y 5 seg despues
    i_clases = i_clases_[tt] # tipo de clase
    
    #i_muestras = i_muestras_ # indices en muestra del inicio estimulo -> tomar 2 seg antes y 5 seg despues
    #i_clases = i_clases_ # tipo de clase
    
    
    #eli = 1023 
    #ind = i_clases_ != eli
    #i_clases = i_clases_[ind]
    #i_muestras = i_muestras_[ind]
    ni = np.zeros(len(clases))
    for i in range(len(clases)):
        ni[i] = np.sum(i_clases == clases[i]) #izquierda
    
    Xraw = np.zeros((int(np.sum(ni)),len(Ch),int(sfreq*(vt[1]+vt[0]))))
    y = np.zeros(int(np.sum(ni)))
    ii = 0
    for i in range(len(clases)):
        for j in range(len(i_clases)):
            if i_clases[j] == clases[i]:
                rc = raw[:,int(i_muestras[j]-vt[0]*sfreq):int(i_muestras[j]+vt[1]*sfreq)][0]
                rc = rc - np.mean(rc)
                Xraw[ii,:,:] = rc[Ch,:]
                y[ii] = int(i+1)
                ii += 1
    
    return i_muestras, i_clases, raw, Xraw, y, ni, m

#%%
def leer_bci42a_test_full(path_filename,clases,Ch,vt):
    
    raw = read_raw_edf(path_filename,preload=False)
    sfreq=raw.info['sfreq']
    
    #raw.save('tempraw.fif',overwrite=True)#, tmin=3, tmax=5,overwrite = True)
    #rawo = mne.io.read_raw_fif('tempraw.fif', preload=True)  # load data  
    # depurar canales
    #rawo.plot()
    
    #clases_b = [769,770,771,772] #codigo clases
    i_muestras_   = raw._raw_extras[0]['events'][1]           # Indices de las actividades.
    i_clases_ = raw._raw_extras[0]['events'][2]           # Marcadores de las actividades.
    
    #remov   = np.ndarray.tolist(i_clases_)                 # Quitar artefactos.
#    Trials_eli = 1023                                   # Elimina los trials con artefactos.
#    m       = np.array([i for i,x in enumerate(remov) if x==Trials_eli])   # Identifica en donde se encuentra los artefactos.
#    m_      = m+1
#    tt      = np.array(raw._raw_extras[0]['events'][0]*[1],dtype=bool)
#    tt[m]   = False
#    tt[m_]  = False
#    i_muestras = i_muestras_[tt] # indices en muestra del inicio estimulo -> tomar 2 seg antes y 5 seg despues
#    i_clases = i_clases_[tt] # tipo de clase
#    
    i_muestras = i_muestras_ # indices en muestra del inicio estimulo -> tomar 2 seg antes y 5 seg despues
    i_clases = i_clases_ # tipo de clase
    
    
    ni = np.zeros(len(clases))
    for i in range(len(clases)):
        ni[i] = np.sum(i_clases == clases[i]) #izquierda
    
    Xraw = np.zeros((int(np.sum(ni)),len(Ch),int(sfreq*(vt[1]+vt[0]))))
    #y = np.zeros(int(np.sum(ni)))
    ii = 0
    for i in range(len(clases)):
        for j in range(len(i_clases)):
            if i_clases[j] == clases[i]:
                rc = raw[:,int(i_muestras[j]-vt[0]*sfreq):int(i_muestras[j]+vt[1]*sfreq)][0]
                rc = rc - np.mean(rc)
                Xraw[ii,:,:] = rc[Ch,:]
                #y[ii] = int(clases[i])
                ii += 1
    
    return i_muestras, i_clases, raw, Xraw

#%% Filters

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b,a,data)#lfilter(b, a, data)
    return y

#%% Bank filter
def bank_filter_epochsEEG(Xraw, fs, f_frec): #Xraw[nepochs,nchannels]
    nf,ff = f_frec.shape
    epochs,channels,T = Xraw.shape
    Xraw_f = np.zeros((epochs,channels,T,nf))
    for f in range(nf):
        lfc = f_frec[f,0]
        hfc = f_frec[f,1]
        b,a = butter_bandpass(lfc, hfc, fs)
        zi = lfilter_zi(b, a)
        Xraw_f[:,:,:,f] = filtfilt(b,a,Xraw,axis=2)
        #for n in range(epochs):
        #    for c in range(channels):
                #print(c)
        #        zi = lfilter_zi(b, a)
        #        Xraw_f[n,c,:,f] = lfilter(b, a, Xraw[n,c,:],zi = zi*Xraw[n,c,0])[0]
                #Xraw_f[n,c,:,f] = lfilter(b, a, Xraw[n,c,:])
    return Xraw_f

#%% CSP epochs
def CSP_epochsEEG(Xraw, y, ncomp): #Xraw[nepochs,nchannels]
    
    csp = CSP(n_components=ncomp, reg='empirical', log=True, norm_trace=False) 
    epochs,channels,T,nf = Xraw.shape
    Xcsp = np.zeros((epochs,ncomp,nf))
    csp_l = []
    for f in range(nf):
        
        csp_l.append(csp.fit(Xraw[:,:,:,f],y))
        Xcsp[:,:,f] = csp_l[f].transform(Xraw[:,:,:,f])
    
    return csp_l, Xcsp

#%% CSP custom sklearn

#from sklearn.metrics import pairwise_distances  
#from scipy.spatial.distance import squareform 


from sklearn.base import  BaseEstimator, TransformerMixin
class CSP_epochs_filter_extractor(TransformerMixin,BaseEstimator):
    def __init__(self, fs,f_frec=[4,30], ncomp=4,reg='empirical',PCov=False):
        self.reg = reg
        self.fs = fs
        self.PCov=PCov
        self.f_frec = f_frec
        self.ncomp = ncomp
        
    def _averagingEEG(self,X):
        epochs,channels,T = X.shape
        Xc = np.zeros((epochs,channels,T))
        for i in range(epochs):
            Xc[i,:,:] = X[i,:,:] - np.mean(X[i,:,:])
        return Xc    
        
    def _bank_filter_epochsEEG(self,X):
        nf,ff = self.f_frec.shape
        epochs,channels,T = X.shape
        X_f = np.zeros((epochs,channels,T,nf))
        for f in range(nf):
            lfc = self.f_frec[f,0]
            hfc = self.f_frec[f,1]
            b,a = butter_bandpass(lfc, hfc, self.fs)
            X_f[:,:,:,f] = filtfilt(b,a,X,axis=2)
        return X_f  
    def _CSP_epochsEEG(self,Xraw, y,*_):
        ncomp = self.ncomp
        mne.set_log_level('WARNING')
        if self.PCov==True:
            epochs,P,nf = Xraw.shape
            channels = int((1+np.sqrt(1+8*P))/2)
        else:  
            epochs,channels,T,nf = Xraw.shape
        Xcsp = np.zeros((epochs,self.ncomp,nf))
        self.filters  =np.zeros((self.ncomp,channels,nf))
        csp_l = []
        for f in range(nf):
            if self.PCov == True:
                csp_l+= [CSP(n_components=ncomp, reg=self.reg, log=True,transform_into='average_power',PCov = self.PCov).fit(Xraw[:,:,f],y)]
                Xcsp[:,:,f] = csp_l[f].transform(Xraw[:,:,f])
            else:
                csp_l+= [CSP(n_components=ncomp, reg=self.reg, log=True,transform_into='average_power',PCov = self.PCov).fit(Xraw[:,:,:,f],y)]
                Xcsp[:,:,f] = csp_l[f].transform(Xraw[:,:,:,f])
            self.filters[:,:,f] = csp_l[f].filters_[:self.ncomp]
        return csp_l, Xcsp

    def fit(self,Xraw,y, *_):
        if self.PCov == True:
            self.csp_l, self.Xcsp = self._CSP_epochsEEG(Xraw, y)
        else:
            Xraw = self._averagingEEG(Xraw)
            self.csp_l, self.Xcsp = self._CSP_epochsEEG(self._bank_filter_epochsEEG(Xraw), y)
        return self    

    
    def transform(self, Xraw, *_):
        if self.PCov == False:
            Xraw = self._averagingEEG(Xraw)
            Xraw = self._bank_filter_epochsEEG(Xraw)
            epochs,channels,T,nf = Xraw.shape
        else:
            #Xwtmpha = tfr_array_morlet(Xraw,sfreq=self.fs,freqs=np.mean(self.f_frec,axis=1),n_cycles=self.n_cycles,output="phase")
            epochs,P,nf  = Xraw.shape
            #epochs,channels,nf,T = Xwtmpha.shape
        ncomp = self.ncomp    
        result = np.zeros((epochs,ncomp,nf))   
        for f in range(nf):
            #if self.PCov == 'kernel':
            #    Cov = np.zeros((epochs,channels,channels))
                #for epoch in range(epochs):
                #Cov[epoch,:,:] = self.Kg(Xraw_f[epoch,:,:,f]) 
            #    Cov = np.array(Parallel(n_jobs=-1)(delayed(self.Kg)(Xraw_f[n,:,:,f]) for n in range(epochs))) 
            #    result[:,:,f] =  self.csp_l[f].transform(Cov)
            #elif self.PCov == 'PLV':
            #    Cov = np.array(Parallel(n_jobs=-1)(delayed(self.plv_phase_distance)(Xwtmpha[:,:,f,:],n) for n in range(epochs)))
            #    result[:,:,f] =  self.csp_l[f].transform(Cov)
            if self.PCov == True:
                result[:,:,f] =  self.csp_l[f].transform(Xraw[:,:,f]) 
            else:
                result[:,:,f] =  self.csp_l[f].transform(Xraw[:,:,:,f]) 
        result = result.reshape(np.size(result,0),-1)  
        return result 

def eeg_nor(Xraw,sca=1e5): #Xraw[epochs,ch,T]
    epochs,chs,T = Xraw.shape
    Xrawp = np.zeros((epochs,chs,T))
    for ep in range(epochs):
        for c in range(chs):
            Xrawp[ep,:,:] = sca*(Xraw[ep,:,:] - Xraw[ep,:,:].mean(axis=0))
    return Xrawp

def plot_eeg(data,sample_rate,channels_names,sca=0.75): #data[channels, samples]
    #  Como se conoce la frecuencia de muestreo es posible recuperar el vector del tiempo
    time = np.linspace(0, data.shape[1] / sample_rate, data.shape[1])

    fig = plt.gcf()#plt.figure(figsize=(16, 9), dpi=90)
    sumf = sca*np.max(sca*(data-matlib.repmat(data.mean(axis=1).reshape(-1,1),1,data.shape[1])))
     # Se reemplazan los valores numéricos del eje Y por los nombres de los canales
    plt.yticks(np.arange(0, sumf*len(channels_names),sumf),channels_names)
    color = sns.color_palette('husl',n_colors=data.shape[0])
    # Como los datos están en vertical (columnas) se reorientan con la transpuesta para poder visualizar los canales
    for i in range(data.shape[0]):  # se ignora la última columna
        # Para que no queden los canales sobrepuestos, antes de graficar se centra y se le suma un entero para desplazarlo ligeramente hacia arriba.
        plt.plot(time, (data[i,:] - data[i,:].mean()) + sumf*i,color=color[i])
    return


def plot_confusion_matrix_MS(cm_m, cm_s, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    
    fig, ax = plt.subplots()
    im = ax.imshow(cm_m, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm_m.shape[1]),
           yticks=np.arange(cm_m.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm_m.max() / 2.
    for i in range(cm_m.shape[0]):
        for j in range(cm_m.shape[1]):
            s = format(cm_m[i, j],'.1f') + "$\pm$" + format(cm_s[i, j],'.1f')
            ax.text(j, i, s,ha="center", va="center",
                    color="white" if cm_m[i, j] > thresh else "black",fontsize=12)
    fig.tight_layout()
    return ax

#%%

class Window_band_CSP_eppoch(TransformerMixin,BaseEstimator):
    def __init__(self,fs,vtw=[2.5,4.5],f_frec=[4,40],ncomp=6,reg='empirical',PCov=False):
        self.fs=fs
        self.ncomp=ncomp
        self.PCov=PCov
        self.vtw=vtw
        self.f_frec=f_frec
        self.reg=reg

    def fit_CSP_Xraw_time_samples(self,Xraw,y,vtw,fs,f_frec,ncomp):
        self.csp_c = [None]*len(vtw)
        self.filters = [None]*len(vtw)
        for i in range(len(vtw)):
            if self.PCov == False:
                X = Xraw[:,:,int(vtw[i][0]*fs):int(vtw[i][1]*fs)]
            else:
                X = Xraw[:,:,i,:]
            self.csp_c[i] = CSP_epochs_filter_extractor(fs=fs,f_frec=f_frec,ncomp=ncomp,reg=self.reg,PCov=self.PCov)
            self.csp_c[i].fit(X,y)
            self.filters[i] = self.csp_c[i].filters

    def fit(self,Xraw,y,*_):
        if self.PCov == True:
            Xraw = Xraw.reshape((len(Xraw),-1,len(self.vtw),len(self.f_frec)))
        self.fit_CSP_Xraw_time_samples(Xraw,y,self.vtw,self.fs,self.f_frec,self.ncomp) 
        return self  
    
    def transform(self, Xraw, *_):
        Xf = [None]*len(self.csp_c)
        if self.PCov == True:
            Xraw = Xraw.reshape((len(Xraw),-1,len(self.vtw),len(self.f_frec)))
        for i in range(len(self.csp_c)):
            if self.PCov == False:
                X = Xraw[:,:,int(self.vtw[i][0]*self.fs):int(self.vtw[i][1]*self.fs)]
            else: 
                X = Xraw[:,:,i,:]
            Xf[i] = self.csp_c[i].transform(X)
        return Xf #
    #def fit_transform(self,Xraw,y,*_):
    #    self.fit(Xraw,y)
    #    return self.transform(Xraw)
#%%
"""
class Window_band_MM_eppoch(TransformerMixin,BaseEstimator):
    def __init__(self,fs,vtw=[[0,2],[2,4]],f_frec=np.array([[8,30]])):
        self.fs=fs
        self.vtw=vtw
        self.f_frec=f_frec
    def fit(self,Xraw,y,*_):
        pass 
    def transform(self, Xraw, *_):
        Xfil = bank_filter_epochsEEG(Xraw,self.fs,self.f_frec)
        for i in range(len(self.vtw)):
            Xfil_r = Xfil[:,:,int(self.fs*self.vtw[i][0]):int(self.fs*self.vtw[i][1]),:]
            Xm =np.zeros((Xfil_r.shape[0],Xfil_r.shape[1],len(self.vtw),len(self.f_frec),5))
            for band in range(len(self.f_frec)):
                Xm[:,:,i,band,0]=Xfil_r[:,:,:,band].mean(axis=-1)
                Xm[:,:,i,band,1]=Xfil_r[:,:,:,band].var(axis=-1)
                Xm[:,:,i,band,2]=Xfil_r[:,:,:,band].max(axis=-1)
                Xm[:,:,i,band,3]=Xfil_r[:,:,:,band].min(axis=-1)
                Xm[:,:,i,band,4]=np.median(Xfil_r[:,:,:,band],axis=-1)
            Xm = Xm.reshape((Xfil_r.shape[0],-1))
        return Xm
    def fit_transform(self,Xraw,y,*_):
        self.transform(Xraw,y)
        return self.transform(Xraw)
#%%
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class elastic_net_feats(TransformerMixin,BaseEstimator):
  def __init__(self,alpha=0,l1_ratio=0):
    self.alpha=alpha
    self.l1_ratio=l1_ratio
  def fit(self,X,y,*_):
    self.elatic = ElasticNet(alpha=self.alpha,l1_ratio=self.l1_ratio,random_state=0)
    #if len(np.unique(y))>2:
    mlb = OneHotEncoder()
    y=mlb.fit_transform(y.reshape((-1,1))).toarray()
    self.elatic.fit(X,y)
  def transform(self,X,*_):
    if len(self.elatic.coef_.shape)>1:
        Xr = np.dot(X,self.elatic.coef_.T)
    else:
        Xr = np.dot(X,self.elatic.coef_)
        Xr = Xr.reshape((-1,1))
    return Xr
  def fit_transform(self,X,y,*_):
    self.fit(X,y)
    return self.transform(X)

class Lasso_feats(TransformerMixin,BaseEstimator):
  def __init__(self,alpha=0):
    self.alpha=alpha
  def fit(self,X,y,*_):
    self.lasso = Lasso(alpha=self.alpha)
    self.lasso.fit(X,y)
  def transform(self,X,*_):
    Xr = X[:,np.abs(self.lasso.coef_)> 0]
    return Xr
  def fit_transform(self,X,y,*_):
    self.fit(X,y)
    return self.transform(X)
from numpy import swapaxes
"""
class flatt(TransformerMixin,BaseEstimator):
    def __init__(self,axis=0):
        self.axis=axis
    def fit(self,X,*_):
        for i in range(len(X)):
            if i==0:
                Xx = X[i]
            else:
                Xx = np.concatenate((Xx,X[i]),axis=1)  
            if self.axis==1:
                Xx = Xx.T
        return Xx 
    def transform(self,X,*_):
        for i in range(len(X)):
            if i==0:
                Xx = X[i]
            else:
                Xx = np.concatenate((Xx,X[i]),axis=1)  
            if self.axis==1:
                Xx = Xx.T
        return Xx 
    def fit_transform(self,X,*_):
        for i in range(len(X)):
            if i==0:
                Xx = X[i]
            else:
                Xx = np.concatenate((Xx,X[i]),axis=1)  
            if self.axis==1:
                Xx = Xx.T
        return Xx

"""
class Swapax_csp(TransformerMixin,BaseEstimator):
    def __init__(self,vtw,f_frec):
        self.vtw=vtw
        self.f_frec=f_frec
    def fit(self,X,*_):
        newkl = X.reshape((len(X),len(self.vtw),-1,len(self.f_frec)))
        newkl =np.swapaxes(newkl,1,2)
        return  newkl.reshape((len(X),-1))
    def transform(self,X,*_):
        newkl = X.reshape((len(X),len(self.vtw),-1,len(self.f_frec)))
        newkl =np.swapaxes(newkl,1,2)
        return  newkl.reshape((len(X),-1))
    def fit_transform(self,X,*_):
        newkl = X.reshape((len(X),len(self.vtw),-1,len(self.f_frec)))
        newkl =np.swapaxes(newkl,1,2)
        return  newkl.reshape((len(X),-1))

class Swapax(TransformerMixin,BaseEstimator):
    def __init__(self,col1=0,col2=1):
        self.col1 = col1
        self.col2 = col2
    def fit(self,X,y,*_):
        Xr = swapaxes(X,self.col1,self.col2)
        return Xr
    def transform(self,X,*_):
        Xr = swapaxes(X,self.col1,self.col2)
        return Xr
    def fit_transform(self,X,y,*_):
        Xr = swapaxes(X,self.col1,self.col2)
        return Xr 

class concat(TransformerMixin,BaseEstimator):
    def __init__(self,col1=0):
        self.col1 = col1
    def fit(self,X,y,*_):
        Xr = np.reshape(X,(np.shape(X)[0],-1))
        return Xr
    def transform(self,X,*_):
        Xr = np.reshape(X,(np.shape(X)[0],-1))
        return Xr
    def fit_transform(self,X,y,*_):
        Xr = np.reshape(X,(np.shape(X)[0],-1))
        return Xr     

class selectCSPwin_freq(TransformerMixin,BaseEstimator):
    def __init__(self,windows,freqs,f_frec):
        self.windows = windows
        self.freqs = freqs
        self.f_frec=f_frec
    def fit(self,X,y,*_):
        Xr=[]
        Wind,ind_inv = np.unique(self.windows,return_inverse=True)
        for ix in range(len(Wind)):
            tmp = X[Wind[ix]]
            tmp = tmp.reshape((tmp.shape[0],-1,len(self.f_frec)))
            Xr.append(tmp[:,:,self.freqs[ind_inv==ix]].reshape((tmp.shape[0],-1)))
        return Xr
    def transform(self,X,*_):
        Xr=[]
        Wind,ind_inv = np.unique(self.windows,return_inverse=True)
        for ix in range(len(Wind)):
            tmp = X[Wind[ix]]
            tmp = tmp.reshape((tmp.shape[0],-1,len(self.f_frec)))
            Xr.append(tmp[:,:,self.freqs[ind_inv==ix]].reshape((tmp.shape[0],-1)))
        return Xr
    def fit_transform(self,X,y,*_):
        Xr=[]
        Wind,ind_inv = np.unique(self.windows,return_inverse=True)
        for ix in range(len(Wind)):
            tmp = X[Wind[ix]]
            tmp = tmp.reshape((tmp.shape[0],-1,len(self.f_frec)))
            Xr.append(tmp[:,:,self.freqs[ind_inv==ix]].reshape((tmp.shape[0],-1)))
        return Xr  

#%%
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
import time
class Cov_epochs_filter_extractor(TransformerMixin,BaseEstimator):
    def __init__(self, fs,f_frec=[4,30],gamma=1,Normalize=False,alpha=1,eta=1):
        self.gamma = gamma
        self.fs = fs
        self.alpha=alpha
        self.eta=eta
        self.f_frec = f_frec
        self.Normalize = Normalize
        
    def _averagingEEG(self,X):
        epochs,channels,T = X.shape
        Xc = np.zeros((epochs,channels,T))
        for i in range(epochs):
            Xc[i,:,:] = X[i,:,:] - X[i,:,:].mean(axis=0)
        return Xc    
        
    def _bank_filter_epochsEEG(self,X):
        nf,ff = self.f_frec.shape
        epochs,channels,T = X.shape
        X_f = np.zeros((epochs,channels,T,nf))
        for f in range(nf):
            lfc = self.f_frec[f,0]
            hfc = self.f_frec[f,1]
            b,a = butter_bandpass(lfc, hfc, self.fs)
            X_f[:,:,:,f] = filtfilt(b,a,X,axis=2)
        return X_f    

    def _cov_epochsEEG(self,Xraw,*_):
        mne.set_log_level('WARNING')
        epochs,channels,T,nf = Xraw.shape
        #Xcov = np.zeros((epochs,int(channels*(channels-1)/2),nf))
        Xcov = np.zeros((epochs,int(channels*self.alpha),nf))
        self.epochs = epochs
        self.channels  = channels
        for f in range(nf):
            for i in  range(epochs):
                #C = pairwise_distances(Xraw[i,:,:,f],Xraw[i,:,:,f])
                #C = (C+C.T)/2 # ensure symmetry matrix
                #C = C-np.diag(np.diag(C))
                #np.corrcoef(Xraw[i,:,:,f])#Xraw[i,:,:,f].dot(Xraw[i,:,:,f].T) 
                #gamma0 = 1/np.median(squareform(C)**2)
                #C = np.exp(-.5*self.gamma*gamma0*(C**2))
                kpca = KernelPCA(n_components=self.alpha,kernel='rbf',gamma=self.eta)
                C = kpca.fit_transform(Xraw[i,:,:,f])
                #w ,v = np.linalg.eig(C)
                #indx = np.argsort(w)
                #w = w[indx]
                #v = v[:,indx]
                #cus = np.cumsum(w)/np.sum(w)<self.eta
                #C = v[:,cus].dot(np.diag(w[cus])).dot(v[:,cus].T)
                #C = (C+C.T)/2 # ensure symmetry matrix
                #C = C.dot(v[:,cus]).dot(np.diag(w[cus]))
                #Xcov[i,:,f] = squareform(C-np.diag(np.diag(C)))
                Xcov[i,:,f] = C.ravel()
        return Xcov.reshape(epochs,-1)
    
    def _cov_vec2mat(self,Xv):
        return Xv.reshape(self.epochs,int(self.channels*(self.channels-1)/2),len(self.f_frec))

    def fit(self,Xraw,y, *_):
        if self.Normalize==True:
            Xraw = self._averagingEEG(Xraw)
        Xraw_f = self._bank_filter_epochsEEG(Xraw)
        self.Xcov_v = self._cov_epochsEEG(Xraw_f)
        return self    
    
    def transform(self, Xraw, *_):
        if self.Normalize==True:
            Xraw = self._averagingEEG(Xraw)
        Xraw_f = self._bank_filter_epochsEEG(Xraw)
        return self._cov_epochsEEG(Xraw_f)
#%%
class Concat_epochs_filter_extractor(TransformerMixin,BaseEstimator):
    def __init__(self, fs,f_frec=[4,30],Normalize=False,StaditicFeatures=False):
        self.fs = fs
        self.f_frec = f_frec
        self.Normalize = Normalize
        self.StaditicFeatures = StaditicFeatures
        
    def _averagingEEG(self,X):
        epochs,channels,T = X.shape
        Xc = np.zeros((epochs,channels,T))
        for i in range(epochs):
            Xc[i,:,:] = X[i,:,:] - X[i,:,:].mean(axis=0)
        return Xc    
        
    def _bank_filter_epochsEEG(self,X):
        nf,ff = self.f_frec.shape
        epochs,channels,T = X.shape
        X_f = np.zeros((epochs,channels,T,nf))
        for f in range(nf):
            lfc = self.f_frec[f,0]
            hfc = self.f_frec[f,1]
            b,a = butter_bandpass(lfc, hfc, self.fs)
            X_f[:,:,:,f] = filtfilt(b,a,X,axis=2)
        return X_f    

    def _cov_epochsEEG(self,Xraw,*_):
        mne.set_log_level('WARNING')
        epochs,channels,T,nf = Xraw.shape
        self.epochs = epochs
        self.channels  = channels
        if self.StaditicFeatures == False:
            return Xraw.reshape((epochs,-1))
        else:
            Xst = np.zeros((epochs,channels,5,nf))
            for f in range(nf):
                for i in  range(epochs):
                    for c in range(channels):
                        Xst[i,c,0,f] = np.mean(Xraw[i,c,:,f])
                        Xst[i,c,1,f] = np.std(Xraw[i,c,:,f])
                        Xst[i,c,2,f] = np.min(Xraw[i,c,:,f])
                        Xst[i,c,3,f] = np.max(Xraw[i,c,:,f])
                        Xst[i,c,4,f] = np.median(Xraw[i,c,:,f])
            return Xst.reshape((epochs,-1))
    
    def fit(self,Xraw,y, *_):
        if self.Normalize==True:
            Xraw = self._averagingEEG(Xraw)
        Xraw_f = self._bank_filter_epochsEEG(Xraw)
        return self    
    
    def transform(self, Xraw, *_):
        if self.Normalize==True:
            Xraw = self._averagingEEG(Xraw)
        Xraw_f = self._bank_filter_epochsEEG(Xraw)
        return self._cov_epochsEEG(Xraw_f)

'''
def rho_topoplot(rho,info,channels_names,show_names=False,countours=0, cmap='jet',ax =None,fig=None,sca=1,colorbar=True,vmin=0,vmax=1):
    
    if ax == None: ax = plt.gca()
    if fig == None: fig = plt.gcf()
    rhoc = sca*rho
    if colorbar:
        cax = fig.add_axes([0.95, 0.15, 0.05, 0.75])
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm,cax=cax)
    mne.viz.plot_topomap(rhoc,info, names=channels_names, 
                          show_names=show_names,contours=0,cmap=cmap,axes=ax,vmin=vmin,vmax=vmax,res=128)
    return

'''
from sklearn.metrics import pairwise_distances  
from scipy.spatial.distance import squareform 
from mne.viz import plot_topomap
from scipy.stats import kurtosis
import pywt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import periodogram, welch
from scipy.stats import spearmanr
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mne.time_frequency import tfr_array_morlet, csd_array_morlet, csd_array_fourier
from joblib import Parallel, delayed
from scipy.stats import kendalltau
#from mne.viz.topomap import _check_outlines, _draw_outlines
#from mne.viz.utils import plt_show,tight_layout
#from mne.io.pick import _picks_to_idx
import itertools
import numpy as np
import itertools
from sklearn.base import  BaseEstimator, TransformerMixin
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt #, freqz
from joblib import Parallel, delayed
import numpy as np
import itertools
from sklearn.base import  BaseEstimator, TransformerMixin
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt #, freqz
from joblib import Parallel, delayed
from itertools import islice
from scipy.spatial.distance import cdist
from itertools import permutations
import math
from scipy import special
class FB_feats(TransformerMixin,BaseEstimator):
    def __init__(self, fs=250,f_frec=np.array([[8,12],[12,30]]),vtw=np.array([[2.5,4.5],[3,5]]),gammad=1,
                feat='kcon',ncomp = 6,method='fft',normalize=True,n_cycles=7.0,motifs_transform=False,stride=1,
                     points=3,minus_resting=False,over_connexions=False,leg_order=10,m=4,
                     smoothing=1e-5,laplacian_montage=mne.channels.read_montage('standard_1005'),
                     channels_names=None):
        self.vtw = vtw
        self.fs = fs
        self.f_frec = f_frec
        self.gammad = gammad
        self.feat = feat
        self.method = method
        self.normalize = normalize
        self.ncomp = ncomp
        self.n_cycles = n_cycles
        self.motifs_transform=motifs_transform
        self.stride=stride
        self.points=points
        self.minus_resting=minus_resting
        self.over_connexions=over_connexions
        self.leg_order=leg_order
        self.laplacian_montage=laplacian_montage
        self.smoothing = smoothing
        self.m=m
        self.channels_names=channels_names
##################
    def surface_laplacian(self, data): 
        #x = self.laplacian_montage.pos[:,0]
        #y = self.laplacian_montage.pos[:,1]
        #z = self.laplacian_montage.pos[:,2]

        #####################
        #montage = mne.channels.read_montage('standard_1005')
        layout = mne.channels.read_layout('EEG1005')
        #montage = mne.channels.read_montage('biosemi128')
        #layout = montage.ch_names
        info = mne.create_info(layout.names, sfreq=self.fs, ch_types="eeg",
                                montage=self.laplacian_montage)
        pos = self.laplacian_montage.pos
        #pos = layout.pos#np.array([(p[0] + p[2] / 2., p[1] + p[3] / 2.) for p in layout.pos])
        # pick channels
        picks = _picks_to_idx(info,self.channels_names)
        pos = pos[picks]
        # adjust positions
        #pos, outlines = _check_outlines(pos, 'head')
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
        ##########################
        #arrange data
        data = np.moveaxis(data, 0, -1)
        orig_data_size = np.squeeze(data.shape)

        numelectrodes = len(x)
        
        # normalize cartesian coordenates to sphere unit
        def cart2sph(x, y, z):
            hxy = np.hypot(x, y)
            r = np.hypot(hxy, z)
            el = np.arctan2(z, hxy)
            az = np.arctan2(y, x)
            return az, el, r

        junk1, junk2, spherical_radii = cart2sph(x,y,z)
        maxrad = np.max(spherical_radii)
        x = x/maxrad
        y = y/maxrad
        z = z/maxrad
        
        # compute cousine distance between all pairs of electrodes
        cosdist = np.zeros((numelectrodes, numelectrodes))
        for i in range(numelectrodes):
            for j in range(i+1,numelectrodes):
                cosdist[i,j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)/2)

        cosdist = cosdist + cosdist.T + np.identity(numelectrodes)

        # get legendre polynomials
        legpoly = np.zeros((self.leg_order, numelectrodes, numelectrodes))
        for ni in range(self.leg_order):
            for i in range(numelectrodes):
                for j in range(i+1, numelectrodes):
                    #temp = special.lpn(8,cosdist[0,1])[0][8]
                    legpoly[ni,i,j] = special.lpn(ni+1,cosdist[i,j])[0][ni+1]

        legpoly = legpoly + np.transpose(legpoly,(0,2,1))

        for i in range(self.leg_order):
            legpoly[i,:,:] = legpoly[i,:,:] + np.identity(numelectrodes)

        # compute G and H matrixes
        twoN1 = np.multiply(2, range(1, self.leg_order+1))+1
        gdenom = np.power(np.multiply(range(1, self.leg_order+1), range(2, self.leg_order+2)), self.m, dtype=float)
        hdenom = np.power(np.multiply(range(1, self.leg_order+1), range(2, self.leg_order+2)), self.m-1, dtype=float)

        G = np.zeros((numelectrodes, numelectrodes))
        H = np.zeros((numelectrodes, numelectrodes))

        for i in range(numelectrodes):
            for j in range(i, numelectrodes):

                g = 0
                h = 0

                for ni in range(self.leg_order):
                    g = g + (twoN1[ni] * legpoly[ni,i,j]) / gdenom[ni]
                    h = h - (twoN1[ni] * legpoly[ni,i,j]) / hdenom[ni]

                G[i,j] = g / (4*math.pi)
                H[i,j] = -h / (4*math.pi)

        G = G + G.T
        H = H + H.T

        G = G - np.identity(numelectrodes) * G[1,1] / 2
        H = H - np.identity(numelectrodes) * H[1,1] / 2

        data = np.reshape(data, (orig_data_size[0], np.prod(orig_data_size[1:3])))

        # compute C matrix
        Gs = G + np.identity(numelectrodes) * self.smoothing
        GsinvS = np.sum(np.linalg.inv(Gs), 0)
        dataGs = np.dot(data.T, np.linalg.inv(Gs))
        C = dataGs - np.dot(np.atleast_2d(np.sum(dataGs, 1)/np.sum(GsinvS)).T, np.atleast_2d(GsinvS))

        # apply transform
        surf_lap = np.reshape(np.transpose(np.dot(C,np.transpose(H))), orig_data_size)

        return np.moveaxis(surf_lap, -1, 0)
    def takens_delay_embedding(self,x):
        '''
        takens delay embedding of a time serie
        INPUT
        -----
            1. x: (1D array) unidimensional time serie in R^{T}
            2. tau: (int) time delay embedding in N
            3. dim: (int) embedding dimension in N
        OUTPUT
        ------
            1. x_emb: (2D array) embedding time serie in R^{row_emb,dim}
        '''
        tau = int(self.stride)
        dim = int(self.points)
        if tau <= 0 or dim <= 0:
            raise ValueError("The time delay embedding and embedding dimension have to be greater than zero")
        len_    = x.shape[0]
        row_emb = len_ - (dim-1)*tau
        if row_emb <= 0:
            raise ValueError("The embeddings dimension and time delay embedding doesn't fit the time serie")
        x_emb = np.zeros((row_emb,dim))
        for i,j in enumerate(np.arange(0,tau*dim,tau)):
            x_emb[:,i] = x[np.arange(j,j + row_emb)]
        return x_emb

    def motif_representation_EEG_DB(self,X):
        degree=self.points
        lag=self.stride
        motifs = np.array(list(permutations(np.arange(degree, dtype=np.ushort), degree)), dtype=np.ushort)
        motifs_order = np.argsort(motifs, axis=1)
        X_emb = np.zeros((X.shape[0], X.shape[1], (X.shape[-1] - (degree-1)*lag), degree))
        for n in np.arange(X.shape[0]):
            for ch in np.arange(X.shape[1]):
                X_emb[n, ch, :, :] = self.takens_delay_embedding(X[n, ch, :])
        X_emb = X_emb.reshape(-1, degree)
        X_emb_order = np.argsort(X_emb,axis=1)
        X_motif = np.zeros(X_emb.shape[0])
        for motif in np.arange(motifs_order.shape[0]):
            X_motif[(X_emb_order==motifs_order[motif]).prod(axis=1).astype(bool)]=motif
        return X_motif.reshape(X.shape[0], X.shape[1], (X.shape[-1] - (degree-1)*lag))
#################
    def butter_bandpass(self, lowcut, highcut, order=5):
      nyq = 0.5 * self.fs
      low = lowcut / nyq
      high = highcut / nyq
      b, a = butter(order, [low, high], btype='band')
      return b, a

    def _bank_filter_epochsEEG(self,X):
        X_f = np.zeros((X.shape[0],X.shape[1],X.shape[2],self.f_frec.shape[0])) #epochs, Ch, Time, bands
        for f in range(self.f_frec.shape[0]):
            lfc = self.f_frec[f,0]
            hfc = self.f_frec[f,1]
            b,a = self.butter_bandpass(lfc, hfc)
            X_f[:,:,:,f] = filtfilt(b,a,X,axis=2)
        return X_f    
    def calculo_cwt(self,x,fs):
        wname     = 'cmor'
        delta     = 1/fs
        coef,freq = pywt.cwt(x.T,np.arange(1,32),wname,delta)
        return coef, freq

    def _cwt_feat_extraction(self,Xraw):
        Xfeat = np.zeros((Xraw.shape[0],Xraw.shape[1],len(self.vtw),len(self.f_frec)))
        for tr in range(Xraw.shape[0]):#loop across trials
            for ch in range(Xraw.shape[1]):#loop across channels
                for w in range(self.vtw.shape[0]): #windows
                    coef, freq = self.calculo_cwt(np.squeeze(Xraw[tr,ch,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1])]),self.fs)
                    coef       = np.abs(coef)
                    for fb in range(self.f_frec.shape[0]):#loop across filter bands
                        coef_mat           = coef[np.where((freq > self.f_frec[fb,0]) & (freq <self.f_frec[fb,1])),:]
                        coef_mat           = np.squeeze(coef_mat[0,:,:])
                        Xfeat[tr,ch,w,fb]    = np.mean(coef_mat.flatten())
        return Xfeat
    def _FB_PLVEEG(self,Xraw):
        Xcov = np.zeros((Xraw.shape[0],int(0.5*Xraw.shape[1]*(Xraw.shape[1]-1)),self.vtw.shape[0],self.f_frec.shape[0]))
        for w in range(self.vtw.shape[0]): #windows
            if self.feat == 'PLV' or self.feat == 'pli':
                Xwtmpha = tfr_array_morlet(Xraw[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1])],sfreq=self.fs,freqs=np.mean(self.f_frec,axis=1),n_cycles=self.n_cycles,output="phase")
                for f in range(self.f_frec.shape[0]): #frequencies
                    Xcov[:,:,w,f] = np.array(Parallel(n_jobs=-1)(delayed(self.plv_phase_distance)(Xwtmpha,f,n) for n in range(Xraw.shape[0])))
            elif self.feat == 'cds':
                tmpfrec = np.mean(self.f_frec,axis=1)
                for trial in range(len(Xraw)):
                    Xcdstmp = csd_array_morlet(Xraw[trial:trial+1,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1])],sfreq=self.fs,frequencies=np.mean(self.f_frec,axis=1),n_cycles=self.n_cycles)
                    for f in range(self.f_frec.shape[0]):
                        #Xcdstmp = csd_array_fourier(Xraw[trial:trial+1,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1])],sfreq=self.fs,fmin=self.f_frec[f,0],fmax=self.f_frec[f,1])
                        k = np.real(Xcdstmp.get_data(frequency=tmpfrec[f]))
                        k = k - np.diag(np.diag(k))
                        k = 0.5*(k+k.T)
                        Xcov[trial,:,w,f] = squareform(k)
        return Xcov
    def plv(self,x,y):
        er = np.exp(1j*(x-y))
        return abs(np.mean(er))
    def pli(self,x,y):
        er = np.sign(np.sin(x-y))
        #er = np.sign((x-y))
        return abs(np.mean(er))
    def plv_phase_distance(self,Xwtmpha,f,n):
        if self.feat == 'PLV':
            k = pairwise_distances(Xwtmpha[n,:,f,:],Xwtmpha[n,:,f,:],metric=self.plv)
        else:
            k = pairwise_distances(Xwtmpha[n,:,f,:],Xwtmpha[n,:,f,:],metric=self.pli)
        k = 0.5*(k+k.T)
        k -= np.diag(np.diag(k))
        return squareform(k)     
    def _FB_momentsEEG(self,Xraw_f):
        self.P = 6
        Xfeat = np.zeros((Xraw_f.shape[0],Xraw_f.shape[1],self.P,self.vtw.shape[0],self.f_frec.shape[0]))  # epochs, Ch, 6moments, windows, bands
        for w in range(self.vtw.shape[0]): #windows
          for f in range(self.f_frec.shape[0]): #bands
              Xfeat[:,:,0,w,f] = Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f].mean(axis=-1)
              Xfeat[:,:,1,w,f] = Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f].var(axis=-1)
              Xfeat[:,:,2,w,f] = Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f].min(axis=-1)
              Xfeat[:,:,3,w,f] = Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f].max(axis=-1)
              Xfeat[:,:,4,w,f] = np.median(Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f],axis=-1)
              Xfeat[:,:,4,w,f] = kurtosis(Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f],axis=-1)
        return Xfeat
    def kcov(self,X): #epochs, Ch, T
      Xcov = np.zeros((X.shape[0],int(0.5*X.shape[1]*(X.shape[1]-1))))
      utri_ind =  np.triu_indices(X.shape[1], 1)
      #tmp = np.ones((X.shape[1],X.shape[1]))-np.eye(X.shape[1],X.shape[1])
      for n in range(X.shape[0]):
        if self.feat == 'kcon_surr':
            if n == X.shape[0]-1:
                #dd = pairwise_distances(X[n],X[0])
                dd = cdist(X[n],X[0],'euclidean')
            else:
                #dd = pairwise_distances(X[n],X[n+1])
                dd = cdist(X[n],X[n+1],'euclidean')
        else:
            #dd = pairwise_distances(X[n],X[n])
            dd = cdist(X[n],X[n],'euclidean')
        sigma = np.median(dd[utri_ind])
        #dd = 0.5*(dd + dd.T)*tmp
        #g = self.gammad/(2*np.median(squareform(dd))**2)
        #k = np.exp(-g*(dd**2))
        #N = np.shape(k)[0]
        #h = np.eye(N) - (1.0/N)*np.ones([N,1])*np.ones([1,N]) #matrix for centered kernel
        #kc = np.dot(h,np.dot(k,h))
        #k = k - np.diag(np.diag(k))
        #k = 0.5*(k+k.T)
        #k = k - np.diag(np.diag(k))
        K = np.exp(-1*(self.gammad/(2*sigma**2))*(dd**2))        
        Xcov[n,:] = K[utri_ind]#squareform(k)
      return Xcov
    def kcon_pearson(self,X):
        Xcov = np.zeros((X.shape[0],int(0.5*X.shape[1]*(X.shape[1]-1))))
        for n in range(X.shape[0]):
            k = np.corrcoef(X[n])
            k = k - np.diag(np.diag(k))
            k = 0.5*(k+k.T) 
            Xcov[n,:] = squareform(k)
        return Xcov
    def kcon_spearman(self,X):
        Xcov = np.zeros((X.shape[0],int(0.5*X.shape[1]*(X.shape[1]-1))))
        for n in range(X.shape[0]):
            k =  spearmanr(X[n].T)[0]
            k = k - np.diag(np.diag(k))
            k = 0.5*(k+k.T)    
            Xcov[n,:] = squareform(k)
        return Xcov
    def kendal_d1(self,x,y):
        return kendalltau(x,y)[0]
    def kendal_d2(self,X,n):
        k = pairwise_distances(X[n],X[n],metric=self.plv)
        k = k - np.diag(np.diag(k))
        k = 0.5*(k+k.T)
        return squareform(k)    
    def kcon_kendall(self,X):
        #Xcov = np.zeros((X.shape[0],int(0.5*X.shape[1]*(X.shape[1]-1))))
        #for n in range(X.shape[0]):
        tmp=[]
        for i in range(len(X[n])):
            for j in np.arange(len(X[n])-1-i)+1:
                tmp.append(kendalltau(X[n][i],X[n][j])[0])
            #Xcov[n,:] = np.asarray(tmp)
        return np.asarray(tmp)        
    def _FB_KcovEEG(self,Xraw_f):
      self.P = int(0.5*Xraw_f.shape[1]*(Xraw_f.shape[1]-1))
      Xfeat = np.zeros((Xraw_f.shape[0],self.P,self.vtw.shape[0],self.f_frec.shape[0]))  # epochs, Ch(Ch-1)/2, windows, bands
      for w in range(self.vtw.shape[0]): #windows
        for f in range(self.f_frec.shape[0]): #bands
            if self.feat=='kcon' or self.feat == 'kcon_surr':
                if self.motifs_transform:
                    Xfeat[:,:,w,f] =  self.kcov(self.motif_representation_EEG_DB(Xraw_f[:,:,:,w,f]))
                else:
                    Xfeat[:,:,w,f] =  self.kcov(Xraw_f[:,:,:,w,f])
                    #Xfeat[:,:,w,f] =  self.kcov(Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f])
            elif self.feat=='Kcon_pearson':
                if self.motifs_transform:
                    Xfeat[:,:,w,f] =  self.kcon_pearson(self.motfis_trasnform(Xraw_f[:,:,:,w,f]))
                else:
                    Xfeat[:,:,w,f] =  self.kcon_pearson(Xraw_f[:,:,:,w,f])
                #Xfeat[:,:,w,f] =  self.kcon_pearson(Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f])
            elif self.feat=='Kcon_spearman':
                if self.motifs_transform:
                    Xfeat[:,:,w,f] =  self.kcon_spearman(self.motfis_trasnform(Xraw_f[:,:,:,w,f]))
                else:
                    Xfeat[:,:,w,f] =  self.kcon_spearman(Xraw_f[:,:,:,w,f])                
                #Xfeat[:,:,w,f] =  self.kcon_spearman(Xraw_f[:,:,:,w,f]) 
                #Xfeat[:,:,w,f] =  self.kcon_spearman(Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f]) 
            elif self.feat=='Kcon_kendall':
                if self.motifs_transform:
                    XXtmp = self.motfis_trasnform(Xraw_f[:,:,:,w,f])
                else:
                    XXtmp = Xraw_f[:,:,:,w,f]
                #XXtmp = Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f]
                Xfeat[:,:,w,f] =  np.array(Parallel(n_jobs=-1)(delayed(self.kendal_d2)(XXtmp,n) for n in range(Xraw_f.shape[0])))  
      return Xfeat  
    def _spectral_entropy(self,x,fs):
        if self.method == 'fft':
            _, psd = periodogram(x, fs)
        elif self.method == 'welch':
            _, psd = welch(x,fs, nperseg=None)
        psd_norm = np.divide(psd, psd.sum())
        Xse = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        if self.normalize:
            Xse /= np.log2(psd_norm.size)        
        return Xse
    def _FBse(self,Xraw_f):
        Xfeat = np.zeros((Xraw_f.shape[0],Xraw_f.shape[1],self.vtw.shape[0],self.f_frec.shape[0]))
        for tr in range(Xraw_f.shape[0]):#loop across trials
            for ch in range(Xraw_f.shape[1]):#loop across channels
                for w in range(self.vtw.shape[0]): #windows
                    for f in range(self.f_frec.shape[0]): #bands  
                        Xfeat[tr,ch,w,f] = self._spectral_entropy(Xraw_f[tr,ch,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),f],self.fs)   

        return Xfeat
    def _averagingEEG(self,X):
        epochs,channels,T = X.shape
        Xc = np.zeros((epochs,channels,T))
        for i in range(epochs):
            Xc[i,:,:] = X[i,:,:] - np.mean(X[i,:,:])
        return Xc 
    def fit(self,Xraw,y, *_): #epochs, Ch, T
      self.Ch = Xraw.shape[1]
      self.y=y
      if self.feat == 'cwt+csp':
          self.csp = Window_band_CSP_eppoch(fs=self.fs,vtw=self.vtw,f_frec=self.f_frec,ncomp=self.ncomp)
          if self.normalize:
              Xraw = self._averagingEEG(Xraw)
          if self.channels_names is not None:
              Xraw = self.surface_laplacian(Xraw)              
          self.csp.fit(Xraw,y)
      return    
    def make_time_window(self,Xraw_f):
        X_tiw = np.zeros((Xraw_f.shape[0],Xraw_f.shape[1],int(self.fs*(self.vtw[0,1]-self.vtw[0,0])),self.vtw.shape[0],Xraw_f.shape[3]))
        for w in range(self.vtw.shape[0]):
           X_tiw[:,:,:,w,:] = Xraw_f[:,:,int(self.fs*self.vtw[w,0]):int(self.fs*self.vtw[w,1]),:]
        return X_tiw
    def transform(self, Xraw, *_):
        scaler = MinMaxScaler()
        if self.channels_names is not None:
            Xraw = self.surface_laplacian(Xraw)
        if self.normalize:
            Xraw = self._averagingEEG(Xraw)      
        if self.feat == 'moments':
            Xfeat = self._FB_momentsEEG(self._bank_filter_epochsEEG(Xraw))
        elif self.feat == 'Dwt':
            Xfeat = self._cwt_feat_extraction(Xraw)
        elif self.feat == 'spectral entropy':
            Xfeat = self._FBse(self._bank_filter_epochsEEG(Xraw))
        elif self.feat == 'cwt+csp':
            ft = flatt()
            Xcsp = scaler.fit_transform(ft.fit_transform(self.csp.transform(Xraw)).reshape(Xraw.shape[0],-1))
            Xcwt = scaler.fit_transform(self._cwt_feat_extraction(Xraw).reshape(Xraw.shape[0],-1))
            Xfeat = np.concatenate((Xcwt,Xcsp),axis=1)
        elif self.feat == 'PLV' or self.feat == 'pli' or self.feat == 'cds':
            Xfeat = self._FB_PLVEEG(Xraw)
        else:
            X_tiw =  self.make_time_window(self._bank_filter_epochsEEG(Xraw))
            Xfeat = self._FB_KcovEEG(X_tiw)
        Xfeat_return=Xfeat.reshape(Xraw.shape[0],-1)
        if self.minus_resting:
            Xfeat_return=self.minusresting(Xfeat_return)
        return Xfeat_return.reshape(Xraw.shape[0],-1)
    def minusresting(self,X):
        xr = X.reshape(X.shape[0],-1,len(self.vtw),len(self.f_frec))
        if self.over_connexions:
            sumxrw = xr
            feat_xrm = np.ones((X.shape[0],xr.shape[1],len(self.vtw),len(self.f_frec)))
        else:
            xrm = np.ones((X.shape[0],self.Ch,self.Ch,len(self.vtw),len(self.f_frec)))
            for w in range(len(self.vtw)):
                for f in range(len(self.f_frec)):
                    xrm[:,:,:,w,f] = np.asarray([squareform(x_) for x_ in xr[:,:,w,f]])                
            sumxrw = xrm.sum(axis=1)
            feat_xrm = np.ones((X.shape[0],self.Ch,len(self.vtw),len(self.f_frec)))
        try:
            self.resting_mn = sumxrw[self.y==2].mean(axis=2).mean(axis=0)
            #self.resting_mn = sumxrw[self.y==3].mean(axis=2).mean(axis=0)
        except:
            tmp=0
        for w in range(len(self.vtw)):
            for f in range(len(self.f_frec)):
                feat_xrm[:,:,w,f] = np.asarray([ (self.resting_mn[:,f]-x_)/self.resting_mn[:,f] for x_ in sumxrw[:,:,w,f]])
        return feat_xrm
    def fit_transform(self, Xraw, y, *_):
      self.fit(Xraw,y)
      return self.transform(Xraw)
    def vreshape(self,rho): #rho in F
        if self.feat == 'moments':# Ch, 5moments, windows, bands
            rhoM = rho.reshape(self.Ch,self.P,self.vtw.shape[0],self.f_frec.shape[0])
            rhoM = rhoM.sum(axis=1) #Ch,windows,bands
        elif np.logical_or(self.feat == 'Dwt',self.feat == 'spectral entropy'):
            rhoM = rho.reshape(self.Ch,self.vtw.shape[0],self.f_frec.shape[0]) #Ch,windows,bands
        elif self.feat == 'cwt+csp':  
            Ncwt =  self.Ch*self.vtw.shape[0]*self.f_frec.shape[0]
            rhoM = rho[:Ncwt].reshape(self.Ch,self.vtw.shape[0],self.f_frec.shape[0])
            filters = self.csp.filters
            ncomp = self.csp.ncomp
            rho = rho[Ncwt:]
            rho = rho.reshape((len(self.vtw),ncomp,-1))
            rhocka = np.zeros((self.Ch,len(self.vtw),len(self.f_frec)))
            for i in range(len(self.vtw)):
                for j in range(len(self.f_frec)):
                    rhocka[:,i,j] =  np.sum(np.diag(np.abs(rho[i,:,j])).dot(np.abs(filters[i][:,:,j])),0)
            rhoM = np.concatenate((rhoM,rhocka),axis=0)
        else:
          self.P = int(0.5*self.Ch*(self.Ch-1))
          rhoMc = rho.reshape(self.P,self.vtw.shape[0],self.f_frec.shape[0]) # Ch(Ch-1)/2, windows, bands
          rhoM = np.zeros((self.Ch,self.vtw.shape[0],self.f_frec.shape[0]))#Ch,windows,bands
          for w in range(self.vtw.shape[0]):
              for f in range(self.f_frec.shape[0]):
                rhoM[:,w,f] = squareform(rhoMc[:,w,f]).sum(axis=1)

        rhoM = rhoM - rhoM.min()
        rhoM /= rhoM.max() #rho [0,1]
        self.rhoMat = rhoM ##Ch,windows,bands
        return rhoM

    def Wt_FB_plot_topomap(self,rhoM,info,figsize=(5,5),save=True,sbj=0,acc=0,format='png',path='s'):
        #f,ax = plt.subplots(len(self.f_frec),len(self.vtw),figsize=figsize)
        cmap='jet'
        f=plt.figure(figsize=figsize)
        itr=1
        for w in range(len(self.vtw)):
          for ff in range(len(self.f_frec)):
            ax = f.add_subplot(len(self.f_frec),len(self.vtw),itr)
            itr+=1              
            plot_topomap(rhoM[:,w,ff], info, axes=ax, show=False,cmap=cmap,vmin=0,vmax=1)
            ax.set_title(str(self.f_frec[ff])+'[Hz]'+str(self.vtw[w])+'[s]')
        cax = f.add_axes([0.95, 0.15, 0.02, 0.75])
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm,cax=cax)
        if save ==True:
            plt.savefig(str(path)+'sbj'+str(sbj)+'acc'+str(acc)+'.'+format,format=format)
        return
    def rho_plot(self,rho,info,figsize=(5,5),save=True,sbj=0,acc=0,format='png',path='',Ch='s'):
        if Ch !='s':
            self.Ch = Ch
        rhoM = self.vreshape(rho)
        if self.feat == 'cwt+csp': 
            self.Wt_FB_plot_topomap(rhoM[:int(self.Ch)],info,figsize,save,'cwt'+str(sbj),acc,format)
            self.Wt_FB_plot_topomap(rhoM[int(self.Ch):],info,figsize,save,'csp'+str(sbj),acc,format)
        else:
            self.Wt_FB_plot_topomap(rhoM,info,figsize,save,sbj,acc,format,path)
        return
    def connection_rho(self,rho,fs,channels_names,thr=0.9,mean_by_time_windows=None,
                        mean_by_frequencies=None,figsize=(30,30),save=True,sbj='',acc='',
                        format='png',path='',Ch='s',cmap_tplt='jet',normalizate=True,
                        ch_ext=None,size_names=15):
        # whether mean_by_time_windows or mean_by_frequencies are True, the rho vector will be averaged over time or frequency.
        #but, also it can be a vector, such as, [0,0,1,1,2,2] that indicates  which indices of rho will be averaged.
        if Ch !='s':
            self.Ch = Ch
        self.P = int(0.5*self.Ch*(self.Ch-1))
        if ch_ext is not None:
            pCh = len(ch_ext)
            pp = int(0.5*pCh*(pCh-1))
            tmprho = rho.reshape(pp,-1)
            srhoMc= np.asarray([squareform(tmprho[:,i]) for i in range(tmprho.shape[-1])]).T
            chpt = np.isin(channels_names,ch_ext)
            tmpch = np.zeros((len(channels_names),len(channels_names),srhoMc.shape[-1]))
            for p in range(srhoMc.shape[-1]):
                k=0
                for i in range(len(chpt)):
                    if chpt[i]==True:
                        j=0
                        for ii in range(len(chpt)):
                            if chpt[ii]==True:
                                tmpch[i,ii,p] = srhoMc[j,k,p]
                                j+=1
                            else:
                                tmpch[i,ii,p] = 0
                        k+=1
                    else:
                        tmpch[i,:,p]=0
            ghj = np.asarray([squareform(tmpch[:,:,i]) for i in range(tmpch.shape[-1])]).T
            rho = ghj.ravel()
        rhoMc = rho.reshape(self.P,self.vtw.shape[0],self.f_frec.shape[0]) # Ch(Ch-1)/2, windows, bands
        if mean_by_frequencies is None:
            mean_by_frequencies = np.arange(self.f_frec.shape[0])
        if mean_by_time_windows is None:
            mean_by_time_windows = np.arange(self.vtw.shape[0])
        times = len(np.unique(mean_by_time_windows))
        frecs = len(np.unique(mean_by_frequencies))
        rhof=[]
        for i in np.unique(mean_by_frequencies):
            rhof.append(np.mean(rhoMc[:,:,mean_by_frequencies==i],axis=-1))
        rhof=np.array(rhof)
        #rhof = np.swapaxes(rhof,0,-1).squeeze()
        rhof = np.moveaxis(rhof, [0], [-1])
        rho = []
        for i in np.unique(mean_by_time_windows):
            rho.append(np.mean(rhof[:,mean_by_time_windows==i,:],axis=-2))
        rho=np.array(rho)
        #rho = np.swapaxes(rho,0,2).squeeze()
        rho = np.moveaxis(rho, [0], [-2])
        montage = mne.channels.read_montage('standard_1005')
        layout = mne.channels.read_layout('EEG1005')
        #montage = mne.channels.read_montage('biosemi128')
        #layout = montage.ch_names
        info = mne.create_info(layout.names, sfreq=fs, ch_types="eeg",
                                montage=montage)
        pos = np.array([(p[0] + p[2] / 2., p[1] + p[3] / 2.) for p in layout.pos])
        # pick channels
        picks = _picks_to_idx(info,channels_names)
        pos = pos[picks]
        # adjust positions
        pos, outlines = _check_outlines(pos, 'head')
        pos_x = pos[:,0]
        pos_y = pos[:,1]
        names = np.array(layout.names)[picks]
        if normalizate:
            rho=((rho-rho.min())/(rho.max()-rho.min()))
        cmap = plt.cm.jet
        #if normalizate:
        cNorm  = colors.Normalize(vmin=np.min(rho), vmax=np.max(rho))
        #else:
            #cNorm  = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
        rhotopo = np.zeros((Ch,frecs,times))
        for time in np.arange(times):
            for frec in np.arange(frecs):
                rhotopo[:,frec,time] = squareform(rho[:,time,frec]).sum(axis=1)
        rhotopo = rhotopo-rhotopo.min()
        rhotopo /= rhotopo.max()
        #f,ax = plt.subplots(frecs,times,figsize=figsize)
        fig=plt.figure(figsize=figsize)
        itr=1
        for time in np.arange(times): 
            for frec in np.arange(frecs):
                ax = fig.add_subplot(frecs,times,itr)
                itr+=1
                rhok = squareform(rho[:,time,frec])
                #plot_topomap(rhotopo[:,frec,time], pos, axes=ax[frec,time], cmap=cmap_tplt, show=False, contours=0, sensors=False,vmin=0,vmax=1)
                plot_topomap(rhotopo[:,frec,time], pos, axes=ax, cmap=cmap_tplt, show=False, contours=0, sensors=False,vmin=0,vmax=1)
                if thr >1:
                    indx_pct = np.where(np.triu(rhok)>np.percentile(rho.ravel()[rho.ravel()>1e-8], thr))
                else:
                    indx_pct = np.where(np.triu(rhok)>thr)
                #ax[frec,time].set(xticks=[], yticks=[], aspect='equal')
                ax.set(xticks=[], yticks=[], aspect='equal')
                #ax[frec,time].scatter(pos[:,0],pos[:,1],100)
                #_draw_outlines(ax[frec,time], outlines)
                _draw_outlines(ax, outlines)
                for i in np.arange(np.shape(indx_pct)[-1]):
                    ch1=indx_pct[0][i]
                    ch2=indx_pct[1][i]     
                    #ax[frec,time].arrow(pos_x[ch1],pos_y[ch1],pos_x[ch2]-pos_x[ch1],
                            #pos_y[ch2]-pos_y[ch1],head_width=0.0,length_includes_head=False,
                            #width=0.003,color=scalarMap.to_rgba(rhok[ch1,ch2]))
                    ax.arrow(pos_x[ch1],pos_y[ch1],pos_x[ch2]-pos_x[ch1],
                            pos_y[ch2]-pos_y[ch1],head_width=0.0,length_includes_head=False,   
                            width=0.003,color=scalarMap.to_rgba(rhok[ch1,ch2]))
                for ch1 in  np.unique(np.ravel(indx_pct)):
                    #ax[frec,time].scatter(pos_x[ch1],pos_y[ch1],50*np.sum(np.ravel(indx_pct)==ch1),'gray')
                    ax.scatter(pos_x[ch1],pos_y[ch1],50*np.sum(np.ravel(indx_pct)==ch1),'gray')
                    #ax[frec,time].annotate(names[ch1], xy=pos[ch1,:],size=15)
                    ax.annotate(names[ch1], xy=pos[ch1,:],size=size_names)
        #cax_con=fig.add_axes([ax[-1,-1].get_position().x1 + 0.05,axs[-1,-1].get_position().y0,0.02,axs[0,-1].get_position().y1-axs[-1,-1].get_position().y0])
        #cax_tplt=fig.add_axes([ax[-1,0].get_position().x0,axs[-1,0].get_position().y0 - 0.05,axs[-1,-1].get_position().x1-axs[-1,0].get_position().x0,0.02])
        cax_con = fig.add_axes([0.95, 0.15, 0.02, 0.75])
        cax_tplt = fig.add_axes([0.13, 0.13, 0.75, 0.02])
        # Mappeable objects for connectivities and topomaps colorbars
        sm1 = plt.cm.ScalarMappable(norm=cNorm,cmap=cmap)
        sm1.set_array(rho.ravel())
        sm2 = plt.cm.ScalarMappable(cmap=cmap_tplt)
        sm2.set_array(rhotopo.ravel())
        fig.colorbar(sm1,cax=cax_con)
        fig.colorbar(sm2,cax=cax_tplt, orientation='horizontal')
        if save ==True:
            plt.savefig(str(path)+'Cxsbj'+str(sbj)+'acc'+str(acc)+'.'+format,format=format)
        return
# %%
"""
