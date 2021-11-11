from __future__ import print_function
import numpy as np
import scipy 
import os, glob
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os, os.path
from scipy.io.wavfile import read as wavread
import math

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * 
np.arange(iWindowLength)))

def compute_spectrogram(xb):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(scipy.fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 
#let's be pedantic about normalization
    
    return X

def get_f0_from_Hps(X,fs,order):
    '''
    Computes computes the block-wise fundamental frequency f0 given the magnitude spectrogram X 
    and the samping rate based on a HPS approach of order order.

    THIS COMPUTES HPS FOR EACH BLOCK
    '''
    f_min = 200
    f = np.zeros(X.shape[0])
    Length =  int((X.shape[1]-1)/order)
    afHps = X[::,np.arange(0,Length)]
    #Convert Frequency to bins
    min_bin = int(round(f_min / fs * 2 * (X.shape[1] - 1)))


    for i in range(1,order):
             X_d = X[::,::i+1]
             afHps *= X_d[::,np.arange(0,Length)]
    f = np.argmax(afHps[::,np.arange(min_bin,afHps.shape[1])], axis=1)
    f0 = (f + min_bin) / (X.shape[1] - 1) * fs / 2
    
    return f0

def track_pitch_hps(x,blockSize,hopSize,fs):
    '''
    estimates the fundamental frequency f0 of the audio signal based on  HPS based approach. 
    Use blockSize = 1024 in compute_spectrogram(). Use order = 4 for get_f0_from_Hps()
    '''
    f0=0
    timeInSec = 0
    
    return f0,timeInSec