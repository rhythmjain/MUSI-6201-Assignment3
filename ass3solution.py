from __future__ import print_function
import numpy as np
import math
import scipy 
import os, glob
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os, os.path
from scipy import signal
from scipy.io.wavfile import read as wavread


def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2 ** (nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return samplerate, audio
def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])

    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t

def compute_hann_window(window_size):
    return 0.5*(1-(np.cos(2*np.pi*(np.arange(window_size)/window_size))))

def compute_spectogram2(xb,fs):
    [NumBlocks, blockSize] = xb.shape
    X_spec= np.zeros((NumBlocks, blockSize//2 + 1))

    def fourier(x):
        # Get Symmetric fft
        w = signal.windows.hann(np.size(x))
        windowed = x * w
        w1 = int((x.size + 1) // 2)
        w2 = int(x.size / 2)
        fftans = np.zeros(x.size)

        # Centre to make even function
        fftans[0:w1] = windowed[w2:]
        fftans[w2:] = windowed[0:w1]
        X = fft(fftans)
        magX = abs(X[0:int(x.size // 2 + 1)])
        return magX
    for block in range(xb.shape[0]):
        X_spec[::,block] = fourier(xb[block])
    fInHz = np.arange(0, blockSize, dtype=int)*fs/blockSize
    return X_spec,fInHz


def compute_spectogram(xb,fs):
    '''
    - Computes magnitude spectrum for each block of audio in xb, returns the magnitude spectogram X (blockSize/2+1 X NumOfBlocks)
    - A frequency vector fHz (dim blockSize/2+1) containing central frequency at each bim
    Apply a von-Hann window of appropriate length to the blocks
    '''
    [NumBlocks, blockSize] = xb.shape
    X = np.zeros((NumBlocks, blockSize//2 + 1))
    hann = compute_hann_window(blockSize)
    for block in range(NumBlocks):
        windowed_block = np.multiply(hann,xb[block])
        X[block] = np.abs(np.fft.fft(windowed_block)[:int(blockSize // 2 + 1)])

    fInHz = np.arange(0, blockSize, dtype=int)*fs/blockSize

    return X.T, fInHz

def track_pitch_fftmax(x, blockSize,hopSize,fs):
    '''
    Estimates fundamental frequency f0 of the audio signal based on a block-wise maxiumum spectral peak finding approach
    using compute_spectogram

    QUESTION: If the blockSize = 1024 for blocking what is the exact time resolution of your pitch tracker, 
    Can this be improved without changing the block-size? If yes, how? If no, why? (Use a sampling rate of 44100Hz for all calculations).
    '''
    f0 = 0
    timeInSec = 0
    return f0, timeInSec

#HPS


def get_f0_from_Hps(X,fs,order):
    '''
    Computes the block-wise fundamental frequency f0 given the magnitude spectrogram X 
    and the samping rate based on a HPS approach of order.

    THIS COMPUTES HPS FOR EACH BLOCK
    '''
    f_min = 300
    f0 = np.zeros(X.shape[1])
    max_length = int((X.shape[0]-1)/order)
    dwnpro = X[np.arange(0,max_length),:]
    #Convert Frequency to bins
    min_bin = int((f_min / fs)*2*(X.shape[0]-1))
    print(min_bin)
    fInHz = (np.arange(0, X.shape[0], dtype=int))*(fs)/2(X.shape[0]-1)
    for i in range(1,order):
        X_dwnsample = X[::i+1,::]
        dwnpro *= X_dwnsample[np.arange(0,max_length),:]
        #plt.figure()
        #plt.plot(dwnpro)
        #plt.show()
    f = np.argmax(dwnpro[np.arange(min_bin,dwnpro.shape[0])],axis=0)
    f0 = fInHz[f]
    
    return f0

def track_pitch_hps(x,blockSize,hopSize,fs):
    '''
    estimates the fundamental frequency f0 of the audio signal based on  HPS based approach. 
    Use blockSize = 1024 in compute_spectrogram(). Use order = 4 for get_f0_from_Hps()
    '''
    xb,timeInSec = block_audio(x,blockSize,hopSize,fs)
    X,finHz = compute_spectogram2(xb,fs)
    f0 = get_f0_from_Hps(X,fs,4)
    return f0,timeInSec

#************************------------------------------************************---------------------------******#
#Part 2: Voicing Detection

def extract_rms(xb):
    rmsDb = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        rmsDb[block] = np.sqrt(np.mean(np.square(xb[block])))
        threshold = 1e-5  # truncated at -100dB
        if rmsDb[block] < threshold:
            rmsDb[block] = threshold
        rmsDb[block] = 20 * np.log10(rms[block])
    return rmsDb

def create_voicing_mask(rmsDb, thesholdDb):
    '''
    takes a vector of decibel values for the different blocks of audio and creates a binary mask based on the threshold parameter. 
    Note: A binary mask in this case is a simple column vector of the same size as 'rmsDb' containing 0's and 1's only. 
    The value of the mask at an index is 0 if the rmsDb value at that index is less than 'thresholdDb' 
    and the value is 1 if 'rmsDb' value at that index is greater than or equal to the threshold. '''
    mask = np.ones((5,2))
    return mask

def apply_voicing_mask(f0,mask):
    '''
    which applies the voicing mask to the previously computed f0 so that the 
    f0 of blocks with low energy is set to 0.
    '''
    f0Adj = 0
    return f0Adj

#************************------------------------------************************---------------------------******#

#D Different Evaluation Metrics
def eval_voiced_fp(estimation,annotation):
    '''computes the percentage of false positives for your fundamental frequency estimation 
    False Positive : The denominator would be the number of blocks for which annotation = 0. 
    The numerator would be how many of these blocks were classified as voiced (with a fundamental frequency not equal to 0) is your estimation. 
    '''
    pfp=0
    return pfp

def eval_voiced_fn(estimation,annotation):
    '''
    computes the percentage of false negatives for your fundamental frequency estimation
    False Negative: In this case the denominator would be number of blocks which have non-zero fundamental frequency in the annotation. 
    The numerator would be number of blocks out of these that were detected as zero is the estimation.
    '''
    pfn =0
    return pfn

def eval_pitchtrack_v2(estimation,annotation):
    '''
    return all the 3 performance metrics for your fundamental frequency estimation.  
    Note: the errorCentRms computation might need to slightly change now considering that your estimation might also contain zeros.

    '''
    errCentRms=0
    pfp=0
    pfn=0
    return errCentRms,pfp,pfn


#************************------------------------------************************---------------------------******#
#E EVALUATION

def executeassign3():
    '''
    generate a test signal (sine wave, f = 441 Hz from 0-1 sec and f = 882 Hz from 1-2 sec), apply your track_pitch_fftmax(), (blockSize = 1024, hopSize = 512) and plot the f0 curve. 
    Also, plot the absolute error per block and discuss the possible causes for the deviation. Repeat for track_pitch_hps() with the same signal and parameters. 
    Why does the HPS method fail with this signal?

    [5 points] Next use (blockSize = 2048, hopSize = 512) and repeat the above experiment (only for the max spectra method). Do you see any improvement in performance? 
    '''
    return 0
'''
[5 points] Evaluate your track_pitch_fftmax() using the development set (see assignment 1) and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set.
[5 points] Evaluate your track_pitch_hps() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set. 
[6 points] Implement a MATLAB wrapper function [f0Adj, timeInSec] = track_pitch(x, blockSize, hopSize, fs, method, voicingThres) that takes audio signal ‘x’ and related paramters (fs, blockSize, hopSize), calls the appropriate pitch tracker based on the method parameter (‘acf’,‘max’, ‘hps’) to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter. 
[6 points] Evaluate your track_pitch() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512) over all 3 pitch trackers (acf, max and hps) and report the results with two values of threshold (threshold = -40, -20)
'''
def track_pitch(x,blockSize,hopSize,fs,method,voicingThres):
    '''
    that takes audio signal ‘x’ and related paramters (fs, blockSize, hopSize), 
    calls the appropriate pitch tracker based on the method parameter (‘acf’,‘max’, ‘hps’) to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter.
    '''
    f0Adj = 0
    timeInSec = 0
    return f0Adj,timeInSec


#************************------------------------------************************---------------------------******#
#BONUS
def track_pitch_mod(x,blockSize,hopSize,fs):
    '''
    [10 points, capped at max] 
    Implement a function: [f0, timeInSec] = track_pitch_mod(x, blockSize, hopSize, fs) that combines ideas from different pitch trackers you have tried thus far 
    and thereby provides better f0 estimations. You may include voicing detection within this method with parameters of your choosing. 
    Please explain your approach in the report. Your function will be tested using a testing set (not provided) with a block size of 1024 and a hopsize of 512, 
    and points will be given based on its performance compared to the other groups. Best performing group gets 10 points and worst performing group gets 1 point. 
    '''
    f0=0
    timeInSec = 0

    return f0,timeInSec
