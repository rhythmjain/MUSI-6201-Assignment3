from __future__ import print_function
import numpy as np
import math
import scipy as sc 
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
"""
def compute_spectrogram2(xb,fs):
    [NumBlocks, blockSize] = xb.shape
    X_spec= np.zeros((blockSize//2 + 1,NumBlocks))

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
        magX = abs(X[0:int(blockSize//2 + 1)])
        return magX
    for block in range(xb.shape[0]):
        X_spec[::,block] = fourier(xb[block])
    fInHz = np.arange(0, blockSize, dtype=int)*fs/blockSize
    return X_spec,fInHz
"""

def compute_spectrogram(xb,fs):
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
    xb, t = block_audio(x, blockSize, hopSize, fs)
    X, freq= compute_spectrogram(xb, fs)
    #print(t.size, freq.size, X.shape)
    f0 = np.zeros(xb.shape[0])
    for i in range(X.shape[1]):
        f0[i] = freq[np.argmax(X[:,i])]
    #print(f0.size)
    return f0, t

def get_f0_from_Hps(X,fs,order):
    '''
    Computes the block-wise fundamental frequency f0 given the magnitude spectrogram X 
    and the samping rate based on a HPS approach of order.

    THIS COMPUTES HPS FOR EACH BLOCK
    '''
    f_min = 200
    f0 = np.zeros(X.shape[1])
    max_length = int((X.shape[0]-1)/order)
    dwnpro = X[np.arange(0,max_length),:]
    #Convert Frequency to bins
    min_bin = int((f_min / fs)*2*(X.shape[0]-1))
    #sprint(min_bin)
    fInHz = (np.arange(0, X.shape[0], dtype=int))*(fs)/(2*(X.shape[0]-1))

    for i in range(1,order):
        X_dwnsample = X[::i+1,::]
        dwnpro *= X_dwnsample[np.arange(0,max_length),:]
        #plt.figure()
        #plt.plot(dwnpro)
        #plt.show()
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
    X,finHz = compute_spectrogram(xb,fs)
    f0 = get_f0_from_Hps(X,fs,4)
    return f0,timeInSec

def comp_acf(inputVector, bIsNormalized = True):
    if bIsNormalized:
        norm = np.dot(inputVector, inputVector)
    else:
        norm = 1
    afCorr = np.correlate(inputVector, inputVector, "full") / norm
    afCorr = afCorr[np.arange(inputVector.size-1, afCorr.size)]
    return (afCorr)

def get_f0_from_acf (r, fs):
    eta_min = 1
    afDeltaCorr = np.diff(r)
    eta_tmp = np.argmax(afDeltaCorr > 0)
    eta_min = np.max([eta_min, eta_tmp])
    f = np.argmax(r[np.arange(eta_min + 1, r.size)])
    f = fs / (f + eta_min + 1)
    return (f)

def track_pitch_acf(x,blockSize,hopSize,fs):
    
    # get blocks
    [xb,t] = block_audio(x,blockSize,hopSize,fs)
    # init result
    f0 = np.zeros(xb.shape[0]
                  )
    # compute acf
    for n in range(0, xb.shape[0]):
        r = comp_acf(xb[n,:])
        f0[n] = get_f0_from_acf(r,fs)
    return (f0,t)
        
        
#************************------------------------------************************---------------------------******#
#Part 2: Voicing Detection

def extract_rms(xb):
    rmsDb = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        rmsDb[block] = np.sqrt(np.mean(np.square(xb[block])))
        threshold = 1e-5  # truncated at -100dB
        if rmsDb[block] < threshold:
            rmsDb[block] = threshold
        rmsDb[block] = 20 * np.log10(rmsDb[block])
    return rmsDb

def create_voicing_mask(rmsDb, thresholdDb):
    '''
    takes a vector of decibel values for the different blocks of audio and creates a binary mask based on the threshold parameter. 
    Note: A binary mask in this case is a simple column vector of the same size as 'rmsDb' containing 0's and 1's only. 
    The value of the mask at an index is 0 if the rmsDb value at that index is less than 'thresholdDb' 
    and the value is 1 if 'rmsDb' value at that index is greater than or equal to the threshold. '''
    mask = np.zeros_like(rmsDb)
    mask = [1 if i>=thresholdDb else 0 for i in rmsDb]
    return mask

def apply_voicing_mask(f0,mask):
    '''
    which applies the voicing mask to the previously computed f0 so that the 
    f0 of blocks with low energy is set to 0.
    '''
    f0Adj = np.multiply(mask,f0)
    return f0Adj

#************************------------------------------************************---------------------------******#

#D Different Evaluation Metrics
def eval_voiced_fp(estimation,annotation):
    '''computes the percentage of false positives for your fundamental frequency estimation 
    False Positive : The denominator would be the number of blocks for which annotation = 0. 
    The numerator would be how many of these blocks were classified as voiced (with a fundamental frequency not equal to 0) is your estimation. 
    '''
    
    numerator = np.count_nonzero(estimation[annotation==0])
    denominator = np.count_nonzero([annotation==0])
    false_positives_percentage = -1 #Need to check for these in the calling functions
    try:
        false_positives_percentage = (numerator/denominator)*100
    except ZeroDivisionError:
        print("Denominator is equal to zero!")    
    
    return false_positives_percentage

def eval_voiced_fn(estimation,annotation):
    '''
    computes the percentage of false negatives for your fundamental frequency estimation
    False Negative: In this case the denominator would be number of blocks which have non-zero fundamental frequency in the annotation. 
    The numerator would be number of blocks out of these that were detected as zero is the estimation.
    '''
    numerator = np.count_nonzero([estimation[annotation!=0]==0])
    denominator = np.count_nonzero([annotation!=0])
    false_negatives_percentage = -1 #Need to check for these in the calling functions
    try:
        false_negatives_percentage = (numerator/denominator)*100
    except ZeroDivisionError:
        print("Denominator is equal to zero!")  
    
    return false_negatives_percentage

def convert_freq2midi(fInHz, fA4InHz = 440):
        def convert_freq2midi_scalar(f, fA4InHz):
 
            if f <= 0:
                return 0
            else:
                return (69 + 12 * np.log2(f/fA4InHz))
        fInHz = np.asarray(fInHz)
        if fInHz.ndim == 0:
            return convert_freq2midi_scalar(fInHz,fA4InHz)
        midi = np.zeros(fInHz.shape)
        for k,f in enumerate(fInHz):
            midi[k] =  convert_freq2midi_scalar(f,fA4InHz)
                
        return (midi)
    
def eval_pitchtrack(estimateInHz, groundtruthInHz):
        if np.abs(groundtruthInHz).sum() <= 0:
            return 0
        # truncate longer vector
        if groundtruthInHz.size < estimateInHz.size:
            estimateInHz = estimateInHz[np.arange(0,groundtruthInHz.size)]
        elif estimateInHz.size < groundtruthInHz.size:
            groundtruthInHz = groundtruthInHz[np.arange(0,estimateInHz.size)]
        diffInCent = 100*(convert_freq2midi(estimateInHz) - 
    convert_freq2midi(groundtruthInHz))
        rms = np.sqrt(np.mean(diffInCent**2))
        return (rms)
    
def eval_pitchtrack_v2(estimation,annotation):
    '''
    return all the 3 performance metrics for your fundamental frequency estimation.  
    Note: the errorCentRms computation might need to slightly change now considering that your estimation might also contain zeros.

    '''    
    pfp= eval_voiced_fp(estimation,annotation)
    pfn= eval_voiced_fn(estimation,annotation)
    errCentRms= eval_pitchtrack(estimation, annotation)
    
    return errCentRms,pfp,pfn


#************************------------------------------************************---------------------------******#
#E EVALUATION
def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs):
    t = np.arange(0, length_secs,1.0/(sampling_rate_Hz))
    x = amplitude * np.sin(2*np.pi*frequency_Hz*t)
    return t,x

def plot(x,y,title="Plot", xlabel="xlabel", ylabel="ylabel"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.show()
    
def executeassign3():
    '''
    generate a test signal (sine wave, f = 441 Hz from 0-1 sec and f = 882 Hz from 1-2 sec), apply your track_pitch_fftmax(), (blockSize = 1024, hopSize = 512) and plot the f0 curve. 
    Also, plot the absolute error per block and discuss the possible causes for the deviation. Repeat for track_pitch_hps() with the same signal and parameters. 
    Why does the HPS method fail with this signal?

    [5 points] Next use (blockSize = 2048, hopSize = 512) and repeat the above experiment (only for the max spectra method). Do you see any improvement in performance? 
    '''
    
    fs = 44100
    A = 1
    sn1 = generateSinusoidal(A, fs, 441, 1)
    sn2 = generateSinusoidal(A, fs, 882, 1)
    t = np.array(sn1[0])
    t= np.append(t, sn2[0])

    signal = sn1[1]
    signal = np.append(signal, sn2[1])

    #blockSize 1024 and hopSize 512
    
    #Run track_pitch_fftmax
    f0Est1, timeEst1 = track_pitch_fftmax(signal, 1024, 512, fs)
    groundtruth = [441 if i < (f0Est1.size/2) - 1 else 882 for i in range(f0Est1.size)]
    abs_error1 = np.abs(groundtruth - f0Est1)
    
    #plot the f0 curve
    plot(np.arange(f0Est1.size), f0Est1, "F0 curve with FFTMAX - test signal,  blockSize 1024 and hopSize 512", "Block number", "F0(Hz)")
    
    #plot the absolute error per block
    plot(np.arange(f0Est1.size), abs_error1, "Absolute error per block - test signal, blockSize 1024 and hopSize 512", "Block number", "Absolute Error(Hz)")
    
    #Run track_pitch_hps
    f0Est2, timeEst2 = track_pitch_hps(signal, 1024, 512, fs)
    groundtruth2 = [441 if i < (f0Est2.size/2) - 1 else 882 for i in range(f0Est2.size)]
    abs_error_hps = np.abs(groundtruth2 - f0Est2)
    
    #plot the f0 curve for hps
    
    plot(np.arange(f0Est2.size), f0Est2, "F0 curve with HPS - test signal, blockSize 1024 and hopSize 512", "Block number", "F0(Hz)")
    
    #plot the absolute error per block for hps
    plot(np.arange(f0Est2.size), abs_error_hps, "Absolute error per block with HPS - test signal, blockSize 1024 and hopSize 512", "Block number", "Absolute Error(Hz)")
    
    #blockSize 2048 and hopSize 512
    
    #Run track_pitch_fftmax with 
    f0Est3, timeEst3=track_pitch_fftmax(signal, 2048, 512, fs)
    groundtruth3 = [441 if i < (f0Est3.size/2) - 1 else 882 for i in range(f0Est3.size)]
    abs_error_fftmax2 = np.abs(groundtruth3 - f0Est3)
    
    #plot the f0 curve for track_pitch_fftmax
    plot(np.arange(f0Est3.size), f0Est3, "F0 curve with FFTMAX - test signal, blockSize 2048 and hopSize 512", "Block number", "F0(Hz)")
    
    #plot the absolute error per block for track_pitch_fftmax
    plot(np.arange(f0Est3.size), abs_error_fftmax2, "Absolute error per block with FFTMAX - test signal, blockSize 2048 and hopSize 512", "Block number", "Absolute Error(Hz)")
    
    
    """
    
    fs = 44100
    f1= 441
    f2 = 882
    duration = [1,1]
    x = np.zeros(2*fs)
    x[0:fs] = np.arange(0,fs*duration[0])*(f1/fs)
    x[fs:]= np.arange(0,fs*duration[1])*(f2/fs)
    SineWave = np.sin(2*np.pi*x)
    xb, t = block_audio(SineWave, 1024, 512, fs)
    f0_gnd = [f1 if t<=1 else f2 for i in xb]
    f0_fft,t_fft =  track_pitch_fftmax(SineWave, 1024,512,fs)
    f0_hps, t_hps = track_pitch_hps(SineWave,1024,512,fs)
    ferr_fft = np.abs(np.diff((f0_fft,f0_gnd)))
    plt.figure()
    plt.plot(f0_fft)
    plt.title('Fft max F0')
    plt.xlabel('Blocks')
    plt.ylabel('F0')    
    plt.figure()
    plt.plot(f0_hps)
    plt.title('HPS F0')
    plt.xlabel('Blocks')
    plt.ylabel('F0')
    plt.plot(ferr_fft)
    plt.title('Fft Max F0 Error')
    plt.xlabel('Blocks')
    plt.ylabel('Error')
    

    #Experiment with (blockSize = 2048, hopSize = 512)

    f0_fft2,t_fft2 =  track_pitch_fftmax(SineWave, 1024,512,fs)
    xb2, t2 = block_audio(SineWave, 2048, 512, fs)
    f0_gnd2 = [f1 if t<=1 else f2 for i in xb2]
    ferr_fft = np.abs(np.diff((f0_fft2,f0_gnd2)))
    """ 
    return 0
'''
[5 points] Evaluate your track_pitch_fftmax() using the development set (see assignment 1) and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set.
[5 points] Evaluate your track_pitch_hps() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set. 
[6 points] Implement a MATLAB wrapper function [f0Adj, timeInSec] = track_pitch(x, blockSize, hopSize, fs, method, voicingThres) that takes audio signal ‘x’ and related paramters (fs, blockSize, hopSize), calls the appropriate pitch tracker based on the method parameter (‘acf’,‘max’, ‘hps’) to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter. 
[6 points] Evaluate your track_pitch() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512) over all 3 pitch trackers (acf, max and hps) and report the results with two values of threshold (threshold = -40, -20)
'''
def run_evaluation_fftmax(complete_path_to_data_folder):
    # init
    rmsAvg = 0
    pfp=0
    pfn=0
    pfp_sum = 0
    pfn_sum = 0
    rmsAvg_sum = 0

    iNumOfFiles = 0
    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)
            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(complete_path_to_data_folder + os.path.splitext(file)[0]+'.f0.Corrected.txt')
        else:
            continue
        # extract pitch
        [f0,t] = track_pitch_fftmax(afAudioData,1024,512,fs)
        # compute rms and accumulate
        rmsAvg,pfp,pfn = eval_pitchtrack_v2(f0, refdata[:,2])
        rmsAvg_sum +=rmsAvg
        pfp_sum += pfp
        pfn_sum += pfn
        
    if iNumOfFiles == 0:
        return -1
    return rmsAvg_sum/iNumOfFiles , pfp_sum/iNumOfFiles , pfn_sum/iNumOfFiles

def run_evaluation_hps(complete_path_to_data_folder):
    # init
    rmsAvg = 0
    pfp=0
    pfn=0
    pfp_sum = 0
    pfn_sum = 0
    rmsAvg_sum = 0

    iNumOfFiles = 0
    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)
            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(complete_path_to_data_folder + os.path.splitext(file)[0]+'.f0.Corrected.txt')
        else:
            continue
        # extract pitch
        [f0,t] = track_pitch_hps(afAudioData,1024,512,fs)
        # compute rms and accumulate
        rmsAvg,pfp,pfn = eval_pitchtrack_v2(f0, refdata[:,2])
        rmsAvg_sum +=rmsAvg
        pfp_sum += pfp
        pfn_sum += pfn

    if iNumOfFiles == 0:
        return -1
    return rmsAvg_sum/iNumOfFiles , pfp_sum/iNumOfFiles , pfn_sum/iNumOfFiles

def track_pitch(x,blockSize,hopSize,fs,method,voicingThres):
    '''
    that takes audio signal ‘x’ and related paramters (fs, blockSize, hopSize), 
    calls the appropriate pitch tracker based on the method parameter (‘acf’,‘max’, ‘hps’) to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter.
    '''
    
    if method == 'acf':
        f0, timeInSec = track_pitch_acf(x,blockSize, hopSize, fs)
    elif method == 'hps':
        f0, timeInSec = track_pitch_hps(x,blockSize, hopSize, fs)
    elif method == 'max':
        f0, timeInSec = track_pitch_fftmax(x,blockSize, hopSize, fs)

    xb, t = block_audio(x, blockSize, hopSize, fs)
    rmsDb = extract_rms(xb)
    voicing_mask = create_voicing_mask(rmsDb, voicingThres)
    f0Adj = apply_voicing_mask(f0, voicing_mask)
    
    return f0Adj, timeInSec

def evaluate_trackpitch(complete_path_to_data_folder):
    methods = ['acf','hps','max']
    thresholds = [-40,-20]
    iNumOfFiles = 0
    rms_avg = np.zeros((3,2))
    pfp_avg = np.zeros((3,2))
    pfn_avg = np.zeros((3,2))
    # for loop over files
    for tr in range (len(thresholds)):
        for met in range(len(methods)):
            rmsAvg = 0
            pfp=0
            pfn=0
            pfp_sum = 0
            pfn_sum = 0
            rmsAvg_sum = 0

            iNumOfFiles = 0
            for file in os.listdir(complete_path_to_data_folder):
                if file.endswith(".wav"):
                    iNumOfFiles += 1
                    # read audio
                    [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)
                    # read ground truth (assume the file is there!)
                    refdata = np.loadtxt(complete_path_to_data_folder + os.path.splitext(file)[0]+'.f0.Corrected.txt')
                else: #this is redundant
                    continue

                # extract pitch
                f0Adj, timeInSec = track_pitch(afAudioData,1024,512,fs,methods[met],thresholds[tr])
                # compute rms and accumulate
                rmsAvg,pfp,pfn = eval_pitchtrack_v2(f0Adj, refdata[:,2])
                rmsAvg_sum +=rmsAvg
                pfp_sum += pfp
                pfn_sum += pfn
            rms_avg[met,tr] = rmsAvg_sum/iNumOfFiles
            pfp_avg[met,tr] = pfp_sum/iNumOfFiles
            pfn_avg[met,tr] = pfn_sum/iNumOfFiles

    return rms_avg,pfp_avg,pfn_avg


#************************------------------------------************************---------------------------******#
#BONUS

# def zero_padding(xb):
#     xz_b = np.zeros(xb.shape[0],xb.shape[1]+int(xb.shape[1]//2))
#     for i in xb:
#     return xz_b
def track_pitch_mod(x,blockSize,hopSize,fs):
    '''
    [10 points, capped at max] 
    Function that combines ideas from different pitch trackers you have tried thus far and thereby provides better f0 estimations. You may include voicing 
    detection within this method with parameters of your choosing. Please explain your approach in the report. Your function will be tested using a testing set 
    (not provided) with a block size of 1024 and a hopsize of 512, and points will be given based on its performance compared to the other groups. Best performing
    group gets 10 points and worst performing group gets 1 point. 
    '''
    def zero_padding(xb):
     xz_b = np.zeros((xb.shape[0],xb.shape[1]+int(xb.shape[1])))
     for i in range(xb.shape[1]):
        xz_b[i,:][0:xb.shape[1]] = xb[i,:]
     return xz_b
       
    x = np.sign(x)
    xb,timeInSec = block_audio(x,blockSize,hopSize,fs)
    xz_b = zero_padding(xb)
    XZ, fInHz = compute_spectrogram(xz_b,fs)
    f0 = np.zeros((timeInSec.shape[0]))
    r_arr = np.zeros(xb.shape)
    for i in range(XZ.shape[1]):
        r_arr[i] = comp_acf(XZ[:,i])
        f0[i] = get_f0_from_acf(r,fs)
    return (f0,r_arr,timeInSec)


complete_path_to_data_folder = "/Users/rhythmjain/Desktop/GTStuff/1-2/AudioContent/A03/MUSI-6201-Assignment3/trainData/"
evaluate_trackpitch(complete_path_to_data_folder)
