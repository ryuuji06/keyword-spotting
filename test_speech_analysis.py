# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import numpy as np
from glob import glob # module for finding pathnames
from time import time

import librosa
import tensorflow as tf
from matplotlib import pyplot as plt

# execute file from python interpreter
# exec(open('test_speech_analysis.py').read())

# =================================================================
# =================================================================
# TEST DATA PREPROCESSING: SPEECH ANALYSIS
# =================================================================
# =================================================================

# implement FFT (single window of STFT)
# np.fft

# HTK: library to implement hidden markov models

# problem: even a single keyword may have variable duration
#  - how to deal with it?
#  - DTW - dynamic warping?


# ---------------------------------------------------------------------

# dataset folder directory
dataset_path = 'C:\\datasets\\speech_commands'


# keywords are the own target value (to be one hot encoded)
keywords = ['house', 'five', 'down', 'left', 'learn', 'nine', 'one', 'cat']
MAX_SAMPLES = 500 # max number of samples per keyword

WORD_NUM = 2 # select keyword to exhibit
NUM_SHOW = 4 # number of examples to show

# ANALYSIS SETTINGS
# STFT specification
WINDOW_LEN = 320 # 200, 320
HOP_LEN = 80 # 80, 160
FFT_LEN = 512
# mel-frequency and MFCC specification
NUM_MEL = 60
NUM_MFCC = 20


# =================================================
print('\n(1) Load Data (Google Speech Commands)')
# =================================================

K = len(keywords)
input_paths = [] # will be a list of lists

# create list of input path, list of target values (own keywords)
for k in range(K):
	print(keywords[k])
	count = 0
	folder_files = []
	for file in os.listdir(os.path.join(dataset_path, keywords[k])):
		count += 1
		folder_files.append(os.path.join(dataset_path, keywords[k], file))
		if count >= MAX_SAMPLES:
			break
	input_paths.append(folder_files)
	#target_path = glob(('{}\\%s\\*.wav'%keywords[k]).format(dataset_path), recursive=True)
	#input_paths.append(target_path)

# for i in range(100):
# 	print(input_paths[0][i])


# =================================================
print('\n(2) Read Audio File')
# =================================================

# # with librosa

# # (A) get sampling rate and maximum number of samples
# sampling_rate = librosa.get_samplerate(input_paths[WORD_NUM][0])
# audio_lengths = np.zeros(NUM_SHOW,dtype=np.int64)
# for i in range(NUM_SHOW):
# 	audio_lengths[i] = sampling_rate * librosa.get_duration(filename=input_paths[WORD_NUM][i]) + 1
# AUDIO_LEN = np.max(audio_lengths)

# # (B) get audio signals
# sample_audio = np.zeros((NUM_SHOW,AUDIO_LEN))
# for i in range(NUM_SHOW):
# 	audio, _ = librosa.load(input_paths[WORD_NUM][i], sr=None)
# 	el = audio.shape[0]
# 	sample_audio[i,:el] = audio

# with tensorflow

# (A) PRE-ARANGE INTO LIST, AND MEASURE MAX LENGTH
audio_list = []
audio_lengths = np.zeros(NUM_SHOW,dtype=np.int64)
for i in range(NUM_SHOW):
	audio_bin = tf.io.read_file(input_paths[WORD_NUM][i])
	audio, sampling_rate = tf.audio.decode_wav(audio_bin, 1)
	audio_lengths[i] = audio.shape[0]
	audio_list.append(tf.squeeze(audio, axis=-1).numpy())
AUDIO_LEN = np.max(audio_lengths)

# (B) MOVE TO NUMPY ARRAY
sample_audio = np.zeros((NUM_SHOW,AUDIO_LEN))
for i in range(NUM_SHOW):
	audio, _ = librosa.load(input_paths[WORD_NUM][i], sr=None)
	el = audio_list[i].shape[0]
	sample_audio[i,:el] = audio_list[i]
del audio_list


# example plot
print('Example word: ', keywords[WORD_NUM])
plt.figure(1); plt.clf()
for i in range(NUM_SHOW):
	plt.subplot(NUM_SHOW,1,i+1); plt.grid()
	plt.plot(sample_audio[i,:])



# # ==============================================================
# print('\n(3) Test analysis: spectrogram and MFCC (librosa)')
# # ==============================================================

# # test direct mfcc
# test_mfcc1 = librosa.feature.mfcc(y=sample_audio[1,:], sr=sampling_rate,
# 	win_length=WINDOW_LEN, hop_length=HOP_LEN, n_fft=FFT_LEN, window='hann',
# 	n_mels=NUM_MEL, power=2.0,
# 	n_mfcc=20, dct_type=2, norm='ortho', lifter=0)

# # test melspec and mfcc
# test_mel_freq2 = librosa.feature.melspectrogram(y=sample_audio[1,:], sr=sampling_rate,
# 	win_length=WINDOW_LEN, hop_length=HOP_LEN, n_fft=FFT_LEN, window='hann',
# 	power=2.0, n_mels=NUM_MEL )
# test_mfcc2 = librosa.feature.mfcc(S=10*np.log10(test_mel_freq2), n_mfcc=NUM_MFCC, dct_type=2)

# # test spec, melspec and mfcc
# test_stft3 = librosa.stft( sample_audio[1,:], win_length=WINDOW_LEN, hop_length=HOP_LEN, n_fft=FFT_LEN, window='hann' )
# test_mel_freq3 = librosa.feature.melspectrogram(S=np.abs(test_stft3)**2, sr=sampling_rate, power=2.0, n_mels=NUM_MEL)
# test_mfcc3 = librosa.feature.mfcc(S=10*np.log10(test_mel_freq3), n_mfcc=NUM_MFCC, dct_type=2)

# plt.figure(1); plt.clf()
# plt.subplot(3,1,1)
# plt.imshow(test_mfcc1, cmap=plt.cm.jet, aspect='auto')
# plt.subplot(3,1,2)
# plt.imshow(test_mfcc2, cmap=plt.cm.jet, aspect='auto')
# plt.subplot(3,1,3)
# plt.imshow(test_mfcc3, cmap=plt.cm.jet, aspect='auto')

# plt.figure(2); plt.clf()
# plt.subplot(3,1,2)
# plt.imshow(10*np.log10(test_mel_freq2), cmap=plt.cm.jet, aspect='auto')
# plt.subplot(3,1,3)
# plt.imshow(10*np.log10(test_mel_freq3), cmap=plt.cm.jet, aspect='auto')

# plt.figure(3); plt.clf()
# plt.subplot(3,1,3)
# plt.imshow(10*np.log10(np.abs(test_stft3)**2), cmap=plt.cm.jet, aspect='auto')

# print('Audio size:', AUDIO_LEN)
# print('STFT length (expected):', (AUDIO_LEN-WINDOW_LEN+HOP_LEN)//HOP_LEN)
# print('STFT size:', test_stft3.shape)
# print('MFCC size:', test_mfcc3.shape)


# ==============================================================
print('\n(3) Test analysis: spectrogram and MFCC (tensorflow)')
# ==============================================================

# specifications for mel-frequencies
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80

# by default, this function uses von Hann window
# window_fn=tf.signal.hann_window
spectro = tf.signal.stft(audio, frame_length=WINDOW_LEN, frame_step=HOP_LEN, fft_length=FFT_LEN)

# mel-spectrogram
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    NUM_MEL, FFT_LEN//2+1, sampling_rate, lower_edge_hertz, upper_edge_hertz)
mel_spectrograms = tf.tensordot( tf.abs(spectro), linear_to_mel_weight_matrix, 1)
mel_spectrograms.set_shape( spectro.shape[:-1].concatenate(
	linear_to_mel_weight_matrix.shape[-1:]) )
# Compute a stabilized log to get log-magnitude mel-scale spectrograms.
log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

# Compute MFCCs from log_mel_spectrograms and take the first 13.
mfcc_full = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)

print('Audio size:', AUDIO_LEN)
print('STFT length (expected):', (AUDIO_LEN-WINDOW_LEN+HOP_LEN)//HOP_LEN)
print('STFT size:', spectro.shape)
print('MF size:', log_mel_spectrograms.shape)


plt.figure(2)
plt.subplot(3,1,1); plt.grid()
plt.plot(audio); plt.ylabel('Audio signal')
plt.subplot(3,1,2)
plt.imshow(10*np.log10(np.abs(tf.transpose(spectro))**2), cmap=plt.cm.jet, aspect='auto')
plt.ylabel('Spectrogram')
plt.subplot(3,1,3)
plt.imshow(tf.transpose(mfcc_full[:,:13]), cmap=plt.cm.jet, aspect='auto')
plt.ylabel('MFCC')

plt.show()



# ==================================================================