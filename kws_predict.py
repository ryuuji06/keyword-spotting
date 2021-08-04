# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import pickle
from glob import glob # module for finding pathnames
from time import time

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers

from auxiliar_functions import english_standardization, librispeech_pair_samples
from auxiliar_functions import speech_cmds_pair_samples, speech_cmds_edit_pair_samples, arrange_data
from models import build_model_1, build_model_2
from matplotlib import pyplot as plt

# execute file from python interpreter
# exec(open('kws_predict.py').read())


# =================================================================
# =================================================================
# KEYWORD SPOTTING USING CNN-RNN-CTC
# PREDICTION AND EVALUATION STAGE
# =================================================================
# =================================================================

# during training
# - in dataset, perform map to add random delay at the beginning and add noise signal
# - record long noise track

# implement posterior handling, in order to evaluate accuracy
#  - remove null-elements, remove zeros: if match -> sum 0.6
#  - insert back zeros: if match number and positions -> sum 0.4

#  - correct number and order of tokens: 1
#  - absence of keyword token (when there is) or wrong token: 0
#  - correct token, but wrong number of zeros: [0,1]

# how to verify alignment?
#  - consider strides along network
#  - manually, exporting audio superposed to sinusoid

# much larger/deeper network


# --------------------------------------------
# (1) Datapaths and keywords
# --------------------------------------------

# cmds1_path = 'D:\\Users\\Administrador\\Documents\\DLtests\\datasets\\speech_commands'
# cmds2_path = 'D:\\Users\\Administrador\\Documents\\DLtests\\datasets\\speech_commands_edit'
# libri1_path = "D:\\Users\\Administrador\\Documents\\DLtests\\datasets\\LibriSpeech\\dev-clean"
# libri2_path = "D:\\Users\\Administrador\\Documents\\DLtests\\datasets\\LibriSpeech\\test-clean"
cmds1_path = 'C:\\datasets\\speech_commands'
cmds2_path = 'C:\\datasets\\speech_commands_edit'
libri1_path = "C:\\datasets\\LibriSpeech\\dev-clean"
libri2_path = "C:\\datasets\\LibriSpeech\\test-clean"

#keywords = ['house', 'five', 'down', 'left', 'learn', 'nine', 'one', 'cat']
keywords = ['house', 'right', 'down', 'left', 'no', 'five', 'one', 'three']
NUM_KWD = len(keywords)
# total of NUM_KWD+2 outputs: keywords, unknown and CTC-null

# select model to load
#folder, model_name = 'results02', 'model-100-0.668.h5'
folder, model_name = 'results03', 'model-78-1.129.h5'


# --------------------------------------------
# (2) Dataset and Training Configuration
# --------------------------------------------

MAX_NUM_ISOLATED_KWD = 1000
MAX_TEXT_LEN = 40  # max num of tokens
NUM_NONKWD_SAMP = 500

# --------------------------------------------
# (3) Feature Extraction Parameters
# --------------------------------------------

SAMPLING_RATE = 16000 # 16000.0, 22050.0

# STFT SPECIFICATION
FRAME_LENGTH = 256 # 200
FRAME_STEP = 128 # 80
FFT_LENGTH = 256

# MFCC SPECIFICATION
LOWER_FREQ = 80.0
UPPER_FREQ = 7600.0
N_MEL_BINS = 80
N_MFCC_BINS = 13

FEAT_DIM = N_MFCC_BINS



# ----------------------------------------------
#   F E A T U R E   E X T R A C T I O N
# ----------------------------------------------

# argument is a batch of pairs (audio_path,target_text)
@tf.function
def path2audio(path):
	audio_bin = tf.io.read_file(path)
	audio, _ = tf.audio.decode_wav(audio_bin, 1)
	return tf.squeeze(audio, axis=-1)

# @tf.function
# def audio2spec(audio):
#     # spectrogram using stft
#     # by default, this function uses von Hann window (window_fn=tf.signal.hann_window)
#     x = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
#     # option 1: matlibplot style - squared module, and take log
#     x = tf.math.log( tf.math.pow( tf.abs(x),2 ) + 1e-8 )
#     # option 2: simple square root
#     # x = tf.math.pow(tf.abs(x), 0.5)
#     return x

@tf.function
def audio2mfcc(audio):
	# spectrogram using stft
	# by default, this function uses von Hann window (window_fn=tf.signal.hann_window)
	stfts = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
	# mel-frequencies spectrogram
	linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
		N_MEL_BINS, stfts.shape[-1], SAMPLING_RATE, LOWER_FREQ, UPPER_FREQ)
	mel_spectrograms = tf.tensordot(
		tf.abs(stfts), linear_to_mel_weight_matrix, 1)
	mel_spectrograms.set_shape( stfts.shape[:-1].concatenate(
		linear_to_mel_weight_matrix.shape[-1:]) )
	# compute and return MFCCs
	return tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectrograms + 1e-8))[:,:N_MFCC_BINS]

text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=english_standardization,
    max_tokens=None,
    vocabulary=keywords)



# =====================================================
print('\n(1) LOAD DATA INTO LISTS')
# =====================================================
# (has separate train-test folders, but I use only one)

# (a) VERIFY SAMPLING RATE

_, sr1 = librosa.load(cmds1_path+'\\bed\\00f0204f_nohash_0.wav', sr=None)
_, sr2 = librosa.load(cmds2_path+'\\0-zero-zero-house-down-one.wav', sr=None)
_, sr3 = librosa.load(libri1_path+'\\84\\121123\\84-121123-0001.wav', sr=None)
_, sr4 = librosa.load(libri2_path+'\\61\\70968\\61-70968-0000.wav', sr=None)
#_, sr3 = librosa.load(lj_path+'\\wavs\\LJ001-0002.wav', sr=None)
print('Verify sampling rate:', sr1, sr2, sr3)


# (b) READ DATA (input audio file path and label text)

t1 = time()
input_paths_1, target_texts_1 = speech_cmds_pair_samples(cmds1_path, MAX_NUM_ISOLATED_KWD, keywords)
input_paths_2, target_texts_2 = speech_cmds_edit_pair_samples(cmds2_path)
input_paths_3, target_texts_3 = librispeech_pair_samples(libri1_path)
input_paths_4, target_texts_4 = librispeech_pair_samples(libri2_path)

t2 = time(); print('a) Load and arrange transcription pairs: %.3fs'%(t2-t1))
print('Single speech commands size:', len(input_paths_1))
print('Edited speech commands size:', len(input_paths_2))

# (c) FILTER SAMPLES BY MAXIMUM LABEL LENGTH

kwd_occs_3, kwd_inds_3, nonkwd_inds_3 = arrange_data(
	input_paths_3, target_texts_3, text_processor, NUM_KWD, MAX_TEXT_LEN, NUM_NONKWD_SAMP)

input_paths_3_kwd  = [input_paths_3[i] for i in kwd_inds_3]
target_texts_3_kwd = [target_texts_3[i] for i in kwd_inds_3]
input_paths_3  = [input_paths_3[i] for i in nonkwd_inds_3] # update existing list
target_texts_3 = [target_texts_3[i] for i in nonkwd_inds_3]

print('Librispeech dev size (keywords):', len(input_paths_3_kwd))
print('Librispeech dev size (nonkeyword):', len(input_paths_3))

kwd_occs_4, kwd_inds_4, nonkwd_inds_4 = arrange_data(
	input_paths_4, target_texts_4, text_processor, NUM_KWD, MAX_TEXT_LEN, NUM_NONKWD_SAMP)

input_paths_4_kwd  = [input_paths_4[i] for i in kwd_inds_4]
target_texts_4_kwd = [target_texts_4[i] for i in kwd_inds_4]
input_paths_4  = [input_paths_4[i] for i in nonkwd_inds_4] # update existing list
target_texts_4 = [target_texts_4[i] for i in nonkwd_inds_4]

print('Librispeech test size (keywords):', len(input_paths_4_kwd))
print('Librispeech test size (nonkeyword):', len(input_paths_4))

print('Number of samples per kerword')
for i in range(NUM_KWD):
	 print('',keywords[i],kwd_occs_3[i]+kwd_occs_4[i])


# (d) LOAD MODEL

# load model loss history
with open(folder+"/hist.pickle", 'rb') as f:
    hist = pickle.load(f)

# load model
model_train, model_pred = build_model_2(FEAT_DIM, NUM_KWD)
model_pred.load_weights(folder+'/'+model_name)



# =====================================================
print('\n(2) SELECT AND ARRANGE DATA TO PREDICTION')
# =====================================================

N = 4 # pick up N examples at random

# choose dataset
#inp_path, tar_text = input_paths_1, target_texts_1
#inp_path, tar_text = input_paths_2, target_texts_2
#inp_path, tar_text = input_paths_3, target_texts_3
#inp_path, tar_text = input_paths_4, target_texts_4
inp_path, tar_text = input_paths_3_kwd, target_texts_3_kwd
#inp_path, tar_text = input_paths_4_kwd, target_texts_4_kwd

rnd_ind = np.random.choice(200,N) # choose N among 200 first samples
inp_path = [ inp_path[i] for i in rnd_ind ]
tar_text = [ tar_text[i] for i in rnd_ind ]

# (B) ARRANGE DATA INTO A SINGLE PADDED BATCH

audio_ds = tf.data.Dataset.from_tensor_slices(inp_path)
audio_ds = audio_ds.map(path2audio)
feats_ds = audio_ds.map(audio2mfcc)
audio_ds = audio_ds.padded_batch(N)
feats_ds = feats_ds.padded_batch(N)

text_ds = tf.data.Dataset.from_tensor_slices(tar_text)
text_ds = text_ds.map(lambda x: text_processor(x)-1)
text_ds = text_ds.padded_batch(N, padding_values=np.int64(NUM_KWD+1))

for audio_batch in audio_ds:
	print('Done')
for feats_batch in feats_ds:
	print('Done')
for text_batch in text_ds:
	print('Done')


# =====================================================
print('\n(3) PERFORM PREDICTION')
# =====================================================

# prediction: probabilty sequence of each token
# tensor of shape ( N, seq_len, NUM_KWD+2 )
token_prob = model_pred(feats_batch)

# categorical prediction: take the most probable token at each time instant
# tensor of shape ( N, seq_len )
tokens_seq1 = tf.argmax(token_prob,axis=-1)

# posterior handling of predicted tokens (have variable lengths)
#  - remove serially repeated tokens, then remove null element (9)

tokens_seq2 = []
for i in range(N):
	q = []
	q.append(tokens_seq1[i][0].numpy())
	for j in range(1,len(tokens_seq1[i])):
		if tokens_seq1[i][j].numpy() != q[-1]:
			q.append(tokens_seq1[i][j].numpy())
	tokens_seq2.append(q)
for i in range(N):
	tokens_seq2[i] = [s for s in tokens_seq2[i] if s != NUM_KWD+1]

# convert pred_tokens to time scale (from tokens_seq1)
#  - undo CNN compression, convert frame scale to time (sample) scale
#  - the i-th element in tokens_seq1 corresponds to the gap between the i and i+1 values
TIME_COMP_RATIO = 4
token_edges_in_time = TIME_COMP_RATIO*FRAME_STEP*np.arange(tokens_seq1.shape[-1]+1)



# =====================================================
print('\n(4) PLOTS')
# =====================================================

# turn text_batch in to lists (remove '9')
true_tokens = []
for i in range(N):
	q = []
	for j in range(len(text_batch[i])):
		if text_batch[i][j] != NUM_KWD+1:
			q.append(text_batch[i][j].numpy())
	true_tokens.append(q)


for i in range(N):
	plt.figure(i+1)
	
	plt.subplot(2,1,1); plt.grid()
	for j in range(NUM_KWD+2):
		plt.plot(token_prob[i,:,j])
	#plt.plot(tokens_seq1[i,:]/9,'ko')
	plt.ylabel('Token probability')
	plt.title( 'True tokens: ' + str(true_tokens[i]) + '\nPredicted: ' + str(tokens_seq2[i]) )
	plt.legend(['0 [unk]','1','2','3','4','5','6','7','8','null'])

	plt.subplot(2,1,2); plt.grid()
	plt.plot(audio_batch[i])
	#plt.plot(token_edges_in_time,np.zeros(tokens_seq1.shape[-1]+1),'ko')
	plt.ylabel('Speech signal')

plt.figure(N+1); plt.grid()
plt.plot(10*np.log10(hist[0]))
plt.plot(10*np.log10(hist[1]),'o')
plt.legend(['Training loss','Validation loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss (dB)')

plt.show()


# plt.figure(1); plt.clf()
# plt.subplot(3,1,1)
# plt.imshow(tf.transpose(feat[0,:,:]), cmap=plt.cm.jet, aspect='auto')
# plt.subplot(3,1,2)
# plt.imshow(tf.transpose(feat[1,:,:]), cmap=plt.cm.jet, aspect='auto')
# plt.subplot(3,1,3)
# plt.imshow(tf.transpose(feat[2,:,:]), cmap=plt.cm.jet, aspect='auto')
# plt.show()





# ==================================================================
