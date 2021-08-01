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

from auxiliar_functions import english_standardization, librispeech_pair_samples
from auxiliar_functions import speech_cmds_pair_samples, speech_cmds_edit_pair_samples, arrange_data

from models import build_model_1, build_model_2
#from matplotlib import pyplot as plt


# execute file from python interpreter
# exec(open('kws_train.py').read())

# =================================================================
# =================================================================
# KEYWORD SPOTTING USING CNN-RNN-CTC
# TRAINING STAGE
# =================================================================
# =================================================================

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

resultfolder = 'results_xx'
if os.path.exists(resultfolder):
	raise AssertionError('Folder already exists. Be sure to use a proper name for the test.')
else:
	os.makedirs(resultfolder)

descript = """
TRAINING DESCRIPTION

keywords = ['house', 'right', 'down', 'left', 'no', 'five', 'one', 'three']
MAX_NUM_ISOLATED_KWD = 1000
MAX_TEXT_LEN = 40  # max num of tokens
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 64

FRAME_LENGTH = 256
FRAME_STEP = 128
FFT_LENGTH = 256
N_MEL_BINS = 80
N_MFCC_BINS = 13
(only MFCC features, and no deltas)

RNN-CTC MODEL
 - starts with batch normalization
 - two CNN layer blocks, convolutions with unit stride and ReLU, and each block followed by MaxPool stride 2
   (1: 2 layer 32 kernels of size 3)
   (2: 2 layer 64 kernels of size 3)
 - two BiLSTM, both with dropout 0.25
   (1: 128 units)
   (2: 128 units)
"""

with open(resultfolder+'/description.txt', 'w') as f:
	f.write(descript)


# --------------------------------------------
# (2) Dataset and Training Configuration
# --------------------------------------------

MAX_NUM_ISOLATED_KWD = 1000
MAX_TEXT_LEN = 40  # max num of tokens
NUM_NONKWD_SAMP = 500
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 64
EPOCHS = 100

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

@tf.function
def join_input_label(inputs, labels):
	return {'input': inputs, 'target': labels}

def create_speech_dataset(inputs, targets, vectorize_text, batch_size=32, train_test_split=0.8):
	# (1) extract audio features
	audio_ds = tf.data.Dataset.from_tensor_slices(inputs)
	audio_ds = audio_ds.map(path2audio)
	#audio_ds = audio_ds.map(audio2spec) # num_parallel_calls=tf.data.experimental.AUTOTUNE
	audio_ds = audio_ds.map(audio2mfcc,num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# (2) tokenize text
	text_ds = tf.data.Dataset.from_tensor_slices(target_texts)
	text_ds = text_ds.map(lambda x: vectorize_text(x)-1)
	# (3) batch inputs and targets, and zip
	audio_ds = audio_ds.padded_batch(batch_size)
	text_ds = text_ds.padded_batch(BATCH_SIZE, padding_values=np.int64(NUM_KWD+1))

	ds = tf.data.Dataset.zip((audio_ds, text_ds))
	ds = ds.map(join_input_label)
	ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

	# (3) split into train/valid datasets
	num_train = int( len(audio_ds)*train_test_split )
	return ds.take(num_train), ds.skip(num_train)

# -----------------------------------------------------------

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
print('Verify sampling rate:', sr1, sr2, sr3, sr4)



# (b) READ DATA (input audio file path and label text)

t1 = time()
input_paths_1, target_texts_1 = speech_cmds_pair_samples(cmds1_path, MAX_NUM_ISOLATED_KWD, keywords)
input_paths_2, target_texts_2 = speech_cmds_edit_pair_samples(cmds2_path)
input_paths_3, target_texts_3 = librispeech_pair_samples(libri1_path)
input_paths_4, target_texts_4 = librispeech_pair_samples(libri2_path)
#input_paths_3, target_texts_3 = ljspeech_pair_samples(lj_path)
t2 = time(); print('a) Load and arrange transcription pairs: %.3fs'%(t2-t1))


# (c) FILTER SAMPLES BY MAXIMUM LABEL LENGTH

kwd_occs_3, kwd_inds_3, nonkwd_inds_3 = arrange_data(
	input_paths_3, target_texts_3, text_processor, NUM_KWD, MAX_TEXT_LEN, NUM_NONKWD_SAMP)

input_paths_3_kwd  = [input_paths_3[i] for i in kwd_inds_3]
target_texts_3_kwd = [target_texts_3[i] for i in kwd_inds_3]
input_paths_3  = [input_paths_3[i] for i in nonkwd_inds_3] # update existing list
target_texts_3 = [target_texts_3[i] for i in nonkwd_inds_3]

print('Librispeech test size (nonkeyword):', len(input_paths_3))
print('Librispeech test size (keywords):', len(input_paths_3_kwd))

kwd_occs_4, kwd_inds_4, nonkwd_inds_4 = arrange_data(
	input_paths_4, target_texts_4, text_processor, NUM_KWD, MAX_TEXT_LEN, NUM_NONKWD_SAMP)

input_paths_4_kwd  = [input_paths_4[i] for i in kwd_inds_4]
target_texts_4_kwd = [target_texts_4[i] for i in kwd_inds_4]
input_paths_4  = [input_paths_4[i] for i in nonkwd_inds_4] # update existing list
target_texts_4 = [target_texts_4[i] for i in nonkwd_inds_4]

print('Librispeech dev size (nonkeyword):', len(input_paths_4))
print('Librispeech dev size (keywords):', len(input_paths_4_kwd))

print('Number of samples per kerword')
for i in range(NUM_KWD):
	 print('',keywords[i],kwd_occs_3[i]+kwd_occs_4[i])


# (d) JOIN DATA AND SHUFFLE

input_paths = input_paths_1 + input_paths_2 + input_paths_3 + input_paths_3_kwd + input_paths_4 + input_paths_4_kwd
target_texts = target_texts_1 + target_texts_2 + target_texts_3 + target_texts_3_kwd + target_texts_4 + target_texts_4_kwd

NUM_SAMPLES = len(input_paths)
p = np.random.permutation(NUM_SAMPLES)
input_paths = [ input_paths[i] for i in p ]
target_texts = [ target_texts[i] for i in p ]
print(f'Number of transcription pairs: {NUM_SAMPLES}')




# =====================================================
print('\n(2) CREATE DATSET OBJECT')
# =====================================================

t1 = time()
train_ds, valid_ds = create_speech_dataset(input_paths, target_texts,
	text_processor, batch_size=BATCH_SIZE, train_test_split=TRAIN_TEST_SPLIT)
t2 = time(); print('b) Arrange dataset object: %.3fs'%(t2-t1))

print('Dataset examples')
t1 = time()
for pair in train_ds.take(1):
	for i in range(10):
		print(pair['input'][i].shape, pair['target'][i])



# =====================================================
print('\n(3) BUILD MODEL AND TRAIN')
# =====================================================

# Get the model
model_train, model_pred = build_model_1(FEAT_DIM, NUM_KWD)
model_train.summary()

pickle_path = resultfolder+'/hist.pickle'
save_model_path = resultfolder+'/model-{epoch:02d}-{val_loss:.3f}.h5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(
	save_model_path, save_best_only=True, save_weights_only=True, verbose=0)

# Train the model
h = model_train.fit(train_ds, validation_data=valid_ds,
	epochs=EPOCHS, callbacks=[checkpointer])

hist = [h.history['loss'], h.history['val_loss']]

# save model loss history
with open(resultfolder+"/hist.pickle", 'wb') as f:
	pickle.dump(hist, f)

model_pred.save(resultfolder+'/prediction_model.h5')



# ==================================================================