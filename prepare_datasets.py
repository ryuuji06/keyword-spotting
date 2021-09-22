# # clear existing user defined variables
# for element in dir():
#     if element[0:2] != "__":
#         del globals()[element]

import os
import pickle
from time import time

import numpy as np
import tensorflow as tf

from functions_preprocessing import english_standardization
from functions_dataset import handle_speech_cmds, handle_speech_cmds_edit, handle_librispeech_pair

# =================================================================
# =================================================================
# PREPARE DATASETS FILES FOR TRAINING/EVALUATION
# =================================================================
# =================================================================

# reads datasets, selecting data according to some specified parameters,
# and save pickle files for parameters and for paired data (audio-path, target text)
# for each dataset

# in particular, we use datasets Google Speech Commands, a modified version of it,
# and Librispeech (Clean 100 hours)


# --------------------------------------------------
#   S E T T I N G S
# --------------------------------------------------

# dataset paths
cmds1_path = 'C:\\datasets\\speech_commands'
cmds2_path = 'C:\\datasets\\mod_gsc'
libri_path = "C:\\datasets\\LibriSpeech\\train-clean-100"
noisefolder = 'C:\\datasets\\MS-SNSD'
# cmds1_path = '/home/thiago/datasets/speech_commands'
# cmds2_path = '/home/thiago/datasets/mod_gsc'
# libri_path = '/home/thiago/datasets/LibriSpeech/train-clean-100'
# noisefolder = '/home/thiago/datasets/MS-SNSD'

# name of folder with the prepared data
selected_data_folder = 'data2'

# select keywords
#keywords = ['house', 'five', 'down', 'left', 'learn', 'nine', 'one', 'cat']
keywords = ['house', 'right', 'down', 'left', 'no', 'five', 'one', 'three']

# maximum number of samples with isolated keywords, for each keyword
# (speech commands samples)
max_num_isolated_kwd = 1000
# maximum number of tokens per sample (for librispeech)
max_text_len = 40
# maximum number of samples containing no keywords (for librispeech)
num_nonkwd_samp = 1000 + 4679

# this actually must be verified beforehand
sampling_rate = 16000

# noise augmentation settings
# (instead of decoding a 1-5 min. signal for every training sample, reads and concatenates
# audios into a single signal once before reading training samples)
# folder of noise signal files
# select noise files
noisefiles = ['AirConditioner_9.wav', 'Babble_2.wav', 'Typing_8.wav',
              'CafeTeria_1.wav', 'CopyMachine_8.wav', 'SqueakyChair_4.wav',
              'SqueakyChair_6.wav', 'VacuumCleaner_9.wav', 'WasherDryer_2.wav']
# amplification for each noise file
noise_amp = [0.05, 0.07, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


# --------------------------------------------------
#   S A V E   I N I T I A L   F I L E S
# --------------------------------------------------

num_kwd = len(keywords) # the total number of tokens is num_kwd+1 (plus the filler token)

descript = f"""SELECTED DATA PROPERTIES

keywords = {keywords}
Max. number of isolated keywords (Speech Commands): {max_num_isolated_kwd}
Max. sentence length (in tokens)(for Librispeech): {max_text_len}
Number of sentence samples with no keywords (for Librispeech): {num_nonkwd_samp}"""

params = {'keywords': keywords,
          'num_kwd': num_kwd,
          'sampling_rate': sampling_rate,
          'max_num_isolated_kwd': max_num_isolated_kwd,
          'max_text_len': max_text_len,
          'num_nonkwd_samp': num_nonkwd_samp}

if os.path.exists(selected_data_folder):
    raise AssertionError('Data folder already exists. Make sure not to overwrite it.')
else:
    os.makedirs(selected_data_folder)

with open(os.path.join(selected_data_folder,'description.txt'), 'w') as f:
    f.write(descript)

with open(os.path.join(selected_data_folder,'parameters.pickle'), 'wb') as f:
        pickle.dump(params, f)

# input apparently must be tensor, but sometimes it works for numpy input
text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=english_standardization,
    max_tokens=None,
    vocabulary=keywords)


# --------------------------------------------------
#   P R E P A R E   D A T A S E T S
# --------------------------------------------------

# save selected noise paths and properties
t1 = time()
for i in range(len(noisefiles)):
	noisefiles[i] = os.path.join(noisefolder,noisefiles[i])
noise_settings = {'path':noisefiles, 'amp':noise_amp}
with open(os.path.join(selected_data_folder,'noise_aug.pickle'), 'wb') as f:
    pickle.dump(noise_settings, f)
# verify size of noise signal
noise, sr = tf.constant([]), 1
for f, a in zip(noise_settings['path'], noise_settings['amp']):
    audio_bin = tf.io.read_file(f)
    audio, sr = tf.audio.decode_wav(audio_bin, 1)
    noise = tf.concat([noise,a*tf.squeeze(audio, axis=-1)],0)
dur = noise.shape[0]/sr.numpy()
with open(os.path.join(selected_data_folder,'description.txt'), 'a') as f:
    f.write('\nLength of noise signal: {:02d}h {:02d}min {:02d}sec'.format(
    	int(dur//3600), int(dur//60), int(dur%60)))
t2 = time(); print('Noise verified: %.3fs'%(t2-t1))

t1 = time()
handle_speech_cmds(
    cmds1_path, selected_data_folder,
    'speech_commands_dict',
    max_num_isolated_kwd, keywords)
t2 = time(); print('a) Arrange data and count keywords: %.3fs'%(t2-t1))

t1 = time()
handle_speech_cmds_edit(
    cmds2_path, selected_data_folder,
    'speech_commands_edit_dict',
    text_processor, num_kwd+1)
t2 = time(); print('b) Arrange data and count keywords: %.3fs'%(t2-t1))

t1 = time()
handle_librispeech_pair(
    libri_path, selected_data_folder,
    'librispeech_dict', text_processor,
    num_kwd+1, max_text_len, num_nonkwd_samp)
t2 = time(); print('c) Arrange data and count keywords: %.3fs'%(t2-t1))



# --------------------------------------------------
#   T E S T   F I L E S
# --------------------------------------------------

with open(os.path.join(selected_data_folder,"speech_commands_dict.pickle"), 'rb') as f:
    data1 = pickle.load(f)

with open(os.path.join(selected_data_folder,"speech_commands_edit_dict.pickle"), 'rb') as f:
    data2 = pickle.load(f)

with open(os.path.join(selected_data_folder,"librispeech_dict.pickle"), 'rb') as f:
    data3 = pickle.load(f)

with open(os.path.join(selected_data_folder,"parameters.pickle"), 'rb') as f:
    parameters = pickle.load(f)

print(data1['count_occur'])
print(data2['count_occur'])
print(data3['count_occur'])




# ==================================================================