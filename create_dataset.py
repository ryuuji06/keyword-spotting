# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import numpy as np
from glob import glob # module for finding pathnames
from time import time
from matplotlib import pyplot as plt

import scipy.io.wavfile as wav
import librosa
import tensorflow as tf

# execute file from python interpreter
# exec(open('create_dataset.py').read())

# =================================================================
# =================================================================
# CREATE ADDITIONAL DATASET FROM SPEECH COMMANDS
# =================================================================
# =================================================================

# - create additional dataset for keywords, where each sample are composed
# mostly of keywords, and some unknown words (unused commands of Speech Commands)

# (1) read speech command file paths, count elements
# (2) select elements to compose the dataset (do not use the first 1000 of each keyword)
# (3) arrange random number of words in audio signal (in order of the list)
#  - from 2 to 6 words
#  - remove silent tracks of signals (silent tracks have noise?)
#  - insert random length gaps between words


# ---------------------------------------------------------------------

# dataset folder directory
dataset_path = 'C:\\datasets\\speech_commands'
edit_path = 'C:\\datasets\\speech_commands_edit' # dataset to be created


# keywords are the own target value (to be one hot encoded)
# these keywords have at least 2000 samples in original speech dataset
keywords = ['house', 'right', 'down', 'left', 'no', 'five', 'one', 'three']
fillers = ['backward', 'bird', 'bed', 'cat', 'eight', 'go', 'follow', 'forward', 'happy', 'marvin',
	'on', 'off', 'sheila', 'zero', 'yes', 'up', 'tree', 'seven' ]

L1 = len(keywords)
L2 = len(fillers)

NUM_SAMPLES = 4000

def load_and_concatenate(paths,remove):
	sig = np.zeros(0)
	for k in range(len(paths)):
		audio, _ = librosa.load(paths[k], sr=None) # numpy array
		if remove:
			interv = librosa.effects.split(audio, top_db=35)
			tracks = []
			for m in range(len(interv)):
				tracks.append(audio[interv[m][0]:interv[m][1]])
			audio = np.hstack(tracks)
		sig = np.hstack([sig,audio])
	return sig


# =================================================
print('\n(1) Load Data (Google Speech Commands)')
# =================================================

keyword_paths = [] # will be a list of lists
filler_paths = []

# create list of input path, list of target values (own keywords)
for k in range(L1):
	print(keywords[k])
	count = 0
	folder_files = []
	for file in os.listdir(os.path.join(dataset_path, keywords[k])):
		count += 1
		folder_files.append(os.path.join(dataset_path, keywords[k], file))
	keyword_paths.append(folder_files)

for k in range(L2):
	print(fillers[k])
	count = 0
	folder_files = []
	for file in os.listdir(os.path.join(dataset_path, fillers[k])):
		count += 1
		folder_files.append(os.path.join(dataset_path, fillers[k], file))
		if count >= 200:
			break
	filler_paths.append(folder_files)

# for i in range(L1):
# 	print(keywords[i],len(keyword_paths[i]))
# for i in range(L2):
# 	print(fillers[i],len(filler_paths[i]))


# # =================================================
# print('\n(2) Test audio data')
# # =================================================

# # verify silent tracks, remove silent tracks

# audio, _ = librosa.load(keyword_paths[2][199], sr=None)
# interv = librosa.effects.split(audio, top_db=32)

# plt.figure(); plt.grid()
# plt.plot(audio)
# for i in range(len(interv)):
# 	plt.plot([interv[i][0],interv[i][0]],[-0.5,0.5],'k')
# 	plt.plot([interv[i][1],interv[i][1]],[-0.5,0.5],'k')
# plt.show()


# =================================================
print('\n(3) Arrange Data')
# =================================================

# 1000 (at least) of each keyword, 200 of each filler
# randomly remove silent tracks ou keep whole signal

single_word_paths = []
targets = []
for i in range(L1):
	q = keyword_paths[i][1000:3000]
	single_word_paths = single_word_paths + q
	targets = targets + len(q)*[keywords[i]]
for i in range(L2):
	q = filler_paths[i]
	single_word_paths = single_word_paths + q
	targets = targets + len(q)*[fillers[i]]
print('Number of single words:',len(single_word_paths))

# shuffle
p = np.random.permutation(len(single_word_paths))
single_word_paths = [ single_word_paths[i] for i in p ]
targets = [ targets[i] for i in p ]

# for i in range(L1):
# 	print(keywords[i],'\t',i+1)
# for i in range(100):
# 	print(single_word_paths[i],'\t',targets[i])


# group into list of lists

num_tokens = np.random.choice(a=[2,3,4,5,6],size=NUM_SAMPLES)
print('Total tokens:',np.sum(num_tokens))
if np.sum(num_tokens) > len(single_word_paths):
	print('Problem: requiring more samples than existing')

#plt.figure(1); plt.grid()

count = 0
for i in range(NUM_SAMPLES): # NUM_SAMPLES

	signal_paths = single_word_paths[count:count+num_tokens[i]]
	words = targets[count:count+num_tokens[i]]
	count += num_tokens[i]
	rem_silent = np.random.choice([0,1])

	# load signals and concatenate
	audio = load_and_concatenate(signal_paths,rem_silent)
	# save
	pathname = edit_path + '\\' + str(i) + '-' + '-'.join(words) + '.wav'
	tensor_audio = tf.expand_dims(tf.cast(tf.constant(audio),dtype=tf.float32),axis=1)
	bin_wav = tf.audio.encode_wav(tensor_audio,16000)
	tf.io.write_file(pathname,bin_wav)
	#wav.write( pathname, 16000, audio.astype(np.float32) )

	# plt.subplot(2,2,i+1)
	# plt.plot(audio)
	# plt.title(str(num_tokens[i])+', cuts:'+str(rem_silent))

	if i%20 == 0:
		print(i)

#plt.show()





# ==================================================================