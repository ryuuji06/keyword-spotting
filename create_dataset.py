# # clear existing user defined variables
# for element in dir():
#     if element[0:2] != "__":
#         del globals()[element]

import os
import numpy as np
import librosa
from functions_dataset import read_subfolders, flatten_list
from functions_dataset import generate_num_tokens_per_sample, create_modified_dataset
#from matplotlib import pyplot as plt

# =================================================================
# =================================================================
# CREATE ADDITIONAL DATASET FROM SPEECH COMMANDS
# =================================================================
# =================================================================

# Create additional dataset for keywords, where each sample is composed
# mostly of keywords, and some unknown words (unused commands from Speech Commands)

# This dataset is intended to work along some original samples of Speech Commands.
# So, in order not to repeat the same samples used in raw Speech Commands, by default
# considered the first samples in the dataset, one can set skip_samples to skip these
# samples in this modified dataset.


# -----------------------------------------------------
#   S E T T I N G S
# -----------------------------------------------------

# keywords and non-keywords to compose the dataset
# (must be present in Google Speech Commands)
keywords = ['house', 'right', 'down', 'left', 'no', 'five', 'one', 'three']
fillers = ['backward', 'bird', 'bed', 'cat', 'eight', 'go', 'follow', 'forward',
    'happy', 'marvin', 'on', 'off', 'sheila', 'zero', 'yes', 'up', 'tree', 'seven' ]

source_folder = 'C:\\datasets\\speech_commands' # path to speech commands dataset
target_folder = 'C:\\datasets\\mod_gsc' # path to created dataset

num_keywords = 2000 # max. number of samples per keyword
skip_samples = 1000 # gap of samples skipped at the beginning of each keyword list
num_fillers = 200 # max. number of samples per filler

# number of audio samples to be concatenated
# to form each new sample (variable length)
num_audio_concat = list(range(2,7))

# ---------------------------------------------------------

if not os.path.exists(target_folder):
    os.makedirs(target_folder)



# =================================================
print('(1) Load Paths (Google Speech Commands)')
# =================================================

# these are list of lists (grouped by keywords)
keyword_paths, keyword_labels = read_subfolders(source_folder, keywords,
                                offset=skip_samples, max_number=num_keywords)
filler_paths, filler_labels = read_subfolders(source_folder, fillers,
                               offset=0, max_number=num_fillers)

# get sampling rate
_, sampling_rate = librosa.load(keyword_paths[0][0], sr=None)

print('Number of samples per word.')
for i in range(len(keywords)):
    print(keywords[i], len(keyword_paths[i]))
for i in range(len(fillers)):
    print(fillers[i], len(filler_paths[i]))

# flatten list of paths/labels into a flat list
single_word_paths = flatten_list(keyword_paths) + flatten_list(filler_paths)
single_word_labels = flatten_list(keyword_labels) + flatten_list(filler_labels)
print('Number of single words:',len(single_word_paths))

# shuffle list
p = np.random.permutation(len(single_word_paths))
single_word_paths = [ single_word_paths[i] for i in p ]
single_word_labels = [ single_word_labels[i] for i in p ]

# for i in range(100):
#   print(single_word_paths[i],'\t',targets[i])


# # =================================================
# print('\n(2) Test audio data')
# # =================================================

# # verify silent tracks, remove silent tracks

# audio, _ = librosa.load(keyword_paths[2][199], sr=None)
# interv = librosa.effects.split(audio, top_db=32)

# plt.figure(); plt.grid()
# plt.plot(audio)
# for i in range(len(interv)):
#   plt.plot([interv[i][0],interv[i][0]],[-0.5,0.5],'k')
#   plt.plot([interv[i][1],interv[i][1]],[-0.5,0.5],'k')
# plt.show()


# =================================================
print('(2) Arrange New Dataset')
# =================================================

# set number of concatenated audios for each created sample
# total number of audios is `len(single_word_paths)`
num_tokens_per_sample = generate_num_tokens_per_sample(len(single_word_paths), num_audio_concat)
print('Number of samples for the new dataset:', len(num_tokens_per_sample))

create_modified_dataset(
    single_word_paths,
    single_word_labels,
    num_tokens_per_sample,
    target_folder,
    sampling_rate)



# ==================================================================