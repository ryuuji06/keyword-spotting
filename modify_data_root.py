# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import pickle
import librosa
from shutil import copyfile
import numpy as np


# Modify the data files to be used in a different machine,
# where the datasets are located in different folder path
# (potentially of different operation system)


sourcefolder = 'data2_ubuntu'
newfolder = 'data2'
new_root = 'C:\\'
#new_root = 'D:\\Users\\Administrador\\Documents\\DLtests'
num_rem_root = 3 # number of root path components to remove
old_separator = '/' # current separator

def edit_files_root(datafile, newfile, cur_sep, num_rem_root, new_root):
    with open(datafile, 'rb') as f:
        data = pickle.load(f)
    # os.path.split()
    for i in range(len(data['input_path'])):
        old = data['input_path'][i].split(cur_sep)
        data['input_path'][i] = os.path.join(new_root,*old[num_rem_root:])
    with open(newfile, 'wb') as f:
        pickle.dump(data,f)


# ----------------------------------------------------------
# COPYING PARAMETERS AND DESCRIPTION
# ----------------------------------------------------------

if os.path.exists(newfolder):
    raise AssertionError('Data folder already exists. Make sure not to overwrite it.')
else:
    os.makedirs(newfolder)

copyfile(os.path.join(sourcefolder,'description.txt'),
         os.path.join(newfolder,'description.txt'))
copyfile(os.path.join(sourcefolder,'parameters.pickle'),
         os.path.join(newfolder,'parameters.pickle'))

edit_files_root(
    os.path.join(sourcefolder,'speech_commands_dict.pickle'),
    os.path.join(newfolder,'speech_commands_dict.pickle'),
    old_separator, num_rem_root, new_root)

edit_files_root(
    os.path.join(sourcefolder,'speech_commands_edit_dict.pickle'),
    os.path.join(newfolder,'speech_commands_edit_dict.pickle'),
    old_separator, num_rem_root, new_root)

edit_files_root(
    os.path.join(sourcefolder,'librispeech_dict.pickle'),
    os.path.join(newfolder,'librispeech_dict.pickle'),
    old_separator, num_rem_root, new_root)

# for noise files
with open(os.path.join(sourcefolder,'noise_aug.pickle'), 'rb') as f:
    data = pickle.load(f)
# os.path.split()
for i in range(len(data['path'])):
    old = data['path'][i].split(old_separator)
    data['path'][i] = os.path.join(new_root,*old[num_rem_root:])
with open(os.path.join(newfolder,'noise_aug.pickle'), 'wb') as f:
    pickle.dump(data,f)

print('Edition complete.')
    
# ----------------------------------------------------------
# TESTING
# ----------------------------------------------------------

with open(os.path.join(newfolder,"librispeech_dict.pickle"), 'rb') as f:
    data3 = pickle.load(f)

for i in range(100):
    _, sampling_rate = librosa.load(data3['input_path'][i], sr=None)
    print(data3['input_path'][i])


# ==================================================================