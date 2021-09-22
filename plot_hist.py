# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import numpy as np
#from functions_postprocessing import read_hist
from matplotlib import pyplot as plt

# =================================================================
# =================================================================
# KEYWORD SPOTTING USING CNN-RNN-CTC
# PREDICTION AND EVALUATION STAGE
# =================================================================
# =================================================================

# read history file generated during training
def read_hist(filepath):
    # read file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # copy to data array
    lines[0] = lines[0].replace('\n','')
    lines[0] = lines[0].replace('\t\t','\t')
    heads = lines[0].split('\t')
    array_data = np.zeros((len(heads),len(lines)-1))
    for i in range(1,len(lines)):
        line = lines[i].replace('\n','')
        line = line.split('\t')
        array_data[:,i-1] = list(map(float,line))
    # move to dictionary
    hist = {}
    for i in range(len(heads)):
        hist[heads[i]] = array_data[i,:]
    return hist


folders = ['result_model02',
           'result_model02_noise',
           'result_model03',
           'result_model04',
]

description = ['Model 2',
               'Model 2 noise',
               'Model 3',
               'Model 4',
]

# -------------------------------------------
#  Load Performance History
# -------------------------------------------

hist = [read_hist(os.path.join(f,"hist.txt")) for f in folders]


# -------------------------------------------
#   P  L  O  T
# -------------------------------------------

plt.figure(1)

plt.subplot(1,2,1); plt.grid()
for i in range(len(hist)):
    plt.plot(10*np.log10(hist[i]['loss']))
plt.xlabel('Epochs')
plt.ylabel('Loss (dB)')
plt.title('Training Loss')
plt.legend(description)

plt.subplot(1,2,2); plt.grid()
for i in range(len(hist)):
    plt.plot(10*np.log10(hist[i]['val_loss']))
plt.xlabel('Epochs')
plt.ylabel('Loss (dB)')
plt.title('Validation Loss')
plt.legend(description)

plt.show()


# ==================================================================