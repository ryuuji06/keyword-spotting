
import os
from glob import glob # module for finding pathnames
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers


# ----------------------------------------------
#   P R E P R O C E S S I N G
# ----------------------------------------------

# remove punctuation, lower ()
# turn plural to singular, past verb to standard
# [optional] turn consecutive zeros into one single zero
def english_standardization(text):
	# lower case
	text = tf.strings.lower(text)
	# remove punctuation and other symbols
	text = tf.strings.regex_replace(text, "[()*'?!,.:;\n]", '')
	# substitute some expressions
	text = tf.strings.regex_replace(text, "houses",  "house")
	text = tf.strings.regex_replace(text, "rights",  "right")
	text = tf.strings.regex_replace(text, "ones", "one")
	# Strip whitespace.
	#text = tf_strings.strip(text)
	return text


# ----------------------------------------------
#   R E A D I N G   F I L E S  
# ----------------------------------------------

# for LibriSpeech Dataset
# return list of every single training sample (sample audio and corresp. transcription)
def librispeech_pair_samples(datapath):
	# track transcription files (list of strings)
	target_path = glob("{}/**/**/*.txt".format(datapath), recursive=True)
	input_path = []
	target_text = []
	for i, t in enumerate(target_path): # for each txt file
		with open(t, 'r') as f: # for each line in txt
			for line in f:
				code, raw_text = line.split(' ', maxsplit=1)
				ids = code.split('-')
				audio_path = datapath + f'\\{ids[0]}\\{ids[1]}\\{code}.wav'
				if os.path.exists(audio_path):
					input_path.append(audio_path)
					target_text.append(raw_text)
				else:
					print(f'Audio file of code {code} did not found!')

	return input_path, target_text


def speech_cmds_pair_samples(datapath, max_num, kw):
	input_path = []
	target_text = []
	# create list of input path, list of target values (own keywords)
	for k in range(len(kw)):
		count = 0
		folder_files = []
		for file in os.listdir(os.path.join(datapath, kw[k])):
			count += 1
			folder_files.append(os.path.join(datapath, kw[k], file))
			if count >= max_num:
				break
		input_path.extend(folder_files)
		target_text.extend([kw[k]]*count)

	return input_path, target_text


def speech_cmds_edit_pair_samples(datapath):

	input_path = glob("{}/*.wav".format(datapath), recursive=True) # list
	target_text = []
	# create list of input path, list of target values (own keywords)
	for path in input_path:
		x1 = path.split('\\')[-1] # wav file name
		x2 = x1.split('.')[0] # remove .wav
		x3 = x2.split('-')[1:] # list of words (remove first number)
		target_text.append(' '.join(x3))
	return input_path, target_text


# path and texts: array of strings
def arrange_data(paths, texts, vectorize_text, num_kwd, max_text_len, num_empty_samps):

	# compute length of each text line
	target_tokens = vectorize_text(tf.constant(texts))
	target_tokens_length = tf.cast(target_tokens!=0,dtype=np.int64)
	target_tokens_length = tf.reduce_sum(target_tokens_length,axis=1)

	# remove samples with text length > max_text_len
	idx = np.where(target_tokens_length>max_text_len)[0]
	for i in sorted(idx, reverse=True):
		del paths[i]
		del texts[i]
	target_tokens = target_tokens[target_tokens_length<=max_text_len]-1

	# find keywords indices
	kwd_occurence = np.zeros(num_kwd)
	kwd_ind = []
	for i in range(num_kwd):
		x = np.where(target_tokens==i+1)
		kwd_occurence[i] = len(x[0])
		kwd_ind = kwd_ind + list(x[0])
	kwd_ind = sorted(set(kwd_ind))

	# find nonkeywords indices
	nonkwd_ind = []
	i = 0 # num to inset to list
	j = 0 # num to read kwd_ind
	while len(nonkwd_ind)<num_empty_samps:
		if kwd_ind[j] > i:
			nonkwd_ind.append(i)
			i+=1
		else:
			j+=1
			i+=1

	return kwd_occurence, kwd_ind, nonkwd_ind




# ==================================================================