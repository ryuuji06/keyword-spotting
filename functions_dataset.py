
import os
from glob import glob # module for finding pathnames
import pickle
import librosa
import numpy as np
import tensorflow as tf


# ----------------------------------------------
#   R E A D I N G   P A T H S
# ----------------------------------------------

def read_subfolders(mainfolder, subfolders, offset=0, max_number=1_000_000):
    """Return list of lists; each sublist contains the content of each subfolder."""
    paths = [] # will be a list of lists
    labels = []
    for k in range(len(subfolders)):
        folder_files = []
        for i, file in enumerate(sorted(os.listdir(os.path.join(mainfolder, subfolders[k])))):
            if i>=offset+max_number:
                break
            elif i>=offset:
                folder_files.append(os.path.join(mainfolder, subfolders[k], file))
        paths.append(folder_files)
        labels.append([subfolders[k]]*len(folder_files))
    return paths, labels

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


# -------------------------------------------------------
#   G E N E R A T E   A U X I L I A R   D A T A S E T
# -------------------------------------------------------

def load_and_concatenate(paths,remove):
    """Load audio file whose paths are in list `paths`, and return
    the concatenation of them. Optionally removes silent tracks according
    to the boolean `remove`."""
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

def generate_num_tokens_per_sample(total_tokens, length_range):
    num_tokens_per_sample = length_range * (total_tokens//sum(length_range))
    rem = total_tokens%sum(length_range)
    i = 0
    while rem != 0:
        if rem >= length_range[i]:
            num_tokens_per_sample.append(length_range[i])
            rem -= length_range[i]
            i += 1
        else:
            num_tokens_per_sample.append(rem)
            rem = 0
    return num_tokens_per_sample


def create_modified_dataset(
        src_paths, src_labels,
        num_tokens_per_sample,
        folder, sampling_rate
    ):
    
    count = 0
    for i in range(len(num_tokens_per_sample)):

        # select filepaths and targets
        signal_paths = src_paths[count:count+num_tokens_per_sample[i]]
        words = src_labels[count:count+num_tokens_per_sample[i]]
        count += num_tokens_per_sample[i]
        rem_silent = np.random.choice([0,1]) # random removing of silent track
        # load signals and concatenate
        audio = load_and_concatenate(signal_paths,rem_silent)
        # save audio file (filename is the composing target words)
        filename = '{:04d}-'.format(i) + '-'.join(words) + '.wav'
        pathname = os.path.join(folder, filename)
        tensor_audio = tf.expand_dims(tf.cast(tf.constant(audio),dtype=tf.float32),axis=1)
        bin_wav = tf.audio.encode_wav(tensor_audio, sampling_rate)
        tf.io.write_file(pathname,bin_wav)

        if i%100 == 0:
            print('sample', i)



# -------------------------------------------------
#   G E N E R A T E   P R E - R E A D   F I L E S
# -------------------------------------------------

def append_to_decript(selected_path, name, num_samples, num_kwd, count_occur, count_samps):
    """Append data of a given dataset to the description file, writing number of
    occurencce for each keyword (num_kwd also include non-keyword token)"""
    with open(os.path.join(selected_path,'description.txt'), 'a') as f:
        f.write(f'\n\n{name}')
        f.write(f'\nTotal number of samples: {num_samples}')
        f.write('\nOccurences per keywords\n')
        for i in range(num_kwd):
            f.write(str(count_occur[i])+'\t')
        f.write('\nSamples containing keywords\n')
        for i in range(num_kwd):
            f.write(str(count_samps[i])+'\t')


def handle_speech_cmds(datapath, selected_path, filename, max_num, keywords):
    """Make preparation of Speech Commands dataset and save pickle with
    audio paths and target texts."""
    
    # create list of input path, list of target values (the very keywords)
    input_path, target_text = read_subfolders(datapath, keywords,
                                offset=0, max_number=max_num)
    # count samples and keyword occurences
    count_occur = np.array([0] + [len(input_path[i]) for i in range(len(keywords))])
    count_samps = count_occur
    # flatten list into a single list
    input_path = flatten_list(input_path)
    target_text = flatten_list(target_text)

    # write counts to descript
    append_to_decript(selected_path, 'Speech Commands (isolated words)',
        len(input_path), len(keywords)+1, count_occur, count_samps)

    # save to file
    pack = {'input_path':input_path, 'target_text':target_text,
            'count_occur':count_occur, 'count_samps':count_samps}
    with open(os.path.join(selected_path,filename+'.pickle'), 'wb') as f:
        pickle.dump(pack, f)


def handle_speech_cmds_edit(datapath, selected_path, filename, vectorizer, num_kwd):
    """Make preparation of Speech Commands edited dataset and save pickle with
    audio paths and target texts (num_kwd also include the non-keyword token)."""

    input_path = sorted(glob("{}/*.wav".format(datapath), recursive=True))
    target_text = [] # each element is a list of strings
    count_occur = np.zeros(num_kwd, dtype=np.int64) # counts total number of keywords
    count_samps = np.zeros(num_kwd, dtype=np.int64)
    if len(input_path)==0:
        print('Found empty folder.')
    else:
        # create list of input path, list of target values (own keywords)
        for i in range(len(input_path)):
            x1 = os.path.split(input_path[i])[-1] # wav file name
            x2 = x1.split('.')[0] # remove .wav
            x3 = x2.split('-')[1:] # list of words (remove first number)
            target_text.append(' '.join(x3))

        # count keywords
        tokens = vectorizer(tf.constant(target_text))-1

        for i in range(num_kwd): # from 0 to 8
            s1 = tf.cast(tokens==i,tf.int64) # spot specific key
            s2 = tf.reduce_sum(s1,axis=-1) # count total by sample
            count_occur[i] = tf.reduce_sum(s2)
            count_samps[i] = tf.reduce_sum(tf.cast(s2!=0,tf.int64))

    # write counts to descript
    append_to_decript(selected_path, 'Speech Commands (edited)',
        len(input_path), num_kwd, count_occur, count_samps)

    # save to file
    pack = {'input_path':input_path, 'target_text':target_text,
            'count_occur':count_occur, 'count_samps':count_samps}
    with open(os.path.join(selected_path, filename+".pickle"), 'wb') as f:
        pickle.dump(pack, f)


def handle_librispeech_pair(datapath, selected_path, filename,
    vectorize_text, num_kwd, max_text_len, num_empty_samps):
    """Make preparation of Librispeech dataset and save pickle with
    audio paths and target texts (num_kwd also include the non-keyword token)."""
    
    # get list of input paths and target texts
    target_path = sorted(glob("{}/**/**/*.txt".format(datapath), recursive=True))
    if len(target_path)==0:
        raise AssertionError('Error reading folder files.')
    #target_path = glob("{}/**/**/*.txt".format(datapath), recursive=True)
    input_path = []
    target_text = []
    count_occur = np.zeros(num_kwd, dtype=np.int64) # counts total number of keywords
    count_samps = np.zeros(num_kwd, dtype=np.int64)
    for i, t in enumerate(target_path): # for each txt file
        with open(t, 'r') as f: # for each line in txt
            for line in f:
                code, raw_text = line.split(' ', maxsplit=1)
                ids = code.split('-')
                #audio_path = datapath + f'/{ids[0]}/{ids[1]}/{code}.wav'
                audio_path = os.path.join(datapath, ids[0], ids[1], code+'.wav')
                if os.path.exists(audio_path):
                    input_path.append(audio_path)
                    target_text.append(raw_text)
                else:
                    print(f'Audio file of code {code} did not found!')
    print(' paths and targets read, total:', len(input_path))

    # compute length of each text line
    target_tokens = vectorize_text(tf.constant(target_text))-1
    target_tokens_length = tf.cast(target_tokens!=-1,dtype=np.int64)
    target_tokens_length = tf.reduce_sum(target_tokens_length,axis=1)

    # remove samples with text length > max_text_len
    idx = np.where(target_tokens_length>max_text_len)[0]
    for i in sorted(idx, reverse=True):
        del input_path[i]
        del target_text[i]
    target_tokens = target_tokens[target_tokens_length<=max_text_len]
    print(' very long sentences removed, total:', len(input_path))

    # find and count keywords occurences
    kwd_presence = np.zeros((num_kwd,len(input_path)),dtype=bool)
    for i in range(1,num_kwd): # exclude [unk] keyword by now
        count_per_samp = tf.reduce_sum(tf.cast(target_tokens==i,tf.int64),axis=-1)
        kwd_presence[i] = count_per_samp!=0
        count_occur[i] = tf.reduce_sum(count_per_samp)
        count_samps[i] = tf.reduce_sum(tf.cast(kwd_presence[i],tf.int64))

    # verify samples that do contain keywords (fill first column of kwd_presence)
    for i in range(1,num_kwd):
        kwd_presence[0] = kwd_presence[0] | kwd_presence[i]
    kwd_on_ind = np.where(kwd_presence[0])[0]
    kwd_off_ind = np.where(~kwd_presence[0])[0]
    print(' samples with keywords, total:',len(kwd_on_ind))

    # filter dataset
    input_path_2  =  [input_path[i] for i in kwd_on_ind]
    target_text_2  = [target_text[i] for i in kwd_on_ind]
    #input_paths_off, target_text_off = [], []
    for i in range(num_empty_samps):
        input_path_2.append(input_path[kwd_off_ind[i]])
        target_text_2.append(target_text[kwd_off_ind[i]])

    # count number of non-keywords
    tokens_kwd_on = target_tokens[kwd_presence[0]]
    tokens_kwd_off = target_tokens[~kwd_presence[0]]
    tokens_kwd_off = tokens_kwd_off[:num_empty_samps]
    target_tokens = tf.concat([tokens_kwd_on,tokens_kwd_off],0)
    s2 = tf.reduce_sum(tf.cast(target_tokens==0,tf.int64),axis=-1) # count total by sample
    count_occur[0] = tf.reduce_sum(s2)
    count_samps[0] = tf.reduce_sum(tf.cast(s2!=0,tf.int64))

    #return input_path_2, target_text_2, target_tokens

    # write counts to descript
    append_to_decript(selected_path, 'Librispeech',
        len(input_path_2), num_kwd, count_occur, count_samps)

    # save to file
    pack = {'input_path':input_path_2, 'target_text':target_text_2,
            'count_occur':count_occur, 'count_samps':count_samps}
    with open(os.path.join(selected_path, filename+".pickle"), 'wb') as f:
        pickle.dump(pack, f)




# ==================================================================