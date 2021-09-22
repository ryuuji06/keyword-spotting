import os
import pickle
import numpy as np
import tensorflow as tf


# --------------------------------------------------------
#  T E X T   P R E P R O C E S S I N G
# --------------------------------------------------------

def english_standardization(text):
    """Preprocessing of target texts: remove punctuation, lowercase,
    turn plural to singular, past verb to standard
    (consider keywords: house, right, down, left, no, five, one, three)"""
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



# -----------------------------------------------------------
#   A U D I O   P R E P R O C E S S I N G
# -----------------------------------------------------------

@tf.function
def path2audio(path):
    """Load and return the audio signal from file `path`."""
    audio_bin = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio_bin, 1)
    return tf.squeeze(audio, axis=-1)

@tf.function
def additional_noise(audio, noise, prob):
    q = tf.random.uniform([], minval=0, maxval=tf.shape(noise)[0]-tf.shape(audio)[0], dtype=tf.int32)
    return tf.cond(
        tf.math.less(tf.random.uniform([],minval=0,maxval=1),prob),
        lambda: audio + noise[q : q + tf.shape(audio)[0]],
        lambda: audio
    )

@tf.function
def audio2melspec(audio, mel_matrix, frame_len=256, frame_hop=128, fft_len=256,
        num_mel_bins=80):
    """Extract the log mel-frequency spectrogram of the audio signal `audio`."""
    # spectrogram using stft
    # uses by default von Hann window (window_fn=tf.signal.hann_window)
    stfts = tf.signal.stft(audio, frame_length=frame_len, frame_step=frame_hop, fft_length=fft_len)
    # mel-frequencies spectrogram
    mel_spectrograms = tf.tensordot(tf.abs(stfts), mel_matrix, 1)
    # take log
    mel_spectrograms = tf.math.log(mel_spectrograms + 1e-8)
    return tf.expand_dims(mel_spectrograms,axis=-1)

@tf.function
def audio2mfcc(audio, mel_matrix, frame_len=256, frame_hop=128, fft_len=256,
    num_mel_bins=80, num_mfcc=13):
    """Extract the MFCCs of the audio signal `audio`."""
    # spectrogram using stft
    # uses by default von Hann window (window_fn=tf.signal.hann_window)
    stfts = tf.signal.stft(audio, frame_length=frame_len, frame_step=frame_hop, fft_length=fft_len)
    # mel-frequencies spectrogram
    mel_spectrograms = tf.tensordot(tf.abs(stfts), mel_matrix, 1)
    # compute and return MFCCs
    mel_spectrograms = tf.math.log(mel_spectrograms + 1e-8)
    return tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrograms)[:,:num_mfcc]

@tf.function
def join_input_label(inputs, labels):
    """Join data into dictionary of audio features input and target text label."""
    return {'input': inputs, 'target': labels}


def create_dataset(paths, texts, featurizer=None, noise=None, noise_prob=0.0,
        vectorizer=None, batch_size=32, num_kwd=10):
    """Create dataset object for either train or test set."""
    # (1) extract audio features
    audio_ds = tf.data.Dataset.from_tensor_slices(paths)
    audio_ds = audio_ds.map(path2audio)
    if noise_prob > 0.0 and noise_prob<= 1.0:
        audio_ds = audio_ds.map(lambda x: additional_noise(x, noise, noise_prob))
    audio_ds = audio_ds.map(featurizer, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # (2) tokenize text
    text_ds = tf.data.Dataset.from_tensor_slices(texts)
    text_ds = text_ds.map(lambda x: vectorizer(x)-1)

    # (3) batch inputs and targets, and zip
    audio_ds = audio_ds.padded_batch(batch_size)
    text_ds = text_ds.padded_batch(batch_size, padding_values=np.int64(num_kwd+1))
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(join_input_label)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

def create_train_eval_datasets(inputs, targets, batch_size=32, train_test_split=0.8,
    feature_type='mfcc', vectorizer=None, sr=16000, frame_len=256, frame_hop=128,
    fft_len=256, num_mel_bins=80, lower_freq=80.0, upper_freq=7600.0, num_mfcc=13,
    noise_settings=None, noise_prob=0.0):
    """Create dataset object for training and testing from `input` path list and `target` labels list."""

    # text vectorizer includes [UNK] and an empty token
    num_kwd = len(vectorizer.get_vocabulary())-2
    # number of samples for training and testing
    num_train = int( len(inputs)*train_test_split )

    # mel-filter bank matrix
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, fft_len//2+1, sr, lower_freq, upper_freq)

    # generate noise signal
    noise = tf.constant([])
    if noise_prob > 0.0 and noise_prob <= 1.0:
        # concatenate noise signal
        for f, a in zip(noise_settings['path'], noise_settings['amp']):
            audio_bin = tf.io.read_file(f)
            audio, _ = tf.audio.decode_wav(audio_bin, 1)
            noise = tf.concat([noise,a*tf.squeeze(audio, axis=-1)],0)

    if feature_type=='mfcc':
        train_ds = create_dataset(inputs[:num_train], targets[:num_train],
            batch_size=batch_size, num_kwd=num_kwd, vectorizer=vectorizer,
            noise=noise, noise_prob=noise_prob,
            featurizer=lambda sig: audio2mfcc( sig, mel_matrix,
            frame_len=frame_len, frame_hop=frame_hop, fft_len=fft_len,
            num_mel_bins=num_mel_bins, num_mfcc=num_mfcc))
        valid_ds = create_dataset(inputs[num_train:], targets[num_train:],
            batch_size=batch_size, num_kwd=num_kwd, vectorizer=vectorizer,
            noise=noise, noise_prob=noise_prob,
            featurizer=lambda sig: audio2mfcc( sig, mel_matrix,
            frame_len=frame_len, frame_hop=frame_hop, fft_len=fft_len,
            num_mel_bins=num_mel_bins, num_mfcc=num_mfcc))
    elif feature_type=='melspec':
        train_ds = create_dataset(inputs[:num_train], targets[:num_train],
            batch_size=batch_size, num_kwd=num_kwd, vectorizer=vectorizer,
            noise=noise, noise_prob=noise_prob,
            featurizer=lambda sig: audio2melspec( sig, mel_matrix,
            frame_len=frame_len, frame_hop=frame_hop, fft_len=fft_len,
            num_mel_bins=num_mel_bins))
        valid_ds = create_dataset(inputs[num_train:], targets[num_train:],
            batch_size=batch_size, num_kwd=num_kwd, vectorizer=vectorizer,
            noise=noise, noise_prob=noise_prob,
            featurizer=lambda sig: audio2melspec( sig, mel_matrix,
            frame_len=frame_len, frame_hop=frame_hop, fft_len=fft_len,
            num_mel_bins=num_mel_bins))

    return train_ds, valid_ds


def create_eval_dataset(inputs, targets, batch_size=32, train_test_split=0.8,
    feature_type='mfcc', vectorizer=None, sr=16000, frame_len=256, frame_hop=128,
    fft_len=256, num_mel_bins=80, lower_freq=80.0, upper_freq=7600.0, num_mfcc=13,
    noise_settings=None, noise_prob=0.0):
    """Create dataset object for training and testing from `input` path list and `target` labels list."""

    # text vectorizer includes [UNK] and an empty token
    num_kwd = len(vectorizer.get_vocabulary())-2
    # number of samples for training and testing
    num_train = int( len(inputs)*train_test_split )

    # mel-filter bank matrix
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, fft_len//2+1, sr, lower_freq, upper_freq)

    # generate noise signal
    noise = tf.constant([])
    if noise_prob > 0.0 and noise_prob <= 1.0:
        # concatenate noise signal
        for f, a in zip(noise_settings['path'], noise_settings['amp']):
            audio_bin = tf.io.read_file(f)
            audio, _ = tf.audio.decode_wav(audio_bin, 1)
            noise = tf.concat([noise,a*tf.squeeze(audio, axis=-1)],0)

    if feature_type=='mfcc':
        valid_ds = create_dataset(inputs[num_train:], targets[num_train:],
            batch_size=batch_size, num_kwd=num_kwd, vectorizer=vectorizer,
            noise=noise, noise_prob=noise_prob,
            featurizer=lambda sig: audio2mfcc( sig, mel_matrix,
            frame_len=frame_len, frame_hop=frame_hop, fft_len=fft_len,
            num_mel_bins=num_mel_bins, num_mfcc=num_mfcc))
    elif feature_type=='melspec':
        valid_ds = create_dataset(inputs[num_train:], targets[num_train:],
            batch_size=batch_size, num_kwd=num_kwd, vectorizer=vectorizer,
            noise=noise, noise_prob=noise_prob,
            featurizer=lambda sig: audio2melspec( sig, mel_matrix,
            frame_len=frame_len, frame_hop=frame_hop, fft_len=fft_len,
            num_mel_bins=num_mel_bins))

    return valid_ds


def visualize_prediction(inputs, targets, noise_settings=None, noise_prob=0.0,
    feature_type='mfcc', vectorizer=None, sr=16000, frame_len=256, frame_hop=128,
    fft_len=256, num_mel_bins=80, lower_freq=80.0, upper_freq=7600.0, num_mfcc=13,
    ):
    """Create dataset object for training and testing from `input` path list and `target` labels list."""

    # text vectorizer includes [UNK] and an empty token
    num_kwd = len(vectorizer.get_vocabulary())-2

    # mel-filter bank matrix
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, fft_len//2+1, sr, lower_freq, upper_freq)

    # generate noise signal
    noise = tf.constant([])
    if noise_prob > 0.0 and noise_prob <= 1.0:
        # concatenate noise signal
        for f, a in zip(noise_settings['path'], noise_settings['amp']):
            audio_bin = tf.io.read_file(f)
            audio, _ = tf.audio.decode_wav(audio_bin, 1)
            noise = tf.concat([noise,a*tf.squeeze(audio, axis=-1)],0)

    # (1) audio dataset
    audio_ds = tf.data.Dataset.from_tensor_slices(inputs)
    audio_ds = audio_ds.map(path2audio)
    if noise_prob > 0.0 and noise_prob<= 1.0:
        audio_ds = audio_ds.map(lambda x: additional_noise(x, noise, noise_prob))

    # (2) feature dataset
    if feature_type=='mfcc':
        feat_ds = audio_ds.map(lambda sig: audio2mfcc( sig, mel_matrix, frame_len=frame_len,
            frame_hop=frame_hop, fft_len=fft_len, num_mel_bins=num_mel_bins,
            num_mfcc=num_mfcc), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif feature_type=='melspec':
        feat_ds = audio_ds.map(lambda sig: audio2melspec( sig, mel_matrix,
            frame_len=frame_len, frame_hop=frame_hop, fft_len=fft_len,
            num_mel_bins=num_mel_bins), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # (3) tokenize text
    text_ds = tf.data.Dataset.from_tensor_slices(targets)
    text_ds = text_ds.map(lambda x: vectorizer(x)-1)

    # (4) batch inputs and targets, and zip
    ds = tf.data.Dataset.zip((audio_ds, feat_ds, text_ds))
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds



# ==================================================================