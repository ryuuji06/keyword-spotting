# Real-Time Keyword Spotting


## Introduction

In this repository, I implement in Python/Keras a system for detecting specific spoken words in speech signals, task known as **keyword spotting** (KWS). When reading a speech signal, I detect not only the presence, but also the time position of the keyword. For this purpose, I use a CNN-RNN network, with a CTC (Connectionist Temporal Classification) loss function [1]. As the system is causal (I use only unidirectional LSTMs, and convolutional operations are causal with some delay), I also make an implementation in real-time.

Traditionally, keyword spotting and other speech detection tasks are usually implemented with hidden Markov models (HMM). Since [2,3,4], there has been much more attention in solving this problem with deep convolutional or recurrent networks, performing with higher accuracy and with smaller footprint. Some of these approaches [2,3], however, depends on the annotation of every frame of the input signal as one of the keywords or as a non-keyword frame. In these works, the annotation is performed at a pre-training on a HMM system, from which the frame-wise labeling is obtained, and then the deep neural network can be trained and predict with few computational burden.

In this work, I use a CNN-RNN network with a CTC loss function [1], as in [5]. Such approach exempts the need of a label on each input signal frame, since the CTC forward-backward algorithm aligns the input sequence with the label sequence so as to minimize its loss function.

The input speech signal is preprocessed to extract 13 mel-frequency cepstral coefficents (MFCC) for each frame. The MFCCs are preferably used in speech recognition tasks because it captures well the distinctions between phonemes and is more compact than a direct usage of the spectrogram.

In this test, I specifically trained the model to identify the following eight keywords (arbitrarily chosen from publicly available datasets): `house`, `right`, `down`, `left`, `no`, `five`, `one` and `three`. To train the network, I use a subset of the Google Speech Commands dataset [6] that contains isolated speech data for these keywords, and to train the detection of the keywords within entire spoken sentences, I use the Librispeech dataset [7].

As the occurences of the keywords in Librispeech is very low, the network is induced to predict non-keywords almost all the time, yielding an undesirable behavior. To remedy this issue, I restrict the number of input samples of Librispeech that does not contain any keyword, and create a third dataset: I concatenate multiple speech signals from Speech Commands, keywords or not, to form sequences with multiple occurences of keywords in the same signal sample. This additional modification of the dataset improved a lot the prediction performance of the model.


## Usage Instructions

### (1) Download Datasets

First, download the datasets you will use to train your model. Any dataset can be used, although it is easier to use Google Speech Commands [6] and LibriSpeech [7] (I used the `train-clean-100.tar.gz` subset), because I already implemented functions to read these specific datasets in `functions_datasets.py`.

Optionally, you can also download a dataset of additional background noise. I used MS-SNSD [8] (`noise-train`) in my tests.

(This additional noise would intuitively improve the robustness of the detection. However, in my tests, it did not improve the evauation metrics, and actually even degraded the online performance I tested with my microphone.)

### (2) Convert LibriSpeech Files to WAV

LibriSpeech files are originally in FLAC format. As I use Tensorflow audio decoder, the files must be in WAV format. Just move `flac2wav.sh` to the dataset folder and run it (it uses avconv program from libav http://builds.libav.org/windows/nightly-gpl/, in case you don't already have it). If you are in Unix system, just run

`./flac2wav.sh`

or, if you are in Windows,

`powershell ./flac2wav.sh`

### (3) Prepare Datasets

To improve performance, I generate an additional dataset with multiple keywords per audio sample out of Google Speech Commands. You can generate it by running `create_dataset.py`. In the first lines of the code, you can edit some settings, such as the folder paths, the keywords you want to use, the number of samples per keyword, etc.

When training or evaluating the model, the codes read a file with the path and transcription of previously selected audio files. You generate this file by running `prepare_datasets.py`, which also generates a file containing the some info of the datasets, such as the number of samples per keyword. You can edit some settings, such as dataset paths, selected keywords and samples restriction in the first lines of  the code.


### (4) Train model

To train the model, just run `kws_train.py`. Training parameters are set via parsing, such as

`python kws_train.py -d data -r result_01 -m 2 --train_test_split 0.7 --epochs 50`

where `data` is the folder generated by `prepare_datasets.py`, `result_01` is the folder to store the result files (model weights and history), 2 is the chosen network model, a fraction of 0.7 of the dataset is taken as training set, and the training runs 50 epochs. Other parameters can be verified at the beggining of the code in `kws_train.py`.

Currently, there is only a few non-parametric built-in network models, defined in `models.py`, all of them with convolutional and recurrent layers. Some of them use Conv1D, intended to be used with MFCC feature, and others use Conv2D, intended to be used along with mel-spectrogram feature.


### (5) Evaluation and Prediction

-

### (6) Online Implementation

Currently implemented only for model MFCC 1.

## About the model

The tested model begins with a batch normalization step. Then, the batch normalized MFCCs propagates through five layers:

 - Conv1D layer, 32 kernels of size 5x5, unit stride, followed by ReLU activation and Max Pooling stride 2;
 - Conv2D layer, 64 kernels of size 5x5, unit stride, followed by ReLU activation and Max Pooling stride 2;
 - Bidirectional LSTM layer with 128 units, with dropout;
 - Bidirectional LSTM layer with 128 units, with dropout;
 - Dense layer with 10 units (8 keyword tokens, 1 non-keyword token and CTC null token) with softmax activation.

## Sample results

The figure below illustrates a speech signal and their features (spectrogram and MFCCs). It is an example for the word "down". While the spectrogram requires 257 coefficients to represent a single signal frame, we can compactly represent it with 13 MFCCs.

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/images/ex_speech_features.png" width="700">

The next figure show the learning curve during training, considering the modified dataset strategy (history and models stored in folder `results03`). 

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/images/ex_hist.png" width="400">

The next figure shows the model outputs (probabilities of each token) when inputing a sample signal from Librispeech, with the sentence

`He was young. No spear had touched him. No poison lurked in his wine`

The last token (cyan) is the null character token, inherent of the CTC algorithm, and encodes no actual character.  A posterior handling of the predicted tokens is performed to remove token duplicates in sequence, which is necessary when using CTC. Note that the model detected correctly the keywords and non-keywords present in the speech, and resonably aligned their positions to the actual moment they are spoken. The only alignment problem occurs with the first token, which is early predicted at the beginning of the output sequence.

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/images/ex_prediction.png" width="800">


## References

[1] A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks". Proceedings of the 23rd International Conference on Machine Learning (ICML'06), p.369-376, 2006

[2] G. Chen, C. Parada, and G. Heigold, "Small-footprint keyword spotting using deep neural networks". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), p. 4087–4091, 2014

[3] T.N. Sainath and C. Parada. "Convolutional neural networks for small-footprint keyword spotting". Proceedings of the INTERSPEECH, 2015, p. 1478–1482.

[4] S.O. Arik, M. Kliegl, R. Child, J. Hestness, A. Gibiansky, C. Fougner, R. Prenger and A. Coates. "Convolutional recurrent neural networks for small-footprint keyword spotting". Proceedings of the INTERSPEECH, 2017, pp. 1606–1610.

[5] S. Fernandez, A. Graves, and J. Schmidhuber. "An application of recurrent neural networks to discriminative keyword spotting". Proceedings of the International Conferencce on Artificial Neural Networks (ICANN), 2007, pp. 220–229.

[6] P. Warden. "Speech commands: a dataset for limited-vocabulary speech recognition". 2018. http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

[7] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. "Librispeech: an ASR corpus based on public domain audio books". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, p. 5206-5210. https://www.openslr.org/12

[8] Microsoft Scalable Noisy Speech Dataset. https://github.com/microsoft/MS-SNSD
