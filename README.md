# Real-Time Keyword Spotting


## Introduction

In this repository, I implement in Python/Keras a system for detecting specific spoken words in speech signals, task known as **keyword spotting** (KWS). When reading a speech signal, I detect not only the presence, but also the time position of the keyword. For this purpose, I use a CNN-RNN network, with a CTC (Connectionist Temporal Classification) loss function [1]. As the system is causal (I use only unidirectional LSTMs, and convolutional operations are causal with some delay), I also make an implementation in real-time.

Traditionally, keyword spotting and other speech detection tasks are usually implemented with hidden Markov models (HMM). Since [2,3,4], there has been much more attention in solving this problem with deep convolutional or recurrent networks, performing with higher accuracy and with smaller footprint. Some of these approaches [2,3], however, depends on the annotation of every frame of the input signal as one of the keywords or as a non-keyword frame. In these works, the annotation is performed at a pre-training on a HMM system, from which the frame-wise labeling is obtained, and then the deep neural network can be trained and predict with few computational burden.

In this work, I use a CNN-RNN network with a CTC loss function [1], as in [5]. Such approach exempts the need of a label on each input signal frame, since the CTC forward-backward algorithm aligns the input sequence with the label sequence so as to minimize its loss function.

The input speech signal is preprocessed to extract 13 mel-frequency cepstral coefficents (MFCC) for each frame. The MFCCs are preferably used in speech recognition tasks because it captures well the distinctions between phonemes and is more compact than a direct usage of the spectrogram.

In this test, I specifically trained the model to identify the following eight keywords (arbitrarily chosen from publicly available datasets): `house`, `right`, `down`, `left`, `no`, `five`, `one` and `three`. To train the network, I use a subset of the Google Speech Commands dataset [6] that contains isolated speech data for these keywords, and to train the detection of the keywords within entire spoken sentences, I use the Librispeech dataset [7].

As the occurences of the keywords in Librispeech is very low, the network is induced to predict non-keywords almost all the time, yielding an undesirable behavior. To remedy this issue, I restrict the number of input samples of Librispeech that does not contain any keyword, and create a third dataset: I concatenate multiple speech signals from Speech Commands, keywords or not, to form sequences with multiple occurences of keywords in the same signal sample. This additional modification of the dataset improved a lot the prediction performance of the model.


## Usage Instructions

### (1) Install Required Python Packages

Install the required packages by running the following:

`pip install -r requirements.txt`

If you are in Windows, you might find trouble installing PyAudio (required only for real-time implementation). If this is the case, you must download the appropriate `.whl` file directly from the PyAudio repository, as instructed in

https://stackoverflow.com/questions/52283840/i-cant-install-pyaudio-on-windows-how-to-solve-error-microsoft-visual-c-14


### (2) Download Datasets

First, download the datasets you will use to train your model. Any dataset can be used, although it is easier to use Google Speech Commands [6] and LibriSpeech [7] (I used the `train-clean-100.tar.gz` subset), because I already implemented functions to read these specific datasets in `functions_datasets.py`.

Optionally, you can also download a dataset of additional background noise. I used MS-SNSD [8] (`noise-train`) in my tests.

(This additional noise would intuitively improve the robustness of the detection. However, in my tests, it did not improve the evauation metrics, and actually even degraded the online performance I tested with my microphone.)

### (3) Convert LibriSpeech Files to WAV

LibriSpeech files are originally in FLAC format. As I use Tensorflow audio decoder, the files must be in WAV format. Just move `flac2wav.sh` to the dataset folder and run it (it uses avconv program from libav http://builds.libav.org/windows/nightly-gpl/, in case you don't already have it). If you are in Unix system, just run

`./flac2wav.sh`

or, if you are in Windows,

`powershell ./flac2wav.sh`

### (4) Prepare Datasets

To improve performance, I generate an additional dataset with multiple keywords per audio sample out of Google Speech Commands. You can generate it by running `create_dataset.py`. In the first lines of the code, you can edit some settings, such as the folder paths, the keywords you want to use, the number of samples per keyword, etc.

When training or evaluating the model, the codes read a file with the path and transcription of previously selected audio files. You generate this file by running `prepare_datasets.py`, which also generates a file containing the some info of the datasets, such as the number of samples per keyword. You can edit some settings, such as dataset paths, selected keywords and samples restriction in the first lines of  the code.


### (5) Train model

To train the model, just run `kws_train.py`. Training parameters are set via parsing, such as

`python kws_train.py -d data -r result_01 -m 2 --train_test_split 0.7 --epochs 50`

where `data` is the folder generated by `prepare_datasets.py`, `result_01` is the folder to store the result files (model weights and history), 2 is the chosen network model, a fraction of 0.7 of the dataset is taken as training set, and the training runs 50 epochs. Other parameters can be verified at the beggining of the code in `kws_train.py`.

Currently, there is only a few non-parametric built-in network models, defined in `models.py`, all of them with convolutional and recurrent layers. Some of them use Conv1D, intended to be used with MFCC feature, and others use Conv2D, intended to be used along with mel-spectrogram feature.


### (6) Evaluation and Prediction

To evaluate the model, I count the number of correct detections (true positives), the number of false positives and false negatives at each test sample, for each keyword, and compute precision, recall and F1-score. The global metrics for the model is then computed as the average of the metrics for each keyword.

To evaluate the model, run `kws_evaluate.py`, for example, by

`python kws_evaluate.py -d data -r result_01`

It uses only the portion of data considered as validation during the training of the referenced result folder. It also generates a file `performance.txt` in the result folder recording the performance results.

In order to visualize the network outputs, you can run `kws_predict.py`, which prints, for some selected samples of the dataset, the audio signal and the network output token probabilities with Matplotlib, along with the label token sequence and the predicted sequence.


### (7) Real-time Implementation

I also made a real-time implementation of the network, using PyAudio to capture the audio stream from a microphone device. It can be run as, for example,

`python kws_realtime.py -r result_01`

which runs using the model parameters stored in the result folder. Currently, it only uses the model number 2, which apparently gives the best result.


## About the model

I have tested some different network models, but they all have a similar concept of using some convolutional layers followed by recurrent ones. In the following, I describe the model number 2, which has given the best results in the tests.

 - MFCC feature extraction;
 - Batch normalization;
 - Convolutional block below, with N = 32 filters;
 - Convolutional block below, with N = 64 filters;
 - Convolutional block below, with N = 128 filters;
 - Unidirectional LSTM layer with 128 units, with dropout rate 0.25;
 - Unidirectional LSTM layer with 128 units, with dropout rate 0.25;
 - Dense layer with 10 units (8 keyword tokens, 1 non-keyword token and CTC null token) with softmax activation.

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/examples/figures/block_diagram_conv1.png" width="150">


## Sample results

The figure below illustrates a speech signal and their features (mel-spectrogram and MFCCs). It is an example for the spoken word "left".

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/examples/figures/illust_features_1.png" width="500">

The figure below shows the learning curve for the models within the `example` folder. Note that model 2 performs the best. All models are trained with no additive augmentation noise, except `Model 2 noise`, in which a background noise was added in 50% of the training and validations samples.

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/examples/figures/learning_curves.png" width="700">

The table below depicts the performance of each model, considering the metrics precision, recall and F1-score. These metrics were computed by counting the total number of true positive, false positives and false negatives detections for each keyword, and taking the mean.

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/examples/figures/performance_table.png" width="350">

The next figure shows the network outputs (probabilities of each token) when inputing a sample signal from Librispeech, with the sentence

`certainly not to walk three miles or four miles or five miles.`

The last token (dashed black line) is the null character token, inherent of the CTC algorithm, and encodes no actual character.  A posterior handling of the predicted tokens is performed to remove token duplicates in sequence, which is necessary when using CTC. Note that the model detected correctly the two keywords, and have not detected one of the fillers. This represents error though, since we are only interested in the keywords.

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/examples/figures/prediction_example_2.png" width="600">

The next figure illustrates the real-time application. It captures the instant when I said "no one left the house". Note that it did not detected the keyword "one". In this real-time application, the network seemed not to perform as well as the metrics above ensure.

<img src="https://github.com/ryuuji06/keyword-spotting/blob/main/examples/figures/online_test.png" width="400">


## References

[1] A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks". Proceedings of the 23rd International Conference on Machine Learning (ICML'06), p.369-376, 2006

[2] G. Chen, C. Parada, and G. Heigold, "Small-footprint keyword spotting using deep neural networks". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), p. 4087–4091, 2014

[3] T.N. Sainath and C. Parada. "Convolutional neural networks for small-footprint keyword spotting". Proceedings of the INTERSPEECH, 2015, p. 1478–1482.

[4] S.O. Arik, M. Kliegl, R. Child, J. Hestness, A. Gibiansky, C. Fougner, R. Prenger and A. Coates. "Convolutional recurrent neural networks for small-footprint keyword spotting". Proceedings of the INTERSPEECH, 2017, pp. 1606–1610.

[5] S. Fernandez, A. Graves, and J. Schmidhuber. "An application of recurrent neural networks to discriminative keyword spotting". Proceedings of the International Conferencce on Artificial Neural Networks (ICANN), 2007, pp. 220–229.

[6] P. Warden. "Speech commands: a dataset for limited-vocabulary speech recognition". 2018. http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

[7] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. "Librispeech: an ASR corpus based on public domain audio books". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, p. 5206-5210. https://www.openslr.org/12

[8] Microsoft Scalable Noisy Speech Dataset. https://github.com/microsoft/MS-SNSD
