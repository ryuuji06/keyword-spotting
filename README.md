# Spotting Keywords in Speech Signals 

In this repository, I implement a system for detecting specific spoken words from a piece of speech signal, task known as **keyword spotting**. When reading a speech signal, I detect not only the presence, but also the time position of the keyword. For this purpose, I use a CNN-RNN network, with a CTC (Connectionist Temporal Classification) loss function [1].

Traditionally, keyword spotting and other speech detection tasks are performed with hidden Markov models (HMM). Since [2,3,4], there has been much more attention in solving this problem with deep convolutional or recurrent networks, performing with higher accuracy and with smaller footprint. Some of these approaches [2,3], however, depends on the annotation of every frame of the input signal as one of the keywords or as a non-keyword frame. In these works, the annotation is performed at a pre-training on a HMM system, from which the frame-wise labeling is obtained, and then the deep neural network can be trained and predict with few computational burden.

In this work, I use a CNN-RNN network with a CTC loss function [1], as in [5]. Such approach exempts the need of a label on each input signal frame, since the CTC forward-backward algorithm aligns the input sequence with the label sequence so as to minimize its loss function.

The input speech signal is preprocessed to extract 13 mel-frequency cepstral coefficents (MFCC) for each frame. The MFCCs are preferably used in speech recognition tasks because it captures well the distinctions between phonemes and is more compact than a direct usage of the spectrogram.

In this test, I specifically trained the model to identify the following eight keywords: `house`, `right`, `down`, `left`, `no`, `five`, `one`, `three`. Also, there is the token for non-keywords and the additional "null-element" token of the CTC algorithm, totalizing 10 output tokens. To train the network, I use a subset of the Google Speech Commands dataset [6] that contains isolated speech data for these keywords, and to train the detection of the keywords within entire spoken sentences, I use the Librispeech dataset [7].

As the occurences of the keywords in Librispeech is very low, the network is induced to predict non-keywords almost all the time, yielding an undesirabl behavior. To remedy this issue, I restrict the number of input samples of Librispeech that does not contain any keyword, and create a third dataset: I concatenate multiple speech signals from Speech Commands, keywords or not, to form sequences with multiple occurences of keywords in the same signal sample. This additional modification of the dataset improved a lot the prediction performance of the model.

## About the model

## Sample results

this

visualize features of word "down".

## References

[1] A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks". Proceedings of the 23rd International Conference on Machine Learning (ICML'06), p.369-376, 2006

[2] G. Chen, C. Parada, and G. Heigold, "Small-footprint keyword spotting using deep neural networks". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), p. 4087–4091, 2014

[3] T.N. Sainath and C. Parada. "Convolutional neural networks for small-footprint keyword spotting". Proceedings of the INTERSPEECH, 2015, p. 1478–1482.

[4] S.O. Arik, M. Kliegl, R. Child, J. Hestness, A. Gibiansky, C. Fougner, R. Prenger and A. Coates. "Convolutional recurrent neural networks for small-footprint keyword spotting". Proceedings of the INTERSPEECH, 2017, pp. 1606–1610.

[5] S. Fernandez, A. Graves, and J. Schmidhuber. "An application of recurrent neural networks to discriminative keyword spotting". Proceedings of the International Conferencce on Artificial Neural Networks (ICANN), 2007, pp. 220–229.

[6] P. Warden. "Speech commands: a dataset for limited-vocabulary speech recognition". 2018.

[7] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. "Librispeech: an ASR corpus based on public domain audio books". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, p. 5206-5210.

