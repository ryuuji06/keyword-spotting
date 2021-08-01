
#import os
#from glob import glob # module for finding pathnames
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class CTCLayer(layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model_1(feat_dim, num_kwd):

	# Inputs to the model
	input_feat = layers.Input(shape=(None, feat_dim), name="input", dtype="float32")
	labels = layers.Input(name="target", shape=(None,), dtype="float32")

	# initial batch normalization
	x = layers.BatchNormalization(axis=-1)(input_feat)

	# First conv block
	x = layers.Conv1D(32, 5, strides=1, activation="relu",
		kernel_initializer="he_normal", padding="same")(input_feat)
	x = layers.MaxPooling1D(2, strides=2)(x)
	# Second conv block
	x = layers.Conv1D(64, 5, strides=1, activation="relu",
		kernel_initializer="he_normal", padding="same")(x)
	x = layers.MaxPooling1D(2, strides=2)(x)

	# RNNs
	x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
	x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

	# Output layer
	output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

	# Add CTC layer for calculating CTC loss at each step
	ctcloss = CTCLayer(name="ctc_loss")(labels, output)

	# Define the model
	model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss, name="ocr_model_v1" )
	model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
	# Optimizer
	opt = tf.keras.optimizers.Adam()
	# Compile the model and return
	model_train.compile(optimizer=opt)
	return model_train, model_pred


# test VGG strategy: sequence of conv layers with small kernels
def build_model_2(feat_dim, num_kwd):

	# Inputs to the model
	input_feat = layers.Input(shape=(None, feat_dim), name="input", dtype="float32")
	labels = layers.Input(name="target", shape=(None,), dtype="float32")

	# initial batch normalization
	x = layers.BatchNormalization(axis=-1)(input_feat)

	# First conv block
	x = layers.Conv1D(32, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(input_feat)
	x = layers.Conv1D(32, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
	x = layers.MaxPooling1D(2, strides=2)(x)
	# Second conv block
	x = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
	x = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
	x = layers.MaxPooling1D(2, strides=2)(x)

	# RNNs
	x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
	x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

	# Output layer
	output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

	# Add CTC layer for calculating CTC loss at each step
	ctcloss = CTCLayer(name="ctc_loss")(labels, output)

	# Define the model
	model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss, name="ocr_model_v1" )
	model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
	# Optimizer
	opt = tf.keras.optimizers.Adam()
	# Compile the model and return
	model_train.compile(optimizer=opt)
	return model_train, model_pred



# ==================================================================