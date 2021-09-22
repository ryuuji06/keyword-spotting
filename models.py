
import os
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

# Callback objects access the caller model as attribute (self.model)
# Save model parameters, overwriting the former one
class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, folderpath):
        self.best = np.inf
        self.folder = folderpath
        # store prediction model as parameter, to save at each epoch

    def on_epoch_end(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        val_loss = logs['val_loss']
        # write to hist.txt file
        line = str(epoch) + '\t' + '{:.4e}'.format(logs['loss']) + '\t' + '{:.4e}'.format(val_loss) + '\n'
        with open(self.folder+'/hist.txt', 'a') as f:
            f.write(line)
        if val_loss <= self.best:
            self.best = val_loss
            # save model weights
            self.model.save_weights(os.path.join(self.folder,'model_weights.h5'))
            

def build_model(num, feat_dim, num_kwd):
    """From models 0 to 4, uses 'mfcc' as feature type. Models 5 and 6
    uses 'melspec' as feature type."""
    if num==0:
        return model_test(feat_dim, num_kwd)
    elif num==1:
        return model_1(feat_dim, num_kwd)
    elif num==2:
        return model_2(feat_dim, num_kwd)
    elif num==3:
        return model_3(feat_dim, num_kwd)
    elif num==4:
        return model_4(feat_dim, num_kwd)
    elif num==5:
        return model_5(feat_dim, num_kwd)
    elif num==6:
        return model_6(feat_dim, num_kwd)
    else:
        return 0



# ----------------------------------------------------------------
#  C N N - R N N   A R C H I T E C T U R E S
# ----------------------------------------------------------------

# Three convolutional blocks, and two recurrent (LSTM) layers
# Convolutional blocks: two Conv1D 3x3, unit stride with ReLU
# Batch normalization at the beginning and after each conv block
def model_test(feat_dim, num_kwd):

    # Inputs to the model
    input_feat = layers.Input(shape=(None, feat_dim), name="input", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv1D(5, 3, strides=2, activation="relu", kernel_initializer="he_normal", padding="same")(input_feat)
    x = layers.Conv1D(5, 3, strides=2, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(5, 3, strides=2, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.LSTM(5, return_sequences=True, dropout=0.25)(x)

    # Output layer
    output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss)
    model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred

# Three convolutional blocks, and two recurrent (LSTM) layers
# Convolutional blocks: two Conv1D 3x3, unit stride with ReLU
# Batch normalization at the beginning and after each conv block
def model_1(feat_dim, num_kwd):

    # Inputs to the model
    input_feat = layers.Input(shape=(None, feat_dim), name="input", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    # First conv block
    x = layers.BatchNormalization(axis=-1)(input_feat)
    x = layers.Conv1D(32, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(32, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # Second conv block
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # Third conv block
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv1D(128, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(128, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # RNNs
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)

    # Output layer
    output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss)
    model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred


# Three convolutional blocks, and two recurrent (LSTM) layers
# Convolutional blocks: two Conv1D 3x3, unit stride, BN before ReLU
# Batch normalization at the beginning (and BN after each conv operation)
def model_2(feat_dim, num_kwd):

    # Inputs to the model
    input_feat = layers.Input(shape=(None, feat_dim), name="input", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    x = layers.BatchNormalization(axis=-1)(input_feat)
    
    # First conv block  
    x = layers.Conv1D(32, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(32, 3, strides=1, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # Second conv block
    x = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(64, 3, strides=1, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # Third conv block
    x = layers.Conv1D(128, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(128, 3, strides=1, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # RNNs
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)

    # Output layer
    output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss )
    model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred


# Three convolutional blocks, and two recurrent (LSTM) layers
# Convolutional blocks: Conv1D 5x5, stride 2, BN before ReLU
# Batch normalization at the beginning (and BN after each conv operation)
def model_3(feat_dim, num_kwd):

    # Inputs to the model
    input_feat = layers.Input(shape=(None, feat_dim), name="input", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    x = layers.BatchNormalization(axis=-1)(input_feat)

    # First conv block
    x = layers.Conv1D(32, 5, strides=2, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    # Second conv block
    x = layers.Conv1D(64, 5, strides=2, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    # Third conv block
    x = layers.Conv1D(128, 5, strides=2, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    # RNNs
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)

    # Output layer
    output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss )
    model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred


# Two convolutional blocks, and three recurrent (LSTM) layers
# Convolutional blocks: Conv1D 3x3, unit stride, BN before ReLU
# Batch normalization at the beginning (and BN after each conv operation)
def model_4(feat_dim, num_kwd):

    # Inputs to the model
    input_feat = layers.Input(shape=(None, feat_dim), name="input", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    x = layers.BatchNormalization(axis=-1)(input_feat)
    
    # First conv block  
    x = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(64, 3, strides=1, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # Second conv block
    x = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv1D(64, 3, strides=1, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2)(x)

    # RNNs
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)

    # Output layer
    output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss )
    model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred


# for inputing mel-spectrogram
# Three convolutional blocks, and two recurrent (LSTM) layers
# Convolutional blocks: two Conv2D 3x3, unit stride, BN before ReLU
# Batch normalization at the beginning (and BN after each conv operation)
def model_5(feat_dim, num_kwd):

    # Inputs to the model
    input_feat = layers.Input(shape=(None, feat_dim, 1), name="input", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    x = layers.BatchNormalization(axis=-1)(input_feat)
    
    # First conv block  
    x = layers.Conv2D(32, 5, strides=2, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    #x = layers.MaxPooling2D(2, strides=2)(x)

    # Second conv block
    x = layers.Conv2D(64, 5, strides=2, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    # RNNs
    #x = layers.Reshape(target_shape=(None, 128*(feat_dim//8)), name="reshape")(x)
    x = layers.Lambda(lambda inp: tf.reduce_sum(inp, axis=-2))(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)

    # Output layer
    output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss )
    model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred

# for inputing mel-spectrogram
# Three convolutional blocks, and two recurrent (LSTM) layers
# Convolutional blocks: two Conv2D 3x3, unit stride, BN before ReLU
# Batch normalization at the beginning (and BN after each conv operation)
def model_6(feat_dim, num_kwd):

    # Inputs to the model
    input_feat = layers.Input(shape=(None, feat_dim, 1), name="input", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    x = layers.BatchNormalization(axis=-1)(input_feat)
    
    # First conv block  
    x = layers.Conv2D(32, 3, strides=2, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    #x = layers.MaxPooling2D(2, strides=2)(x)

    # Second conv block
    x = layers.Conv2D(64, 3, strides=2, activation=None, kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    # RNNs
    #x = layers.Reshape(target_shape=(None, 128*(feat_dim//8)), name="reshape")(x)
    x = layers.Lambda(lambda inp: tf.reduce_sum(inp, axis=-2))(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.25)(x)

    # Output layer
    output = layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_feat, labels], outputs=ctcloss )
    model_pred = tf.keras.models.Model( inputs=input_feat, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred

# ==================================================================