
import numpy as np
import pyaudio
import tensorflow as tf
from models import build_model
from numpy_ringbuffer import RingBuffer


# ----------------------------------------------
#  NETWORK MODELS FOR ONLINE IMPLEMENTATION
# ----------------------------------------------

def my_relu(x):
    """Implementation of ReLU function (for numpy array)."""
    return x * (x > 0)


class TimeDistributedConv(object):
    """Implements a time-distributed (online) version of the following convolutional block:
    
    x = tf.keras.layers.Conv1D(f2, kernel_size, strides=1, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(f2, kernel_size, strides=1, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2, strides=2)(x)

    Uses ring buffers to store data from previous iterations. `f1` is the number of channels
    of the data input to the block.

    """
    def __init__(self, kernel_size, f1, f2, init_pool, w_conv1, w_conv2, w_bn):
        #super(ClassName, self).__init__()
        self.inp_buf   = RingBuffer(capacity=kernel_size, dtype=(np.float32,f1))
        self.conv1_buf = RingBuffer(capacity=kernel_size, dtype=(np.float32,f2))
        self.conv2_buf = RingBuffer(capacity=kernel_size, dtype=(np.float32,f2))

        self.WC1_1 = w_conv1[0].numpy()
        self.WC1_0 = w_conv1[1].numpy()
        self.WC2_1 = w_conv2[0].numpy()
        self.WC2_0 = w_conv2[1].numpy()
        self.WBN_1 = w_bn[1].numpy()
        self.WBN_2 = w_bn[2].numpy()
        self.WBN_const = w_bn[0].numpy() / np.sqrt(w_bn[3].numpy() + 0.001)

        self.pool_out = init_pool

        for k in range(kernel_size):
            self.inp_buf.append(np.zeros(f1,dtype=np.float32))
            self.conv1_buf.append(np.zeros(f2,dtype=np.float32))
            self.conv2_buf.append(np.zeros(f2,dtype=np.float32))

    def insert(self,x):
        """Input data to the convolutional block."""
        self.inp_buf.append(x)
        self.conv1_buf.append(my_relu(np.sum(
            self.WC1_1 * np.expand_dims(np.array(self.inp_buf), axis=-1), axis=(0,1)) + self.WC1_0))
        conv_out = np.sum(self.WC2_1 * np.expand_dims(np.array(self.conv1_buf), axis=-1), axis=(0,1)) + self.WC2_0
        self.conv2_buf.append(my_relu(self.WBN_const*(conv_out-self.WBN_2) + self.WBN_1))

        self.pool_out = not self.pool_out

    def get(self):
        """Get the convolutional block output."""
        return np.max(np.array(self.conv2_buf)[-2:], axis=0)     


def convolutional_block(feat_dim, kernel_size):
    """Convolutional block used in the main model."""

    input_feat = tf.keras.layers.Input(shape=(None, feat_dim), name="input", dtype="float32")

    x = tf.keras.layers.BatchNormalization()(input_feat)

    # First convolutional block
    x = tf.keras.layers.Conv1D(32, kernel_size, strides=1, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(32, kernel_size, strides=1, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2, strides=2)(x)

    # Second convolutional block
    x = tf.keras.layers.Conv1D(64, kernel_size, strides=1, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(64, kernel_size, strides=1, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2, strides=2)(x)

    # Third convolutional block
    x = tf.keras.layers.Conv1D(128, kernel_size, strides=1, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(128, kernel_size, strides=1, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2, strides=2)(x)

    return tf.keras.models.Model( inputs=input_feat, outputs=x )


def recurrent_block(feat_dim, num_kwd):
    """Recurrent block used in the main model."""
    input_feat = tf.keras.layers.Input(batch_input_shape=(1, None, feat_dim), name="input", dtype="float32")
    x = tf.keras.layers.LSTM(feat_dim, return_sequences=True, stateful=True, dropout=0.25)(input_feat)
    x = tf.keras.layers.LSTM(feat_dim, return_sequences=True, stateful=True, dropout=0.25)(x)
    # Output dense layer
    output = tf.keras.layers.Dense(num_kwd + 2, activation="softmax", name="dense2")(x)

    return tf.keras.models.Model( inputs=input_feat, outputs=output )


# resultfolder+'/'+model_name
def load_model_layers(model_path):

    _, model_pred = build_model(2, 13, 8)
    model_pred.load_weights(model_path)
    trained = model_pred.get_weights()

    # initial batch normalization
    WBN0_0 = model_pred.layers[1].weights[0].numpy()
    WBN0_1 = model_pred.layers[1].weights[1].numpy()
    WBN0_2 = model_pred.layers[1].weights[2].numpy()
    WBN0_3 = model_pred.layers[1].weights[3].numpy()
    WBN0_const = WBN0_0/np.sqrt(WBN0_3+0.001)
    # WBN0_const*(conv_out-WBN0_2) + WBN0_1
    init_batch_norm = lambda x: WBN0_const * (x-WBN0_2) + WBN0_1

    conv_layer_1 = TimeDistributedConv(3, 13, 32, True,
                        model_pred.layers[2].weights,
                        model_pred.layers[3].weights,
                        model_pred.layers[4].weights)
    conv_layer_2 = TimeDistributedConv(3, 32, 64, False,
                        model_pred.layers[7].weights,
                        model_pred.layers[8].weights,
                        model_pred.layers[9].weights)
    conv_layer_3 = TimeDistributedConv(3, 64, 128, True,
                        model_pred.layers[12].weights,
                        model_pred.layers[13].weights,
                        model_pred.layers[14].weights)
    
    rec_layers = recurrent_block(128, 8)
    rec_layers.set_weights(trained[28:36])

    return init_batch_norm, conv_layer_1, conv_layer_2, conv_layer_3, rec_layers



# ----------------------------------------------
#  FUNCTIONS FOR AUDIO STREAM
# ----------------------------------------------

def open_mic(fmt, channels, sampling_rate, chunk_size):
    """
    open_mic:
    creates a PyAudio object and initializes the mic stream
    ouputs: stream, PyAudio object
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(format = fmt,
                     channels = channels,
                     rate = sampling_rate,
                     input = True,
                     frames_per_buffer = chunk_size)
    return stream, pa


"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""
def get_sample(stream, pa, chunk_size):
    #input_data = stream.read(chunk_size, exception_on_overflow = False)
    input_data = stream.read(chunk_size)
    #data = np.fromstring(input_data,np.int16)
    data = np.frombuffer(input_data,np.int16)
    return data.astype('float32')/32768



class OverlapHandler(object):
    def __init__(self, chunk_size, overlap_size):
        #super(ClassName, self).__init__()
        self.buffer = RingBuffer(capacity=chunk_size+overlap_size, dtype=np.float32)
        self.buffer.extend(np.zeros(chunk_size+overlap_size,dtype=np.float32))
    def insert(self,x):
        self.buffer.extend(x)

    def get(self):
        return np.array(self.buffer)

# ==================================================================