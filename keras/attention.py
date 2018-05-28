""""""

from keras.layers import Layer
from keras.layers import Dense, Multiply
import keras.backend as K


def attention1d(inputs):
    """
    Attention1D

    Args:
        inputs:
            shape: [batch_size, n_features]

    Returns:
        outputs:
            shape == input_shape
    """
    input_dim = K.int_shape(inputs)[-1]

    attn = Dense(input_dim, activation='softmax')(inputs)
    outputs = Multiply()([inputs, attn])

    return outputs


class Attention1D(Layer):
    """"""

    def build(self, input_shape):
        """"""

    def call(self, inputs, **kwargs):
        """"""

    def compute_output_shape(self, input_shape):
        """"""
