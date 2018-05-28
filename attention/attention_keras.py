""""""

from keras.layers import Layer
from keras.layers import Dense, Multiply
from keras.layers import Permute, RepeatVector
from keras.layers import Lambda

import keras.backend as K


def attention1d(inputs, activation='softmax'):
    """
    Attention1D:
        The input shape is [n_features] without batch axis

    Args:
        inputs:
            shape: [batch_size, n_features]
        activation:

    Returns:
        outputs:
            shape == input_shape
    """
    input_dim = K.int_shape(inputs)[-1]

    attn = Dense(input_dim, activation=activation)(inputs)
    outputs = Multiply()([inputs, attn])

    return outputs


def attention2d(inputs, share_attention=True):
    """"""
    input_shape = K.int_shape(inputs)
    input_dim = input_shape[2]
    n_steps = input_shape[1]

    # attention
    attn = Permute([2, 1])(inputs)  # -> [batch_size, n_features, n_steps]
    attn = Dense(n_steps)(attn)

    if share_attention:
        attn = Lambda(lambda x: K.mean(x, axis=1))(attn)
        attn = RepeatVector(input_dim)(attn)

    attn = Permute([2, 1])(attn)
    outputs = Multiply()([inputs, attn])

    return outputs


class Attention1D(Layer):
    """"""

    def call(self, inputs, **kwargs):
        """"""
        return attention1d(inputs)

    def compute_output_shape(self, input_shape):
        """"""
        return input_shape
