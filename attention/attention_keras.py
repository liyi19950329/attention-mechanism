""""""
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec

from keras.layers import Layer
from keras.layers import Dense, Multiply
from keras.layers import Permute, RepeatVector
from keras.layers import Lambda

import keras.backend as K


def attention1d(inputs, activation='softmax'):
    """
    Attention1D:

        Input shape:
            [batch_size, n_features]
        Output shape:
            [batch_size, n_features] same as inputs

    Args:
        inputs:
            shape: [batch_size, n_features]
        activation:
            activation function
    """
    input_dim = K.int_shape(inputs)[-1]

    # attention
    # Input shape: [batch_size, n_features]
    attn = Dense(input_dim, activation=activation)(inputs)  # -> [batch_size, n_features]
    outputs = Multiply()([inputs, attn])  # -> [batch_size, n_features]

    return outputs


def attention2d(inputs, share_attention=True):
    """
    Attention2D:

        Input shape:
            [batch_size, n_steps, n_features]
        Output shape:
            [batch_size, n_steps, n_features] same as inputs

    Args:
        inputs:
        share_attention:

    Returns:

    """
    input_shape = K.int_shape(inputs)
    input_dim = input_shape[2]
    n_steps = input_shape[1]

    # attention
    # Input shape: [batch_size, n_steps, n_features]
    attn = Permute([2, 1])(inputs)  # -> [batch_size, n_features, n_steps]
    attn = Dense(n_steps)(attn)  # -> [batch_size, n_features, n_steps]

    if share_attention:
        attn = Lambda(lambda x: K.mean(x, axis=1))(attn)  # -> [batch_size, n_steps]
        attn = RepeatVector(input_dim)(attn)  # -> [batch_size, n_features, n_steps]

    attn = Permute([2, 1])(attn)  # -> [batch_size, n_steps, n_features]
    outputs = Multiply()([inputs, attn])  # -> [batch_size, n_steps, n_features]

    return outputs


class Attention1D(Layer):
    """
    Action:
        input_dim = K.int_shape(inputs)[-1]
        attn = Dense(input_dim, activation=activation)(inputs)
        outputs = Multiply()([inputs, attn])

    References:
        keras.layers.Dense
        keras.layers.Multiply
    """

    def __init__(self,
                 activation='softmax',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """"""
        super(Attention1D, self).__init__(**kwargs)

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        """"""
        super(Attention1D, self).build(input_shape)
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, input_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_dim,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

    def call(self, inputs, **kwargs):
        """"""
        outputs = inputs

        attn = K.dot(inputs, self.kernel)
        if self.use_bias:
            attn = K.bias_add(attn, self.bias)
        if self.activation is not None:
            attn = self.activation(attn)

        outputs *= attn
        return outputs

    def compute_output_shape(self, input_shape):
        """"""
        return input_shape
