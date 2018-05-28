"""

"""

import numpy as np

from data_helper import gen_data
from config import config_dense as config

from keras.models import Model

from keras.layers import Input, Dense

from attention.attention_keras import attention1d, Attention1D

np.random.seed(config.seed)


def build_model():
    """"""
    inputs = Input(shape=(config.input_dim,))

    # attention
    # attn = Dense(config.input_dim, activation='softmax', name='attention_vec')(inputs)
    # attn = Multiply()([inputs, attn])
    attn = attention1d(inputs)
    # attn = Attention1D()(inputs)

    net = Dense(64)(attn)
    outputs = Dense(1, activation='sigmoid')(net)

    model = Model([inputs], [outputs])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    """"""
    x, y = gen_data()

    model = build_model()
    model.summary()

    model.fit(x, y,
              epochs=config.epochs,
              batch_size=config.batch_size,
              validation_split=0.8)


