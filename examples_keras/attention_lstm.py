""""""

import numpy as np

from data_helper import gen_time_data
from config import config_lstm as config

from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras.layers import Flatten

from attention.attention_keras import attention2d

np.random.seed(config.seed)


def build_model():
    """"""
    inputs = Input(shape=(config.time_steps, config.input_dim))

    lstm_out = LSTM(config.lstm_units, return_sequences=True)(inputs)
    attn = attention2d(lstm_out)
    attn = Flatten()(attn)
    output = Dense(1, activation='sigmoid')(attn)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    """"""
    x, y = gen_time_data()

    model = build_model()
    model.summary()

    model.fit(x, y,
              epochs=config.epochs,
              batch_size=config.batch_size,
              validation_split=0.8)
