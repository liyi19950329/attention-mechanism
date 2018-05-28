""""""

SEED = 1337


class config_dense:
    """"""
    data_size = 100000
    input_dim = 32
    attention_column = 1
    seed = SEED

    # train
    batch_size = 64
    epochs = 3

    # dense
    units = 64


class config_lstm:
    """"""
    data_size = 100000
    time_steps = 20
    input_dim = 2
    attention_column = 10
    seed = SEED
