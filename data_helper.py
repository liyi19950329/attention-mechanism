"""
生成随机测试数据
"""
import numpy as np
from config import *


def gen_data(n=config_dense.data_size,
             input_dim=config_dense.input_dim,
             attention_column=config_dense.attention_column):
    """生成随机数据
    数据特征：
        x[attention_column] = y
    网络应该学习到 y = x[attention_column]，这是为了测试 attention 特意构造的数据

    Returns:
        x: [n, input_dim]
        y: [n, 1]
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


def gen_time_data(n=config_lstm.data_size,
                  time_steps=config_lstm.time_steps,
                  input_dim=config_lstm.input_dim,
                  attention_column=config_lstm.attention_column):
    """生成随机数据

    Returns:
        x: [n, time_steps, input_dim]
        y: [n, 1]
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y
