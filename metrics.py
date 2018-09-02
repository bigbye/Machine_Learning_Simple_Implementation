import numpy as np


# 计算预测精确度函数
def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], 'the size of y_true must be equal to the size of y_predict'
    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    ''' mse'''
    assert len(y_true) == len(y_predict), 'the size of y_true must be equal to the size of y_predict'
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    ''' rmse'''
    assert len(y_true) == len(y_predict), 'the size of y_true must be equal to the size of y_predict'
    return np.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    ''' mae'''
    assert len(y_true) == len(y_predict), 'the size of y_true must be equal to the size of y_predict'
    return np.sum(np.abs(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    ''' r2_square'''
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
