import numpy as np


# 将数据集分为测试集与训练集
def train_test_split(x, y, test_ratio=0.2, seed=None):
    '''将数据x、y按照test_ratio分割成x_train、y_train、x_test、y_test'''
    assert x.shape[0] == y.shape[0], 'the size of x and y must be the same'
    assert 0.0 <= test_ratio <= 1.0, 'test_ratio must be valid'
    if seed:
        np.random.seed(seed)  # 用于debug多次random一样

    # 我们将x,y的索引进行随机化，80%用于train，20%用于test
    shuffle_indexes = np.random.permutation(len(x))
    # print(shuffle_indexes)  # 乱序的索引
    test_len = int(len(x) * test_ratio)

    test_indexes = shuffle_indexes[:test_len]
    train_indexes = shuffle_indexes[test_len:]

    x_train = x[train_indexes]
    y_train = y[train_indexes]

    x_test = x[test_indexes]
    y_test = y[test_indexes]

    return x_train, x_test, y_train, y_test
