import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        assert x.ndim == 2, 'the dimension of x must be 2'
        self.mean_ = np.array([np.mean(x[:, i]) for i in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[:, i]) for i in range(x.shape[1])])
        return self

    def transform(self, x):
        '''将x根据这个standardScaler进行均值方差归一化处理'''
        assert x.ndim == 2, 'the dimension of x must be 2'
        assert self.mean_ is not None and self.scale_ is not None, 'must fit before transform'
        assert x.shape[1] == len(self.mean_), 'the feature number of x must be equal to mean_ and std_'
        resX = np.empty(shape=x.shape, dtype=np.float32)
        for col in range(x.shape[1]):
            resX[:, col] = (x[:, col] - self.mean_[col]) / self.scale_[col]
        return resX
