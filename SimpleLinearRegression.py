import numpy as np
from metrics import r2_score

'''
实现向量化运算的简单线性回归，替代for可以大大提高算法效率与性能
'''


class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, 'simple linear regression can only resolve single feature training data'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # n=0.0
        # d=0.0
        # for x, y in zip(x_train, y_train):
        #     n += (x - x_mean) * (y - y_mean)
        #     d += (x - x_mean) ** 2
        n = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = n / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, 'simple linear regressor can only resolve single feature training data'
        assert self.a_ is not None and self.b_ is not None, 'must fit before predict'

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x):
        return self.a_ * x + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)
