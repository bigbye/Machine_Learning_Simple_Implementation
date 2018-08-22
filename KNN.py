import numpy as np
from collections import Counter
from metrics import accuracy_score

'''
本KNN算法的api调用方式完全仿照scikit-learn中的KNeighborsClassifier方法：fit、predict
还有metrics中的accuracy_score、score方法
'''


# knn分类器
class knn_classifier(object):
    def __init__(self, k):
        '''初始化knn分类器'''
        assert k >= 1, 'k must be valid'
        self.k = k
        self.x_train = None  # 首先在创建knn分类器时 将x_train，y_train置空
        self.y_train = None

    # 外部可调用的训练方法
    def fit(self, x_train, y_train):
        '''根据训练数据集x_train,y_train训练knn分类器'''
        assert self.k <= x_train.shape[0], "the size of x_train must be at least k"
        assert x_train.shape[0] == y_train.shape[0], "the size of x_data must equal to the size of y_data"
        self._x_train = x_train
        self._y_train = y_train
        return self

    # 外部可调用的预测整个数据的方法
    def predict(self, x_predict):
        '''给定带预测数据集x_predict,返回表示x_predict的结果向量'''
        assert self._x_train is not None and self._y_train is not None, "must fit before predict"
        assert x_predict.shape[1] == self._x_train.shape[1], 'the feature number of x_predict must be equal to x_train'
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    # 内部预测每个数据的方法
    def _predict(self, x):
        '''给定单个待预测数据x，返回x的预测结果值'''
        assert x.shape[0] == self._x_train.shape[1], 'the feature number of x must be equal to x_train'

        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self._x_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    # 直接设置一个测试算法准确度的方法
    def score(self, x_test, y_test):
        assert self._x_train is not None and self._y_train is not None, "must fit before score"
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)
