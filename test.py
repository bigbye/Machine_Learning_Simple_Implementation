from sklearn import datasets
from model_selection import *
from preprocessing import *
from KNN import *
from metrics import *

'''
这是对自己构造的KNN算法的测试脚本
'''
iris = datasets.load_iris()
x = iris.data
y = iris.target
# 划分训练数据与测试数据
x_train, y_train, x_test, y_test = train_test_split(x, y)
# 均值方差归一化
scaler = StandardScaler()
scaler.fit(x_train)
x_train_standard = scaler.transform(x_train)
x_test_standard = scaler.transform(x_test)
# 生成分类器
clr = knn_classifier(k=4)
clr.fit(x_train_standard, y_train)
y_predict = clr.predict(x_test_standard)
# 对测试结果进行评估、
acc = accuracy_score(y_test, y_predict)
print('acc=', acc)
# 直接对算法精度进行评估
score = clr.score(x_test_standard, y_test)
print('score=', score)
