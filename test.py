from sklearn import datasets
from model_selection import *
from preprocessing import *
from KNN import *
from LinearRegression import *
from metrics import *
from PCA import *
import datetime
import matplotlib.pyplot as plt


def test_KNN():
    '''
    这是对自己构造的KNN算法的测试脚本
    '''
    print('KNN TEST BEGIN')
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    # 划分训练数据与测试数据
    x_train, x_test, y_train, y_test = train_test_split(x, y)
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


def test_Linear_Regression():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    print(X.shape)  # (500,13)

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
    # 如果使用梯度下降需要标准化数据，不然会造成内存溢出
    scaler = StandardScaler()
    scaler.fit(X_train)
    standard_X_train = scaler.transform(X_train)
    standard_X_test = scaler.transform(X_test)
    print('-------------------正规公式计算-----------------------------')
    # 正规公式法求解线性回归
    start_time = datetime.datetime.now()
    reg = LinearRegression()
    reg.fit_normal(X_train, y_train)  # 无需归一化
    end_time = datetime.datetime.now()
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X_test, y_test))
    print('use time:', (end_time - start_time).microseconds)
    print('-------------------批量梯度下降------------------------------')
    #
    # 批次梯度下降法求解线性回归
    start_time = datetime.datetime.now()
    reg2 = LinearRegression()
    reg2.fit_gd(standard_X_train, y_train)
    end_time = datetime.datetime.now()
    print(reg2.coef_)
    print(reg2.intercept_)
    print(reg2.score(standard_X_test, y_test))
    print('use time:', (end_time - start_time).microseconds)
    print('-------------------随机梯度下降------------------------------')

    # 随机梯度下降法求解线性回归
    start_time = datetime.datetime.now()
    reg3 = LinearRegression()
    reg3.fit_sgd(standard_X_train, y_train, n_iters=100)
    end_time = datetime.datetime.now()
    print(reg3.coef_)  # 斜率
    print(reg3.intercept_)  # 截距
    print(reg3.score(standard_X_test, y_test))
    print('use time:', (end_time - start_time).microseconds)
    print('------------------小批量随机梯度下降--------------------------')

    # 小批量随机梯度下降法
    start_time = datetime.datetime.now()
    reg4 = LinearRegression()
    reg4.fit_msgd(standard_X_train, y_train, batch_size=5)
    end_time = datetime.datetime.now()
    print(reg4.coef_)  # 斜率
    print(reg4.intercept_)  # 截距
    print(reg4.score(standard_X_test, y_test))
    print('use time:', (end_time - start_time).microseconds)


def test_PCA():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.components_)

    # 降维
    pca = PCA(n_components=1)
    pca.fit(X)
    X_reduction = pca.transform(X)
    print(X_reduction.shape)
    X_restore = pca.inverse_transform(X_reduction)
    print(X_restore.shape)

    plt.scatter(X[:, 0], X[:, 1], color='b')
    plt.scatter(X_restore[:, 0], X_restore[:, 1], color='r', alpha=0.5)
    plt.show()
    # restore并不能复原原数据，只是再次在高维下表示已降到低维的数据。


# test_KNN()
# test_Linear_Regression()

test_PCA()
