import matplotlib.pyplot as plt
import numpy as np
import random

from keras.datasets import cifar10
# 进行knn计算
from sklearn.neighbors import KNeighborsClassifier

def dataLoad():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3)
    return x_train, y_train, x_test, y_test

def test(x_test, y_test, n, len):
    real = []
    pre = []
    accuracy = 0
    for i in range(n):
        index = random.randint(0, len)
        p = model.predict(x_test[i:i + 1])
        r = y_test[i]
        pre.append(p)
        real.append(y_test[i])
        print(p, r)
        if (p == r): accuracy += 1
    accuracy /= n
    return accuracy, pre, real

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = dataLoad()
    k = [1, 3, 5]
    model = KNeighborsClassifier(n_neighbors=k[0], algorithm='ball_tree', n_jobs=6)
    model.fit(x_train, y_train.ravel())
    accuracy, pre, real = test(x_test, y_test, 20, len(y_test))

    print(accuracy)