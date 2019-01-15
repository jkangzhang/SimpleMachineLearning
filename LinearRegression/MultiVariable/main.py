import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cost_function(X, y, theta):
    m = len(y)
    pred = X.dot(theta)
    error = (pred - y) ** 2
    return 1.0 / (2 * m) * np.sum(error)

def gradient_descent(X, y, theta, alpha, epoch=1500):
    m = len(y)
    js = []
    for i in range(epoch):
        pred = X.dot(theta)
        dd = np.dot(X.transpose(), pred - y)
        theta -= alpha * 1.0 / m * dd
        js.append(cost_function(X, y, theta))
    return theta, js

def feature_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm

def train(data_file):
    data = pd.read_csv(data_file, header=None)
    # print(data.head())
    # print(data.describe())
    data_n = data.values
    m = len(data_n[:, -1])
    X = data_n[:, 0:2].reshape(m, 2)
    X = feature_normalize(X)
    X = np.append(np.ones((m, 1)), X, axis=1)
    y = data_n[:, -1].reshape(m, 1)
    theta = np.zeros((3, 1))
    c = cost_function(X, y, theta)
    theta, js = gradient_descent(X, y, theta, 0.01, 400)
    print(theta)
    plt.plot(js)
    plt.xlabel("iterations")
    plt.ylabel("j-theta")
    plt.title("cost function descent")
    # plt.show()

if __name__ == '__main__':
    import sys
    train(sys.argv[1])
    # predict()

