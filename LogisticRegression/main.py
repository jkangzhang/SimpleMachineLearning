import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmod(z):
    return 1.0 / (1 + np.exp(-z))

def normalization(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    X_norm = (X - mean) / std
    return X_norm

def cost_function(X, y, theta):
    preds = sigmod(X.dot(theta))
    m = len(y)
    error = (-np.log(preds).transpose().dot(y) - np.log(1 - preds).transpose().dot(1 - y))
    return 1.0 / m * np.sum(error)

def gradient_descent(X, y, theta, alpha, epoch=1500):
    m = len(y)
    js = []
    for i in range(epoch):
        pred = sigmod(X.dot(theta))
        dd = np.dot(X.transpose(), pred - y)
        theta -= alpha * 1.0 / m * dd
        js.append(cost_function(X, y, theta))
    return theta, js

def train(file):
    data = pd.read_csv(file, header=None)
    theta = np.zeros((3, 1))
    data_n = data.values
    m = len(data_n)
    print(sigmod(0))
    X = data_n[:, 0:2].reshape(m, 2)
    X = normalization(X)
    X = np.append(np.ones((m, 1)), X, axis = 1)
    y = data_n[:, -1].reshape(m, 1)
    c = cost_function(X, y, theta)
    theta, js = gradient_descent(X, y, theta, 1, 400)
    print(theta)
    plt.plot(js)
    plt.xlabel("iterations")
    plt.ylabel("j-theta")
    plt.title("cost function descent")
    plt.show()

if __name__ == '__main__':
    import sys
    train(sys.argv[1])
