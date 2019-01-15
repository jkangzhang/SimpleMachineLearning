import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def costFunction(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    serror = (predictions - y) ** 2
    return 1.0 / (2 * m) * np.sum(serror)

def gradientDescent(X, y, theta, alpha=0.01, epoch=1500):
    m = len(y)
    js = []
    for i in range(epoch):
        predictions = X.dot(theta)
        # sum of diff
        derivative = np.dot(X.transpose(), predictions - y)
        theta -= alpha * 1.0 / m * derivative
        js.append(costFunction(X, y, theta))
    return theta, js

def normalEquation(X, y):
    A = np.dot(X.T, X) B = np.linalg.inv(A) C = np.dot(B, X.T) D = np.dot(C, y) return D def train(): # read data data = pd.read_csv("ex1data1.txt", header=None) # show image # plt.scatter(data[0], data[1]) # plt.xticks(np.arange(5, 30, step=5))
    # plt.yticks(np.arange(-5, 30, step=5))
    # plt.show()
    # compute cost function
    theta = np.zeros((2, 1))
    data_n = data.values
    m = len(data_n)
    # print(data_n[:,0].reshape(m, 1))
    X = np.append(np.ones((m, 1)), data_n[:,0].reshape(m, 1), axis=1)
    # convert to a vector
    y = data_n[:, 1].reshape(m, 1)
    # calculate cost
    v = costFunction(X, y, theta)
    # train()
    theta, js = gradientDescent(X, y, theta, 0.01, 1500)
    print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

    # Use normalEquation get better result
    theta1 = normalEquation(X, y)
    v2 = costFunction(X, y, theta1)
    v1 = costFunction(X, y, theta)
    print(v1, v2)

    # draw cost function descent
    # plt.plot(js)
    # plt.xlabel("Iteration")
    # plt.ylabel("J-theta")
    # plt.title("Cost function value")
    # plt.show()

    # draw line on it
    plt.scatter(data[0], data[1])
    xvalues = [x for x in range(25)]
    yvalues = [v*theta[1] + theta[0] for v in xvalues]
    plt.plot(xvalues, yvalues, color='r')
    plt.xticks(np.arange(5, 30, step=5))
    plt.yticks(np.arange(-5, 30, step=5))
    # plt.show()

if __name__ == '__main__':
    train()
