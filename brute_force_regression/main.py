import sys
import numpy as np

def generate_train_data():
    w = 1.23
    b = 20.23
    c = [100, 260, 310, 380, 89, 145]
    d = []
    for n in c:
        x = n
        y = w * x + b
        d.append((x, y))
    return d

def train(data):
    min_dif = sys.maxsize
    w = 0.0
    b = 0.0
    for i in np.arange(0.0, 2.0, 0.01):
        for j in np.arange(0.0, 30.0, 0.01):
            cur_sum = 0
            for d in data:
                x = d[0]
                y = d[1]
                _y = i * x + j
                dif = y - _y
                cur_sum = cur_sum + (dif * dif)
            if cur_sum < min_dif:
                min_dif = cur_sum
                w = i
                b = j
    return w, b

def predict(w, b, x):
    return w * x + b

if __name__ == '__main__':
    data = generate_train_data()
    print(data)
    w, b = train(data[:len(data) - 1])
    print('-----')
    print(w)
    print(b)
    print('----')
    y = predict(w, b, data[len(data) - 1][0])
    print("we predict:" + str(y) + " actually:" + str(data[len(data) - 1][1]))
