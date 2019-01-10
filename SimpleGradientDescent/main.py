import sys
import numpy as np

def generate_train_data():
    w = 100.3555
    b = 30444.245522
    c = [100, 260, 329, 322, 400, 98, 288, 111, 250]
    d = []
    for n in c:
        x = n
        y = w * x + b
        d.append((x, y))
    return d

def gradient_function_b(data, w, b):
    gradient = 0.
    m = len(data)
    for d in data:
        x = d[0]
        y = d[1]
        _y = w * x + b
        diff = _y - y
        gradient = gradient + diff
    return (1./ m) * gradient

def gradient_function_w(data, w, b):
    gradient = 0.
    m = len(data)
    for d in data:
        x = d[0]
        y = d[1]
        _y = w * x + b
        diff = (_y - y) * x
        gradient = gradient + diff
    return (1./m) * gradient

def train(data, epoch=30):
    w = 0.0
    b = 19.3333
    a = 1e-5
    m = len(data)
    i = 0
    dw = gradient_function_w(data, w, b)
    db = gradient_function_b(data, w, b)
    print("g", dw, db)
    # while not abs(graident_w) >= 1e-2 or abs(graident_b) >= 1e-2 or i < 30:
    while i < epoch and (abs(dw) >= 1e-3 or abs(db) >= 1e-3):
        print(i)
        dw = gradient_function_w(data, w, b)
        db = gradient_function_b(data, w, b)
        print("g", dw, db)
        w = w - a * dw
        b = b - a * db
        print("a", w, b)
        i = i + 1
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
