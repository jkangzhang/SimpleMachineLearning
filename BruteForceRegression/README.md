# Straight-forward Understanding and Simple Practice in Machine Learning

This blog is inspired by Andrew Wu's Machine Learning course, which aims to give you a most straightfoward concept of Machine Learning. At first it is comeing out with a problem whhich could be solved by using machine learning ideas, then I will elaborate what machine learning is really doing and the definination of Linear Regression base on the example; next we will focus on implementing this simple machine learning algorithm with the very basic library of Python; At last I will talk about the precision and time complexity of this algorithm.

### Solving a problem using machine learning ideas

Imagine that a friend of you is going to sell his house, and he wants you to help him estimate how much money he would charge.

Firstly I collect some data, which each item comprises the size and price of the house. For example, a house with a size of 100m<sup>2</sup> worths 570,000$. (We are just simplifying the problem into smallest one which means the size of a house is the unique factor to the price of it.)

Let's draw the data in a coordinate axis, the x-coordinate denotes the size of houses and y-coordinate denotes the estimate price of the houses. The plot will look like below:

![basic](https://raw.githubusercontent.com/jkangzhang/SimpleMachineLearning/master/BruteForceRegression/images/points.jpg)

Here we keeping going on simplifying the problem, we may consider this is a linear relationship between the size and the price of a house. And what we need to do is unveiling the releationship of these two factors base on the data. More straight-forwardly, we need to find a line which goes through the most points on the plot, as you see, the red color line is the best one among all of the lines.

![lines](https://raw.githubusercontent.com/jkangzhang/SimpleMachineLearning/master/BruteForceRegression/images/lines.jpg)

If I have got this line in some way, I could estimate the price of a house if the size of the house is given. Thus we need to conceive an algorithm to let machine find this line itself.

### Linear Regression

In conclusion, the nature of machine learning is finding a function through an algorithm  which conceived base on some data. This function can use given input, to predict output. Here is a generally conception graph of machine learning. 

![graph](https://raw.githubusercontent.com/jkangzhang/SimpleMachineLearning/master/BruteForceRegression/images/algorithm.jpg)

Apparently, the red line we mentioned above is the function the machine need to find out. The function can denote in this mathematic way:

h(x) = wx + b

This is just a general function of a line, and we define **w** and **b** are the parameters of this function.

It is easy to figure out the essential part of this problem is to get the two parameters **w** and **b**, thus we find need to search a pair of w and b, such that each items of the training data has good minimal difference between h(x) and y. As more generally, such w and b minimize the sum of the whole training data's difference. Here's the mathematic denotion:

$\sum_1^m\(h(x^i)-y^i)^2

For this kind of problem, as it is a linear relationship between input and output, we call it a **Linear Regression Problem** in machine learning, and for this simple one, there is only one variable input, we call it a Unique Linear Regression Problem.

### Implementing the algorithm in brute-force way

Here we want to implement the algorithm in the algorithm, using a most straight-forward way, method of exhaustion.

Firstly we compose a function to generate some training data.

```python
def generate_train_data():
    w = 1.239
    b = 20.223
    c = [100, 260, 310, 380, 89, 145]
    d = []
    for n in c:
        x = n
        y = w * x + b
        d.append((x, y))
    return d
```

As you see, I defined two variables w and b, which represent the underlying regulation of the training data.

Secondly, we come into the training part. In this function, I choose an originial value and enumerate one by one in a stable step. And we use the first m - 1 piece of data for training, leave the last for testifying.

```python
def train(data):
    min_dif = sys.maxsize
    w = 0.0
    b = 0.0
    for i in np.arange(0.0, 2.0, 0.001):
        for j in np.arange(0.0, 30.0, 0.001):
            cur_sum = 0
            for d in data:
                x = d[0]
                y = d[1]
                _y = i * x + j
                dif = y - _y
                # 对应公式 y = 
                cur_sum = cur_sum + (dif * dif)
            if cur_sum < min_dif:
                min_dif = cur_sum
                w = i
                b = j
                print(cur_sum)
    return w, b
```

Here we got the parameter w and b.

Then we test them by another function.


```python
def predict(w, b, x):
    return w * x + b
```

### More

Let's finish our last work, just consider what is the time complexity of this algorithm.

Apparently, the time the machine need to compute depends on the step of the iteration, as it is also the precision of this algorithm. For a 32-bit machine in the biggest precision, the time complexity is 2<sup>32</sup> * 2<sup>32</sup> * m

m denotes the size of the training data.