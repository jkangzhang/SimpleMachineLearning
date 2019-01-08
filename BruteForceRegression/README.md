# 最简单的机器学习入门

这篇文章是受到 Andrew Wu 的机器学习课程启发，基于这个课程的一个实践，目的说明机器学习到底在干什么，以及机器学习算法的最基本的原理，并使用最原始的方法实现一个机器学习的算法。

### 通过机器学习解决一个问题

假设我想预测一个地区的某种面积大小房子对应的房价。  
那么我搜集了一拼数据，数据由房子大小和评价构成。如100平方米的房子，售价是57万；300平方米的房子售价130万。
这样的数据用一个坐标轴进行表示，横轴为面积大小，纵轴为房子的售价。将所有的数据绘制到坐标轴上，如图所示。

![basic](https://raw.githubusercontent.com/jkangzhang/SimpleMachineLearning/master/BruteForceRegression/images/points.jpg)

不妨假设，面积大小与房价存在某种线性关系。那么我们要做的就是通过现有数据找到这样的线性关系，即直观上说是，找到一条直线尽可能通过最多的点。如图所示，红色即为我们想找到的直线。

![lines](https://raw.githubusercontent.com/jkangzhang/SimpleMachineLearning/master/BruteForceRegression/images/lines.jpg)

如果有了这样的一条直线，那么我有了任意一个面积数据就可以计算出它对应的房价了。我们要设计的机器学习算法，就是让机器自己找到这样的一条直线。

### 线性回归的原理

所以机器学习的本质就是根据一些训练数据，通过一个学习算法，让机器学得某个函数。这个函数根据给定输入（房子面积），可以给出输出（房价）。

![graph](https://raw.githubusercontent.com/jkangzhang/SimpleMachineLearning/master/BruteForceRegression/images/algorithm.jpg)


明显地，上面的直线就是要求得函数，这个函数可以表示为：

h(x) = wx + b

w, b 称为这个函数的参数。 

那么问题从求h(x)这个函数转换成求它的两个参数 **w** 和 **b**。  
即找出这样的 w 和 b 让每一个训练数据中的每一个 h(x) 和 y 尽可能的接近。  
对于整体样本(m个）来说，即让所有样本的差异综合最小。

\sum_1^m\(h(x^i)-y^i)^2

这样的问题，由于问题的是线性的，在机器学习中成为**线性回归**问题，且这个问题中只有一个元素(x)，所以这个问题成为**一元线性回归**问题。


### 用最简单的方式实现机器学习算法

下面就就以上面的问题举例，用最简单的方式实现找出 w 和 b，核心思想就是穷举法。  

首先，先写一个函数，模拟这些数据。

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

上面的我在造数据的过程中，我定义了一组 w 和 b，代表这组数据的潜在规律，而下面的训练的算法，就是为了找出这组 w 和 b。

训练算法中，我们选择一个起始值，以固定的步长，进行穷举，输入数据的前 m - 1 个数据，将最后一个数据留下验证学习效果。

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

这样我们就得到了 w 和 b。

然后我们通过 predict 函数验证学习效果。

```python
def predict(w, b, x):
    return w * x + b
```

### 想更多些


那么上面的线性回归算法的时间复杂度是多少呢？  
很明显，是根据你的循环的随着循环的步长指数型增加，而这步长就代表了学习算法的精度，步长设置的越小，精度则越高，时间复杂度就越高，按一台32位的机器的最大精度来算，穷举所有数字的时间复杂度为：

2<sup>32</sup> * 2<sup>32</sup> * m

m 为训练数据的大小。


