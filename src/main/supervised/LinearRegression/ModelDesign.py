# Working example for my blog post at:
# http://danijar.com/variable-sequence-lengths-in-tensorflow/
from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Parameters:
    weights: tf.Tensor
    bias: tf.Tensor

@dataclass(frozen=True)
class Input:
    input: object
    output: object

def inputData(TRUE_W: float, TRUE_b: float, NUM_EXAMPLES: int):
    inputs = tf.random_normal(shape=[NUM_EXAMPLES, 1])
    noise = tf.random_normal(shape=[NUM_EXAMPLES, 1])
    outputs = inputs * TRUE_W + TRUE_b + noise
    return Input(inputs, outputs)

def printInput(prediction, input: Input):
    plt.scatter(input.input, input.output, c='b')
    plt.scatter(input.input, prediction, c='r')
    plt.show()


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class LinearRegression:

    def __init__(self, data, target, epoche=100, learning_rate=0.1):
        self.data = data
        self.epoche = epoche
        self.learning_rate = learning_rate
        self.target=target
        self.weights=tf.Variable(100, name="weights", dtype=tf.float32)
        self.bias=tf.Variable(10, name="bias", dtype=tf.float32)
        self.i=tf.constant(0)
        self.prediction
        self.cost
        self.optimize

    @lazy_property
    def prediction(self):
        prediction = tf.add(tf.multiply(self.data, self.weights), self.bias)
        tf.print(prediction, [prediction])
        return prediction

    @lazy_property
    def cost(self):
        return tf.reduce_mean(tf.square(self.prediction - self.target))/2

    @lazy_property
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)


    @lazy_property
    def training(self):
        def condition(i): return i < 100
        def body(i):
            train_op=self.optimize
            # tf.print(i, [i])
            return tf.tuple([tf.add(i, 1)], control_inputs=[train_op])
        return tf.while_loop(condition, body, [self.i])



if __name__ == '__main__':
    TRUE_W = 2.0
    TRUE_b = 4.0
    NUM_EXAMPLES = 10000
    dataSet = inputData(TRUE_W, TRUE_b, NUM_EXAMPLES)

    data = tf.placeholder(tf.float32, [None,1])
    target = tf.placeholder(tf.float32, [None,1])
    model = LinearRegression(data, target)
    #################################################
    # from tensorflow.python.framework import ops
    # ops.reset_default_graph()
    # g = tf.get_default_graph()
    # print([op.name for op in g.get_operations()])
    #################################################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    input, output = sess.run([dataSet.input,dataSet.output])
    prediction = sess.run(model.prediction, {data: input})
    printInput(prediction, Input(input, output))
    # for epoch in range(1000):
    #     sess.run(model.optimize, {data: input, target: output})
    #     error = sess.run(model.cost, {data: input, target: output})
    #     datas= np.array([[20]])
    #     prediction = sess.run(model.prediction, {data:datas})
    #     #weights, bias=sess.run(model.parameters())
    #     print("huhu", sess.run([model.weights, model.bias]))
    #     print('Epoch {:2d} error {:f}'.format(epoch + 1,  error))
    #     print(prediction)
    # # sess.run(model.training(1000),{data: input, target: output})
    def condition(i): return i < 10000
    i = tf.constant(0)
    def body(i):
        sess.run(model.optimize, {data: input, target: output})
        tf.print(i, [i])
        return tf.tuple([tf.add(i, 1)])

    sess.run(tf.while_loop(condition, body, [i]), {data: input, target: output})
    datas = np.array([[20]])
    prediction = sess.run(model.prediction, {data:datas})
    print(prediction)
    prediction = sess.run(model.prediction, {data: input})
    printInput(prediction, Input(input, output))
    # print(sess.run(model.parameters().weights))