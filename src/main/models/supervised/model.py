from __future__ import print_function
import tensorflow as tf
import numpy as np
from src.main.utils.decorators import lazy_property


class Model:

    def __init__(self, data_dimension, data, epoche=100, learning_rate=0.1):
        self.sess=tf.Session()
        self.train_data = tf.placeholder(tf.float32, [None,data_dimension])
        self.train_target= tf.placeholder(tf.float32, [None,1])
        self.epoche = epoche
        self.data=data
        self.learning_rate = learning_rate
        self.weights=tf.Variable(np.random.rand(data_dimension, 1), name="weights", dtype=tf.float32)
        self.bias=tf.Variable(10, name="bias", dtype=tf.float32)
        self.prediction
        self.cost
        self.optimize

    @lazy_property
    def prediction(self):
        pass

    @lazy_property
    def cost(self):
        pass

    @lazy_property
    def optimize(self):
        pass

    @lazy_property
    def training(self):
        self.sess.run(tf.global_variables_initializer())
        input, output = self.sess.run([self.data.x,self.data.y])
        for epoch in range(self.epoche):
            self.sess.run(self.optimize, {self.train_data: input, self.train_target: output})
            error = self.sess.run(self.cost, {self.train_data: input, self.train_target: output})
            print('Epoch {:2d} error {:f}'.format(epoch + 1, error))