import tensorflow as tf
import numpy as np
from src.main.inputs.inputdata import regression_data
from src.main.models.supervised.model import Model
from src.main.utils.decorators import lazy_property

class LinearRegression(Model):

    def __init__(self, data_dimension, data, epoche=100, learning_rate=0.1):
        Model.__init__(self,  data_dimension, data, epoche, learning_rate)

    @lazy_property
    def prediction(self):
        return tf.add(tf.matmul(self.train_data, self.weights), self.bias)

    @lazy_property
    def cost(self):
        return tf.reduce_mean(tf.square(self.prediction - self.train_target))/2

    @lazy_property
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)


if __name__ == '__main__':
    input = regression_data(TRUE_W=[[17.0], [4]],
                              TRUE_b=4,
                              NUM_EXAMPLES=10000000)
    model = LinearRegression(data=input, data_dimension=2, epoche=300)
    model.training
    print(model.sess.run([model.weights, model.bias]))