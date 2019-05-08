from __future__ import print_function
import tensorflow as tf
from src.main.inputs.inputdata import classification_data
from src.main.models.supervised.model import Model
from src.main.utils.decorators import lazy_property


class LogisticRegression(Model):

    def __init__(self, data_dimension, data, epoche=100, learning_rate=0.1):
        Model.__init__(self,  data_dimension, data, epoche, learning_rate)

    @lazy_property
    def prediction(self):
        return tf.nn.sigmoid(tf.add(tf.matmul(self.train_data, self.weights), self.bias))

    @lazy_property
    def cost(self):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits = self.prediction, labels = self.train_target)

    @lazy_property
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return tf.math.reduce_sum(self.cost)

    @lazy_property
    def accuracy(self):
        correct_prediction = tf.equal(tf.round(self.prediction),self.train_target)
        accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        input, output = self.sess.run([self.data.x, self.data.y])
        return (self.sess.run(accuracy,  {self.train_data: input, self.train_target: output}))

if __name__ == '__main__':
    NUM_EXAMPLES = 100000
    features=10
    dataSet = classification_data(n_samples=NUM_EXAMPLES, n_features=features)

    model = LogisticRegression(data_dimension=features, data=dataSet,  epoche=500)
    model.training
    print(model.sess.run([model.weights, model.bias]))
    print(model.accuracy)