from __future__ import print_function
import tensorflow as tf
from src.main.inputs.inputdata import classification_data
from src.main.models.supervised.model import Model
from src.main.utils.decorators import lazy_property


class KNN(Model):

    def __init__(self, data_dimension, data, epoche=100, learning_rate=0.1):
        Model.__init__(self,  data_dimension, data, epoche, learning_rate)

    @lazy_property
    def prediction(self):
        # manhattan distance
        distance = tf.reduce_sum(tf.abs(tf.subtract(self.train_data, tf.expand_dims(x_data_test, 1))), axis=2)

        # nearest k points
        _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
        top_k_label = tf.gather(self.train_target, top_k_indices)

        sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
        return  tf.argmax(sum_up_predictions, axis=1)

    @lazy_property
    def cost(self):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits = self.prediction, labels = self.train_target)

    @lazy_property
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)


if __name__ == '__main__':
    NUM_EXAMPLES = 1000
    features=4
    dataSet = classification_data(n_samples=NUM_EXAMPLES, n_features=features)

    model = KNN(data_dimension=features, data=dataSet,  epoche=300)
    model.training
    print(model.sess.run([model.weights, model.bias]))