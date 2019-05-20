import tensorflow as tf
from src.main.model.model import Model, Config, Parameters
from src.main.utils.decorators import lazy_property

class SVM(Model):

    def __init__(self, dataset, config: Config, parameters: Parameters):
        Model.__init__(self, dataset=dataset, config=config, parameters=parameters)

    @lazy_property
    def prediction(self):
        return tf.subtract(tf.matmul(self.dataset.features_batch, self.weights), self.bias)

    @lazy_property
    def cost(self):
        l2_norm = tf.reduce_sum(tf.square(self.weights))
        alpha = tf.constant([0.1])
        classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(self.prediction, self.dataset.target_batch))))
        return tf.add(classification_term, tf.multiply(alpha, l2_norm))

    @lazy_property
    def optimization(self):
        optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost