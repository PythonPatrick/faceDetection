import tensorflow as tf


class Lasso:
    def __init__(self, beta):
        self.beta=float(beta)

    def regularization(self, weights):
        return tf.multiply(self.beta,tf.reduce_mean(tf.abs(weights)))

class Ridge:
    def __init__(self, beta):
        self.beta=float(beta)

    def regularization(self, weights):
        return tf.multiply(self.beta,tf.reduce_mean(tf.square(weights)))


class ElasticNet:
    def __init__(self, alpha,beta):
        self.alpha=float(alpha)
        self.beta=float(beta)

    def regularization(self, weights):
        lasso=tf.multiply(self.alpha,tf.reduce_mean(tf.abs(weights)))
        ridge=tf.multiply(self.beta,tf.reduce_mean(tf.square(weights)))
        return tf.sum(lasso,ridge)