import tensorflow as tf
from tensorflow import Tensor


class LinearKernel:

    def __init__(self):
        """Linear Kernel

        """

    def kernel(self,X: Tensor):
        return tf.matmul(X, tf.transpose(X))

class GaussianKernel:

    def __init__(self, gamma: float):
        """Gaussian Kernel

        """
        self.gamma= tf.constant(gamma)

    def kernel(self, X: Tensor):
        rA = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1])
        rB=tf.subtract(rA, tf.multiply(2., tf.matmul(X, tf.transpose(X))))
        pred_sq_dist = tf.add(rB, tf.transpose(rA))
        return tf.exp(tf.multiply(self.gamma, tf.abs(pred_sq_dist)))


