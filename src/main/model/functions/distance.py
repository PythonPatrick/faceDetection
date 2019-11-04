import tensorflow as tf
from tensorflow import Tensor

class L0Norm:

    def __init__(self):
        """L0 norm distance

        """

    def distance(self,X:Tensor, Y:Tensor):
        return tf.count_nonzero(tf.subtract(X,Y), axis=[1])

class Manhatten:

    def __init__(self):
        """Manhatten distance

        """

    def distance(self, X: Tensor, Y: Tensor):
        return tf.reduce_sum(tf.abs(tf.subtract(X, tf.expand_dims(Y, 1))), axis=2)

class L2Norm:

    def __init__(self):
        """L2 Norm distance

        """

    def distance(self, X: Tensor, Y: Tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(X, tf.expand_dims(Y, 1))), reduction_indices = 1))


