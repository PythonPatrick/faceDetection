import tensorflow as tf
from tensorflow import Tensor


class Manhatten:

    def __init__(self):
        """Manhatten distance

        """

    def distance(self,X:Tensor, Y:Tensor):
        return tf.reduce_sum(tf.abs(tf.subtract(X, tf.expand_dims(Y, 1))), axis=2)

