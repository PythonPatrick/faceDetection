import tensorflow as tf


class PearsonCorrelationCoefficient(object):

    def __init__(self, sess=tf.Session()):
        """
        Compute the pairwise Pearson Correlation Coefficient (https://bit.ly/2ipHb9y)
        using TensorFlow (http://www.tensorflow.org) framework.
        :param sess a Tensorflow session
        :usage
        >>> import numpy as np
        >>> x = np.array([[1,2,3,4,5,6], [5,6,7,8,9,9]]).T
        >>> pcc = PearsonCorrelationCoefficient()
        >>> pcc.compute_score(x)
        """

        self.x_ph = tf.placeholder(tf.float32, shape=(None, None))

        x_mean, x_var = tf.nn.moments(self.x_ph, axes=0)

        x_op = self.x_ph - x_mean

        self.w_a = tf.placeholder(tf.int32)

        self.h_b = tf.placeholder(tf.int32)

        self.w_b = tf.placeholder(tf.int32)

        x_sd = tf.sqrt(x_var)

        self.x_sds = tf.reshape(tf.einsum('i,k->ik', x_sd, x_sd), shape=(-1,))

        c = tf.einsum('ij,ik->ikj', x_op, x_op)

        c = tf.reshape(c, shape=tf.stack([self.h_b, self.w_a * self.w_b]))

        self.op = tf.reshape(tf.reduce_mean(c, axis=0) / self.x_sds, shape=tf.stack([self.w_a, self.w_b]))

        self.sess = sess

    def compute_score(self, x):
        """
        Compute the Pearson Correlation Coefficient of the x matrix. It is equivalent to `numpy.corrcoef(x.T)`
        :param x: a numpy matrix containing a variable per column
        :return: Pairwise Pearson Correlation Coefficient of the x matrix.
        """

        if len(x.shape) == 1:
            x = x.reshape((-1, 1))

        assert len(x.shape) == 2 and x.shape[1] > 0

        self.sess.run(tf.global_variables_initializer())

        return self.sess.run(self.op, feed_dict={self.x_ph: x, self.h_b: x.shape[0],
                                                 self.w_a: x.shape[1], self.w_b: x.shape[1]})


import numpy as np


x = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 9],[5, 6, 7, 8, 9, 9]]).T
print(x)
pcc = PearsonCorrelationCoefficient()
print(pcc.compute_score(x))