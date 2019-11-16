import tensorflow as tf
import tensorflow_probability as tfp


class Measures:

    @staticmethod
    def variance(x):
        """ Compute the variance of the values in last dimension of x

        :param x: Tensor or SparseTensor.
        :return:  Integer value.
        """
        return tf.math.reduce_variance(x)

    @staticmethod
    def harmonicMean(x):
        """ Compute the harmonic mean of the values in last dimension of x

        :param x: Tensor or SparseTensor.
        :return:  Integer value.
        """
        return 1. / tf.reduce_mean(1 / x)

    @staticmethod
    def size(x):
        """ Compute number of elements along last dimension of x.

        :param x: Tensor or SparseTensor.
        :return:  Integer value.
        """
        return tf.shape(x)[0]

    @staticmethod
    def intersection(a, b):
        """Compute set intersection of elements in last dimension of a and b.

        :param a: Tensor or SparseTensor of the same type as b. If sparse, indices must be sorted in row-major order.
        :param b: Tensor or SparseTensor of the same type as a. If sparse, indices must be sorted in row-major order.
        :return: A SparseTensor whose shape is the same rank as a and b, and all but the last dimension the same.
                 Elements along the last dimension contain the intersections.
        """
        return tf.sets.intersection(a, b)

    @staticmethod
    def union(a, b):
        """Compute set intersection of elements in last dimension of a and b.

        :param a: Tensor or SparseTensor of the same type as b. If sparse, indices must be sorted in row-major order.
        :param b: Tensor or SparseTensor of the same type as a. If sparse, indices must be sorted in row-major order.
        :return: A SparseTensor whose shape is the same rank as a and b, and all but the last dimension the same.
                 Elements along the last dimension contain the unions.
        """
        return tf.sets.union(a, b)

    def precision(self, a, b):
        """ Compute the precision of the set a (subset of C), where b (subset of C) is the "true" set.
            The precision then is |a intersection b|/|a|.

        :param a: Tensor or SparseTensor of the same type as b. If sparse, indices must be sorted in row-major order.
        :param b: Tensor or SparseTensor of the same type as a. If sparse, indices must be sorted in row-major order.
        :return: Integer value.
        """
        return self.size(self.intersection(a, b))/self.size(a)

    def recall(self, a, b):
        """ Compute the recall of the set a (subset of C), where b (subset of C) is the "true" set.
            The recall then is |a intersection b|/|b|.

        :param a: Tensor or SparseTensor of the same type as b. If sparse, indices must be sorted in row-major order.
        :param b: Tensor or SparseTensor of the same type as a. If sparse, indices must be sorted in row-major order.
        :return: Integer value.
        """
        return self.size(self.intersection(a, b))/self.size(b)

    @staticmethod
    def factorial(n):
        """ Compute the factorial n!

        :return:
        """
        return tf.exp(tf.math.lgamma(n + 1))

    @staticmethod
    def binomialCoefficient(n, k):
        """ Compute the binomial coefficient

        :return:
        """
        return Measures.factorial(n)/tf.multiply(Measures.factorial(k), Measures.factorial(n-k))

    @staticmethod
    def mean(x):
        """Compute the mean vector of a list of vectors in last dimension of x
           mean vector= (x_1 + ... + x_n)/n

        :param x: Tensor or SparseTensor
        :return: Tensor or SparseTensor
        """
        return tf.reduce_sum(x, 0)/tf.cast(Measures.size(x), tf.float64)

    @staticmethod
    def clusterDiameter(x, p):
        """

        :param x:
        :param p:
        :return:
        """
        n = tf.cast(Measures.size(x), tf.float64)
        summation = tf.reduce_sum(tf.pow(tf.math.abs(tf.subtract(x, Measures.mean(x))), p), 1)
        # return tf.pow(tf.reduce_sum(summation)/n, 1/p)
        return tf.reduce_sum(tf.pow(summation, 1/p))/n



