import tensorflow as tf
from main.Clustering.Measures.measures import Measures as measures


class Measurestest(tf.test.TestCase):

    def testVariance(self):
        with self.session():
            x = tf.constant([1., 1., 2., 2., 2., 2., 2., 2.])
            variance = measures.variance(x)
            self.assertAllEqual(variance.eval(), 0.1875)

    def testHarmonicMean(self):
        with self.session():
            x = tf.constant([1., 2., 4., 4.])
            hM = measures.harmonicMean(x)
            self.assertAllEqual(hM.eval(), 2)

    def testSize(self):
        with self.session():
            x = tf.constant([[1., 2., 4., 4.], [1., 2., 4., 4.], [8., 7., 4., 4.]])
            size = measures.size(x)
            self.assertAllEqual(size.eval(), 3)

    # def testIntersection(self):
    #     with self.session():
    #         a = tf.constant([[1., 2., 4., 4.], [1., 2., 4., 4.], [8., 7., 4., 4.]], tf.int64)
    #         b = tf.constant([[1., 2., 4., 4.], [2., 2., 4., 4.], [8., 7., 4., 9.]])
    #         intersection = measures.intersection(a, b)
    #         print(intersection.eval())
    #         self.assertAllEqual(intersection.eval(), [1., 2., 4., 4.])

    def testFactorial(self):
        with self.session():
            self.assertAllEqual(measures.factorial(tf.constant(1, tf.float64)).eval(), 1)
            self.assertAllEqual(measures.factorial(tf.constant(2, tf.float64)).eval(), 2)
            self.assertAllEqual(measures.factorial(tf.constant(3, tf.float64)).eval(), 6)
            self.assertAllEqual(measures.factorial(tf.constant(10, tf.float64)).eval(), 3628800.0000000023)

    def testBinomialCoef(self):
        with self.session():
            n = tf.constant(3, tf.float64)
            k = tf.constant(1, tf.float64)
            self.assertAllEqual(measures.binomialCoefficient(n=n, k=k).eval(), 3)

    def testMean(self):
        with self.session():
            x = tf.constant([[1, 1], [9, 9]], tf.float64)
            print(measures.mean(x).eval())

    def testClusterDiameter(self):
        with self.session():
            x = tf.constant([[2, 2], [5, 5]], tf.float64)
            p = tf.constant(2, tf.float64)
            print(measures.clusterDiameter(x, p).eval())



if __name__ == '__main__':
    tf.test.main()