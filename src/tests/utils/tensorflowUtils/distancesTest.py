import tensorflow as tf
from main.utils.tensorflowUtils.distances import distances
import math


class distancesTest(tf.compat.v1.test.TestCase):

    def testPointToMatrix(self):
        with self.session():
            point = tf.constant([1., 1.], tf.float64)
            matrix = tf.constant([[2., 2.], [3., 3.], [4., 4.]], tf.float64)
            dist = distances.pointToMatrix(point, matrix)
            self.assertAllEqual(dist.eval(), [math.sqrt(2), 2, math.sqrt(18)])


if __name__ == '__main__':
    tf.test.main()
