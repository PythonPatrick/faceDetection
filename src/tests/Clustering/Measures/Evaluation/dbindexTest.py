import tensorflow as tf
from main.Clustering.Measures.Evaluation.dbindex import DBIndex as dbIndex


# class DBIndexTest(tf.test.TestCase):
#
#     def compute_S(self):
#         with self.Session():
#             i = 1
#             sample = tf.constant([[1, 2], [1, 5], [5, 10], [6, 12], [11, 21], [13, 25]], tf.float64)
#             indices = tf.constant([0, 0, 1, 1, 2, 2])
#             partition = tf.dynamic_partition(sample, indices)
#             clusters = tf.constant([[1, 3.5], [5.5, 11], [12, 23]])
#             distance =
#             print(dbIndex.compute_S(i, clusters, partition, distance).eval())