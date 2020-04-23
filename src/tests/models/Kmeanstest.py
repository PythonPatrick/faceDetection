import tensorflow as tf
from main.Clustering.Model.Kmean.kmean import Kmean as kmean
import numpy as np


class Kmeanstest(tf.compat.v1.test.TestCase):

    def setUp(self):
        self.samples = tf.constant(
            [[1., 20.], [3., 31.], [5., 27.], [7., 8.], [9., 10.], [11., 12.], [13., 1.], [15., 2.], [17., 3.]])
        self.kmean_example = kmean(data=self.samples, n_clusters=3)

    # def testNearestAssignation(self):
    #     with self.session():
    #         centroids = tf.constant([[0, 20], [9, 10], [20, 2]])
    #         x = self.kmean_example.assign_to_nearest_initial(centroids)
    #         self.assertAllEqual(x.eval(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
    #
    # def testMissingCluster(self):
    #     with self.session():
    #         indices = tf.constant([2, 2, 2, 2, 2, 2, 2, 2, 2], tf.int64)
    #         centroids = tf.Variable([[3., -26.], [9., 1.], [1., 2.]], tf.float64)
    #         x = self.kmean_example.missingCluster(indices, centroids)
    #         self.assertAllEqual(x.eval(), [0, 1])
    #
    # def testSelectClusterPoints(self):
    #     with self.session():
    #         indices = tf.constant([2, 2, 2, 2, 2, 2, 1, 1, 1], tf.int64)
    #         x = self.kmean_example.mayorCluster(indices)
    #         self.assertAllEqual(x.eval(), 2)
    #
    # def testClusterpoints(self):
    #     with self.session():
    #         indices = tf.constant([2, 2, 2, 2, 2, 2, 1, 1, 1], tf.int32)
    #         points = self.kmean_example.clusterPoints(tf.constant(2, tf.int32), indices)
    #         self.assertAllEqual(points.eval(), [0, 1, 2, 3, 4, 5])
    #
    # def testPointMaximalDistanceInsideCluster(self):
    #     with self.session():
    #         centroid = tf.constant([1., 2.], tf.float64)
    #         clusterIdx = tf.constant([0, 1, 2], tf.int64)
    #         index = self.kmean_example.pointMaximalDistanceInsideCluster(centroid, clusterIdx, False)
    #         self.assertAllEqual(index.eval(), 1)
    #         point = self.kmean_example.pointMaximalDistanceInsideCluster(centroid, clusterIdx, True)
    #         self.assertAllEqual(point.eval(), [3., 31.])
    #
    # def testSubstituteEmptyCluster(self):
    #     with self.session() as sess:
    #         clusters = tf.constant([2, 2, 2, 2, 2, 2, 1, 1, 1], tf.int32)
    #         centroids = tf.Variable([[3., -26.], [9., 1.], [1., 2.]])
    #         sess.run(tf.compat.v1.global_variables_initializer())
    #         clusters, centroids = self.kmean_example.substituteEmptyCluster(clusters, centroids)
    #         self.assertAllEqual(clusters.eval(), [2., 0., 2., 2., 2., 2., 1., 1., 1.])
    #         self.assertAllEqual(centroids.eval(), [[3., 31.], [9., 1.], [1., 2.]])

    def testKMeansIterations(self):
        with self.session() as sess:
            initial_centroids = tf.constant([[1., 10.], [1., 10.], [20., 2.]])
            self.samples = tf.constant(
                [[10., 20.], [10., 24.], [10., 28.], [1., 8.], [1., 10.], [1., 12.], [20., 1.], [20., 2.], [20., 3.]])
            self.kmean_example = kmean(data=self.samples, n_clusters=3, initialCentroids=initial_centroids)
            self.kmean_example.training2(sess)
            self.assertAllEqual(self.kmean_example.centroids.eval(), [[1., 10.], [10., 24.], [20., 2. ]])
            self.assertAllEqual(self.kmean_example.clusters.eval(), [1, 1, 1, 0, 0, 0, 2, 2, 2])


    def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
        """
        Step 1: Initialisation of a sample
        """
        np.random.seed(seed)
        slices = []
        centroids = []
        # Create samples for each cluster
        for i in range(n_clusters):
            samples = tf.random_normal((n_samples_per_cluster, n_features), mean=0.0, stddev=5.0, dtype=tf.float32,
                                       seed=seed, name="cluster_{}".format(i))
            current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor / 2)
            centroids.append(current_centroid)
            samples += current_centroid
            slices.append(samples)
        # Create a big "samples" dataset
        samples = tf.concat(slices, 0, name='samples')
        centroids = tf.concat(centroids, 0, name='centroids')
        return centroids, samples


if __name__ == '__main__':
    tf.test.main()
