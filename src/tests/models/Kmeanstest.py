import tensorflow as tf
from src.main.model.clustering.Kmean.kmean import Kmean as kmean
import numpy as np


class Kmeanstest(tf.compat.v1.test.TestCase):

    def setUp(self):
        self.samples = tf.constant(
            [[1., 20.], [3., 31.], [5., 27.], [7., 8.], [9., 10.], [11., 12.], [13., 1.], [15., 2.], [17., 3.]])
        self.kmean_example = kmean(data=self.samples, n_clusters=3)

    def testNearestAssignation(self):
        with self.session():
            centroids = tf.constant([[0, 20], [9, 10], [20, 2]])
            x = self.kmean_example.assign_to_nearest(centroids)
            self.assertAllEqual(x.eval(), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def testMissingCluster(self):
        with self.session():
            indices = tf.constant([2, 2, 2, 2, 2, 2, 2, 2, 2], tf.int64)
            centroids = tf.Variable([[3., -26.], [9., 1.], [1., 2.]], tf.float64)
            x = self.kmean_example.missingCluster(indices, centroids)
            self.assertAllEqual(x.eval(), [0, 1])

    def testSelectClusterPoints(self):
        with self.session():
            indices = tf.constant([2, 2, 2, 2, 2, 2, 1, 1, 1], tf.int64)
            centroids = tf.Variable([[3., -26.], [9., 1.], [1., 2.]], tf.float64)
            x = self.kmean_example.mayorCluster(indices)
            print(x.eval())
            self.assertAllEqual(x.eval(), 2)

    def testClusterpoints(self):
        with self.session():
            indices = tf.constant([2, 2, 2, 2, 2, 2, 1, 1, 1], tf.int32)
            points = self.kmean_example.clusterPoints(tf.constant(2, tf.int32), indices)
            self.assertAllEqual(points.eval(), [0, 1, 2, 3, 4, 5])

    def testPointMaximalDistanceInsideCluster(self):
        with self.session():
            centroid = tf.constant([1., 2.], tf.float64)
            clusterIdx = tf.constant([0, 1, 2], tf.int64)
            index = self.kmean_example.pointMaximalDistanceInsideCluster(centroid, clusterIdx, False)
            self.assertAllEqual(index.eval(), 1)
            point = self.kmean_example.pointMaximalDistanceInsideCluster(centroid, clusterIdx, True)
            self.assertAllEqual(point.eval(), [3., 31.])

    def testSubstituteEmptyCluster(self):
        with self.session():
            clusters = tf.constant([2, 2, 2, 2, 2, 2, 1, 1, 1], tf.int32)
            centroids = tf.constant([[3., -26.], [9., 1.], [1., 2.]])
            clusters, centroids, z = self.kmean_example.substituteEmptyCluster(clusters, centroids)
            print(clusters.eval(), centroids.eval(), z.eval())





    # def testNearestAssignation2(self):
    #     with self.session():
    #         samples = tf.constant(
    #             [[1., 20.], [3., 31.], [5., 27.], [7., 8.], [9., 10.], [11., 12.], [13., 1.], [15., 2.], [17., 3.]])
    #         kmean_example = kmean(data=samples, n_clusters=3)
    #         centroids = tf.Variable([[3., -26.], [9., 1.], [1., 2.]])
    #         x = kmean_example.assign_to_nearest(centroids)
    #         tf.compat.v1.global_variables_initializer().run()
    #         print("esooooooooooo", x.eval(), [self.samples[0].eval()])
    #         o = tf.scatter_update(centroids, x, [self.samples[0]])
    #         print(o.eval())
    #         a = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2], tf.int64)
    #         g = tf.concat([x, a], 0)
    #         a_vecs = tf.unstack(centroids, axis=0)
    #         print(a_vecs[1].eval())
    #
    #         def function(x):
    #             del a_vecs[x]
    #             return a_vecs
    #
    #         tf.map_fn(lambda a: function(a), x)
    #         a_new = tf.stack(a_vecs, 0)
    #
    #         print(a_new.eval())
    #         self.assertAllEqual(x.eval(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
    #
    # def testCentroidUpdate(self):
    #     with self.session():
    #         nearest_indices = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #         x = self.kmean_example.update_centroids(nearest_indices)
    #         print(x.eval())
    #         self.assertAllEqual(x.eval(), [[3., 26.], [9., 10.], [15., 2.]])
    #
    # def testCentroidUpdate2(self):
    #     with self.session():
    #         nearest_indices = tf.constant([1, 1, 0, 1, 1, 1, 2, 2, 2])
    #         x = self.kmean_example.update_centroids(nearest_indices)
    #         print(x.eval())
    #         self.assertAllEqual(x.eval(), [[3., 26.], [9., 10.], [15., 2.]])
    #
    # def testKMeansIterations(self):
    #     with self.session():
    #         nearest_indices = tf.constant([1, 1, 0, 1, 1, 1, 2, 2, 2])
    #         initial_centroids = tf.constant([[3., -26.], [9., 1.], [1., 2.]])
    #         x, y, z = self.kmean_example.kmneansItertions(nearest_indices, initial_centroids)
    #         tf.run(tf.compat.v1.global_variables_initializer())
    #         print(x.eval(), y.eval(), z.eval())
    #         self.assertAllEqual(x.eval(), [0, 0, 0, 1, 1, 1, 2, 2, 2])

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
