import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from functools import wraps
from src.main.utils.tensorflowUtils.distances import distances
from src.main.utils.tensorflowUtils.tflow import Utils


class Kmean(object):
    def __init__(self, data, n_clusters, initialCentroids=None):
        self.samples = data
        self.n_clusters = n_clusters
        self.initialCentroids = initialCentroids

    class IndexToSamplePoints(object):
        def __init__(self, flag):
            self.flag = flag

        def __call__(self, function):
            @wraps(function)
            def wrapped(sel, *f_args, **f_kwargs):
                idx = function(sel, *f_args, **f_kwargs)
                if not f_args[self.flag]:
                    return idx
                return tf.gather(sel.samples, idx)
            return wrapped


    @property
    def random_centroids(self):
        """
        Step 0: Initialisation: Select `n_clusters` number of random points
        """
        n_samples = tf.shape(self.samples)[0]
        """ create an arrow with all indices from 0 to n_samples in an arbitrary order"""
        random_indices = tf.random_shuffle(tf.range(0, n_samples))
        """ begin with index 0 and select a length of n_clusters """
        begin = [0, ]
        size = [self.n_clusters, ]
        """ select the first n_clusters out of the arrow random_indices """
        centroid_indices = tf.slice(random_indices, begin, size)
        """ collect the points for the associated indices centroid_indices """
        initial_centroids = tf.gather(self.samples, centroid_indices)
        return initial_centroids

    @property
    def centroids(self):
        """
        Variable for centroids of K-mean model
        """
        if self.initialCentroids:
            return tf.Variable(self.initialCentroids)
        else:
            return tf.Variable(self.random_centroids)

    @property
    def clusters(self):
        """
        Variable for cluster indices of K-mean model
        """
        return tf.Variable(self.assign_to_nearest(self.centroids))

    def assign_to_nearest(self, centroids):
        """
        Finds the nearest centroid for each sample
        """
        expanded_vectors = tf.cast(tf.expand_dims(self.samples, 0), tf.float64)
        expanded_centroids = tf.cast(tf.expand_dims(centroids, 1), tf.float64)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
        return tf.argmin(distances, 0)

    def update_centroids(self, nearest_indices):
        """
        Updates the centroid to be the mean of all samples associated with it
        """
        nearest_indices = tf.cast(nearest_indices, tf.int32)
        partitions = tf.dynamic_partition(self.samples, nearest_indices, self.n_clusters)
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
        return new_centroids

    def substituteEmptyCluster(self, clusters, centroid):
        """
        if there is an empty cluster, this empty cluster will be assigned with those point which is
        from the biggest cluster and most far away from its centroid.
        """
        missing = self.missingCluster(clusters, centroid)

        def changeInput(cluster, centroids):
            biggestCluster = self.mayorCluster(cluster)
            centroidBiggestCluster = tf.gather(centroids, biggestCluster)
            clusterIdx = self.clusterPoints(clusters, biggestCluster)
            point = self.pointMaximalDistanceInsideCluster(centroidBiggestCluster, clusterIdx, False)
            pointValue = self.pointMaximalDistanceInsideCluster(centroidBiggestCluster, clusterIdx, True)
            # return Utils.replaceValues(cluster, point, missing), Utils.replaceValues(centroids, missing, pointValue)
            return cluster, tf.expand_dims(point, 0),  Utils.replaceValues(cluster, tf.expand_dims(point, 0), missing)

        def changeNothing(cluster, centroids): return cluster, centroids, centroids

        if tf.shape(missing)[0] != 0:
            return changeInput(clusters, centroid)
        return changeNothing(clusters, centroid)







    @staticmethod
    def missingCluster(clusters, centroids):
        """
        detect empty clusters
        """
        clusters = tf.cast(clusters, tf.int64)
        given, b = tf.unique(clusters)
        clusters = tf.constant(list(range(centroids.shape[0])), tf.int64)
        return tf.setdiff1d(clusters, given).out

    @staticmethod
    def mayorCluster(clusters):
        """
        select cluster with mayor number of elements
        """
        y, idx, count = tf.unique_with_counts(clusters)
        # select biggest cluster
        maximum = tf.math.argmax(count, axis=0)
        y = tf.reshape(y, [1, ])
        x = tf.gather_nd(y, tf.reshape(maximum, [1, ]))
        # select indices of points in cluster
        return x

    def clusterPoints(self, clusterIndex, sampleIndex):
        """
        obtain all points of a given cluster
        """
        return tf.reshape(tf.where(tf.equal(sampleIndex, clusterIndex)), [-1])

    @IndexToSamplePoints(flag=2)
    def pointMaximalDistanceInsideCluster(self, centroid, clusterIdx, idx: bool):
        """
        returns point in cluster with maximal distances to centroid point
        """
        cluster = tf.gather(self.samples, clusterIdx)
        index = distances.pointToMatrix(centroid, cluster)
        return tf.gather(clusterIdx, tf.argmax(index, 0))

    @staticmethod
    def condition(indices, centroids, i): return i < 0

    def training(self, initial_indices, initial_centroids):
        """
        iteration with training set
        """
        indices = tf.Variable(tf.cast(initial_indices, tf.int64))
        centroids = tf.Variable(initial_centroids)

        def body(indices_inner, centroids_inner, i_inner):
            ind = self.assign_to_nearest(centroids_inner)
            cen = self.update_centroids(ind)
            tf.print(i_inner.eval, [i_inner])
            return ind, cen, i_inner + 1

        i = tf.constant(0, tf.int32)
        return tf.while_loop(self.condition, body, [indices, centroids, i])

