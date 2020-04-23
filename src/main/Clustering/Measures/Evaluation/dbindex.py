import tensorflow as tf


class DBIndex:

    @staticmethod
    def compute_S(i: int, clusters: tf.Tensor, partition: list, distance):
        """ Compute distance between each point in x_i and cluster ci.
            The sum of all these distances is the final diameter of cluster i.

        """
        ci = tf.gather(clusters, i)
        return distance(partition[i], ci)

    @staticmethod
    def compute_Rij(i: int, j: int, clusters: tf.Tensor, partition: list, dist):
        """ Compute a ratio measure of how much the two clusters i and j are far away from each
            other compared to the "inter distances" of each cluster i and j, respectively.
        """
        ci = tf.gather(clusters, i)
        cj = tf.gather(clusters, j)
        d = dist(ci, cj)
        Rij = (DBIndex.compute_S(i, clusters, partition, dist) + DBIndex.compute_S(j, clusters, partition, dist)) / d
        return Rij

    @staticmethod
    def compute_Ri(i: int, clusters: tf.Tensor, partition: list, n: int, dist):
        """

        """
        list_r = []
        for j in range(n):
                if i is not j:
                    temp = DBIndex.compute_Rij(i, j, clusters, partition, dist)
                    list_r.append(temp)
        return tf.reduce_max(tf.stack(list_r))

    @staticmethod
    def compute_DB_index(clusters, partition, n: int, dist):
        """

        """
        list_r=[]
        for i in range(n):
            list_r.append(DBIndex.compute_Ri(i, clusters, partition, n, dist))
        return tf.multiply(1 / n, tf.reduce_sum(list_r))






