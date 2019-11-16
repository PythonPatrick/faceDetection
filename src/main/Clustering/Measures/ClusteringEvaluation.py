from main.Clustering.Measures.Evaluation.dbindex import DBIndex as dbIndex


class Evaluation(object):

    def dbIndex(self, n, partition, clusters, distance):
        """ Davies-Bouldin index is an internal evaluation method for clustering algorithms.
            Lower values indicate tighter clusters that are better separated.
        """
        return dbIndex.compute_DB_index(clusters, partition, n, distance)


