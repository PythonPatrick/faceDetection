import tensorflow as tf
import numpy as np
from src.main.model.model import Model, Config, Parameters
from src.main.utils.decorators import lazy_property
from src.main.dataset.datasets import Datasets

class kNearestNeighbors(object):
    """k-Nearest Neighbors Model implementation

    Attributes
    ----------
    dataset : Class
        dataset input pipeline
    distance: Class
        distance class with distance as an internal method
    k: int
        parameter k for k-NN model

    Methods
    -------
    prediction : tf.Tensor
        Model prediction hypothesis
    accuracy : float
        Model accuracy calculation
    save:
        Save training and testing results in json file.


    """

    def __init__(self, dataset: Datasets, distance, k):
        self.dataset = dataset
        self.distanceClass=distance
        self.k=k


    def distance(self, feature):
        """k-Nearest Neighbor distance function

        """
        return self.distanceClass.distance(self.dataset.feature_data,feature)


    def prediction(self, feature):
        """k-Nearest Neighbor Hypothesis

        """
        _, top_k_indices = tf.nn.top_k(tf.negative(self.distance(feature)), k=self.k)
        top_k_label = tf.gather(self.dataset.target_data, top_k_indices)
        labels, indexes, counts = tf.unique_with_counts(top_k_label)
        return tf.gather(labels, tf.argmax(counts))


    def accuracy(self, session):
        """k-Nearest Neighbor Model accuracy

        """
        session.run([self.dataset.test_data_op.initializer])
        sample_size=10000
        num=0
        while True:
            try:
                print("HOLA")
                num+=1
                print("prediction",session.run(self.prediction(self.dataset.test_data_next.feature)), num)
                tf.reset_default_graph()
            except tf.errors.OutOfRangeError:
                break
        return 0