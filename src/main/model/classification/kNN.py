import tensorflow as tf
import numpy as np
from src.main.model.model import Model, Config, Parameters
from src.main.utils.decorators import lazy_property

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

    def __init__(self, dataset, distance, k):
        self.dataset = dataset
        self.distanceClass=distance
        self.k=k
        self.distance
        self.prediction
        self.accuracy


    @lazy_property
    def distance(self):
        """k-Nearest Neighbor distance function

        """
        return self.distanceClass.distance(self.dataset.feature_data, self.dataset.feature_data)


    @lazy_property
    def prediction(self):
        """k-Nearest Neighbor Hypothesis

        """
        _, top_k_indices = tf.nn.top_k(tf.negative(self.distance), k=self.k)
        top_k_label = tf.gather(self.dataset.target_data, top_k_indices)

        sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
        return tf.argmax(sum_up_predictions, axis=1)


    @lazy_property
    def accuracy(self, session):
        """k-Nearest Neighbor Model accuracy

        """
        prediction, target=session.run([self.prediction, self.dataset.target_data])
        accuracy = sum([1 if  pred == np.argmax(actual) else 0 for pred, actual in zip(prediction, target)])
        return accuracy/len(prediction)