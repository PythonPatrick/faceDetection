import tensorflow as tf
from tensorflow import Tensor
from src.main.utils.decorators import lazy_property

class Datasets(object):
    """Input pipeline using tf.data.dataset API

    Dataset input pipline which will be used as an input for a model.
    This class separates your input into training and test data and
    cluster your data into batches with size batch_size.

     Attributes
    ----------
    features : tf.Tensor
        features data. This will define the feature space X.
    target : tf.Tensor
        target data. This defines the target space Y.
    batch_size : int
        Batch size for batch/mini-batch optimization. If this
        value is setted to None (by default), the optimization
        will be batch optimization. If setted to -1, this will be
        gradient optimization. If setted to a natural number, mini
        batch optimization will be applied.
    training_size: float
        Percentage of desired training set. For example, if this
        parameter is fullfied by the value 0.8, 80 percentage of
        the input data will be used as the training data.

    Methods
    -------
    sample_size : int
        Dataset sample size. If the given feature and target
        inputs do not concur in their sample size, this method
        will rise an ValueError and this object will be useless.
    dataset : tf.Tensor
        Dataset. The dataset is defined as a zipped dataset with
        the structure (feature, target) for each sample.
    training_data : tf.Tensor
        Training dataset. 
    test_data : tf.Tensor
        Testing dataset
    """

    def __init__(self, features: Tensor, target: Tensor,
                 batch_size=None, training_size=0.8):
        self.feature_data=features
        self.target_data=target
        self.batch_size=batch_size
        self.training_size=training_size
        self.sample_size
        self.dataset
        self.training_data
        self.test_data

    @lazy_property
    def sample_size(self):
        """Sample size

        Sample size of dataset. Since features and target data are
        independent class arguments, their size does not necessarily
        match. In the case they do not, a ValueError exception is
        raised.
        """
        feature_size=self.feature_data.shape[0]

        target_size = self.target_data.shape[0]
        if feature_size != target_size:
            raise ValueError("Feature sample size {} and target sample size {} must be identically!".format(feature_size, target_size))
        else:
            return int(feature_size)

    @lazy_property
    def dataset(self):
        """Dataset

        Shuffled dataset with feature and target set zipped sample-wise and
        with a seperation into batches with size batch_size.
        """
        feature=tf.data.Dataset.from_tensor_slices(self.feature_data)
        target=tf.data.Dataset.from_tensor_slices(self.target_data)
        return tf.data.Dataset.zip((feature, target))

    @lazy_property
    def training_data(self):
        """Dataset for training

        Training dataset with size equal to the integer part of training
        size (percentaje of entire dataset) multiplied by dataset sample size.
        """
        size=int(self.training_size*self.sample_size)
        if self.batch_size is not None:
            return self.dataset.take(size).batch(self.batch_size)
        return self.dataset.take(size)

    @lazy_property
    def test_data(self):
        """Dataset for testing

        Testing dataset with size equal to sample size minus the
        training dataset size.
        """
        size = int(self.training_size * self.sample_size)
        return self.dataset.skip(size)