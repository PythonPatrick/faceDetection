from unittest import TestCase
import tensorflow as tf
import pandas as pd
from src.main.dataset.datasets import Datasets

class TestDatasets(tf.test.TestCase):
    def setUp(self):
        """Set up all datasets in order to

        """
        feature_data= [[9, 10], [11, 15], [10, 14]]
        target_data = [[1], [0], [1]]
        feature_df=pd.DataFrame(feature_data, columns=["X_1", "X_2"])
        target_df=pd.DataFrame(target_data, columns=["Y"])
        self.feature_tensor=tf.convert_to_tensor(feature_df)
        self.target_tensor=tf.convert_to_tensor(target_df)
        self.batch_size=30
        self.training_size=0.8
        self.dataset=Datasets(self.feature_tensor,
                              self.target_tensor,
                              self.batch_size,
                              self.training_size)

    def test_sample_size(self):
        """Sample size test

        Since the sample size is 3, this unit test checks that the class
        attribute self.dataset.sample_size concurs with this value.
        """
        self.assertEqual(self.dataset.sample_size, 3)

    def test_dataset(self):
        """Dataset test

        This unit test checks that the class attribute dataset of
        the class self.dataset contains the expected data.
        """
        features=tf.data.Dataset.from_tensor_slices(self.feature_tensor)
        target=tf.data.Dataset.from_tensor_slices(self.target_tensor)
        datasetZipped=tf.data.Dataset.zip((features,target))
        self.assertEqual(self.dataset.dataset, datasetZipped)

    def test_training_data(self):
        """Training dataset test

        This unit test checks that the class attribute training_data
        of the class self.dataset contains the expected data.
        """
        feature_size = self.dataset.training_data.output_shapes[1]
        self.assertEqual(feature_size, 2)

    def test_test_data(self):
        """Test dataset test

        This unit test checks that the class attribute test_data
        of the class self.dataset contains the expected data.
        """
        test_size = self.dataset.test_data.output_shapes[1]
        print(test_size)
        self.assertEqual(test_size, 1)
