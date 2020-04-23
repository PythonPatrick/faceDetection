import tensorflow as tf
import numpy as np
from src.main.dataset.inputdata import Input
from src.main.models.supervised.LinearRegression.model import LinearRegression

class TestLinearRegression(tf.compat.v1.test.TestCase):

    def setUp(self):
        self.input=np.random.random_sample((3, 2))
        self.target=np.random.random_sample((3, 1))
        self.model=LinearRegression(data=Input(self.input, self.target),data_dimension=2)

    def test_prediction(self):
        with self.session():
            self.run(tf.global_variables_initializer())
            self.assertEqual(LinearRegression.prediction, self.input*self.model.weights.eval()+self.bias)

        self.fail()

    def test_cost(self):
        self.fail()

    def test_optimize(self):
        self.fail()

    def test_error(self):
        self.fail()
