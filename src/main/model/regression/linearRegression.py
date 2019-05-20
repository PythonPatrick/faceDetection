import tensorflow as tf
from src.main.model.model import Model, Config, Parameters
from src.main.utils.decorators import lazy_property, regularization

class LinearRegression(Model):
    """Linear Regression Model

    The linear multilinear regression model attempts to find parameters W (vector)
    and bias b such that XW+b with X the feature space is a proficient predictor
    for the target vector Y. For optimization the Gradient descent method is used.
    """

    def __init__(self, dataset, config: Config, parameters: Parameters, regularization=None):
        self.regularization = regularization
        Model.__init__(self, dataset=dataset, config=config, parameters=parameters)

    @lazy_property
    def prediction(self):
        """Prediction hypotesis for this model.

        This method returns the formula X*W+b, where X is just a batch of the data.
        """
        return tf.add(tf.matmul(self.dataset.features_batch, self.weights), self.bias) # returns: tf.Tensor

    @lazy_property
    @regularization
    def cost(self):
        """Cost function for this model.

        This method returns the formula sum((X*W+b-Y)^2)/n, where X is just a batch of
        the feature data, Y is a batch of the target data, and n is the dimension.
        """
        return tf.reduce_mean(tf.square(self.prediction - self.dataset.target_batch))/2

    @lazy_property
    def optimization(self):
        """Gradient descent optimization.

        After each batch, this optimization method updates the weights and bias.
        """
        optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost