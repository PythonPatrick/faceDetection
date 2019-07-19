import tensorflow as tf
from src.main.model.model import Model, Config, Parameters
from src.main.utils.decorators import lazy_property
import sklearn.linear_model

class softmax(Model):

    def __init__(self, dataset, config: Config, parameters: Parameters):
        Model.__init__(self, dataset=dataset, config=config, parameters=parameters)

    @lazy_property
    def prediction(self):
        """Softmax Regression Hypothesis.

        Computes sigmoid of `X*W+b` element-wise. Specifically,
        `prediction = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

        Returns:
            A `Tensor`. Has the same type and shape as `logits`.
        """
        linear=tf.add(tf.matmul(self.dataset.training_data_next.feature, self.weights), self.bias)
        return tf.nn.softmax(linear)

    @lazy_property
    def cost(self):
        """Softmax Regression Cost function

        Computes sigmoid cross entropy given `logits`.Measures the probability error
        in discrete classification tasks in which each class is independent and not
        mutually exclusive.

        Returns:
            A `Tensor` of the same shape as `logits` with the componentwise logistic losses.
        """
        logits=self.prediction
        labels=self.dataset.training_data_next.target
        return tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)

    @lazy_property
    def optimization(self):
        """Gradient descent optimization.

        After each batch, this optimization method updates the weights and bias.

        Returns:
            An Operation that updates the variables in `var_list`.  If `global_step`
            was not `None`, that operation also increments `global_step`.
        """
        optimization = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        return optimization.minimize(self.cost)

    @lazy_property
    def error(self):
        return tf.math.reduce_sum(self.cost)

    @lazy_property
    def accuracy(self):
        linear=tf.add(tf.matmul(self.dataset.feature_data, self.weights), self.bias)
        prediction=tf.nn.sigmoid(linear)
        correct_prediction = tf.equal(tf.round(prediction),self.dataset.target_data)
        return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))