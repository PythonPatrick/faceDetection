import tensorflow as tf
from tensorflow import Tensor
from functools import partial
from inspect import signature
from src.main.model.model import Model, Config, Parameters
from src.main.utils.decorators import lazy_property
from src.main.model.functions.kernels import Kernels

class SVM(Model):

    def __init__(self, dataset, config: Config, parameters: Parameters, kernel: str="gaussian", gamma: float=-50.0):
        self.kernelname = kernel
        self.gamma = gamma
        self.alpha=tf.Variable(tf.random_normal(shape=[1, dataset.batch_size]))
        Model.__init__(self, dataset=dataset, config=config, parameters=parameters)


    def predictions(self, Y: Tensor, kernel):

        prediction_output = partial(tf.matmul(tf.multiply(tf.transpose(Y), self.alpha), kernel))
        return tf.sign(prediction_output-tf.reduce_mean(prediction_output))

    def kernel(self):
        var=signature(getattr(Kernels, self.kernelname))
        dataarguments=["X"]
        parameters=[param for param in var.parameters if param not in dataarguments]
        values={k: getattr(self, k) for (k,v) in var.parameters.items() if k not in dataarguments}
        func=partial(getattr(Kernels, self.kernelname),**values)
        return func

    @lazy_property
    def prediction(self):
        """Prediction function

        """
        my_kernel = self.kernel()(self.dataset.features)
        return self.predictions(self.dataset.target_batch, my_kernel)


    @lazy_property
    def cost(self):
        my_kernel=self.kernel()(self.dataset.features_batch)
        first_term = tf.reduce_sum(self.bias)
        alpha_vec_cross = tf.matmul(tf.transpose(self.alpha), self.alpha)
        y_target_cross = tf.matmul(self.dataset.target_batch, tf.transpose(self.dataset.target_batch))
        second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(alpha_vec_cross, y_target_cross)))
        return tf.negative(tf.subtract(first_term, second_term))

    @lazy_property
    def optimization(self):
        optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost

    @lazy_property
    def accuracy(self):
        sum=tf.cast(tf.equal(tf.squeeze(self.prediction),tf.squeeze(self.dataset.target_data)), tf.float32)
        return tf.reduce_mean(sum)