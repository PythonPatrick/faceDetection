from src.main.model.regression.linearRegression import LinearRegression
from src.main.model.model import Config, Parameters
import tensorflow as tf
from src.main.dataset.inputdata import regression_data
from src.main.model.functions.regularization import Ridge

# model configurations
config = Config(feature_num=2, batch_size=50, learning_rate=0.01, epoche=500)

# dataset
dataset = regression_data(TRUE_W=[[17.0], [4]],
                          TRUE_b=4,
                          NUM_EXAMPLES=10000,
                          batch_size=100)

# implemented model
sess = tf.Session()
lr = LinearRegression(dataset=dataset, config=config, parameters=Parameters())
sess.run(tf.global_variables_initializer())
lr.training(session=sess)
print(sess.run(lr.weights))
