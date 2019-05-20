from src.main.model.classification.logisticRegression import LogisticRegression
from src.main.model.model import Config, Parameters
import tensorflow as tf
from src.main.dataset.inputdata import classification_data

# model configurations
config = Config(feature_num=10, batch_size=50, learning_rate=0.01)

# dataset
NUM_EXAMPLES=10000000
dataset = classification_data(batch_size=1000000, n_samples=NUM_EXAMPLES, n_features=10)

# model
sess = tf.Session()
model = LogisticRegression(dataset=dataset, config=config, parameters=Parameters())
sess.run(tf.global_variables_initializer())
model.training(session=sess)
print(sess.run([model.weights, model.bias]))
print(sess.run(model.accuracy))