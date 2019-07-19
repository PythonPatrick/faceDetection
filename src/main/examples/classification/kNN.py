from src.main.dataset.datasets import Datasets
import tensorflow as tf
from sklearn import datasets
from tensorflow import keras
from src.main.model.classification.kNN import kNearestNeighbors
from src.main.model.functions.distance import L0Norm


print(tf.Session())
#data
fashion_mnist=keras.datasets.fashion_mnist
(training, test)=fashion_mnist.load_data()

feature_training=training[0][1:100]
target_training=training[1][1:100]
feature_test=test[0]
target_test=test[1]

dataset=Datasets(features=tf.reshape(tf.clip_by_value(tf.convert_to_tensor(feature_training, dtype=tf.float64),0,1),[99,-1]),
                 target=tf.convert_to_tensor(target_training, dtype=tf.float64),
                 features_test=tf.reshape(tf.clip_by_value(tf.convert_to_tensor(feature_test, dtype=tf.float64),0,1),[10000,-1]),
                 target_test=tf.convert_to_tensor(target_test, dtype=tf.float64),
                 batch_size=100)

# model
sess = tf.Session()
model = kNearestNeighbors(dataset=dataset, k=3, distance=L0Norm())
sess.run(tf.global_variables_initializer())

# option 1: execute code with extra process
p = multiprocessing.Process(target=run_tensorflow)
p.start()
p.join()

# wait until user presses enter key
raw_input()

# option 2: just execute the function
run_tensorflow()

# wait until user presses enter key
raw_input()

model.accuracy(session=sess)
