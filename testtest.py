import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(101)
tf.set_random_seed(101)

# Genrating random linear data
# There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

# create the training datasets
dx_train = tf.data.Dataset.from_tensor_slices(x)
# apply a one-hot transformation to each label for use in the neural network
dy_train = tf.data.Dataset.from_tensor_slices(y)
# zip the x and y training data together and shuffle, batch etc.
train_dataset = tf.data.Dataset.zip((dx_train, dy_train))

W = tf.Variable(np.random.randn(), name="W", dtype=tf.float64)
b = tf.Variable(np.random.randn(), name="b", dtype=tf.float64)

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
next_element = iterator.get_next()

def LinearModel(X,W,b):
    return tf.add(tf.multiply(X, W), b)

def training(epochs, learning_rate):
    gs = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    i = tf.constant(0)

    def condition(i): return i < epochs

    def body(i):
        X, Y= next_element
        y_pred = LinearModel(X, W, b)
        cost = tf.reduce_mean(tf.pow(y_pred - Y, 2))/2
        train_op = optimizer.minimize(cost, global_step=gs)
        return tf.tuple([tf.add(i, 1)], control_inputs=[train_op])

    loop = tf.while_loop(condition, body, [i])
    return loop


from tensorflow.python.framework import ops
ops.reset_default_graph()
g = tf.get_default_graph()
print([op.name for op in g.get_operations()])
# Starting the Tensorflow Session
with tf.Session() as sess:

    X = tf.placeholder(tf.float64)
    Y = tf.placeholder(tf.float64)
    # Initializing the Variables

    print(sess.run(training(100,0.1)))

    sess.run(tf.global_variables_initializer())