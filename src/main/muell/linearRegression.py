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

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
next_element = iterator.get_next()

n = len(x)  # Number of data points

# Plot of Training Data
plt.scatter(x, y)
plt.xlabel('x')
plt.xlabel('y')
plt.title("Training Data")
plt.show()

gs = tf.Variable(0)
i = tf.Variable(0)
epochs=100
learning_rate = 0.01
training_epochs = 1000




def Linear(X):
    W = tf.Variable(np.random.randn(), name = "W", dtype=tf.float64)
    b = tf.Variable(np.random.randn(), name = "b",  dtype=tf.float64)
    # Hypothesis
    y_pred = tf.add(tf.multiply(X, W), b)
    # Mean Squared Error Cost Function
    cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)
    return W, b,cost

def training(epochs):
    W = tf.Variable(np.random.randn(), name="W", dtype=tf.float64)
    b = tf.Variable(np.random.randn(), name="b", dtype=tf.float64)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    def condition(i): return i < epochs

    def body(i):
        X, Y= next_element
        y_pred = tf.add(tf.multiply(X, W), b)
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
    W, b,cost=Linear(X)

    # Initializing the Variables
    sess.run(tf.global_variables_initializer())

    # Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Iterating through all the epochs
    for epoch in range(training_epochs):

        # Feeding each data point into the optimizer using Feed Dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X: _x, Y: _y})

            # Displaying the result after every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Calculating the cost a every epoch
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))

            # Storing necessary values to be used outside the Session
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

    print(sess.run(training(100)))

# Calculating the predictions
predictions = weight * x + bias
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

# Plotting the Results
plt.plot(x, y, 'ro', label ='Original data')
plt.plot(x, predictions, label ='Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.show()


