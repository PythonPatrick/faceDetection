import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('display_step', 100, 'Display logs per step.')


def run_training(train_X, train_Y):
    X = tf.placeholder(tf.float32, [m, n])
    Y = tf.placeholder(tf.float32, [m, 1])

    # weights
    W = tf.Variable(tf.zeros([n, 1], dtype=np.float32), name="weight")
    b = tf.Variable(tf.zeros([1], dtype=np.float32), name="bias")

    # linear model
    activation = tf.add(tf.matmul(X, W), b)
    cost = tf.reduce_mean(tf.square(activation - Y))
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(FLAGS.max_steps):

            sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

            if step % FLAGS.display_step == 0:
                print("Step:", "%04d" % (step+1), "Cost=", "{:.2f}".format(sess.run(cost, \
                    feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})), "W=", sess.run(W), "b=", sess.run(b))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})
        print("Training Cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        print("Predict.... (Predict a house with 1650 square feet and 3 bedrooms.)")
        predict_X = np.array([1650, 3], dtype=np.float32).reshape((1, 2))

        # Do not forget to normalize your features when you make this prediction
        predict_X = (predict_X - mean)/std

        predict_Y = tf.add(tf.matmul(predict_X, W),b)
        print("House price(Y) =", sess.run(predict_Y))


def read_data(filename, read_from_file = True):
    global m, n

    if read_from_file:
        with open(filename) as fd:
            data_list = fd.read().splitlines()

            m = len(data_list) # number of examples
            n = 2 # number of features

            train_X = np.zeros([m, n], dtype=np.float32)
            train_Y = np.zeros([m, 1], dtype=np.float32)

            for i in range(m):
                datas = data_list[i].split(",")
                for j in range(n):
                    train_X[i][j] = float(datas[j])
                train_Y[i][0] = float(datas[-1])
    else:
        m = 47
        n = 2

        train_X = np.array( [[  2.10400000e+03,   3.00000000e+00],
           [  1.60000000e+03,   3.00000000e+00],
           [  2.40000000e+03,   3.00000000e+00],
           [  1.41600000e+03,   2.00000000e+00],
           [  3.00000000e+03,   4.00000000e+00],
           [  1.98500000e+03,   4.00000000e+00],
           [  1.53400000e+03,   3.00000000e+00],
           [  1.42700000e+03,   3.00000000e+00],
           [  1.38000000e+03,   3.00000000e+00],
           [  1.49400000e+03,   3.00000000e+00],
           [  1.94000000e+03,   4.00000000e+00],
           [  2.00000000e+03,   3.00000000e+00],
           [  1.89000000e+03,   3.00000000e+00],
           [  4.47800000e+03,   5.00000000e+00],
           [  1.26800000e+03,   3.00000000e+00],
           [  2.30000000e+03,   4.00000000e+00],
           [  1.32000000e+03,   2.00000000e+00],
           [  1.23600000e+03,   3.00000000e+00],
           [  2.60900000e+03,   4.00000000e+00],
           [  3.03100000e+03,   4.00000000e+00],
           [  1.76700000e+03,   3.00000000e+00],
           [  1.88800000e+03,   2.00000000e+00],
           [  1.60400000e+03,   3.00000000e+00],
           [  1.96200000e+03,   4.00000000e+00],
           [  3.89000000e+03,   3.00000000e+00],
           [  1.10000000e+03,   3.00000000e+00],
           [  1.45800000e+03,   3.00000000e+00],
           [  2.52600000e+03,   3.00000000e+00],
           [  2.20000000e+03,   3.00000000e+00],
           [  2.63700000e+03,   3.00000000e+00],
           [  1.83900000e+03,   2.00000000e+00],
           [  1.00000000e+03,   1.00000000e+00],
           [  2.04000000e+03,   4.00000000e+00],
           [  3.13700000e+03,   3.00000000e+00],
           [  1.81100000e+03,   4.00000000e+00],
           [  1.43700000e+03,   3.00000000e+00],
           [  1.23900000e+03,   3.00000000e+00],
           [  2.13200000e+03,   4.00000000e+00],
           [  4.21500000e+03,   4.00000000e+00],
           [  2.16200000e+03,   4.00000000e+00],
           [  1.66400000e+03,   2.00000000e+00],
           [  2.23800000e+03,   3.00000000e+00],
           [  2.56700000e+03,   4.00000000e+00],
           [  1.20000000e+03,   3.00000000e+00],
           [  8.52000000e+02,   2.00000000e+00],
           [  1.85200000e+03,   4.00000000e+00],
           [  1.20300000e+03,   3.00000000e+00]]
        ).astype('float32')

        train_Y = np.array([[ 399900.],
           [ 329900.],
           [ 369000.],
           [ 232000.],
           [ 539900.],
           [ 299900.],
           [ 314900.],
           [ 198999.],
           [ 212000.],
           [ 242500.],
           [ 239999.],
           [ 347000.],
           [ 329999.],
           [ 699900.],
           [ 259900.],
           [ 449900.],
           [ 299900.],
           [ 199900.],
           [ 499998.],
           [ 599000.],
           [ 252900.],
           [ 255000.],
           [ 242900.],
           [ 259900.],
           [ 573900.],
           [ 249900.],
           [ 464500.],
           [ 469000.],
           [ 475000.],
           [ 299900.],
           [ 349900.],
           [ 169900.],
           [ 314900.],
           [ 579900.],
           [ 285900.],
           [ 249900.],
           [ 229900.],
           [ 345000.],
           [ 549000.],
           [ 287000.],
           [ 368500.],
           [ 329900.],
           [ 314000.],
           [ 299000.],
           [ 179900.],
           [ 299900.],
           [ 239500.]]
        ).astype('float32')

    return train_X, train_Y


def feature_normalize(train_X):

    global mean, std
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)

    return (train_X - mean) / std

import sys

def main(argv):
    if not argv:
        print("Enter data filename.")
        sys.exit()

    #filename = argv[1]

    train_X, train_Y = read_data("", False)
    train_X = feature_normalize(train_X)
    run_training(train_X, train_Y)

if __name__ == '__main__':
    tf.app.run()