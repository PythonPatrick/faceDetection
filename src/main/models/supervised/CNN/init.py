import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###########################
####### IMPORT DATA #######
###########################
train_df = pd.read_csv('/home/pguagliardo/personal/FacialDetection/training.csv')
test_df = pd.read_csv('/home/pguagliardo/personal/FacialDetection/test.csv')

######################################
##### CONVERT PIXEL TO PICTURE #######
######################################
img = np.asarray([int(v) for v in train_df['Image'][1000].split(" ")])
imgres = img.reshape(96, 96)

#########################################################
############ CONVOLUTION WITH FIRST IMAGE ###############
#########################################################

######### mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [9216, ])
y = tf.placeholder(tf.float32, [1, 96, 96, 1])
######### dynamically reshape the input
x_shaped = tf.reshape(x, [-1, 96, 96, 1])
y_shaped = tf.reshape(y, [96, 96])
######### A filter (affects the convolution).
k = tf.constant([
    [0, -10, -108, -602],
    [2, 400, 0, 94],
    [-10, 7, 0, 51],
    [-60, 19, 71, -1]
], dtype=tf.float32, name='k')

######### KERNEL
kernel = tf.reshape(k, [4, 4, 1, 1], name='kernel')
img2 = tf.reshape(img, [-1, 96, 96, 1])

######### CONVOLUTION
res = tf.nn.conv2d(x_shaped, kernel, [1, 1, 1, 1], "SAME")
out_layer = tf.nn.relu(res)

######### POOL
ksize = [1, 3, 3, 1]
strides = [1, 2, 2, 1]
pool = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

######### setup the initialisation operator
init_op = tf.global_variables_initializer()
fig = plt.figure()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    pic = sess.run(x_shaped, feed_dict={x: img})
    out_layer = sess.run(res, feed_dict={x_shaped: pic})
    out_layer2 = sess.run(tf.squeeze(sess.run(res, feed_dict={x_shaped: pic})))
    # out_layer3=sess.run(pool, feed_dict={out_layer: sess.run(tf.nn.relu(out_layer))})
    out_layer3 = sess.run(tf.nn.relu(out_layer))
    out = sess.run(tf.squeeze(sess.run(tf.nn.max_pool(out_layer3, ksize=ksize, strides=strides, padding='SAME'))))
    # out=sess.run(tf.squeeze(sess.run(pool, feed_dict={out_layer: sess.run(tf.nn.relu(out_layer))})))
    # apply a ReLU non-linear activation

    ######################################
    ######## VISUALIZE EXAMPLE ###########
    ######################################
    fig.add_subplot(1, 3, 1)
    plt.imshow(imgres)
    fig.add_subplot(1, 3, 2)
    plt.imshow(out_layer2)
    fig.add_subplot(1, 3, 3)
    plt.imshow(out)
    plt.show()