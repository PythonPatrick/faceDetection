import tensorflow as tf
import numpy as np


a = tf.constant([1, 2, 3, 4, 5])
y = tf.Variable([1, 2, 3, 4, 5])
centroids=tf.Variable([[3.,-26.],[9.,1.],[1.,2.]])
b = tf.strided_slice(centroids, [1], [3])
c = tf.strided_slice(a, [1], [-2], [1])
o=tf.scatter_update(centroids,[0],[[20,2]])

h = tf.constant([1, 2, 3, 4, 5,1,1])
d,e=tf.unique(h)

with tf.Session():
    tf.global_variables_initializer().run()
    print(b.eval())
    print(c.eval())
    print(o.eval())
