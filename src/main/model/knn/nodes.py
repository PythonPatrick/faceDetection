"""A Python implemntation of a kd-tree

This package provides a simple implementation of a kd-tree in Tensorflow.

"""
import tensorflow as tf
from src.main.utils.decorators import lazy_property
import multiprocessing as mp
import numpy as np
import math
import ray

def median(input, dimension):
    return tf.contrib.distributions.percentile(tf.gather(input, dimension, axis=1), 50.0)

def left(input, median, dimension):
    return tf.boolean_mask(input, tf.less(tf.gather(input, dimension, axis=1), median))

def right(input, median, dimension):
    return tf.boolean_mask(input, tf.greater_equal(tf.gather(input, dimension, axis=1), median))

def splitTensor(tensor: tf.Tensor, splits: int):
    row, col= tf.Session().run(tf.shape(tensor))
    batch_size=math.ceil(row/splits)
    return tf.reshape(tensor, [-1, batch_size, col])


@ray.remote
def KDTreeSimple(input: tf.Tensor, k=100):
    build=NodeBuilder(input, k)
    if build.is_leaf:
        inp=build.input
        splitted_set=splitTensor(inp, 8)
        def calculate(input, dimension=build.dimension):
            m = median(input, dimension)
            l = left(input, m, dimension)
            r = right(input, m, dimension)
            return (l, r)

        result=tf.map_fn(fn=calculate, elems=splitted_set, dtype=(tf.float64, tf.float64), parallel_iterations=50)
        return Node(is_leaf=build.is_leaf,
                    size=build.size,
                    dimension=build.dimension,
                    left=tf.reshape(result[0], [-1, 3]),
                    right=tf.reshape(result[1], [-1, 3]))
    else:
        return Node(is_leaf=build.is_leaf,
                    size=build.size,
                    data=build.input)


def Main(input: tf.Tensor, k: int):
    ray.init()
    output=ray.get(KDTreeSimple.remote(input, k))
    p=mp.Pool(mp.cpu_count())
    data = np.random.rand(10000, 3)
    input = tf.convert_to_tensor(data)
    data = np.random.rand(10000, 3)
    input2 = tf.convert_to_tensor(data)
    l=output.left
    r=output.right
    out=ray.get([KDTreeSimple.remote() for i in [input,input2]])
    print(out)
    p.close()
    return out




def KDTree(input: tf.Tensor, k=100):
    build=NodeBuilder(input, k)
    if build.is_leaf:
        inp=build.input
        splitted_set=splitTensor(inp, 8)
        # p=mp.Pool(mp.cpu_count())
        #
        # out=p.map(calculate,[splitted_set[0], splitted_set[1], splitted_set[2], splitted_set[3]])
        # print(out)
        # p.close()
        def calculate(input, dimension=build.dimension):
            m = median(input, dimension)
            l = left(input, m, dimension)
            r = right(input, m, dimension)
            return (l, r)

        result=tf.map_fn(fn=calculate, elems=splitted_set, dtype=(tf.float64, tf.float64), parallel_iterations=50)
        return Node(is_leaf=build.is_leaf,
                    size=build.size,
                    dimension=build.dimension,
                    left=KDTree(tf.reshape(result[0], [-1, 3]), k),
                    right=KDTree(tf.reshape(result[1], [-1, 3]), k))
    else:
        return Node(is_leaf=build.is_leaf,
                    size=build.size,
                    data=build.input)



# def subnodes(self):
#     num=tf.div(self.size, 2)
#     _, index_left=tf.nn.top_k(tf.gather(self.input, self.dimension, axis=1), num)
#     set = tf.range(0, self.size)
#     tile_multiples = tf.concat([tf.ones(tf.shape(tf.shape(set)), dtype=tf.int32), tf.shape(index_left)], axis=0)
#     x_tile = tf.tile(tf.expand_dims(set, -1), tile_multiples)
#     condition = tf.reduce_any(tf.equal(x_tile, index_left), -1)
#     index_right=tf.where(tf.equal(condition, tf.constant(False)))
#     return tf.gather_nd(self.input,tf.reshape(index_left, [-1,1])), tf.gather_nd(self.input, index_right)

class Node(object):
    def __init__(self, is_leaf,size, dimension=None,
                 data=None, left=None, right=None):
        self.size=size
        if is_leaf:
            self.dimension = dimension
            self.left=left
            self.right=right
        else:
            self.data=data


class NodeBuilder(object):
    """
       Attributes
       ----------
       input : Tensor
           Input data

       Methods
       ----------
       node :
       dimension :

       """

    def __init__(self, input: tf.Tensor = None, k: int = 10):
        self.input = input
        self.k = k

    @lazy_property
    def size(self):
        return tf.shape(self.input)[0]

    @lazy_property
    def is_leaf(self):
        return tf.Session().run(tf.less(tf.constant(self.k), self.size))

    @lazy_property
    def moments(self):
        return tf.nn.moments(self.input, axes=[0])

    @lazy_property
    def dimension(self):
        _, var=self.moments
        return tf.argmax(var, axis=0)





