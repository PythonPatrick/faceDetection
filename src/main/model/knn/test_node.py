import tensorflow as tf
import src.main.model.knn.nodes as nodes
import numpy as np
import time

class KDTreeTest(tf.test.TestCase):

    # def testDimension(self):
    #     with self.test_session():
    #         input = tf.constant([[1., 1., 3.], [3., 23., 2.], [3., 23., 1000.]])
    #         tree=nodes.KDTree(input)
    #         dim, mean=tree.dimension
    #         self.assertAllEqual(dim.eval(), 2)
    #         self.assertAllEqual(mean.eval(), 335.)
    #
    # def testNode(self):
    #     with self.test_session():
    #         input = tf.constant([[1., 1., 3.], [3., 23., 2.], [3., 23., 1000.]])
    #         tree=nodes.KDTree(input)
    #         self.assertAllEqual(tree.node.left.eval(), np.array([[ 1.,  1.,  3.],[ 3., 23.,  2.]]))
    #         self.assertAllEqual(tree.node.right.eval(), np.array([ [3., 23., 1000.]]))
    #         self.assertAllEqual(tree.node.dimension, 2)
    #         self.assertAllEqual(tree.node.mean, 335.)

    def testNode1(self):
        with self.test_session():
            data=np.random.rand(10000,3)
            input = tf.convert_to_tensor(data)
            start_time = time.time()
            tree=nodes.Main(input, 100)
            elapsed_time = time.time() - start_time
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            print(tree.size)

    # def testNode2(self):
    #     with self.test_session():
    #         data=np.random.rand(30000,3)
    #         input = tf.convert_to_tensor(data)
    #         start_time = time.time()
    #         tree=nodes.KDTree(input, 100)
    #         elapsed_time = time.time() - start_time
    #         print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    #         print(tree.size)
    #
    # def testNode3(self):
    #     with self.test_session():
    #         data=np.random.rand(40000,3)
    #         input = tf.convert_to_tensor(data)
    #         start_time = time.time()
    #         tree=nodes.KDTree(input, 100)
    #         elapsed_time = time.time() - start_time
    #         print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    #         print(tree.size)
    #
    # def testNode(self):
    #     with self.test_session():
    #         data=np.random.rand(1000,3)
    #         input = tf.convert_to_tensor(data)
    #         start_time = time.time()
    #         splits=nodes.splitTensor(input, 4)
    #         elapsed_time = time.time() - start_time
    #         print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    #         print(splits)
