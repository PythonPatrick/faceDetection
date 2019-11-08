import tensorflow as tf
from main.utils.tensorflowUtils.tflow import Utils as utils
import numpy as np


class utilsTest(tf.compat.v1.test.TestCase):

    def testReplaceValue(self):
        with self.session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            ins = tf.convert_to_tensor(np.array([[4.0, 43.0, 45.0], [2.0, 22.0, 6664.0], [-4543.0, 0.0, 43.0]]), tf.float64)
            ind = tf.convert_to_tensor([[1, 1]], tf.int64)
            vals = tf.convert_to_tensor([45], tf.float64)
            outs = utils.replaceValues(ins, ind, vals)
            self.assertAllEqual(outs.eval(), np.array([[4.0, 43.0, 45.0], [2.0, 45.0, 6664.0], [-4543.0, 0.0, 43.0]]))

    def testSparseSlice(self):
        with self.session():
            indices = tf.constant([[0, 0], [1, 0], [2, 0], [2, 1], [2, 5]])
            values = tf.constant([1.0, 1.0, 0.3, 0.7, 0.8], dtype=tf.float32)
            needed_row_ids = tf.constant([0, 2])
            ind, val = utils.sparse_slice(indices, values, needed_row_ids)
            self.assertAllEqual(ind.eval(), [[0, 0], [2, 0], [2, 1], [2, 5]])
            self.assertAllEqual(val.eval(), np.array([1.0, 0.3, 0.7, 0.8], np.float32))


if __name__ == '__main__':
    tf.test.main()