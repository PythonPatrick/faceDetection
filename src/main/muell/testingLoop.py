import tensorflow as tf
import numpy as np


def for_loop(condition, modifier, body_op, idx=0):
    idx = tf.convert_to_tensor(idx)

    def body(i):
        with tf.control_dependencies([body_op(i)]):
            return [modifier(i)]

    # do the loop:
    loop = tf.while_loop(condition, body, [idx])
    return loop


x = np.arange(10)

data = tf.data.Dataset.from_tensor_slices(x)
data = data.repeat()

iterator = data.make_initializable_iterator()
smpl = iterator.get_next()

loop = for_loop(
    condition=lambda i: tf.less(i, 5),
    modifier=lambda i: tf.add(i, 1),
    body_op=lambda i: tf.print("This is sample:", [iterator.get_next()])
)

sess = tf.InteractiveSession()
sess.run(iterator.initializer)
sess.run(loop)