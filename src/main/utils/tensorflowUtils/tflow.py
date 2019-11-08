import tensorflow as tf


class Utils:

    @staticmethod
    def replaceValues(inputs, indices, values):
        """
        Given an input tensor, the values are replaced in the given indices.

        :param inputs:
        :param indices:
        :param values:
        :return:
        """
        inputs = tf.cast(inputs, tf.float64)
        indices = tf.cast(indices, tf.int64)
        values = tf.cast(values, tf.float64)
        # set elements in "indices" to 0's
        # one 0 for each element in "indices"
        maskValues = tf.tile([0.0], [tf.shape(indices)[0]])
        mask = tf.SparseTensor(indices, maskValues, tf.shape(inputs, out_type=tf.int64))
        # set values in coordinates in "indices" to 0's, leave everything else intact
        maskedInput = tf.multiply(inputs, tf.cast(tf.sparse.to_dense(mask, default_value=1.0), tf.float64))

        # replace elements in "indices" with "values"
        delta = tf.SparseTensor(indices, values, tf.shape(inputs, out_type=tf.int64))
        # add "values" to elements in "indices" (which are 0's so far)
        return tf.add(maskedInput, tf.sparse.to_dense(delta))

    @staticmethod
    def sparse_slice(indices, values, needed_row_ids):
        """

        :param indices:
        :param values:
        :param needed_row_ids:
        :return:
        """
        needed_row_ids = tf.reshape(needed_row_ids, [1, -1])
        num_rows = tf.shape(indices)[0]
        partitions = tf.cast(tf.reduce_any(tf.equal(tf.reshape(indices[:, 0], [-1, 1]), needed_row_ids), 1), tf.int32)
        rows_to_gather = tf.dynamic_partition(tf.range(num_rows), partitions, 2)[1]
        slice_indices = tf.gather(indices, rows_to_gather)
        slice_values = tf.gather(values, rows_to_gather)
        return slice_indices, slice_values
