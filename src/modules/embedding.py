# coding=utf-8


import tensorflow as tf


class Embedding:
    def __init__(self, args):
        super().__init__()
        self.args = args

    @staticmethod
    def set_(sess, value):
        with tf.variable_scope('embedding', reuse=True):
            embedding_matrix = tf.get_variable('embedding_matrix')
        embedding_matrix_input = tf.placeholder(embedding_matrix.dtype, embedding_matrix.get_shape())
        sess.run(embedding_matrix.assign(embedding_matrix_input),
                 feed_dict={
                     embedding_matrix_input: value,
                 })

    def __call__(self, x, dropout_keep_prob):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable('embedding_matrix', dtype=tf.float32, trainable=False,
                                               shape=(self.args.num_vocab, self.args.embedding_dim))
            x = tf.nn.embedding_lookup(embedding_matrix, x)
            x = tf.nn.dropout(x, dropout_keep_prob)
            return x
