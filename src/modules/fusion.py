# coding=utf-8


import tensorflow as tf
from functools import partial
from src.utils.registry import register
from . import dense

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Fusion:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def __call__(self, x, align, _):
        with tf.variable_scope('align', reuse=tf.AUTO_REUSE):
            return dense(tf.concat([x, align], axis=-1), self.args.hidden_size,
                         activation=tf.nn.relu)


@register('full')
class FullFusion:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def __call__(self, x, align, dropout_keep_prob):
        with tf.variable_scope('align', reuse=tf.AUTO_REUSE):
            x = tf.concat([
                dense(tf.concat([x, align], axis=-1), self.args.hidden_size, activation=tf.nn.relu, name='orig'),
                dense(tf.concat([x, x - align], axis=-1), self.args.hidden_size, activation=tf.nn.relu, name='sub'),
                dense(tf.concat([x, x * align], axis=-1), self.args.hidden_size, activation=tf.nn.relu, name='mul'),
            ], axis=-1)
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.hidden_size, activation=tf.nn.relu, name="proj")
            return x
