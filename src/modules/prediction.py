# coding=utf-8


import tensorflow as tf
from functools import partial
from src.utils.registry import register
from . import dense

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Prediction:
    def __init__(self, args):
        self.args = args

    def _features(self, a, b):  # simple mode
        return tf.concat([a, b], axis=-1)

    def __call__(self, a, b, dropout_keep_prob, name='prediction'):
        x = self._features(a, b)
        with tf.variable_scope(name):
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.hidden_size, activation=tf.nn.relu, name='dense_1')
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.num_classes, activation=None, name='dense_2')
            return x


@register('full')
class AdvancedPrediction(Prediction):
    def _features(self, a, b):
        return tf.concat([a, b, a * b, a - b], axis=-1) # TODO  really need   a - b  ？？


@register('symmetric')
class SymmetricPrediction(Prediction):
    def _features(self, a, b):
        return tf.concat([a, b, a * b, tf.abs(a - b)], axis=-1)
