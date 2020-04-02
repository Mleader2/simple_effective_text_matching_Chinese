# coding=utf-8


import tensorflow as tf


def pooling(x, mask):
    return tf.reduce_max(mask * x + (1. - mask) * tf.float32.min, axis=1)
