# coding=utf-8


import tensorflow as tf
from . import conv1d


class Encoder:
    def __init__(self, args, enc_layers=None):
        self.args = args
        self.enc_layers = enc_layers
        if self.enc_layers is None:
            self.enc_layers = self.args.enc_layers

    def __call__(self, x, mask, dropout_keep_prob, name='encoder'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i in range(self.enc_layers):
                x = mask * x
                if i > 0:
                    x = tf.nn.dropout(x, dropout_keep_prob)
                x = conv1d(x, self.args.hidden_size, kernel_size=self.args.kernel_size, activation=tf.nn.relu,
                           name=f'cnn_{i}')
            x = tf.nn.dropout(x, dropout_keep_prob)
            return x
