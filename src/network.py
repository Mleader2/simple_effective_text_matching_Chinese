# coding=utf-8
import tensorflow as tf
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import pooling
from .modules.prediction import registry as prediction
from curLine_file import curLine

class Network:
    def __init__(self, args):
        self.embedding = Embedding(args)
        self.first_block = {
            'encoder': Encoder(args),
            'alignment': alignment[args.alignment](args),
            'fusion': fusion[args.fusion](args),
        }
        self.blocks = [{
            'encoder': Encoder(args, enc_layers=1), # TODO  enc_layers=1
            'alignment': alignment[args.alignment](args),
            'fusion': fusion[args.fusion](args),
        } for _ in range(args.blocks-1)]
        self.connection = connection[args.connection]
        self.pooling = pooling
        self.prediction = prediction[args.prediction](args)

    # 将a,b拼接到一个tensor中,且第一个block减少对a(query)的重复计算
    def __call__(self, a, b, mask_a, mask_b, dropout_keep_prob, batchsize_a, batchsize_b):
        c=tf.concat([a,b], axis=0)
        mask_c = tf.concat([mask_a, mask_b], axis=0)
        c = self.embedding(c, dropout_keep_prob)
        with tf.variable_scope('first_block', reuse=tf.AUTO_REUSE):
            c_enc = self.first_block['encoder'](c, mask_c, dropout_keep_prob)
            c = tf.concat([c, c_enc], axis=-1)
            b = c[batchsize_a:]
            a = c[:batchsize_a]
            # 实际使用时，一个query对应多个候选文本，节约计算量
            a = tf.cond(pred=tf.equal(batchsize_a,tf.constant(1)),
                        true_fn=lambda:tf.tile(a, multiples=tf.stack([batchsize_b, tf.constant(1), tf.constant(1)])),
                        false_fn=lambda:tf.identity(a))
            align_a, align_b = self.first_block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
            align_c = tf.concat([align_a, align_b], axis=0)
            self.c_full = tf.cond(pred=tf.equal(batchsize_a,tf.constant(1)),
                                  true_fn=lambda:tf.concat([a, b], axis=0), false_fn=lambda:tf.identity(c))
            c = self.first_block['fusion'](self.c_full, align_c, dropout_keep_prob)

        if len(self.blocks)>0:
            res_c = c
            mask_c = tf.cond(pred=tf.equal(batchsize_a,tf.constant(1)),
                             true_fn=lambda: tf.concat([tf.tile(mask_a, multiples=tf.stack([batchsize_b, tf.constant(1), tf.constant(1)])), mask_b], axis=0),
                             false_fn=lambda:tf.identity(mask_c))
            for i, block in enumerate(self.blocks, start=1):
                with tf.variable_scope('block-{}'.format(i), reuse=tf.AUTO_REUSE):
                    if i > 0:
                        c = self.connection(c, res_c, i)
                        res_c = c
                    c_enc = block['encoder'](c, mask_c, dropout_keep_prob)
                    c = tf.concat([c, c_enc], axis=-1)
                    b = c[batchsize_b:]  #  在first_block中已经对a进行了tf.tile操作
                    a = c[:batchsize_b]
                    align_a, align_b = block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
                    align_c = tf.concat([align_a, align_b], axis=0)
                    c = block['fusion'](c, align_c, dropout_keep_prob)
        c = self.pooling(c, mask_c)
        a = c[:batchsize_b]
        b = c[batchsize_b:]
        return self.prediction(a, b, dropout_keep_prob)

        # 将a,b拼接到一个tensor中
    # def __call__(self, a, b, mask_a, mask_b, dropout_keep_prob):
    #     c = tf.concat([a, b], axis=0)
    #     mask_c = tf.concat([mask_a, mask_b], axis=0)
    #     c = self.embedding(c, dropout_keep_prob)
    #     res_c = c
    #     for i, block in enumerate(self.blocks):
    #         with tf.variable_scope('block-{}'.format(i), reuse=tf.AUTO_REUSE):
    #             if i > 0:
    #                 c = self.connection(c, res_c, i)
    #                 res_c = c
    #             c_enc = block['encoder'](c, mask_c, dropout_keep_prob)
    #             c = tf.concat([c, c_enc], axis=-1)
    #             a, b = tf.split(c, axis=0, num_or_size_splits=2)
    #             align_a, align_b = block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
    #             align_c = tf.concat([align_a, align_b], axis=0)
    #             c = block['fusion'](c, align_c, dropout_keep_prob)
    #     c = self.pooling(c, mask_c)
    #     a, b = tf.split(c, axis=0, num_or_size_splits=2)
    #     return self.prediction(a, b, dropout_keep_prob)


    # def __call__(self, a, b, mask_a, mask_b, dropout_keep_prob):
    #     a = self.embedding(a, dropout_keep_prob)
    #     b = self.embedding(b, dropout_keep_prob)
    #     res_a, res_b = a, b
    #
    #     for i, block in enumerate(self.blocks):
    #         with tf.variable_scope('block-{}'.format(i), reuse=tf.AUTO_REUSE):
    #             if i > 0:
    #                 a = self.connection(a, res_a, i)
    #                 b = self.connection(b, res_b, i)
    #                 res_a, res_b = a, b
    #             a_enc = block['encoder'](a, mask_a, dropout_keep_prob)
    #             b_enc = block['encoder'](b, mask_b, dropout_keep_prob)
    #             a = tf.concat([a, a_enc], axis=-1)
    #             b = tf.concat([b, b_enc], axis=-1)
    #             align_a, align_b = block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
    #             a = block['fusion'](a, align_a, dropout_keep_prob)
    #             b = block['fusion'](b, align_b, dropout_keep_prob)
    #     a = self.pooling(a, mask_a)
    #     b = self.pooling(b, mask_b)
    #     return self.prediction(a, b, dropout_keep_prob)
