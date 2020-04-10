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
        self.tensor_one = tf.constant(1)

    # 将a,b拼接到一个tensor中,且第一个block减少对a(query)的重复计算
    def __call__(self, a, b, mask_a, mask_b, dropout_keep_prob, batchsize_a, batchsize_b):
        # 将a,b合并为一个tensor，起到并行计算的效果
        c=tf.concat([a,b], axis=0)
        mask_c = tf.concat([mask_a, mask_b], axis=0)
        c = self.embedding(c, dropout_keep_prob)
        # infer_flag为True代表为线上推理模式，即对于一个text1(a) ，预测它与batchsize_b个text2(b)之间的相似度
        infer_flag = tf.equal(batchsize_a, self.tensor_one) & (batchsize_b>self.tensor_one)
        # self.infer_flag = infer_flag
        # 第一个block要特殊处理
        with tf.variable_scope('first_block', reuse=tf.AUTO_REUSE):
            # 并行编码
            c_enc = self.first_block['encoder'](c, mask_c, dropout_keep_prob)
            c = tf.concat([c, c_enc], axis=-1)
            # 将a,b从c中切取出来，为交互模块做准备
            b = tf.slice(c, [batchsize_a, 0, 0], [-1, -1, -1]) # c[batchsize_a:]
            a = tf.slice(c, [0, 0, 0], [batchsize_a, -1, -1]) # c[:batchsize_a]
            # 实际使用时，一个query对应多个候选文本，节约计算量
            a = tf.cond(pred=infer_flag,
                        true_fn=lambda:tf.tile(a, multiples=tf.stack([batchsize_b, self.tensor_one, self.tensor_one])), # 将a的向量复制扩展
                        false_fn=lambda:tf.identity(a))
            # a,b的交互
            align_a, align_b = self.first_block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
            # 并行编码
            align_c = tf.concat([align_a, align_b], axis=0)
            # 如果是线上推理模式，因为原来的c不对，要重新合并c，然后并行编码
            c = tf.cond(pred=infer_flag,
                        true_fn=lambda:tf.concat([a, b], axis=0), false_fn=lambda:tf.identity(c))
            c = self.first_block['fusion'](c, align_c, dropout_keep_prob)
        if len(self.blocks)>0: #  如果布置一个block
            res_c = c
            # 如果是线上推理模式，对mask_a进行复制扩展
            mask_c = tf.cond(pred=infer_flag,
                             true_fn=lambda: tf.concat([tf.tile(mask_a, multiples=tf.stack([batchsize_b, self.tensor_one, self.tensor_one])), mask_b], axis=0),
                             false_fn=lambda:tf.identity(mask_c))
            for i, block in enumerate(self.blocks, start=1):
                with tf.variable_scope('block-{}'.format(i), reuse=tf.AUTO_REUSE):
                    if i > 0:
                        c = self.connection(c, res_c, i)
                        res_c = c
                    c_enc = block['encoder'](c, mask_c, dropout_keep_prob)
                    c = tf.concat([c, c_enc], axis=-1)
                    # 将a,b从c中切取出来，为交互模块做准备
                    b = c[batchsize_b:]
                    a = c[:batchsize_b]  # 在first_block中已经对a进行了tf.tile操作
                    # a,b的交互
                    align_a, align_b = block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
                    align_c = tf.concat([align_a, align_b], axis=0)
                    # 并行编码
                    c = block['fusion'](c, align_c, dropout_keep_prob)
        c = self.pooling(c, mask_c)
        a = c[:batchsize_b]
        b = c[batchsize_b:]
        return self.prediction(a, b, dropout_keep_prob)

    # 将a,b拼接到一个tensor中，与原代码接近。第一个block中会对text1进行重复的编码，线上推理延迟高
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

