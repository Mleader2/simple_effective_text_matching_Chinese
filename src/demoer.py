# coding=utf-8
import os
import random
import json5
import time
import numpy as np
import tensorflow as tf
from pprint import pformat
from .utils.logger import Logger
from .utils.params import validate_params
from .model import Model
from .interface import Interface
from curLine_file import curLine

class Demoer:
    def __init__(self, args, checkpoint_dir):
        self.args = args
        self.log = Logger(self.args)
        tf.reset_default_graph()
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)
            with self.sess.as_default():
                self.model, self.interface, self.states = self.build_model(self.sess)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # self.model_path)
                if ckpt is None:
                    print(curLine(), "%s中没有模型" % checkpoint_dir)
                else:
                    print(curLine(), "restore from %s" % ckpt.model_checkpoint_path)
                    saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
                    saver.restore(self.sess, ckpt.model_checkpoint_path)

                # print(curLine(), "checkpoint_file:", checkpoint_file)
                # self.model.saver.restore(self.sess, checkpoint_file)
                                #          os.path.join(self.args.summary_dir, self.prefix),
                                # global_step=self.updates)

    def serve(self, dev, batch_size=30):
        self.log(f'eval {self.args.eval_file} ({len(dev)})')
        dev_batches = self.interface.pre_process(dev, training=False, batch_size=batch_size)
        predictions = []
        probabilities = []
        start_time = time.time()
        with self.sess.as_default():
            for batch in dev_batches:
                feed_dict = self.model.process_data(batch, training=False)
                pred, prob = self.sess.run(
                    [self.model.pred, self.model.prob],
                    feed_dict=feed_dict)
                predictions.extend(pred.tolist())
                probabilities.extend(prob.tolist())
        inference_time = (time.time() - start_time) * 1000.0
        return predictions, probabilities, inference_time

    def build_model(self, sess):
        states = {}
        interface = Interface(self.args, self.log)
        self.log(f'#classes: {self.args.num_classes}; #vocab: {self.args.num_vocab}')
        if self.args.seed:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            tf.set_random_seed(self.args.seed)

        model = Model(self.args, sess)
        sess.run(tf.global_variables_initializer())
        embeddings = interface.load_embeddings()
        model.set_embeddings(sess, embeddings)

        self.log(f'trainable params: {model.num_parameters():,d}')
        self.log(f'trainable params (exclude embeddings): {model.num_parameters(exclude_embed=True):,d}')
        validate_params(self.args)
        file = os.path.join(self.args.summary_dir, 'args.json5')
        print(curLine(), "save to %s" % file)
        with open(file, 'w') as f:
            args = {k: v for k, v in vars(self.args).items() if not k.startswith('_')}
            json5.dump(args, f, indent=2)
        self.log(pformat(vars(self.args), indent=2, width=120))
        return model, interface, states