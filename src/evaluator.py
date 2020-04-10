# coding=utf-8
import os
from pprint import pprint
import tensorflow as tf
from .model import Model
from .interface import Interface
from .utils.loader import load_data


class Evaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file

    def evaluate(self):
        data = load_data(*os.path.split(self.data_file))

        tf.reset_default_graph()
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)
            with sess.as_default():
                model, checkpoint = Model.load(sess, self.model_path)
                args = checkpoint['args']
                interface = Interface(args)
                batches = interface.pre_process(data, training=False)
                _, stats = model.evaluate(sess, batches)
                pprint(stats)
