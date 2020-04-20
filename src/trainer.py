# coding=utf-8
import os
import random
import json5
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from pprint import pformat
from .utils.loader import load_data
from .utils.logger import Logger
from .utils.params import validate_params
from .model import Model
from .interface import Interface
from curLine_file import curLine

class Trainer:
    def __init__(self, args):
        self.args = args
        self.log = Logger(self.args)

    def train(self, experiment_times):
        start_time = datetime.now()
        startTime = time.time()
        data_dir = self.args.data_dir
        train = load_data(data_dir, 'train')
        dev = load_data(data_dir, self.args.eval_file)
        self.log(f'train ({len(train)}) | {self.args.eval_file} ({len(dev)})')

        tf.reset_default_graph()

        with tf.Graph().as_default():
            break_flag = False
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)
            with sess.as_default():
                model, interface, states = self.build_model(sess)
                train_batches = interface.pre_process(train)
                dev_batches = interface.pre_process(dev, training=False)
                self.log('setup complete: {}s.'.format(str(datetime.now() - start_time).split(".")[0]))
                eval_per_updates = self.args.eval_per_updates \
                    if model.updates > self.args.eval_warmup_steps else self.args.eval_per_updates_warmup
                print(curLine(), "eval_per_updates=", eval_per_updates)
                for epoch in range(states['start_epoch'], self.args.epochs + 1):
                    if break_flag:
                        break
                    print(curLine(), "epoch=%d/%d, experiment_times=%d"%(epoch, self.args.epochs, experiment_times))
                    states['epoch'] = epoch
                    self.log.set_epoch(epoch)
                    batches = interface.shuffle_batch(train_batches)
                    for batch_id, batch in enumerate(batches):
                        stats = model.update(sess, batch)
                        self.log.update(stats)
                        if model.updates % eval_per_updates == 0 \
                                or (self.args.eval_epoch and batch_id + 1 == len(batches)):
                            score, dev_stats = model.evaluate(sess, dev_batches)
                            cost_minutes = (time.time()-startTime)/60.0
                            dev_stats["cost_time"] =cost_minutes
                            if score > states['best_eval']:
                                states['best_eval'], states['best_epoch'], states['best_step'] = \
                                    score, epoch, model.updates
                                if self.args.save:
                                    model.save(states, name=model.best_model_name)
                            self.log.log_eval(dev_stats)
                            if self.args.save_all:
                                model.save(states)
                                model.save(states, name='last')
                            if model.updates - states['best_step'] > self.args.early_stopping \
                                    and model.updates > self.args.min_steps:
                                print(curLine(), "Tolerance reached. Training is stopped early. %d > %d"
                                      % (model.updates - states['best_step'], self.args.early_stopping))
                                break_flag = True
                                break
                        if stats['loss'] > self.args.max_loss:
                            print(curLine(), "Loss exceeds tolerance. Unstable training is stopped early.")
                            break_flag = True
                            break
                            # raise EarlyStop('[Loss exceeds tolerance. Unstable training is stopped early.]')
                        if stats['lr'] < self.args.min_lr - 1e-6:
                            print(curLine(), "Learning rate has decayed below min_lr. Training is stopped early.")
                            break_flag = True
                            break
                    self.log.newline()
                self.log('Training complete.')

                self.log(f'best dev score {states["best_eval"]} at step {states["best_step"]} '
                         f'(epoch {states["best_epoch"]}).')
                self.log(f'best eval stats [{self.log.best_eval_str}]')
                training_time = str(datetime.now() - start_time).split('.')[0]
                self.log(f'Training time: {training_time}.')
        states['start_time'] = str(start_time).split('.')[0]
        states['training_time'] = training_time
        return states, self.log.best_eval

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

        # set initial states
        states['start_epoch'] = 1
        states['best_eval'] = 0.
        states['best_epoch'] = 0
        states['best_step'] = 0

        self.log(f'trainable params: {model.num_parameters():,d}')
        self.log(f'trainable params (exclude embeddings): {model.num_parameters(exclude_embed=True):,d}')
        validate_params(self.args)
        with open(os.path.join(self.args.summary_dir, 'args.json5'), 'w') as f:
            args = {k: v for k, v in vars(self.args).items() if not k.startswith('_')}
            json5.dump(args, f, indent=2)
        self.log(pformat(vars(self.args), indent=2, width=120))
        return model, interface, states


class EarlyStop(Exception):
    pass
