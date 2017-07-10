import numpy as np
import tensorflow as tf

from tf_ops import *
from model import *
from learn import *

class PeriodicCallback(object):
    def __init__(self, func, period):
        self.func = func
        self.period = period
        self.time = 0
    
    def __call__(self, sess):
        if self.time % self.period == 0:
            self.func(sess)
            self.time = 0
        self.time += 1

class Trainer(object):
    def train(self, sess, batch):
        self._train(sess, batch)
    
class FeedDictTrainer(Trainer):
    def __init__(self, learner, callbacks=[]):
        self.learner = learner
        self.callbacks = callbacks

    def _train(self, sess, batch):
        train_op = self.learner.get_train_op()
        global_step = self.learner.get_global_step()
        _, global_step_val = sess.run([train_op, global_step], feed_dict=batch)

        for callback in self.callbacks:
            callback(sess)

        return global_step_val
