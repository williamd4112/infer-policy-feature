import numpy as np
import tensorflow as tf

from tf_ops import *
from model import *
from learn import *

class Trainer(object):
    def train(self, sess, batch):
        self._train(sess, batch)

class FeedDictTrainer(Trainer):
    def __init__(self, learner):
        self.learner = learner

    def _train(self, sess, batch):
        train_op = self.learner.get_train_op()

        sess.run([train_op], feed_dict=batch)
