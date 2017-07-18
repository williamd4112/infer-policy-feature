import numpy as np

import os
import sys
import re
import time
import random
import argparse
import subprocess
import multiprocessing
import threading
from collections import deque

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.RL import *
import tensorflow as tf


class PiTrainer(SimpleTrainer):
    def run_step(self):
        self.hooked_sess.run(self.train_q_op)
        self.hooked_sess.run(self.train_pi_op)

    def _setup(self):
        self._setup_input_source(self._input_source)
        with TowerContext('', is_training=True):
            self.model.build_graph(self._input_source)
            q_cost_grad, pi_cost_grad = self.model.get_cost_and_grad()

        opt = self.model.get_optimizer()
      
        q_cost, q_grad = q_cost_grad
        pi_cost, pi_grad = pi_cost_grad

        self.train_q_op = opt.apply_gradients(q_grad, name='min_op')
        self.train_pi_op = opt.apply_gradients(pi_grad, name='min_op')

def QueueInputPiTrainer(config, input_queue=None):
    if config.data is not None:
        assert isinstance(config.data, QueueInput), config.data
    else:
        config.data = QueueInput(config.dataflow, input_queue)
    config.dataflow = None

    return PiTrainer(config) 
