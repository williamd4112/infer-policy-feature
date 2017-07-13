import numpy as np
import tensorflow as tf
import random

from tf_ops import *
from model import *
from learn import *

class Agent(object):
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model

    def act(self, state):
        return self._act(state)

class DeepQAgent(Agent):
    def __init__(self, sess, model, num_action, eps, decay, decay_period, min_eps):
        super(DeepQAgent, self).__init__(sess, model)
        self.num_action = num_action
        self.eps = eps
        self.decay_period = decay_period
        self.decay = decay
        self.min_eps = min_eps
        self.state = self.model.get_inputs()['state']
        self.q = self.model.get_outputs()['policy_q']
        self.timestep = 0
   
    def _act(self, state):
        self.timestep += 1
        if random.random() < self.eps:
            if self.eps > self.min_eps and self.timestep % self.decay_period == 0:
                self.eps = self.eps * self.decay
            return [random.randint(0, self.num_action - 1)]
        q = self.sess.run([self.q], feed_dict={self.state: state})
        return q[0].argmax(axis=1).astype(np.int32)
