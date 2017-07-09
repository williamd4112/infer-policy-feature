import numpy as np
import tensorflow as tf

from tf_ops import *
from model import *
from learn import *

class Agent(object):
    def __init__(self, model):
        self.model = model

    def act(self, sess, state):
        return self._act(sess, state)

class DeepQAgent(Agent):
    def _act(self, sess, state):
        input = self.model.get_inputs()['state']
        output = self.model.get_outputs()['policy_q']
        q = sess.run([output], feed_dict={input: state})
        return q[0].argmax(axis=1)
