import numpy as np
import tensorflow as tf

from tf_ops import *
from model import *

class DeepQLearner(object):
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        
        self._build_train_op()

    def _build_train_op(self):
        # Model arguments
        num_action = self.model.num_action

        # Inputs
        state = self.model.get_inputs()['state']
        action = self.model.get_inputs()['action']
        reward = self.model.get_inputs()['reward']
        next_state = self.model.get_inputs()['next_state']
        done = self.model.get_inputs()['done']

        # Outputs
        policy_q = self.model.get_outputs()['policy_q']
        target_q = self.model.get_outputs()['target_q']

        # Loss
        action_one_hot = tf.one_hot(action, num_action, 1.0, 0.0)
        pred = tf.reduce_sum(policy_q * action_one_hot, 1)

        target_q_max = tf.reduce_max(target_q, 1)
        target = reward + (1.0 - tf.cast(done, tf.float32)) * self.gamma * tf.stop_gradient(target_q_max)
        loss = tf.reduce_mean(HuberLoss(target - pred), name='loss')

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-3)

        self.train_op = optimizer.minimize(loss)
    
    def get_train_op(self):
        return self.train_op     
 
