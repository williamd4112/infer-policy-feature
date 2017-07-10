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
        self._build_update_target_network_op()
        self._build_optimize_op()
    
    def _build_update_target_network_op(self):
        policy_vars = self.model.get_policy_network_vars()
        target_vars = self.model.get_target_network_vars()
        assign_ops = [tf.assign(t_v, p_v) for p_v, t_v in zip(policy_vars, target_vars)]
        self.update_target_network_op = tf.group(*assign_ops)

    def _build_optimize_op(self):
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
        self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0.0), trainable=False)
        self.train_op = optimizer.minimize(loss, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op
    
    def get_update_target_network_op(self):
        return self.update_target_network_op 

    def get_global_step(self):
        return self.global_step
