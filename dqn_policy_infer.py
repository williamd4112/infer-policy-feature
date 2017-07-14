import numpy as np
import tensorflow as tf

from model import DeepQNetwork
from learn import DeepQLearner
from agent import DeepQAgent
from dqn import DeepQReplayMemory
from tf_ops import *

from data import (FrameStateBuilder, GrayscaleFrameStateBuilder, \
        ResizeFrameStateBuilder, StackedFrameStateBuilder, 
        NamedReplayMemory, StateBuilderProxy)

from util import load_model

class DeepPolicyInferQReplayMemory(NamedReplayMemory):
    def __init__(self, model, capacity=1000000):
        self.model = model
        super(DeepPolicyInferQReplayMemory, self).__init__(capacity=capacity, names=[ 
                                                                self.model.get_inputs()['state'],
                                                                self.model.get_inputs()['action'],
                                                                self.model.get_inputs()['reward'],
                                                                self.model.get_inputs()['next_state'],
                                                                self.model.get_inputs()['done'],
                                                                self.model.get_inputs()['opponent_action']])
class DeepPolicyInferQNetwork(DeepQNetwork): 
    def _init_inputs(self):
        super(DeepPolicyInferQNetwork, self)._init_inputs()
        self.inputs['opponent_action'] = tf.placeholder(dtype=tf.int32, shape=[None,], name='opponent_action')

    def _init_model(self): 
        with tf.variable_scope(self.name, reuse=self.reuse):
            state = tf.cast(self.inputs['state'], tf.float32) / 255.0
            next_state = tf.cast(self.inputs['next_state'], tf.float32) / 255.0            
        
            with tf.variable_scope('policy', reuse=self.reuse):
                with tf.variable_scope('opponent', reuse=self.reuse):
                    opponent_feature, self.opponent_action_prediction_logit = self._build_opponent_feature(state)
                self.policy_q = self._build_q_network(state, opponent_feature)

            if self.with_target_network:
                with tf.variable_scope('target', reuse=self.reuse):
                    with tf.variable_scope('opponent', reuse=self.reuse):
                        opponent_feature, _ = self._build_opponent_feature(next_state)
                    self.target_q = self._build_q_network(next_state, opponent_feature)

    def _init_outputs(self):
        super(DeepPolicyInferQNetwork, self)._init_outputs()
        self.outputs['opponent_action_prediction_logit'] = self.opponent_action_prediction_logit
        
    def _build_opponent_feature(self, state):
        l = state
        l = Conv2D(l, [6, 6], 64, 2, 'VALID', 'conv0')
        l = ReLu(l, 'relu0')
        l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv1')
        l = ReLu(l, 'relu1')
        l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv2')
        l = ReLu(l, 'relu2')
        l = FC(l, 1024, 'fc0')
        l = ReLu(l, 'relu3')
        h = FC(l, 512, 'fc1', initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        y = FC(h, self.num_action, 'fc2')
        return h, y
        
    def _build_q_network(self, state, opponent_feature):
        l = state
        l = Conv2D(l, [8, 8], 32, 4, 'VALID', 'conv0', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu0') 
        l = Conv2D(l, [4, 4], 64, 2, 'VALID', 'conv1', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu1')
        l = Conv2D(l, [3, 3], 64, 1, 'VALID', 'conv2', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu2')
        l = FC(l, 512, 'fc0')
        l = Concat([l, opponent_feature], axis=1)
        l = LeakyReLu(l, 0.01, 'relu3')
        l = FC(l, self.num_action, 'fc1')
        return l

class DeepPolicyInferQLearner(DeepQLearner): 
    def _build_optimize_op(self):
        # Model arguments
        num_action = self.model.num_action

        # Inputs
        state = self.model.get_inputs()['state']
        action = self.model.get_inputs()['action']
        reward = self.model.get_inputs()['reward']
        next_state = self.model.get_inputs()['next_state']
        done = self.model.get_inputs()['done']
        opponent_action = self.model.get_inputs()['opponent_action']

        # Outputs
        policy_q = self.model.get_outputs()['policy_q']
        target_q = self.model.get_outputs()['target_q']
        opponent_action_logit = self.model.get_outputs()['opponent_action_prediction_logit']

        # Bellmen loss
        action_one_hot = tf.one_hot(action, num_action, 1.0, 0.0)
        pred = tf.reduce_sum(policy_q * action_one_hot, 1)

        target_q_max = tf.reduce_max(target_q, 1)
        target = reward + (1.0 - tf.cast(done, tf.float32)) * self.gamma * tf.stop_gradient(target_q_max)
        q_loss = HuberLoss(target - pred)

        # Auxiliary loss
        opponent_action_one_hot = tf.one_hot(opponent_action, num_action, 1.0, 0.0)
        opponent_action_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=opponent_action_one_hot, 
                                                                                logits=opponent_action_logit)
        
        loss = tf.reduce_mean(q_loss + opponent_action_cross_entropy_loss, name='loss')
        self.loss = loss

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-3)
        self.grads_vars = self.optimizer.compute_gradients(loss)
        self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0.0), trainable=False) 
