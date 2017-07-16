import numpy as np
import tensorflow as tf
import logger

from tf_ops import *

class Model(object):
    def __init__(self, name, reuse):
        self.name = name
        self.reuse = reuse

        self._init_inputs()
        self._init_model()
        self._init_outputs()
        self._summary()

    def _summary(self):
        raise NotImplemented()

    def get_inputs(self):
        return self.inputs
    
    def get_outputs(self):
        return self.outputs

class DeepQNetwork(Model):
    def __init__(self, name, reuse, state_shape, num_action, with_target_network=True):
        self.state_shape = state_shape
        self.num_action = num_action
        self.with_target_network = with_target_network
        super(DeepQNetwork, self).__init__(name, reuse)
    
    def _summary(self):
        for p, t in zip(self.get_policy_network_vars(), self.get_target_network_vars()):
            logger.info(p)
            logger.info(t)
            
    def _init_inputs(self):
        self.inputs = { 'state': tf.placeholder(dtype=tf.uint8, shape=[None,] + self.state_shape, name='state'),
                        'action': tf.placeholder(dtype=tf.int32, shape=[None,], name='action'),
                        'reward': tf.placeholder(dtype=tf.float32, shape=[None,], name='reward'),
                        'next_state': tf.placeholder(dtype=tf.uint8, shape=[None,] + self.state_shape, name='next_state'),
                        'done': tf.placeholder(dtype=tf.bool, shape=[None,], name='done')}

    def _init_model(self): 
        with tf.variable_scope(self.name, reuse=self.reuse):
            state = tf.cast(self.inputs['state'], tf.float32) / 255.0
            next_state = tf.cast(self.inputs['next_state'], tf.float32) / 255.0
            with tf.variable_scope('policy', reuse=self.reuse):
                self.policy_q = self._build_q_network(state)
            if self.with_target_network:
                with tf.variable_scope('target', reuse=self.reuse):
                    self.target_q = self._build_q_network(next_state)

    def _init_outputs(self):
        if self.with_target_network:
            self.outputs = { 'policy_q': self.policy_q,
                             'target_q': self.target_q }
        else:
            self.outputs = { 'policy_q': self.policy_q }
        
    def _build_q_network(self, state):
        l = state
        l = Conv2D(l, [8, 8], 32, 4, 'VALID', 'conv0', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu0') 
        l = Conv2D(l, [4, 4], 64, 2, 'VALID', 'conv1', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu1')
        l = Conv2D(l, [3, 3], 64, 1, 'VALID', 'conv2', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu2')
        l = FC(l, 512, 'fc0')
        l = LeakyReLu(l, 0.01, 'relu3')
        l = FC(l, self.num_action, 'fc1')
        return l

    def get_policy_network_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/policy' % self.name)
            
    def get_target_network_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/target' % self.name)           
