import numpy as np
import tensorflow as tf

from tf_ops import *

class Model(object):
    def __init__(self, name, reuse):
        self.name = name
        self.reuse = reuse

        self._init_inputs()
        self._init_model()
        self._init_outputs()

    def get_inputs(self):
        return self.inputs
    
    def get_outputs(self):
        return self.outputs

class DeepQNetwork(Model):
    def __init__(self, name, reuse, state_shape, num_action):
        self.state_shape = state_shape
        self.num_action = num_action
        super(DeepQNetwork, self).__init__(name, reuse)
    
    def _init_inputs(self):
        self.inputs = { 'state': tf.placeholder(dtype=tf.uint8, shape=[None,] + self.state_shape, name='state'),
                        'action': tf.placeholder(dtype=tf.int32, shape=[None,], name='action'),
                        'reward': tf.placeholder(dtype=tf.float32, shape=[None,], name='reward'),
                        'next_state': tf.placeholder(dtype=tf.uint8, shape=[None,] + self.state_shape, name='next_state'),
                        'done': tf.placeholder(dtype=tf.bool, shape=[None,], name='done')}

    def _init_model(self): 
        with tf.variable_scope(self.name, reuse=self.reuse):
            state = self.inputs['state']
            next_state = self.inputs['next_state']
            with tf.variable_scope('policy', reuse=self.reuse):
                self.policy_q = self._build_q_network(state)
            with tf.variable_scope('target', reuse=self.reuse):
                self.target_q = self._build_q_network(next_state)

    def _init_outputs(self):
        self.outputs = { 'policy_q': self.policy_q,
                         'target_q': self.target_q }
        
    def _build_q_network(self, state):
        l = state
        l = tf.cast(l, tf.float32)
        l = l / 255.0
        l = Conv2D(l, [8, 8], 32, 4, 'VALID', 'conv0', reuse=self.reuse)
        l = ReLu(l, 'relu0') 
        l = Conv2D(l, [4, 4], 64, 2, 'VALID', 'conv1', reuse=self.reuse)
        l = ReLu(l, 'relu1')
        l = Conv2D(l, [3, 3], 64, 1, 'VALID', 'conv2', reuse=self.reuse)
        l = ReLu(l, 'relu2')
        l = FC(l, 512, 'fc0')
        l = ReLu(l, 'relu3')
        l = FC(l, self.num_action, 'fc1')
        return l

    def get_policy_network_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/policy' % self.name)
            
    def get_target_network_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/target' % self.name)
       

            
        
       
 
