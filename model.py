import numpy as np
import tensorflow as tf

from tf_ops import *

class Model(object):
    def __init__(self, name, reuse, **kwargs):
        self.name = name
        self.reuse = reuse
        self.args = kwargs

        self._init_inputs()
        self._init_model()
        self._init_outputs()

    def get_inputs(self):
        return self.inputs
    
    def get_outputs(self):
        return self.outputs

    def get_args(self):
        return self.args

class PolicyStyleInferenceNetwork(object):
    def __init__(self):
        self.batchsize = 32
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batchsize, 26, 81])
        self.x_len = tf.placeholder(dtype=tf.int32, shape=[self.batchsize])
        cell = tf.contrib.rnn.LSTMCell(num_units=64, state_is_tuple=True)
        outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=self.x_len,
            inputs=self.x)
        self.y = outputs

class DeepQNetwork(object):      
    def _init_inputs(self):
        self.inputs = { 'state': tf.placeholder(dtype=tf.uint8, (None,) + self.args['state_shape'], name='state'),
                        'reward': tf.placeholder(dtype=tf.float32, (None,), name='reward'),
                        'action': tf.placeholder(dtype=tf.int32, (None,), name='action'),
                        'next_state': tf.placeholder(dtype=tf.uint8, (None,) + self.args['state_shape'], name='next_state'),
                        'done': tf.placeholder(dtype=tf.bool, (None,), name='done')}

    def _init_model(self):
        NUM_ACTION = self.args['num_action']
        
        with tf.variable_scope(self.name, reuse=self.reuse):
            state, reward, action, next_state, done = self.inputs
            with tf.variable_scope('policy', reuse=self.reuse):
                self.policy_q = self._build_q_network(state)
            with tf.variable_scope('target', reuse=self.reuse):
                self.target_q = self._build_q_network(next_state)

    def _init_outputs(self):
        self.outputs = { 'policy_q': self.policy_q,
                         'target_q': self.target_q }
        
    def _build_q_network(self, state):
        l = state
        l = tf.cast(l / 255.0, tf.float32)
        l = Conv2D(l, [8, 8], 32, 4, 'VALID', 'conv0', reuse=self.reuse)
        l = ReLu(l, 'relu0') 
        l = Conv2D(l, [4, 4], 64, 2, 'VALID', 'conv1', reuse=self.reuse)
        l = ReLu(l, 'relu1')
        l = Conv2D(l, [3, 3], 64, 1, 'VALID', 'conv2', reuse=self.reuse)
        l = ReLu(l, 'relu2')
        l = FC(l, 512, 'fc0')
        l = ReLu(l, 'relu3')
        l = FC(l, NUM_ACTION, 'fc1')
        return l
            
        

            
        
       
 
