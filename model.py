import numpy as np
import tensorflow as tf

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
        
       
 
