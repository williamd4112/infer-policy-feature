import tensorflow as tf
import numpy as np
from collections import deque

def get_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

class MovingAverageEpisodeRewardCounter(object):
    def __init__(self, length=1000):
        self.length = length
        self.buff = deque(maxlen=length)

    def add(self, reward):
        self.buff.append(reward)

    def __call__(self):
        np_arr = np.asarray(self.buff)
        return {'min': np_arr.min(), 'mean': np_arr.mean(), 'max': np_arr.max()}
