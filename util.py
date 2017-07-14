import tensorflow as tf
import numpy as np
from collections import deque

def get_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def load_model(sess, path):
    saver = tf.train.import_meta_graph(path)
    # TODO: modify directory of checkpoint
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    return saver

class MovingAverageEpisodeRewardCounter(object):
    def __init__(self):
        self.buff = deque()

    def add(self, reward):
        self.buff.append(reward)

    def __call__(self):
        np_arr = np.asarray(self.buff)
        return {'min': np_arr.min(), 'mean': np_arr.mean(), 'max': np_arr.max()}
