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

class LinearInterpolation(object):
    def __init__(self, start, end):
        '''
        start: (step, val)
        end: (step, val)
        '''
        self.start = start
        self.end = end
  
    def __call__(self, t):
        return np.interp(t, [self.start[0], self.end[0]], [self.start[1], self.end[1]])

class MovingAverageEpisodeRewardCounter(object):
    def __init__(self):
        self.buff = deque()

    def add(self, reward):
        self.buff.append(reward)

    def __call__(self):
        if len(self.buff) == 0:
            return '[Max = %f, Min = %f, Mean = %f]' % (0.0, 0.0, 0.0)
        np_arr = np.asarray(self.buff)
        return  '[Max = %f, Min = %f, Mean = %f]' % (np_arr.max(), np_arr.min(), np_arr.mean())
