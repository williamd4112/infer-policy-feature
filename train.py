import numpy as np
import tensorflow as tf
import argparse, logging
import cPickle as pickle
import os, sys

from model import PolicyStyleInferenceNetwork

def main(args):
    X = np.load(args.X)
    X = np.reshape(X, [X.shape[0], X.shape[1], 81])
    X_len = np.squeeze(np.load(args.X_len))
    T = np.load(args.T)
 
    model = PolicyStyleInferenceNetwork()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(model.y, feed_dict={model.x: X[:32], 
                                    model.x_len: X_len[:32],
                                    })

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--X', help='X', type=str, default='X_small.npy')
    parser.add_argument('--X_len', help='X_len', type=str, default='X_len_small.npy')
    parser.add_argument('--T', help='T', type=str, default='T_small.npy')
    args = parser.parse_args()
 
    main(args) 
