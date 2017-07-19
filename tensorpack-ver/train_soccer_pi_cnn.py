#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train_soccer.py
# Author: zhangwei hong <williamd4112@hotmail.com>

import numpy as np

import os
import sys
import re
import time
import random
import argparse
import subprocess
import multiprocessing
import threading
from collections import deque

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.RL import *
import tensorflow as tf

from DQNPIModel import Model as DQNModel
import common
from common import play_model, Evaluator, eval_model_multithread
from soccer_env import SoccerPlayer
from augment_expreplay import AugmentExpReplay

from tensorpack.tfutils import symbolic_functions as symbf

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = 5e4
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10  # each epoch is 100k played frames
EVAL_EPISODE = 50

NUM_ACTIONS = None
METHOD = None


def get_player(viz=False, train=False):
    pl = SoccerPlayer(image_shape=IMAGE_SIZE[::-1], viz=viz, frame_skip=ACTION_REPEAT)
    if not train:
        # create a new axis to stack history on
        pl = MapPlayerState(pl, lambda im: im[:, :, np.newaxis])
        # in training, history is taken care of in expreplay buffer
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)

        pl = PreventStuckPlayer(pl, 30, 1)
    #pl = LimitLengthPlayer(pl, 30000)
    return pl


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0

        with tf.variable_scope('q'):
            with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True), \
                    argscope(LeakyReLU, alpha=0.01):
                q_l = (LinearWrap(image)
                     # Nature architecture
                     .Conv2D('conv0', out_channel=32, kernel_shape=8, stride=4)
                     .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                     .Conv2D('conv2', out_channel=64, kernel_shape=3)

                     # architecture used for the figure in the README, slower but takes fewer iterations to converge
                     # .Conv2D('conv0', out_channel=32, kernel_shape=5)
                     # .MaxPooling('pool0', 2)
                     # .Conv2D('conv1', out_channel=32, kernel_shape=5)
                     # .MaxPooling('pool1', 2)
                     # .Conv2D('conv2', out_channel=64, kernel_shape=4)
                     # .MaxPooling('pool2', 2)
                     # .Conv2D('conv3', out_channel=64, kernel_shape=3)
                    
                     .FullyConnected('fc0', 512, nl=LeakyReLU)())

        with tf.variable_scope('pi'):
            with argscope(Conv2D, nl=tf.nn.relu):
                    pi_l = Conv2D('conv0', image, out_channel=64, kernel_shape=6, stride=2, padding='VALID')
                    pi_l = Conv2D('conv1', pi_l, out_channel=64, kernel_shape=6, stride=2, padding='SAME')
                    pi_l = Conv2D('conv2', pi_l, out_channel=64, kernel_shape=6, stride=2, padding='SAME')
            pi_l = FullyConnected('fc0', pi_l, 1024, nl=tf.nn.relu)
            pi_h = FullyConnected('fc1', pi_l, 512) 
            pi_y = FullyConnected('fc2', pi_h, self.num_actions, nl=tf.identity)

        l = tf.concat([q_l, pi_h], axis=1)
        
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))

        return tf.identity(Q, name='Qvalue'), tf.identity(pi_y, name='Pivalue')

def get_config():
    M = Model()
    expreplay = AugmentExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return TrainConfig(
        dataflow=expreplay,
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=10000 // UPDATE_FREQ),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=M,
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
        # run the simulator on a separate GPU if available
        predict_tower=[1] if get_nr_gpu() > 1 else [0],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='DQN')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    METHOD = args.algo

    # set num_actions
    NUM_ACTIONS = SoccerPlayer().get_action_space().num_actions()

    if args.task != 'train':
        assert args.load is not None
        cfg = PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue'])
        if args.task == 'play':
            play_model(cfg, get_player(viz=1))
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE, get_player)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN-{}'.format(
                os.path.basename('soccer').split('.')[0])))
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
