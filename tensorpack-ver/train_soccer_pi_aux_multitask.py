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

from DQNPIModel_multitask import Model as DQNModel
import common
from common import play_model, Evaluator, eval_model_multithread
from soccer_env_multitask import SoccerPlayer
from augment_expreplay_multitask import AugmentExpReplay

from tensorpack.tfutils import symbolic_functions as symbf

BATCH_SIZE = None
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = None
ACTION_REPEAT = None   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = 5e4
STEPS_PER_EPOCH = 1000 // UPDATE_FREQ * 10  # each epoch is 100k played frames
EVAL_EPISODE = 50

NUM_ACTIONS = None
NUM_AGENTS = 3
METHOD = None
FIELD = None
AI_SKIP = None
LR = None
LAMB = None
USE_RNN = False
RNN_CELL = None

def get_player(viz=False, train=False):
    pl = SoccerPlayer(image_shape=IMAGE_SIZE[::-1], viz=viz, frame_skip=ACTION_REPEAT, field=FIELD, ai_frame_skip=AI_SKIP)
    if not train:
        # create a new axis to stack history on
        pl = MapPlayerState(pl, lambda im: im[:, :, np.newaxis])
        # in training, history is taken care of in expreplay buffer
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)

        pl = PreventStuckPlayer(pl, 30, 1)
    #pl = LimitLengthPlayer(pl, 30000)
    return pl

def get_rnn_cell():
    if RNN_CELL == 'gru':
        return tf.nn.rnn_cell.GRUCell(num_units=512)
    elif RNN_CELL == 'lstm':
        return tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True)
    else:
        assert 0

class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, NUM_AGENTS, GAMMA, LR, LAMB)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0

        if USE_RNN:
            self.batch_size = tf.shape(image)[0]
            image = tf.transpose(image, perm=[0, 3, 1, 2])
            image = tf.reshape(image, (self.batch_size * self.channel,) + self.image_shape + (1,))

        with tf.variable_scope('q'):
            with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True), \
                    argscope(LeakyReLU, alpha=0.01):
                h = (LinearWrap(image)
                     .Conv2D('conv0', out_channel=32, kernel_shape=8, stride=4)
                     .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                     .Conv2D('conv2', out_channel=64, kernel_shape=3)())                    
                q_l = FullyConnected('fc0-q', h, 512, nl=LeakyReLU)
                pi_l = FullyConnected('fc0-pi', h, 512, nl=LeakyReLU)

                if USE_RNN:
                    # q
                    q_l = tf.reshape(q_l, [self.batch_size, self.channel, 512])
                    q_l, _ = tf.nn.dynamic_rnn(inputs=q_l, 
                                cell=get_rnn_cell(), 
                                dtype=tf.float32, scope='rnn-q')
                    q_l = q_l[:, -1, :]
                    # pi
                    pi_l = tf.reshape(pi_l, [self.batch_size, self.channel, 512])
                    pi_l, _ = tf.nn.dynamic_rnn(inputs=pi_l, 
                                cell=get_rnn_cell(), 
                                dtype=tf.float32, scope='rnn-pi')
                    pi_l = pi_l[:, -1, :]
            pi_ys = []
            for i in range(self.num_agents):
                pi_ys.append(FullyConnected('fct-%d' % i, pi_l, self.num_actions, nl=tf.identity))

        l = tf.multiply(q_l, pi_l)
        
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))

        return tf.identity(Q, name='Qvalue'), [ tf.identity(pi_ys[i], name='Pivalue-%d' % i) for i in range(self.num_agents) ]

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
                                      [(200, 4e-4), (400, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (400, 0.1), (800, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=M,
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=10000,
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
    parser.add_argument('--cell', help='cell',
                        choices=['gru', 'lstm', None], default=None)
    parser.add_argument('--skip', help='act repeat', type=int, required=True)
    parser.add_argument('--ai_skip', help='ai act repeat', type=int, required=True)
    parser.add_argument('--field', help='field type', type=str, choices=['small', 'large'], required=True)
    parser.add_argument('--hist_len', help='hist len', type=int, required=True)
    parser.add_argument('--batch_size', help='batch size', type=int, required=True)
    parser.add_argument('--lr', help='lr', type=float, required=True)
    parser.add_argument('--lamb', help='lamb', type=float, required=True)
    parser.add_argument('--rnn', help='use_rnn', type=str, required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    METHOD = args.algo

    ACTION_REPEAT = args.skip
    FIELD = args.field
    FRAME_HISTORY = args.hist_len
    BATCH_SIZE = args.batch_size
    AI_SKIP = args.ai_skip
    LR = args.lr
    LAMB = args.lamb
    USE_RNN = bool(int(args.rnn))
    RNN_CELL = args.cell

    if USE_RNN:
        MODEL_NAME = 'DRQNPI-%s' % RNN_CELL
    else:
        MODEL_NAME = 'DQNPI'

    logger.info('USE_RNN = {}'.format(USE_RNN))


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
            os.path.join('train_log', 'MT-{}-aux-field-{}-skip-{}-ai_skip-{}-hist-{}-batch-{}-lr-{}-lamb-{}-{}'.format(
                MODEL_NAME, args.field, args.skip, args.ai_skip, args.hist_len, args.batch_size, args.lr, args.lamb, os.path.basename('soccer').split('.')[0])))
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
