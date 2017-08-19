#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from DQNBFPIModel_multi import Model as DQNModel
import common
from common import play_model, Evaluator, eval_model_multithread
from soccer_env_multitask import SoccerPlayer
from augment_expreplay_bfpm import AugmentExpReplay

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
INIT_EXP = 1.0
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10  # each epoch is 100k played frames
EVAL_EPISODE = 50
LAMB = 1.0
FP_DECAY = 0.1
LR = 1e-3
EXP_RATE = None
LR_RATE = None
AI_SKIP = 2


NUM_ACTIONS = None
METHOD = None
FIELD = None
NUM_AGENTS = 1
USE_REG = False

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


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, NUM_AGENTS, GAMMA, lr=LR, lamb=LAMB, fp_decay=FP_DECAY, use_reg=USE_REG)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        self.batch_size = tf.shape(image)[0]

        if USE_RNN:
            image = tf.transpose(image, perm=[0, 3, 1, 2])
            image = tf.reshape(image, (self.batch_size * self.channel,) + self.image_shape + (1,))

        with tf.variable_scope('network'):
            with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True), \
                    argscope(LeakyReLU, alpha=0.01):

                s_l = Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
                s_l = Conv2D('conv1', s_l, out_channel=64, kernel_shape=4, stride=2)
                s_l = Conv2D('conv2', s_l, out_channel=64, kernel_shape=3)

                q_l = FullyConnected('qfc0', s_l, 512, nl=LeakyReLU)
                p_l = FullyConnected('pfc0', s_l, 512, nl=LeakyReLU)

                l = tf.multiply(q_l, p_l, name='mul')


        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))

        bp_ys = []
        pi_ys = []
        fp_ys = []

        for i in range(self.num_agents):
            bp_y = FullyConnected('bp-fc0-%d' % i, p_l, self.num_actions, nl=tf.identity)
            pi_y = FullyConnected('pi-fc0-%d' % i, p_l, self.num_actions, nl=tf.identity)
            fp_conc = tf.concat([Q, bp_y, pi_y, p_l], axis=1)
            fp_y = FullyConnected('fp-fc0-%d' %i, fp_conc, self.num_actions, nl=tf.identity)

            bp_y = tf.identity(bp_y, name='Pivalue-%d' % i)
            pi_y = tf.identity(bp_y, name='Bpvalue-%d' % i)
            fp_y = tf.identity(bp_y, name='Fpvalue-%d' % i)

            bp_ys.append(bp_y)
            pi_ys.append(pi_y)
            fp_ys.append(fp_y)

        return tf.identity(Q, name='Qvalue'), pi_ys, bp_ys, fp_ys

def get_config():
    M = Model()
    expreplay = AugmentExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=INIT_EXP,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        num_agents=NUM_AGENTS
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
                                       LR_RATE),
                                      #[(20, 4e-4), (40, 2e-4)]),
                                      #[(40, 4e-4), (80, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                EXP_RATE,
                #[(0, 1), (10, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                #[(0, 1), (40, 0.1), (80, 0.01)],   # 1->0.1 in the first million steps
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
    parser.add_argument('--skip', help='act repeat', type=int, required=True)
    parser.add_argument('--field', help='field type', type=str, choices=['small', 'large'], default='large')
    parser.add_argument('--hist_len', help='hist len', type=int, required=True)
    parser.add_argument('--batch_size', help='batch size', type=int, required=True)
    parser.add_argument('--lamb', dest='lamb', type=float, default=1.0)
    parser.add_argument('--fp_decay', dest='fp_decay', type=float, default=0.1)
    parser.add_argument('--ai_skip', dest='ai_skip', type=float, default=2)
    parser.add_argument('--rnn', dest='rnn', action='store_true')
    parser.add_argument('--fast', dest='fast', action='store_true')
    parser.add_argument('--mix', dest='mix', action='store_true')
    parser.add_argument('--freq', dest='freq', type=int, default=4)
    parser.add_argument('--reg', dest='reg', action='store_true')
    parser.add_argument('--na', dest='na', type=int, default=1)

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    METHOD = args.algo

    ACTION_REPEAT = args.skip
    FIELD = args.field
    FRAME_HISTORY = args.hist_len
    BATCH_SIZE = args.batch_size
    LAMB = args.lamb
    USE_RNN = args.rnn
    FP_DECAY = args.fp_decay
    AI_SKIP = args.ai_skip
    MIX = args.mix
    UPDATE_FREQ = args.freq
    USE_REG = args.reg
    NUM_AGENTS = args.na

    if args.fast:
        LR_RATE = [(60, 4e-4), (100, 2e-4)]
        EXP_RATE = [(0, 1), (10, 0.1), (320, 0.01)]
    else:
        LR_RATE = [(20, 4e-4), (40, 2e-4)]
        EXP_RATE = [(0, 1), (40, 0.1), (320, 0.01)]

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
            os.path.join('train_log',
                'DQNBFPI-SHARE-COMPACT-field-{}-skip-{}-hist-{}-batch-{}-{}-{}-{}-decay-{}-aiskip-{}-{}-na-{}'.format(
                args.field, args.skip, args.hist_len, args.batch_size, os.path.basename('soccer').split('.')[0], LAMB,
                'fast' if args.fast else 'slow', args.fp_decay, args.ai_skip,
                'reg' if args.reg else '', args.na)))
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
