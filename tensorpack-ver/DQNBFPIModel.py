#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQNModel.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import abc
import tensorflow as tf
from tensorpack import ModelDesc, InputDesc
from tensorpack.utils import logger

from tensorpack.tfutils.gradproc import FilterNoneGrad
from tensorpack.tfutils import (
    collection, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils import symbolic_functions as symbf


class Model(ModelDesc):
    def __init__(self, image_shape, channel, method, num_actions, gamma, lr=1e-3, lamb=1.0, fp_decay=1.0, use_reg=False):
        self.image_shape = image_shape
        self.channel = channel
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.lamb = lamb
        self.fp_decay = fp_decay
        self.use_reg = use_reg

    def _get_inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [InputDesc(tf.uint8,
                          (None,) + self.image_shape + (self.channel + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'reward'),
                InputDesc(tf.bool, (None,), 'isOver'),
                InputDesc(tf.int64, (None, self.channel + 1), 'comb_action_o'),]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        pass

    def _build_graph(self, inputs):

        comb_state, action, reward, isOver, comb_action_o = inputs
        comb_state = tf.cast(comb_state, tf.float32)
        self.batch_size = tf.shape(comb_state)[0]
        reshape_size = (self.batch_size * self.channel,)

        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, self.channel], name='state')
        old_action_o = comb_action_o[:, self.channel-2]
        act_o = comb_action_o[:, self.channel-1]
        next_act_o = comb_action_o[:, self.channel]

        self.predict_value, pi_value, bp_value, fp_value = self._get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)

        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, self.channel], name='next_state')

        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)
        action_o_one_hot = tf.one_hot(act_o, self.num_actions, 1.0, 0.0)
        next_action_o_one_hot = tf.one_hot(next_act_o, self.num_actions, 1.0, 0.0)
        old_action_o_one_hot = tf.one_hot(old_action_o, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), \
                collection.freeze_collection([tf.GraphKeys.TRAINABLE_VARIABLES]):
            targetQ_predict_value, target_pi_value, target_bp_value, target_fp_value = self._get_DQN_prediction(next_state)    # NxA

        if self.method != 'Double':
            # DQN
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN
            sc = tf.get_variable_scope()
            with tf.variable_scope(sc, reuse=True):
                next_predict_value, next_pi_value = self._get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)

        q_cost = (symbf.huber_loss(target - pred_action_value))
        pi_cost = self.lamb * (tf.nn.softmax_cross_entropy_with_logits(labels=action_o_one_hot, logits=pi_value))
        fp_cost = self.fp_decay * self.lamb * (tf.nn.softmax_cross_entropy_with_logits(labels=next_action_o_one_hot, logits=fp_value))
        bp_cost = self.lamb * (tf.nn.softmax_cross_entropy_with_logits(labels=old_action_o_one_hot, logits=bp_value))
        avg_cost = tf.reduce_mean((pi_cost + fp_cost + bp_cost) / 3.0, name='avg_cost')
        reg_coef = tf.stop_gradient((1.0 / avg_cost), name='reg_coef')

        if self.use_reg :
            self.cost = tf.reduce_mean(reg_coef * q_cost + avg_cost, name='total_cost')
        else :
            self.cost = tf.reduce_mean(q_cost + avg_cost, name='total_cost')

        pred_c = tf.argmax(pi_value, axis=1)
        pred_fp = tf.argmax(fp_value, axis=1)
        pred_bp = tf.argmax(bp_value, axis=1)

        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(self.cost)
        summary.add_moving_summary(tf.reduce_mean(pi_cost, name='pi_cost'))
        summary.add_moving_summary(tf.reduce_mean(bp_cost, name='bp_cost'))
        summary.add_moving_summary(tf.reduce_mean(q_cost, name='q_cost'))
        summary.add_moving_summary(tf.reduce_mean(fp_cost, name='fp_cost'))
        summary.add_moving_summary(avg_cost)
        summary.add_moving_summary(reg_coef)
        summary.add_moving_summary(tf.contrib.metrics.accuracy(pred_bp, old_action_o, name='pred_bp_acc'))
        summary.add_moving_summary(tf.contrib.metrics.accuracy(pred_c, act_o, name='pred_c_acc'))
        summary.add_moving_summary(tf.contrib.metrics.accuracy(pred_fp, next_act_o, name='pred_fp_acc'))

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', self.lr, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(10), gradproc.SummaryGradient()])

    @staticmethod
    def update_target_param():
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')


