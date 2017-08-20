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
    def __init__(self, image_shape, channel, method, num_actions, num_agents, gamma, lr=1e-3, lamb=1.0, fp_decay=1.0, use_reg=False, reg_only_pi=False, add_up=True):
        self.image_shape = image_shape
        self.channel = channel
        self.method = method
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.gamma = gamma
        self.lr = lr
        self.lamb = lamb
        self.fp_decay = fp_decay
        self.use_reg = use_reg
        self.reg_only_pi = reg_only_pi
        self.add_up = add_up

    def _get_inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [InputDesc(tf.uint8,
                          (None,) + self.image_shape + (self.channel + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'reward'),
                InputDesc(tf.bool, (None,), 'isOver'),
                InputDesc(tf.int64, (None, self.channel + 1, self.num_agents), 'comb_action_o'),]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        pass

    def _build_graph(self, inputs):

        comb_state, action, reward, isOver, comb_action_o = inputs
        comb_state = tf.cast(comb_state, tf.float32)
        self.batch_size = tf.shape(comb_state)[0]
        reshape_size = (self.batch_size * self.channel,)

        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, self.channel], name='state')
        """
        old_action_o = comb_action_o[:, self.channel-2]
        act_o = comb_action_o[:, self.channel-1]
        next_act_o = comb_action_o[:, self.channel]
        """

        self.predict_value, pi_values, bp_values, fp_values = self._get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)

        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, self.channel], name='next_state')

        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)
        bp_one_hots = []
        pi_one_hots = []
        fp_one_hots = []

        for i in range(self.num_agents):
            bp_one_hots.append(tf.one_hot(comb_action_o[:, self.channel-2, i], self.num_actions, 1.0, 0.0))
            pi_one_hots.append(tf.one_hot(comb_action_o[:, self.channel-1, i], self.num_actions, 1.0, 0.0))
            fp_one_hots.append(tf.one_hot(comb_action_o[:, self.channel-0, i], self.num_actions, 1.0, 0.0))

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

        bp_cost = []
        pi_cost = []
        fp_cost = []
        for i, o in enumerate(zip(bp_one_hots, pi_one_hots, fp_one_hots)):
            bp_cost.append(tf.nn.softmax_cross_entropy_with_logits(labels=o[0], logits=bp_values[i]))
            pi_cost.append(tf.nn.softmax_cross_entropy_with_logits(labels=o[1], logits=pi_values[i]))
            fp_cost.append(tf.nn.softmax_cross_entropy_with_logits(labels=o[2], logits=fp_values[i]))

        if self.add_up :
            bp_cost = self.lamb * tf.add_n(bp_cost)
            pi_cost = self.lamb * tf.add_n(pi_cost)
            fp_cost = self.lamb * self.fp_decay * tf.add_n(fp_cost)
        else :
            bp_cost = self.lamb * tf.reduce_mean(tf.stack(bp_cost, axis=1), axis=1)
            pi_cost = self.lamb * tf.reduce_mean(tf.stack(pi_cost, axis=1), axis=1)
            fp_cost = self.lamb * self.fp_decay * tf.reduce_mean(tf.stack(fp_cost, axis=1), axis=1)

        avg_cost = (pi_cost + fp_cost + bp_cost) / 3.0
        reg_coef = tf.stop_gradient(tf.sqrt(1.0 / avg_cost))

        if self.use_reg :
            if self.reg_only_pi :
                reg_coef = tf.stop_gradient(tf.sqrt(1.0 / pi_cost))
            self.cost = tf.reduce_mean(reg_coef * q_cost + avg_cost, name='total_cost')
        else :
            self.cost = tf.reduce_mean(q_cost + avg_cost, name='total_cost')


        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(self.cost)
        summary.add_moving_summary(tf.reduce_mean(pi_cost, name='pi_cost'))
        summary.add_moving_summary(tf.reduce_mean(bp_cost, name='bp_cost'))
        summary.add_moving_summary(tf.reduce_mean(q_cost, name='q_cost'))
        summary.add_moving_summary(tf.reduce_mean(fp_cost, name='fp_cost'))
        summary.add_moving_summary(tf.reduce_mean(avg_cost, name='avg_cost'))
        summary.add_moving_summary(tf.reduce_mean(reg_coef, name='reg_coef'))

        for i in range(self.num_agents):
            pred_c = tf.argmax(pi_values[i], axis=1)
            pred_fp = tf.argmax(fp_values[i], axis=1)
            pred_bp = tf.argmax(bp_values[i], axis=1)
            summary.add_moving_summary(tf.contrib.metrics.accuracy(pred_bp, comb_action_o[:, self.channel-2, i], name='pred_bp_acc-{}'.format(i)))
            summary.add_moving_summary(tf.contrib.metrics.accuracy(pred_c, comb_action_o[:, self.channel-1, i], name='pred_c_acc-{}'.format(i)))
            summary.add_moving_summary(tf.contrib.metrics.accuracy(pred_fp, comb_action_o[:, self.channel-0, i], name='pred_fp_acc-{}'.format(i)))

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


