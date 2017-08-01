#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQNModel.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import abc
import numpy as np
import tensorflow as tf
from tensorpack import ModelDesc, InputDesc
from tensorpack.utils import logger

from tensorpack.tfutils.gradproc import FilterNoneGrad
from tensorpack.tfutils import (
    collection, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils import symbolic_functions as symbf

ZERO = 1e-8

class Model(ModelDesc):
    def __init__(self, image_shape, channel, method, num_actions, gamma, dist_type='gaussian', lr=1e-3, lamb=0.01, alpha=0.01):
        self.image_shape = image_shape
        self.channel = channel
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.lamb = lamb
        self.alpha = alpha
        self.dist_type = dist_type

    def _get_inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [InputDesc(tf.uint8,
                          (None,) + self.image_shape + (self.channel + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'reward'),
                InputDesc(tf.bool, (None,), 'isOver'),
                InputDesc(tf.int64, (None,), 'action_o')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        pass

    def _build_graph(self, inputs):
        comb_state, action, reward, isOver, action_o = inputs
        comb_state = tf.cast(comb_state, tf.float32)
        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, self.channel], name='state')

        self.predict_value, pi_value, pi_mean, pi_var, pi_livar = self._get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, self.channel], name='next_state')
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)
        action_o_one_hot = tf.one_hot(action_o, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), \
                collection.freeze_collection([tf.GraphKeys.TRAINABLE_VARIABLES]):
            targetQ_predict_value, target_pi_value, _, _, _ = self._get_DQN_prediction(next_state)    # NxA

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

        if self.dist_type == 'gaussian' :
            pi_std = tf.sqrt(tf.exp(pi_mean))
            epsilon = (pi_livar - pi_mean) / (pi_std + ZERO)
            logli = tf.reduce_sum(
                        -0.5 * np.log(2 *np.pi) - tf.log(pi_std + ZERO) - 0.5 * tf.square(epsilon),
                        reduction_indices=1
                     )

            H = -tf.reduce_sum(
                    -0.5 * np.log(2 * np.pi) - tf.log(1 + ZERO) - 0.5 * tf.square(pi_var / (1 + ZERO)),
                    reduction_indices=1
                )
            reg_pvar = tf.nn.l2_loss(pi_var, 'l2_pi')
        else :
            # categorical
            logli = tf.reduce_sum(
                        tf.log(act_prob + ZERO) * act_var,
                        reduction_indices=1
                    )
            H = -tf.reduce_sum(
                    act_prob * tf.log(act_prob + ZERO),
                    reduction_indices=1
                )

        reg_pvar = tf.reduce_mean(tf.reduce_sum(tf.abs(pi_mean) + tf.abs(pi_var) + tf.abs(pi_livar), axis=1))
        logli = tf.reduce_mean(logli, name='logli')
        H = tf.reduce_mean(H, name='H')
        mi_est = -(H + logli)

        q_cost = tf.reduce_mean(symbf.huber_loss(target - pred_action_value), name='q_cost')
        pi_cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=action_o_one_hot, logits=pi_value, name='pi_cost'),
                    name='pi_cost'
                )
        self.cost = tf.reduce_mean(q_cost + self.lamb * pi_cost + self.alpha * mi_est + 0.1 * reg_pvar, name='cost')

        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(self.cost)
        summary.add_moving_summary(reg_pvar)
        summary.add_moving_summary(logli)
        summary.add_moving_summary(q_cost)
        summary.add_moving_summary(pi_cost)
        summary.add_moving_summary(H)



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


