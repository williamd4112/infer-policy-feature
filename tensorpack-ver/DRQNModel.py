#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQNModel.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import abc
import tensorflow as tf
from tensorpack import ModelDesc, InputDesc
from tensorpack.utils import logger
from tensorpack.tfutils import (
    collection, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils import symbolic_functions as symbf


class Model(ModelDesc):
    def __init__(self, image_shape, channel, method, num_actions, gamma, batch_size):
        self.image_shape = image_shape
        self.channel = channel
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size

    def _get_inputs(self):
        # Use a combined state for efficiency.
        # The [:-1, :, :, :] is state, [1:, :, :, :] is next state
        return [InputDesc(tf.uint8,
                          (self.batch_size,) + self.image_shape + (self.channel + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (self.batch_size, self.channel), 'action'),
                InputDesc(tf.float32, (self.batch_size, self.channel), 'reward'),
                InputDesc(tf.bool, (self.batch_size, self.channel), 'isOver')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        pass

    def _build_graph(self, inputs):
        comb_state, action, reward, isOver = inputs
        comb_state = tf.cast(comb_state, tf.float32)
        
        # Reshape for unrolling
        action = tf.reshape(action, [self.batch_size * self.channel])
        reward = tf.reshape(reward, [self.batch_size * self.channel])
        isOver = tf.reshape(isOver, [self.batch_size * self.channel])
         
        # TODO: Add recurrent part
        # DQN state
        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, self.channel], name='state') 
        state = tf.transpose(state, perm=[0, 3, 1, 2])
        state = tf.reshape(state, (self.batch_size * self.channel,) + self.image_shape)
        state = tf.expand_dims(state, -1)
        
        self.predict_value = self._get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)

        # DQN next state
        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, self.channel], name='next_state')
        next_state = tf.transpose(next_state, perm=[0, 3, 1, 2])
        next_state = tf.reshape(next_state, (self.batch_size * self.channel,) + self.image_shape)
        next_state = tf.expand_dims(next_state, -1)

        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)
        print(action_onehot, self.predict_value)
        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), \
                collection.freeze_collection([tf.GraphKeys.TRAINABLE_VARIABLES]):
            targetQ_predict_value = self._get_DQN_prediction(next_state)    # NxA

        if self.method != 'Double':
            # DQN
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN
            sc = tf.get_variable_scope()
            with tf.variable_scope(sc, reuse=True):
                next_predict_value = self._get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)

        self.cost = tf.reduce_mean(symbf.huber_loss(
                                   target - pred_action_value), name='cost')
        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 1e-3, summary=True)
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
