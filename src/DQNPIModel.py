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
    def __init__(self, image_shape, channel, method, num_actions, gamma, 
                lr=1e-3, lamb=1.0, keep_state=False, h_size=512, update_step=1, multi_task=False, num_agents=1, reg=False, mt_type='all'):
        self.image_shape = image_shape
        self.channel = channel
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.lamb = lamb
        self.keep_state = keep_state
        self.h_size = h_size
        self.update_step = update_step
        self.multi_task = multi_task
        self.num_agents = num_agents
        self.reg = reg

        assert mt_type in ['all', 'coop-only', 'opponent-only']
        self.mt_type = mt_type

    def _get_inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        if self.keep_state:
            return [InputDesc(tf.uint8,
                          (None,) + self.image_shape + (self.channel + 1,),
                          'comb_state'),
                    InputDesc(tf.int64, (None, self.channel + 1), 'action'),
                    InputDesc(tf.float32, (None, self.channel + 1), 'reward'),
                    InputDesc(tf.bool, (None, self.channel + 1), 'isOver'),
                    InputDesc(tf.int64, (None, self.channel + 1, self.num_agents), 'action_o'),
                    InputDesc(tf.float32, (None, 2, self.h_size), 'q_rnn_state'),
                    InputDesc(tf.float32, (None, 2, self.h_size), 'pi_rnn_state')] 
        else:
            return [InputDesc(tf.uint8,
                              (None,) + self.image_shape + (self.channel + 1,),
                              'comb_state'),
                    InputDesc(tf.int64, (None, self.channel + 1), 'action'),
                    InputDesc(tf.float32, (None, self.channel + 1), 'reward'),
                    InputDesc(tf.bool, (None, self.channel + 1), 'isOver'),
                    InputDesc(tf.int64, (None, self.channel + 1, self.num_agents), 'action_o')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        pass

    def _build_graph(self, inputs):
        if self.keep_state:
            comb_state, action, reward, isOver, action_o, q_rnn_state, pi_rnn_state = inputs
        else:
            comb_state, action, reward, isOver, action_o = inputs
        self.batch_size = tf.shape(comb_state)[0]

        backward_offset = ((self.channel) - self.update_step)
        action = tf.slice(action, [0, backward_offset], [-1, self.update_step])
        reward = tf.slice(reward, [0, backward_offset], [-1, self.update_step])
        isOver = tf.slice(isOver, [0, backward_offset], [-1, self.update_step])
        action_o = tf.slice(action_o, [0, backward_offset, 0], [-1, self.update_step, self.num_agents])

        action = tf.reshape(action, (self.batch_size * self.update_step,))
        reward = tf.reshape(reward, (self.batch_size * self.update_step,))
        isOver = tf.reshape(isOver, (self.batch_size * self.update_step,))
        action_o = tf.reshape(action_o, (self.batch_size * self.update_step, self.num_agents))

        comb_state = tf.cast(comb_state, tf.float32)
        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, self.channel], name='state')
    
        if self.keep_state:
            self.q_rnn_state = tf.identity(pi_rnn_state, name='q_rnn_state_in')
            self.pi_rnn_state = tf.identity(pi_rnn_state, name='pi_rnn_state_in')

        self.predict_value, pi_value, self.q_rnn_state_out, self.pi_rnn_state_out = self._get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, self.channel], name='next_state')
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), \
                collection.freeze_collection([tf.GraphKeys.TRAINABLE_VARIABLES]):
            targetQ_predict_value, target_pi_value, _, _ = self._get_DQN_prediction(next_state)    # NxA

        if self.method != 'Double':
            # DQN
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN
            sc = tf.get_variable_scope()
            with tf.variable_scope(sc, reuse=True):
                next_predict_value, next_pi_value, _, _ = self._get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)
        
        # q cost
        q_cost = (symbf.huber_loss(target - pred_action_value))
        # pi cost
        action_os = tf.unstack(action_o, self.num_agents, axis=1)
        action_o_one_hots = []
        for o in action_os:
            action_o_one_hots.append(tf.one_hot(o, self.num_actions, 1.0, 0.0))
        pi_costs = []
        for i, o in enumerate(action_o_one_hots):
            scale = 1.0
            # Coop-only: disable opponent loss
            if self.mt_type == 'coop-only' and i > 0:
                scale = 0.0
            # Opponent-only: disable collaborator loss
            if self.mt_type == 'opponent-only' and i == 0:
                scale = 0.0
            pi_costs.append(scale * tf.nn.softmax_cross_entropy_with_logits(labels=o, logits=pi_value[i]))
        pi_cost = self.lamb * tf.add_n(pi_costs)

        if self.reg:
            reg_coff = tf.stop_gradient(tf.sqrt(1.0 / (tf.reduce_mean(pi_cost) + 1e-9)), name='reg')
            self.cost = tf.reduce_mean(reg_coff * q_cost + pi_cost)
            summary.add_moving_summary(reg_coff)
        else:    
            self.cost = tf.reduce_mean(q_cost + pi_cost)

        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(self.cost)
        summary.add_moving_summary(tf.reduce_mean(pi_cost, name='pi_cost'))
        summary.add_moving_summary(tf.reduce_mean(q_cost, name='q_cost'))

        for i, o_t in enumerate(action_os):
            pred = tf.argmax(pi_value[i], axis=1)
            summary.add_moving_summary(tf.contrib.metrics.accuracy(pred, o_t, name='acc-%d' % i))


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


