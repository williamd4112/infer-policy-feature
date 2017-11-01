#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modified from the work of Yuxin Wu <ppwwyyxxc@gmail.com>

import abc
import tensorflow as tf
from tensorpack import ModelDesc, InputDesc
from tensorpack.utils import logger
from tensorpack import *

from tensorpack.tfutils.gradproc import FilterNoneGrad
from tensorpack.tfutils import (
    collection, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils import argscope

def get_rnn_cell():
    if RNN_CELL == 'gru':
        return tf.nn.rnn_cell.GRUCell(num_units=RNN_HIDDEN, activation=tf.nn.relu)
    elif RNN_CELL == 'lstm':
        return tf.nn.rnn_cell.LSTMCell(num_units=RNN_HIDDEN, state_is_tuple=True, activation=tf.nn.relu)
    else:
        assert 0

class MAModel(ModelDesc):
    def __init__(self, image_shape, channel, method, num_actions, gamma, ctrl_size=1,
                lr=1e-3, lamb=1.0, h_size=512, update_step=1, multi_task=False, num_agents=1, reg=True, mt_type='all', models=None, dnp=False):
        self.image_shape = image_shape
        self.channel = channel
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.lamb = lamb
        self.h_size = h_size
        self.update_step = update_step
        self.multi_task = multi_task
        self.num_agents = num_agents
        self.reg = reg
        self.ctrl_size = ctrl_size
        self.dnp = dnp

        assert mt_type in ['all', 'coop-only', 'opponent-only']
        self.mt_type = mt_type
        self.models_type = models
        self.models = [PIModel(rank=0, image_shape=image_shape, channel=channel, method=method,
                       num_actions=num_actions, num_agents=num_agents),
                       DQNModel(rank=1, image_shape=image_shape, channel=channel, num_actions=num_actions,
                       method=method, num_agents=num_agents, dnp=self.dnp)]

    def _get_inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [InputDesc(tf.uint8,
                    (None,) + self.image_shape + (self.channel + 1,),
                    'comb_state'),
                InputDesc(tf.int64, (None, self.channel + 1, 2), 'action'),
                InputDesc(tf.float32, (None, self.channel + 1, 2), 'reward'),
                InputDesc(tf.bool, (None, self.channel + 1), 'isOver'),
                InputDesc(tf.int64, (None, self.channel + 1, self.num_agents), 'action_o')]

    def _get_cost(self):
        return self.cost


    def _get_DQN_prediction(self, image, models):
        qs, pis = [], []
        for i, model in enumerate(models):
            with tf.variable_scope("network-%d" % (i)):
                q, pi, _, _ = model._get_DQN_prediction(image)
            qs.append(q)
            pis.append(pi)
        qs = tf.transpose(tf.stack(qs), [1, 0, 2])

        return qs, pis, None, None

    def _build_graph(self, inputs):
        comb_state, action, reward, isOver, action_o = inputs
        self.batch_size = tf.shape(comb_state)[0]
        self.flatten_size = self.batch_size * self.update_step * self.ctrl_size

        backward_offset = ((self.channel) - self.update_step)
        action = tf.slice(action, [0, backward_offset, 0], [-1, self.update_step, -1])
        reward = tf.slice(reward, [0, backward_offset, 0], [-1, self.update_step, -1])
        isOver = tf.slice(isOver, [0, backward_offset], [-1, self.update_step])
        action_o = tf.slice(action_o, [0, backward_offset, 0], [-1, self.update_step, self.num_agents])
        isOver = tf.tile(tf.expand_dims(isOver, axis=2), [1, 1, 2])

        action = tf.reshape(action, (self.flatten_size, 1))
        reward = tf.reshape(reward, (self.flatten_size, 1))
        isOver = tf.reshape(isOver, (self.flatten_size, 1))
        # assume that we only have one pi agent
        action_o = tf.reshape(action_o, (self.batch_size * self.update_step, self.num_agents))

        comb_state = tf.cast(comb_state, tf.float32)
        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, self.channel], name='state')

        predict_value, pi_value, _, _ = self._get_DQN_prediction(state, self.models)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, self.channel], name='next_state')

        [summary.add_moving_summary(
            tf.reduce_mean(tf.reduce_max(predict_value[:,i], 1), name='pred_r-%d' % i))
        for i in range(self.ctrl_size)]

        self.predict_value = tf.reshape(predict_value, (self.flatten_size, self.num_actions))
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)
        action_onehot = tf.reshape(action_onehot, (self.flatten_size, self.num_actions))
        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)
        pred_action_value = tf.reshape(pred_action_value, (self.flatten_size, 1))

        with tf.variable_scope('target'), \
                collection.freeze_collection([tf.GraphKeys.TRAINABLE_VARIABLES]):
            targetQ_pred_value, target_pi_value, _, _ = self._get_DQN_prediction(next_state, self.models)
            targetQ_pred_value = tf.reshape(targetQ_pred_value, (self.flatten_size, self.num_actions))

        if self.method != 'Double':
            best_v = tf.reduce_max(targetQ_pred_value, 1)
        else:
            sc = tf.get_variable_scope()
            with tf.variable_scope(sc, reuse=True):
                next_predict_value, _, _, _ = self._get_DQN_prediction(next_state, self.models)
            next_predcit_value = tf.reshape(next_predict_value, (self.flatten_size, self.num_actions))
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        reward = tf.reshape(reward, (self.flatten_size, 1))
        best_v = tf.reshape(best_v, (self.flatten_size, 1))
        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)
        action_os = tf.unstack(action_o, self.num_agents, axis=1)
        action_o_one_hots = []
        for o in action_os:
            action_o_one_hots.append(tf.one_hot(o, self.num_actions, 1.0, 0.0))


        pi_costs = []
        pi_value = tf.stack(pi_value)
        # assume only the first agent is pi agent
        for i, o in enumerate(action_o_one_hots):
            scale = 1.0
            # Coop-only: disable opponent loss
            if self.mt_type == 'coop-only' and i > 0:
                scale = 0.0
            # Opponent-only: disable collaborator loss
            if self.mt_type == 'opponent-only' and i == 0:
                scale = 0.0

            pi_costs.append(scale * tf.nn.softmax_cross_entropy_with_logits(labels=o, logits=pi_value[i, :]))

        pi_cost = self.lamb * tf.add_n(pi_costs)
        q_cost = tf.reshape((symbf.huber_loss(target - pred_action_value)), (self.batch_size * self.update_step, self.ctrl_size))
        if self.reg:
            reg_coef = tf.stop_gradient(tf.sqrt(1.0 / (tf.reduce_mean(pi_cost) + 1e-9)), name='reg')
            summary.add_moving_summary(reg_coef)
        else:
            reg_coef = 1.0

        self.cost_pi = tf.reduce_mean(q_cost[:, 0] * reg_coef + pi_cost, name='cost_pi')
        self.cost_dqn = tf.reduce_mean(q_cost[:, 1], name='cost_dqn')

        if self.models_type == 'both':
            self.cost = [self.cost_pi, self.cost_dqn]
        elif self.models_type == 'dqn':
            self.cost = [None, self.cost_dqn]
        else:
            self.cost = [self.cost_pi, None]

        summary.add_moving_summary(tf.reduce_mean(self.cost_pi, name='pi_cost'))
        summary.add_moving_summary(tf.reduce_mean(self.cost_dqn, name='dqn_cost'))

        for i, o_t in enumerate(action_os):
            try:
                pred = tf.argmax(pi_value[i,:], axis=-1)
                summary.add_moving_summary(tf.contrib.metrics.accuracy(pred, o_t, name='acc-%d' % i))
            except:
                print("nothing to log")

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

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', self.lr, summary=True)
        opt = [tf.train.AdamOptimizer(lr, epsilon=1e-3) for i in range(2)]
        return [optimizer.apply_grad_processors(
            opt[i], [gradproc.GlobalNormClip(10), gradproc.SummaryGradient()]) for i in range(2)]

class DQNModel(MAModel):
    def __init__(self, rank, num_actions, h_size=512, dnp=False, **kwargs):
        super(MAModel, self).__init__()
        self.rank = rank
        self.h_size = h_size
        self.num_actions = num_actions
        self.method = kwargs.pop('method')
        self.channel = kwargs.pop('channel')
        self.num_agents = kwargs.pop('num_agents')
        self.image_shape = kwargs.pop('image_shape')
        self.dnp = dnp

        super(MAModel, self).__init__()

    def get_rnn_init_state(self, cell, name):
        return cell.zero_state(self.batch_size, tf.float32)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0

        with tf.variable_scope('q'):
            with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True, padding='SAME'), \
                    argscope(LeakyReLU, alpha=0.01):
                h = (LinearWrap(image)
                     .Conv2D('conv0', out_channel=32, kernel_shape=8, stride=4)
                     .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                     .Conv2D('conv2', out_channel=64, kernel_shape=3)())

                q_l = FullyConnected('fc0-q', h, self.h_size, nl=LeakyReLU)
                pi_l = FullyConnected('fc0-pi', h, self.h_size, nl=LeakyReLU)

        pi_ys = []
        for i in range(self.num_agents):
            pi_ys.append(FullyConnected('fct-%d' % i, pi_l, self.num_actions, nl=tf.identity))

        if self.dnp:
            l = q_l
        else:
            l = tf.multiply(q_l, pi_l)

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))

        pi_values = [ tf.identity(pi_ys[i], name='Pivalue-%d-%d'% (self.rank, i))
                        for i in range(self.num_agents) ]
        """
        pi_values = [ tf.stop_gradient(tf.identity(pi_ys[i], name='Pivalue-%d-%d'% (self.rank, i)))
                        for i in range(self.num_agents) ]
        """
        return tf.identity(Q, name='Qvalue-%d' % self.rank), pi_values, None, None



class PIModel(MAModel):
    def __init__(self, rank, use_rnn=False, rnn_hidden=512, rnn_step=1, h_size=512, num_agents=1, **kwargs):
        self.rank = rank
        self.rnn_hidden = rnn_hidden
        self.use_rnn = use_rnn
        self.rnn_step = rnn_step
        self.num_agents = num_agents
        self.h_size = 512
        self.num_actions = kwargs.pop('num_actions')
        self.method = kwargs.pop('method')
        self.channel = kwargs.pop('channel')
        self.image_shape = kwargs.pop('image_shape')

        super(MAModel, self).__init__()
        """
        self.h_size = h_size
        self.num_agents = num_agents
        self.image_shape = image_shape
        self.channel = channel
        """

    def get_rnn_init_state(self, cell, name):
        return cell.zero_state(self.batch_size, tf.float32)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0

        if self.use_rnn:
            self.batch_size = tf.shape(image)[0]
            image = tf.transpose(image, perm=[0, 3, 1, 2])
            image = tf.reshape(image, (self.batch_size * self.channel,) + self.image_shape + (1,))

        with tf.variable_scope('q'):
            with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True, padding='SAME'), \
                    argscope(LeakyReLU, alpha=0.01):
                h = (LinearWrap(image)
                     .Conv2D('conv0', out_channel=32, kernel_shape=8, stride=4)
                     .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                     .Conv2D('conv2', out_channel=64, kernel_shape=3)())

                q_l = FullyConnected('fc0-q', h, self.h_size, nl=LeakyReLU)
                pi_l = FullyConnected('fc0-pi', h, self.h_size, nl=LeakyReLU)

                if self.use_rnn:
                    # q
                    q_l = tf.reshape(q_l, [self.batch_size, self.channel, self.h_size])
                    q_cell = get_rnn_cell()
                    q_l, q_rnn_state_out = tf.nn.dynamic_rnn(inputs=q_l,
                                cell=q_cell,
                                initial_state=self.get_rnn_init_state(q_cell, 'q'),
                                dtype=tf.float32, scope='rnn-q')
                    q_l = q_l[:, -self.rnn_step:, :]
                    q_l = tf.reshape(q_l, (self.batch_size * self.rnn_step, self.use_rnn))

                    # pi
                    pi_l = tf.reshape(pi_l, [self.batch_size, self.channel, self.h_size])
                    pi_cell = get_rnn_cell()
                    pi_l, pi_rnn_state_out = tf.nn.dynamic_rnn(inputs=pi_l,
                                cell=pi_cell,
                                initial_state=self.get_rnn_init_state(pi_cell, 'pi'),
                                dtype=tf.float32, scope='rnn-pi')
                    pi_l = pi_l[:, -RNN_STEP:, :]
                    pi_l = tf.reshape(pi_l, (self.batch_size * self.rnn_step, self.rnn_hidden))

        pi_ys = []
        for i in range(self.num_agents):
            pi_ys.append(FullyConnected('fct-%d' % i, pi_l, self.num_actions, nl=tf.identity))

        l = tf.multiply(q_l, pi_l)
        #l = tf.concat([q_l, pi_l], axis=1)

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))

        pi_values = [ tf.identity(pi_ys[i], name='Pivalue-%d-%d'% (self.rank, i)) for i in range(self.num_agents) ]

        return tf.identity(Q, name='Qvalue-%d' % self.rank), pi_values, None, None


