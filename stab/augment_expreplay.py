#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modified from the work of Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import copy
from collections import deque, namedtuple
import threading
import six
from six.moves import queue, range

from tensorpack.dataflow import DataFlow
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm, get_rng
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
from tensorpack.callbacks.base import Callback

from expreplay import ReplayMemory, ExpReplay
import time

__all__ = ['AugmentExpReplay', 'AugmentReplayMemory']

AugmentExperience = namedtuple('AugmentExperience',
                        ['state', 'action', 'reward', 'isOver', 'action_o', 'eps', 'episode'])


class AugmentReplayMemory(ReplayMemory):
    def __init__(self, max_size, state_shape, history_len, num_agents):
        super(AugmentReplayMemory, self).__init__(max_size, state_shape, history_len)
        self.num_agents = num_agents
        self.action_o = np.zeros((self.max_size, num_agents), dtype='int32')
        self.eps = np.zeros((self.max_size,), dtype='float32')
        self.episode = np.zeros((self.max_size,), dtype='int32')

    def reset(self):
        self.state[:] = 0
        self.action[:] = 0
        self.reward[:] = 0
        self.isOver[:] = 0
        self.action_o[:] = 0
        self.eps[:] = .99
        self.episode[:] = 0

        self._curr_size = 0
        self._curr_pos = 0
        self._hist = deque(maxlen=self.history_len - 1)

    def sample(self, idx):
        """ return a tuple of (s,r,a,o,a_o),
            where s is of shape STATE_SIZE + (hist_len+1,)"""
        idx = (self._curr_pos + idx) % self._curr_size
        k = self.history_len + 1
        if idx + k <= self._curr_size:
            state = self.state[idx: idx + k]
            reward = self.reward[idx: idx + k]
            action = self.action[idx: idx + k]
            isOver = self.isOver[idx: idx + k]
            action_o = self.action_o[idx: idx + k]
            eps = self.eps[idx: idx + k]
            episode = self.episode[idx: idx + k]
        else:
            end = idx + k - self._curr_size
            state = self._slice(self.state, idx, end)
            reward = self._slice(self.reward, idx, end)
            action = self._slice(self.action, idx, end)
            isOver = self._slice(self.isOver, idx, end)
            action_o = self._slice(self.action_o, idx, end)
            eps = self._slice(self.eps, idx, end)
            episode = self._slice(self.episode, idx, end)
        ret = self._pad_sample(state, reward, action, isOver, action_o, eps, episode)
        return ret

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self, state, reward, action, isOver, action_o, eps, episode):
        for k in range(self.history_len - 2, -1, -1):
            if isOver[k]:
                state = copy.deepcopy(state)
                state[:k + 1].fill(0)
                break
        state = state.transpose(1, 2, 0)
        return (state, reward, action, isOver, action_o, eps, episode)

    def _assign(self, pos, exp):
        self.state[pos] = exp.state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver
        self.action_o[pos] = exp.action_o
        self.eps[pos] = exp.eps
        self.episode[pos] = exp.episode


class AugmentExpReplay(ExpReplay, Callback):
    """
    Implement experience replay in the paper
    `Human-level control through deep reinforcement learning
    <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`_.

    This implementation provides the interface as a :class:`DataFlow`.
    This DataFlow is __not__ fork-safe (thus doesn't support multiprocess prefetching).

    This implementation assumes that state is
    batch-able, and the network takes batched inputs.
    """

    def __init__(self,
                 predictor_io_names,
                 player,
                 state_shape,
                 batch_size,
                 memory_size, init_memory_size,
                 init_exploration,
                 update_frequency, history_len, repop=False, h_size=512, num_agents=1):
        """
        Args:
            predictor_io_names (tuple of list of str): input/output names to
                predict Q value from state.
            player (RLEnvironment): the player.
            history_len (int): length of history frames to concat. Zero-filled
                initial frames.
            update_frequency (int): number of new transitions to add to memory
                after sampling a batch of transitions for training.
        """
        super(AugmentExpReplay, self).__init__(predictor_io_names,
                player,
                state_shape,
                batch_size,
                memory_size,
                init_memory_size,
                init_exploration,
                update_frequency,
                history_len,
                repop)
        self.num_agents = num_agents
        self.h_size = h_size
        self.mem = AugmentReplayMemory(memory_size, state_shape, history_len, num_agents)

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        old_s = self.player.current_state()
        if self.rng.rand() <= self.exploration or (len(self.mem) <= self.history_len):
            act = [self.rng.choice(range(self.num_actions)) for n in range(self.num_agents*2)]
            act = [act[0], act[4]]
        else:
            # build a history state
            history = self.mem.recent_state()
            history.append(old_s)
            history = np.stack(history, axis=2)

            # assume batched network
            start = time.time()
            eps = [self.exploration]* (self.history_len + 1)
            episode = [self.n_epoch]* (self.history_len + 1)
            q_values = self.predictor([[history], [eps], [episode]])  # this is the bottleneck
            #print("inference time: {}".format(time.time() - start))
            act = [np.argmax(q) for q in q_values]

        start = time.time()
        reward, isOver = self.player.action(act)
        #print("env time: {}".format(time.time()-start))
        # NOTE: since modify action interface will destroy the proxy design
        action_o = self.player.get_internal_state()['agent_actions'][1:]
        self.mem.append(AugmentExperience(old_s, act, reward, isOver, action_o, self.exploration, self.n_epoch))

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='uint8')
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        action_o = np.asarray([e[4] for e in batch_exp], dtype='int8')
        eps = np.asarray([e[5] for e in batch_exp], dtype='float32')
        episode = np.asarray([e[6] for e in batch_exp], dtype='int32')

        return [state, action, reward, isOver, action_o, eps, episode]

if __name__ == '__main__':
    import sys

    def predictor(x):
        np.array([1, 1, 1, 1])

    from soccer_env import SoccerPlayer

    player = SoccerPlayer(image_shape=(84, 84), viz=False, frame_skip=4)

    E = AugmentExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=player,
        state_shape=(84, 84),
        batch_size=64,
        memory_size=1000,
        init_memory_size=100,
        init_exploration=1.0,
        update_frequency=4,
        history_len=4
    )

    E._init_memory()

    for k in E.get_data():
        import IPython as IP
        IP.embed(config=IP.terminal.ipapp.load_default_config())
        pass
        # import IPython;
        # IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        # break
