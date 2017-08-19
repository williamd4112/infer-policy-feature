#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import copy
from collections import deque, namedtuple
import threading
import six
from six.moves import queue, range

from tensorpack.dataflow import DataFlow
from tensorpack.utils import logger, get_tqdm, get_rng
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
from tensorpack.callbacks.base import Callback

from expreplay import ReplayMemory, ExpReplay

__all__ = ['AugmentExpReplay']

AugmentExperience = namedtuple('AugmentExperience',
                        ['state', 'action', 'reward', 'isOver', 'action_o'])

NOOP_ACT = 4

class AugmentReplayMemory(ReplayMemory):
    def __init__(self, max_size, state_shape, history_len, num_agents=1):
        super(AugmentReplayMemory, self).__init__(max_size, state_shape, history_len)
        self.action_o = np.zeros((self.max_size, num_agents,), dtype='int32')
        self.num_agents = num_agents

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
        else:
            end = idx + k - self._curr_size
            state = self._slice(self.state, idx, end)
            reward = self._slice(self.reward, idx, end)
            action = self._slice(self.action, idx, end)
            isOver = self._slice(self.isOver, idx, end)
            action_o = self._slice(self.action_o, idx, end)
        ret = self._pad_sample(state, reward, action, isOver, action_o)
        return ret

    def recent_state(self):
        ls = list(self._hist)
        states =[np.zeros(self.state_shape, dtype='uint8')] * (self._hist.maxlen - len(ls))
        states.extend([k.state for k in ls])
        actions = [NOOP_ACT] * (self._hist.maxlen - len(ls))
        actions.extend([k.action_o for k in ls])

        return states, actions

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self, state, reward, action, isOver, action_o):
        for k in range(self.history_len - 2, -1, -1):
            if isOver[k]:
                state = copy.deepcopy(state)
                action_o = copy.deepcopy(action_o)

                state[:k + 1].fill(0)
                action_o[k + 1] = NOOP_ACT
                break
        state = state.transpose(1, 2, 0)
        return (state, reward[-2], action[-2], isOver[-2], action_o)

    def _assign(self, pos, exp):
        self.state[pos] = exp.state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver
        self.action_o[pos] = exp.action_o


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
                 update_frequency, history_len, num_agents=1):
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
                history_len)
        self.mem = AugmentReplayMemory(memory_size, state_shape, history_len, num_agents)
        self.num_agents = num_agents

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        old_s = self.player.current_state()
        old_action_o = self.player.get_internal_state()['agent_actions'][1:self.num_agents+1]
        if self.rng.rand() <= self.exploration or (len(self.mem) <= self.history_len):
            act = self.rng.choice(range(self.num_actions))
        else:
            # build a history state
            history_s, history_a_o = self.mem.recent_state()
            history_s.append(old_s)
            history_a_o.append(old_action_o)

            # assume batched network
            history_s = np.stack(history_s, axis=2)
            #history_a_o = np.asarray(history_a_o)
            q_values = self.predictor([[history_s]])[0][0]  # this is the bottleneck
            act = np.argmax(q_values)
        reward, isOver = self.player.action(act)
        # NOTE: since modify action interface will destroy the proxy design
        action_o = self.player.get_internal_state()['agent_actions'][1:self.num_agents+1]
        self.mem.append(AugmentExperience(old_s, act, reward, isOver, action_o))

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='uint8')
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        action_o = np.asarray([e[4] for e in batch_exp], dtype='int8')
        return [state, action, reward, isOver, action_o]

if __name__ == '__main__':
    import sys

    def predictor(x):
        np.array([1, 1, 1, 1])

    from soccer_env_multitask import SoccerPlayer

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

    """
    for k in E.get_data():
        import IPython as IP
        IP.embed(config=IP.terminal.ipapp.load_default_config())
        pass
        # import IPython;
        # IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        # break
    """