#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import time
import os
import cv2
from collections import deque
import threading
import six
from six.moves import range
import random
from tensorpack.utils import (get_rng, logger, execute_only_once)
from tensorpack.utils.fs import get_dataset_path
from tensorpack.utils.stats import StatCounter

from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace

import pygame_soccer.soccer.soccer_environment as soccer_environment
import pygame_soccer.soccer.soccer_renderer as soccer_renderer

__all__ = ['SoccerPlayer']


class SoccerPlayer(RLEnvironment):
    """
    A wrapper for pygame_soccer emulator.
    Will automatically restart when a real episode ends (isOver might be just
    lost of lives but not game over).
    """
    SOCCER_WIDTH = 288
    SOCCER_HEIGHT = 192

    def __init__(self, viz=0, height_range=(None, None),
                frame_skip=4, image_shape=(84, 84), nullop_start=30, mode=None, team_size=1):
        super(SoccerPlayer, self).__init__()
        self.mode = mode

        self.viz = viz
        if self.viz:
            self.renderer_options = soccer_renderer.RendererOptions(
                show_display=True, max_fps=10, enable_key_events=True)
        else:
            self.renderer_options = None
        
        self.env_options = soccer_environment.SoccerEnvironmentOptions(team_size=1)
        self.env = soccer_environment.SoccerEnvironment(env_options=self.env_options, renderer_options=self.renderer_options)

        self.computer_team_name = self.env.team_names[1]
        self.computer_agent_index = self.env.get_agent_index(self.computer_team_name, 0)

        self.width, self.height = self.SOCCER_WIDTH, self.SOCCER_HEIGHT
        self.actions = self.env.actions

        self.frame_skip = frame_skip
        self.nullop_start = nullop_start
        self.height_range = height_range
        self.image_shape = image_shape

        self.current_episode_score = StatCounter()
        self.restart_episode()

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        self.env.render()
        screenshot = self.env.renderer.get_screenshot()
        return screenshot

    def current_state(self):
        """
        :returns: a gray-scale (h, w) uint8 image
        """
        ret = self._grab_raw_image()
        # max-pooled over the last screen
        #ret = np.maximum(ret, self.last_raw_screen)
        '''
        if self.viz:
            if isinstance(self.viz, float):
                cv2.imshow('soccer', ret)
                cv2.waitKey(1)
        '''
        #ret = ret[self.height_range[0]:self.height_range[1], :].astype('float32')
        # 0.299,0.587.0.114. same as rgb2y in torch/image
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        ret = cv2.resize(ret, self.image_shape)
        return ret.astype('uint8')  # to save some memory

    def get_action_space(self):
        return DiscreteActionSpace(len(self.actions))

    def finish_episode(self):
        self.stats['score'].append(self.current_episode_score.sum)

    def restart_episode(self):
        self.current_episode_score.reset()
        self.env.reset()
        if self.mode is not None:
            self.env.state.set_agent_mode(self.computer_agent_index, self.mode)

        # random null-ops start
        NULL_OP_ACTION = self.env.actions[4]
        n = random.randint(0, self.nullop_start)
        self.last_raw_screen = self._grab_raw_image()

    def action(self, act):
        """
        :param act: an index of the action
        :returns: (reward, isOver)
        """
        r = 0
        for k in range(self.frame_skip):
            if k == self.frame_skip - 1:
                self.last_raw_screen = self._grab_raw_image()
            ret = self.env.take_action(self.env.actions[act])
            r += ret.reward
            if self.env.state.is_terminal():
                break

        self.current_episode_score.feed(r)
        isOver = self.env.state.is_terminal()
        if isOver:
            self.finish_episode()
            self.restart_episode()
        return (r, isOver)

    def get_internal_state(self):
        info = {}
        opponent_act = self.env.state.get_agent_action(self.COMPUTER_AGENT_IDX)
        info['opponent_action'] = self.env.actions.index(opponent_act if opponent_act else 'STAND')
        return info        

if __name__ == '__main__':
    import sys

    def benchmark():
        a = AtariPlayer(sys.argv[1], viz=False, height_range=(28, -8))
        num = a.get_action_space().num_actions()
        rng = get_rng(num)
        start = time.time()
        cnt = 0
        while True:
            act = rng.choice(range(num))
            r, o = a.action(act)
            a.current_state()
            cnt += 1
            if cnt == 5000:
                break
        print(time.time() - start)

    if len(sys.argv) == 3 and sys.argv[2] == 'benchmark':
        import threading
        import multiprocessing
        for k in range(3):
            # th = multiprocessing.Process(target=benchmark)
            th = threading.Thread(target=benchmark)
            th.start()
            time.sleep(0.02)
        benchmark()
    else:
        a = AtariPlayer(sys.argv[1],
                        viz=0.03, height_range=(28, -8))
        num = a.get_action_space().num_actions()
        rng = get_rng(num)
        import time
        while True:
            # im = a.grab_image()
            # cv2.imshow(a.romname, im)
            act = rng.choice(range(num))
            print(act)
            r, o = a.action(act)
            a.current_state()
            # time.sleep(0.1)
            print(r, o)
