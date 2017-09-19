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
import pygame_soccer.util.file_util as file_util

__all__ = ['SoccerPlayer']


class SoccerPlayer(RLEnvironment):
    """
    A wrapper for pygame_soccer emulator.
    Will automatically restart when a real episode ends (isOver might be just
    lost of lives but not game over).
    """
    SOCCER_WIDTH = 288
    SOCCER_HEIGHT = 192

    def __init__(self, viz=0,
                height_range=(None, None),
                field='large', partial=False, radius=2,
                frame_skip=4,
                image_shape=(84, 84),
                nullop_start=30, mode=None, team_size=2, ai_frame_skip=1):
        super(SoccerPlayer, self).__init__()
        self.mode = mode
        self.field = field
        self.partial = partial
        self.viz = viz

        assert mode == None, 'Not impl'
        assert field == 'large', 'No small 2vs2'

        if self.viz:
            self.renderer_options = soccer_renderer.RendererOptions(
                show_display=True, max_fps=10, enable_key_events=True)
        else:
            self.renderer_options = None

        map_path = file_util.resolve_path(__file__, '../data/map/soccer_large.tmx')

        self.team_size = team_size
        self.env_options = soccer_environment.SoccerEnvironmentOptions(team_size=self.team_size, map_path=map_path, ai_frame_skip=ai_frame_skip)
        self.env = soccer_environment.SoccerEnvironment(env_options=self.env_options, renderer_options=self.renderer_options)

        self.computer_team_name = self.env.team_names[1]
        self.player_team_name = self.env.team_names[0]

        # Partial
        if self.partial:
            self.radius = radius
            self.player_agent_index = self.env.get_agent_index(self.player_team_name, 0)

        self.width, self.height = self.SOCCER_WIDTH, self.SOCCER_HEIGHT
        self.actions = self.env.actions

        self.frame_skip = frame_skip
        self.nullop_start = nullop_start
        self.height_range = height_range
        self.image_shape = image_shape

        self.last_info = {}
        self.agent_actions = ['STAND'] * (self.team_size * 2)

        self.current_episode_score = StatCounter()
        self.restart_episode()

    def _get_computer_actions(self):
        # Collaborator
        for i in range(self.team_size):
            index = self.env.get_agent_index(self.player_team_name, i)
            action = self.env.state.get_agent_action(index)
            self.agent_actions[self.team_size * 0 + i] = action
        # Opponent
        for i in range(self.team_size):
            index = self.env.get_agent_index(self.computer_team_name, i)
            action = self.env.state.get_agent_action(index)
            self.agent_actions[self.team_size * 1 + i] = action
        return np.asarray([self.env.actions.index(act if act else 'STAND') for act in self.agent_actions])

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        self.env.render()
        if self.partial:
            screenshot = self.env.renderer.get_po_screenshot(self.player_agent_index, self.radius)
        else:
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
            if k == 0:
                self.last_info['agent_actions'] = self._get_computer_actions()
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
        return self.last_info

