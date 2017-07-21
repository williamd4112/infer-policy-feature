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

from soccer_env import SoccerPlayer

__all__ = ['PartialObserveSoccerPlayer']


class PartialObserveSoccerPlayer(SoccerPlayer):
    def __init__(self, viz=0, height_range=(None, None), field=None, 
                frame_skip=4, image_shape=(84, 84), nullop_start=30, mode=None, team_size=1, radius=2):
        super(PartialObserveSoccerPlayer, self).__init__(viz, height_range, field, frame_skip, image_shape, nullop_start, mode, team_size)
        
        self.radius = radius
        self.player_team_name = self.env.team_names[0]
        self.player_agent_index = self.env.get_agent_index(self.player_team_name, 0)

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        self.env.render()
        screenshot = self.env.renderer.get_po_screenshot(self.player_agent_index, self.radius)
        return screenshot
