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

__all__ = ['SoccerPlayer', 'get_raw_env']

class SoccerSavingBallEnvironment(soccer_environment.SoccerEnvironment): 
   def reset(self):
        super(SoccerSavingBallEnvironment, self).reset()
        player_agent_index = self.get_agent_index('PLAYER', 1)
        ball_pos = self.state.get_ball_possession()
        agent_index = ball_pos['agent_index']
        self.state.switch_ball(agent_index, player_agent_index)
 
   def _get_ai_action(self, team_name, team_agent_index):
    # Get the opponent team name
    opponent_team_name = self.get_opponent_team_name(team_name)
    # Get the agent info
    agent_index = self.get_agent_index(team_name, team_agent_index)
    agent_pos = self.state.get_agent_pos(agent_index)
    agent_ball = self.state.get_agent_ball(agent_index)
    agent_mode = self.state.get_agent_mode(agent_index)
    agent_frame_skip_index = self.state.get_agent_frame_skip_index(
        agent_index)
    # Select the previous action if it's frame skipping
    if agent_frame_skip_index > 0:
      return self.state.get_agent_action(agent_index)

    # Get the position of the nearest opponent
    if team_name == 'COMPUTER':
        nearest_opponent_index = self.get_agent_index('PLAYER', 1)
        nearest_opponent_pos = self.state.get_agent_pos(nearest_opponent_index)
    else:
        nearest_opponent_index = self._get_nearest_opponent_index(
            team_name, team_agent_index)
        nearest_opponent_pos = self.state.get_agent_pos(nearest_opponent_index)

    # Get the position of the defensive target
    defensive_target_agent_index = self._get_defensive_agent_index(
        team_name, team_agent_index)
    defensive_target_agent_pos = self.state.get_agent_pos(
        defensive_target_agent_index)

    if team_name == 'COMPUTER':
        agent_mode = 'OFFENSIVE'
    if team_name == 'PLAYER':
        agent_mode = 'DEFENSIVE'

    # Calculate the target position and the strategic mode
    if agent_mode == 'DEFENSIVE':
      if agent_ball:
        target_pos = nearest_opponent_pos
        strategic_mode = 'AVOID'
      else:
        # Calculate the distance from the agent
        goals = self.map_data.goals[opponent_team_name]
        distances = [self.get_pos_distance(goal_pos, defensive_target_agent_pos)
                     for goal_pos in goals]
        # Select the minimum distance
        min_distance_index = np.argmin(distances)
        target_pos = goals[min_distance_index]
        strategic_mode = 'APPROACH'
    elif agent_mode == 'OFFENSIVE':
      if agent_ball:
        # Calculate the distance from the opponent
        goals = self.map_data.goals[team_name]
        distances = [self.get_pos_distance(goal_pos, nearest_opponent_pos)
                     for goal_pos in goals]
        # Select the maximum distance
        max_distance_index = np.argmax(distances)
        target_pos = goals[max_distance_index]
        strategic_mode = 'APPROACH'
      else:
        target_pos = defensive_target_agent_pos
        strategic_mode = 'INTERCEPT'
    else:
      raise KeyError('Unknown agent mode {}'.format(agent_mode))
    # Get the strategic action
    action = self._get_strategic_action(agent_pos, target_pos, strategic_mode)
    return action


class SoccerPassingBallEnvironment(soccer_environment.SoccerEnvironment): 
   def reset(self):
        super(SoccerPassingBallEnvironment, self).reset()
        player_agent_index = self.get_agent_index('PLAYER', 0)
        ball_pos = self.state.get_ball_possession()
        agent_index = ball_pos['agent_index']
        self.state.switch_ball(agent_index, player_agent_index)
 
   def _get_ai_action(self, team_name, team_agent_index):
    # Get the opponent team name
    opponent_team_name = self.get_opponent_team_name(team_name)
    # Get the agent info
    agent_index = self.get_agent_index(team_name, team_agent_index)
    agent_pos = self.state.get_agent_pos(agent_index)
    agent_ball = self.state.get_agent_ball(agent_index)
    agent_mode = self.state.get_agent_mode(agent_index)
    agent_frame_skip_index = self.state.get_agent_frame_skip_index(
        agent_index)
    # Select the previous action if it's frame skipping
    if agent_frame_skip_index > 0:
      return self.state.get_agent_action(agent_index)

    # Get the position of the nearest opponent
    if team_name == 'COMPUTER':
        nearest_opponent_index = 0
        nearest_opponent_pos = self.state.get_agent_pos(nearest_opponent_index)
    else:
        nearest_opponent_index = self._get_nearest_opponent_index(
            team_name, team_agent_index)
        nearest_opponent_pos = self.state.get_agent_pos(nearest_opponent_index)

    # Get the position of the defensive target
    defensive_target_agent_index = self._get_defensive_agent_index(
        team_name, team_agent_index)
    defensive_target_agent_pos = self.state.get_agent_pos(
        defensive_target_agent_index)

    agent_mode = 'OFFENSIVE'

    # Calculate the target position and the strategic mode
    if agent_mode == 'DEFENSIVE':
      if agent_ball:
        target_pos = nearest_opponent_pos
        strategic_mode = 'AVOID'
      else:
        # Calculate the distance from the agent
        goals = self.map_data.goals[opponent_team_name]
        distances = [self.get_pos_distance(goal_pos, defensive_target_agent_pos)
                     for goal_pos in goals]
        # Select the minimum distance
        min_distance_index = np.argmin(distances)
        target_pos = goals[min_distance_index]
        strategic_mode = 'APPROACH'
    elif agent_mode == 'OFFENSIVE':
      if agent_ball:
        # Calculate the distance from the opponent
        goals = self.map_data.goals[team_name]
        distances = [self.get_pos_distance(goal_pos, nearest_opponent_pos)
                     for goal_pos in goals]
        # Select the maximum distance
        max_distance_index = np.argmax(distances)
        target_pos = goals[max_distance_index]
        strategic_mode = 'APPROACH'
      else:
        target_pos = defensive_target_agent_pos
        strategic_mode = 'INTERCEPT'
    else:
      raise KeyError('Unknown agent mode {}'.format(agent_mode))
    # Get the strategic action
    action = self._get_strategic_action(agent_pos, target_pos, strategic_mode)
    return action

class SoccerPlayer(RLEnvironment):
    """
    A wrapper for pygame_soccer emulator.
    Will automatically restart when a real episode ends (isOver might be just
    lost of lives but not game over).
    """
    SOCCER_WIDTH = 288
    SOCCER_HEIGHT = 192

    def __init__(self, viz=0, 
                field=None, partial=False, radius=2,
                frame_skip=4, 
                image_shape=(84, 84), 
                mode=None, team_size=1, ai_frame_skip=1, raw_env=soccer_environment.SoccerEnvironment):
        super(SoccerPlayer, self).__init__()
        
        if mode != None:
            if team_size > 1:
                self.mode = mode.split(',')
            else:
                self.mode = [ mode ]
        else:
            self.mode = mode
        self.field = field
        self.partial = partial
        self.viz = viz
        if self.viz:
            self.renderer_options = soccer_renderer.RendererOptions(
                show_display=True, max_fps=10, enable_key_events=True)
        else:
            self.renderer_options = None

        if self.field == 'large' :
            map_path = file_util.resolve_path(__file__, 'data/map/soccer_large.tmx')
        else :
            map_path = None

        self.team_size = team_size
        self.env_options = soccer_environment.SoccerEnvironmentOptions(team_size=self.team_size, map_path=map_path, ai_frame_skip=ai_frame_skip)
        self.env = raw_env(env_options=self.env_options, renderer_options=self.renderer_options)

        self.computer_team_name = self.env.team_names[1]
        self.player_team_name = self.env.team_names[0]

        # Partial
        if self.partial:
            self.radius = radius
            self.player_agent_index = self.env.get_agent_index(self.player_team_name, 0)
 
        self.actions = self.env.actions
        self.frame_skip = frame_skip
        self.image_shape = image_shape
        
        self.last_info = {}
        self.agent_actions = ['STAND'] * (self.team_size * 2)
        self.changing_counter = 0
        self.all_changing_counter = 0
        self.all_episode = 0

        self.current_episode_score = StatCounter()
        self.restart_episode()

    def _grab_raw_image(self):
        self.env.render()
        if self.partial:
            screenshot = self.env.renderer.get_po_screenshot(self.player_agent_index, self.radius)
        else:
            screenshot = self.env.renderer.get_screenshot()
        return screenshot

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
    
    def _set_computer_mode(self, mode):
        if mode == None:
            return
        # Collaborator
        for i in range(1, self.team_size):
            index = self.env.get_agent_index(self.player_team_name, i)
            m = mode[self.team_size * 0 + i - 1]
            self.env.state.set_agent_mode(index, m)
        # Opponent
        for i in range(self.team_size):
            index = self.env.get_agent_index(self.computer_team_name, i)
            m = mode[self.team_size * 1 + i - 1]
            self.env.state.set_agent_mode(index, m)
   
    def current_state(self):
        ret = self._grab_raw_image()
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
        self._set_computer_mode(self.mode)
        self.last_raw_screen = self._grab_raw_image()
        self.changing_counter = 0

    def action(self, act):
        ball_pos_agent_old = self.env.state.get_ball_possession()
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
        ball_pos_agent_new = self.env.state.get_ball_possession()
        
        if ball_pos_agent_old['team_name'] == ball_pos_agent_new['team_name'] and ball_pos_agent_new['team_name'] == 'PLAYER':
            if ball_pos_agent_old['team_agent_index'] != ball_pos_agent_new['team_agent_index']:
                self.changing_counter = min(1, self.changing_counter + 1)

        if isOver:
            self.all_episode += 1
            self.all_changing_counter += max(self.changing_counter * r, 0)
            print('Changing rate', float(self.all_changing_counter) / self.all_episode)
            self.finish_episode()
            self.restart_episode()
        return (r, isOver)

    def get_internal_state(self):
        return self.last_info   
    def get_changing_counter(self):
        return self.changing_counter

def get_raw_env(experiment):
    if experiment == 'STANDARD':
        return soccer_environment.SoccerEnvironment
    elif experiment == 'PASSING':
        return SoccerPassingBallEnvironment
    elif experiment == 'SAVING':
        return SoccerSavingBallEnvironment
    assert 0

if __name__ == '__main__':
    pl = SoccerPlayer(image_shape=(84, 84), viz=1, frame_skip=1, field='large', ai_frame_skip=1, 
            team_size=2, raw_env=SoccerPassingBallEnvironment)
    rng = get_rng(5)
    import time
    while True:
        # im = a.grab_image()
        # cv2.imshow(a.romname, im)
        act = rng.choice(range(5))
        act = 4
        r, o = pl.action(act)
        pl.current_state()
        print(pl.get_changing_counter())
        # time.sleep(0.1)
