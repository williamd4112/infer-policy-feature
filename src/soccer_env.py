#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import cv2
from collections import deque
import threading
import six
from six.moves import range
import random
from tensorpack.utils.utils import (get_rng, execute_only_once)
from tensorpack.utils import logger
from tensorpack.utils.fs import get_dataset_path
from tensorpack.utils.stats import StatCounter

from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace

import pygame_rl.scenario.soccer_environment as soccer_environment
import pygame_rl.scenario.soccer_renderer as soccer_renderer
import pygame_rl.util.file_util as file_util
import pdb

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
                frame_skip=4, verbose=False,
                image_shape=(84, 84),
                mode=None, team_size=1, ai_frame_skip=1, train_type='both', raw_env=soccer_environment.SoccerEnvironment):
        super(SoccerPlayer, self).__init__()

        if team_size > 1 and mode != None:
            self.mode = mode.split(',')
        else:
            self.mode = ['DQN_OPPONENT']
        self.field = field
        self.partial = partial
        self.viz = viz
        self.verbose = verbose
        if self.viz:
            self.renderer_options = soccer_renderer.RendererOptions(
                show_display=True, max_fps=10, enable_key_events=True)
        else:
            self.renderer_options = None

        if self.field == 'large' :
            map_path = file_util.resolve_path(__file__, '../data/map/soccer_large.tmx')
        else :
            map_path = None

        self.team_size = team_size
        self.env_options = soccer_environment.SoccerEnvironmentOptions(team_size=self.team_size, map_path=map_path, ai_frame_skip=ai_frame_skip)
        self.env = raw_env(env_options=self.env_options, renderer_options=self.renderer_options)
        self.train_type = train_type

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
        self.timestep = 0
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

    def _set_opponent_mode(self, mode):
        for i in range(self.team_size):
            index = self.env.get_agent_index(self.computer_team_name, i)
            m = mode[i]
            self.env.state.set_agent_mode(index, m)

    def _set_collaborator_mode(self, mode):
        for i in range(1, self.team_size):
            index = self.env.get_agent_index(self.player_team_name, i)
            m = mode[i - 1]
            self.env.state.set_agent_mode(index, m)

    def _set_computer_mode(self, mode):
        if mode[0] == None or len(mode) < self.team_size * 2 - 1:
            return
        if mode[0] in ['OFFENVIE', 'DFFENSIVE']:
            # Collaborator
            if self.team_size >= 2:
                self._set_collaborator_mode(mode[:(self.team_size - 1)])
            # Opponent
            self._set_opponent_mode(mode[(self.team_size - 1):])

    def current_state(self):
        ret = self._grab_raw_image()
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        ret = cv2.resize(ret, self.image_shape)
        return ret.astype('uint8')  # to save some memory

    def get_action_space(self):
        return DiscreteActionSpace(len(self.actions))

    def finish_episode(self):
        score = np.array(self.current_episode_score._values)
        self.stats['score'].append([np.sum(score[:,0]), np.sum(score[:,1])])

    def restart_episode(self):
        self.current_episode_score.reset()
        self.env.reset()
        self._set_computer_mode(self.mode)
        self.last_raw_screen = self._grab_raw_image()
        self.changing_counter = 0
        self.timestep = 0

    def action(self, act):
        ball_pos_agent_old = self.env.state.get_ball_possession()
        r = np.array([0., 0.])
        ball_poss_old = self.env.state.get_ball_possession()['team_name']
        for k in range(self.frame_skip):
            self.timestep += 1
            if k == self.frame_skip - 1:
                self.last_raw_screen = self._grab_raw_image()

            if self.mode[0] == 'DQN_OPPONENT' or type(act) is list:
                actions = {}
                if self.train_type == 'both':
                    for team_name in self.env.team_names:
                        for team_agent_index in range(self.env.options.team_size):
                            agent_index = self.env.get_agent_index(team_name, team_agent_index)
                            actions[agent_index] = self.env.actions[act[agent_index]]
                elif self.train_type == 'dqn':
                    actions[0] = None
                    actions[1] = self.env.actions[act[1]]
                else:
                    actions[0] = self.env.actions[act[0]]
                    actions[1] = None
                if self.verbose:
                    print("actions: {}".format(actions))
                ret = self.env.take_action(actions)

            if self.mode[0] == 'WEAKCOOP':
                actions = {}
                for team_name in self.env.team_names:
                    for team_agent_index in range(self.env.options.team_size):
                        agent_index = self.env.get_agent_index(team_name, team_agent_index)
                        agent_action = self.env._get_ai_action(team_name, team_agent_index)
                        actions[agent_index] = agent_action
                player_index = self.env.get_agent_index(self.player_team_name, 0)
                coop_index = self.env.get_agent_index(self.player_team_name, 1)

                actions[player_index] = self.env.actions[act]
                if random.random() < 0.5:
                    actions[coop_index] = random.choice(self.env.actions)
                ret = self.env.take_all_actions(actions)

            if self.mode[0] == 'OPPONENT_DYNAMIC':
                choices = ['OFFENSIVE', 'DEFENSIVE']
                if self.timestep % random.randint(4, 10) == 0:
                    new_modes = [random.choice(choices) for i in range(self.team_size)]
                    self._set_opponent_mode(new_modes)

            if self.mode[0] == 'COOP_DYNAMIC':
                choices = ['OFFENSIVE', 'DEFENSIVE']
                if self.timestep % random.randint(4, 10) == 0:
                    new_modes = [random.choice(choices) for i in range(self.team_size - 1)]
                    self._set_collaborator_mode(new_modes)

            if self.mode[0] == 'ALL_RANDOM':
                if self.team_size == 1:
                    player_index = self.env.get_agent_index(self.player_team_name, 0)
                    opponent_index = self.env.get_agent_index(self.computer_team_name, 0)
                    actions = {player_index: self.env.actions[act],
                           opponent_index: random.choice(self.env.actions)}
                else:
                    actions = {}
                    for team_name in [self.player_team_name, self.computer_team_name]:
                        for team_index in range(self.team_size):
                            agent_index = self.env.get_agent_index(team_name, team_index)
                            actions[agent_index] = random.choice(self.env.actions)
                    player_index = self.env.get_agent_index(self.player_team_name, 0)
                    actions[player_index] = self.env.actions[act]
                ret = self.env.take_all_actions(actions)
            if k == 0:
                self.last_info['agent_actions'] = self._get_computer_actions()
            r += np.array([ret.reward, -ret.reward])

            if self.env.state.is_terminal():
                break

        self.current_episode_score.feed(r)
        isOver = self.env.state.is_terminal()
        ball_pos_agent_new = self.env.state.get_ball_possession()

        if ball_pos_agent_old['team_name'] == ball_pos_agent_new['team_name'] and ball_pos_agent_new['team_name'] == 'PLAYER':
            if ball_pos_agent_old['team_agent_index'] != ball_pos_agent_new['team_agent_index']:
                self.changing_counter += 1

        if isOver:
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
        return soccer_environment.SoccerPassingBallEnvironment
    elif experiment == 'SAVING':
        return soccer_environment.SoccerSavingBallEnvironment
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
