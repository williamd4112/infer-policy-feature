import numpy as np
import os
import gym

import cv2

from gym import error, spaces
from gym import utils
from gym.utils import seeding

import pygame_soccer.soccer.soccer_environment as soccer_environment
import pygame_soccer.soccer.soccer_renderer as soccer_renderer

import logging
logger = logging.getLogger(__name__)

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import NoopResetEnv, MaxAndSkipEnv, FrameStack, ClippedRewardsWrapper, ScaledFloatFrame

class ProcessSoccerFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessSoccerFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        return ProcessSoccerFrame84.process(obs)

    @staticmethod
    def process(frame):
        img = frame
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class SoccerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    image_shape = (192, 288, 3)

    action_set = [
      'STAND',
      'MOVE_RIGHT',
      'MOVE_UP',
      'MOVE_LEFT',
      'MOVE_DOWN'
    ]

    action_meaning = [
        'NOOP',
        'MOVE_RIGHT',
        'MOVE_UP',
        'MOVE_LEFT',
        'MOVE_DOWN'
    ]

    def __init__(self, frameskip=(2, 5), mode=None, viz=False):
        assert mode in ['OFFENSIVE', 'DEFENSIVE', None]
        self.frameskip = frameskip
        self.mode = mode
        self.viz = viz
        
        renderer_options = None
        if viz:
            renderer_options = soccer_renderer.RendererOptions(
                show_display=True, max_fps=10, enable_key_events=True)

        self.soccer_env = soccer_environment.SoccerEnvironment(renderer_options)
    
        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.image_shape)

    def _get_obs(self):
        self.soccer_env.render()
        return self.soccer_env.renderer.get_screenshot()

    @property
    def _n_actions(self):
        return len(self.action_set)

    def _reset(self):
        self.soccer_env.reset()
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if self.viz:
            self.soccer_env.render()

    def _step(self, a):
        reward = 0.0
        action = self.action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        
        info = {}
        for _ in range(num_steps):
            ret = self.soccer_env.take_action(action)
            # TODO: pass computer action back
            info['computer_action'] = ret.computer_action
            reward += ret.reward
        ob = self._get_obs()

        return ob, reward, self.soccer_env.state.is_terminal(), info
    
    def get_action_meanings(self):
        return self.action_meaning 

def wrap_dqn_for_soccer(env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessSoccerFrame84(env)
    env = FrameStack(env, 4)
    env = ClippedRewardsWrapper(env)
    return env
