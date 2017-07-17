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
from baselines.common.atari_wrappers_deprecated import FrameStack, ScaledFloatFrame

class ProcessSoccerFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessSoccerFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        return ProcessSoccerFrame84.process(obs)

    @staticmethod
    def process(frame):
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        obs = self.env.reset()
        return obs

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
            info['computer_action'] = self.action_set.index(ret.computer_action)
            reward += ret.reward
        ob = self._get_obs()

        return ob, reward, self.soccer_env.state.is_terminal(), info
    
    def get_action_meanings(self):
        return self.action_meaning 

def wrap_dqn_for_soccer(env, skip=4):
    env = SkipEnv(env, skip=skip)
    env = ProcessSoccerFrame84(env)
    env = FrameStack(env, 4)
    return env
