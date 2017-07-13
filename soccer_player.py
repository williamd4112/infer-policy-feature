import numpy as np
import tensorflow as tf
import os, sys

import soccer

from tqdm import *
from util import MovingAverageEpisodeRewardCounter

class SoccerPlayer(object):
    IMAGE_SHAPE = [192, 288, 3]

    def __init__(self, state_builder, frame_skip=4, viz=False, mode=None):
        # Create a renderer options
        renderer_options = soccer.RendererOptions(
                                show_display=True, max_fps=10, enable_key_events=True) if viz else None
        assert mode in ['OFFENSIVE', 'DEFENSIVE', None]

        self.mode = mode
        self.frame_skip = frame_skip
        self.env = soccer.SoccerEnvironment(renderer_options)
        self.state_builder = state_builder
        self.done = False
        self.reward = 0.0
        self.counter = MovingAverageEpisodeRewardCounter()
  
    def observe(self):
        self.env.render()
        obs = self.env.renderer.get_screenshot()
        return obs

    def reset(self):
        self.done = False
        self.reward = 0.0
        self.env.reset()
        self.state_builder.reset()
        if self.mode is not None:
            self.env.state.set_computer_agent_mode(self.mode)
        return self.state_builder(self.observe())

    def step(self, act):
        reward = 0.0
        done = False
        for t in range(self.frame_skip):
            if t == self.frame_skip - 1:       
                next_obs = self.observe()
            response = self.env.take_action(self.env.actions[act])
            reward += response.reward
            done = self.env.state.is_terminal()            
            if done:
                break
        next_state = self.state_builder(next_obs)

        self.done = done
        self.reward += reward

        # Auto reset
        if self.done:
            self.counter.add(self.reward)
            self.reset()

        return next_state, reward, done, None
    
    def stat(self):
        return self.counter()

