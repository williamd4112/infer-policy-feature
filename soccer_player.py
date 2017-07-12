import numpy as np
import tensorflow as tf
import os, sys

import soccer

from tqdm import *
from util import MovingAverageEpisodeRewardCounter

class SoccerPlayer(object):
    IMAGE_SHAPE = [192, 288, 3]

    def __init__(self, state_builder, viz=False):
        # Create a renderer options
        renderer_options = soccer.RendererOptions(
                                show_display=True, max_fps=60, enable_key_events=True) if viz else None

        self.env = soccer.SoccerEnvironment(renderer_options)
        self.state_builder = state_builder
        self.done = False
        self.reward = 0.0
        self.counter = MovingAverageEpisodeRewardCounter(length=100)
  
    def observe(self):
        self.env.render()
        obs = self.env.renderer.get_screenshot()
        return obs

    def reset(self):
        self.done = False
        self.reward = 0.0
        self.env.reset()
        self.state_builder.reset()
        return self.state_builder(self.observe())

    def step(self, act):
        response = self.env.take_action(self.env.actions[act])
        next_obs = self.observe()

        next_state = self.state_builder(next_obs)
        reward = response.reward
        action = response.action
        done = self.env.state.is_terminal()

        self.done = done
        self.reward += reward

        # Auto reset
        if self.done:
            self.reset()
            self.counter.add(self.reward)

        return next_state, reward, done, None
    
    def stat(self):
        return self.counter()

