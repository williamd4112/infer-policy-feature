import numpy as np
import tensorflow as tf
import os, sys

import gym

from tqdm import *
from util import MovingAverageEpisodeRewardCounter

class GymAtariPlayer(object):
    def __init__(self, name, state_builder, viz=False):
        self.env = gym.make(name)
        self.state_builder = state_builder
        self.done = False
        self.reward = 0.0
        self.counter = MovingAverageEpisodeRewardCounter()
  
    def state(self):
        return self.state_builder.get_state()

    def reset(self):
        self.done = False
        self.reward = 0.0
        obs = self.env.reset()
        self.state_builder.reset()
        self.state_builder.set_state(obs)
        return self.state_builder.get_state()

    def step(self, act):
        next_obs, reward, done, _ = self.env.step(act)
        self.state_builder.set_state(next_obs)
        next_state = self.state_builder.get_state()

        self.done = done
        self.reward += reward

        # Auto reset
        if self.done:
            self.counter.add(self.reward)
            self.reset()

        return next_state, reward, done, None
    
    def stat(self):
        return self.counter()

