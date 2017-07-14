import numpy as np
import tensorflow as tf

from model import DeepQNetwork
from learn import DeepQLearner
from agent import DeepQAgent

from data import (FrameStateBuilder, GrayscaleFrameStateBuilder, \
        ResizeFrameStateBuilder, StackedFrameStateBuilder, 
        NamedReplayMemory, StateBuilderProxy)

from util import load_model

class DeepPolicyInferQReplayMemory(NamedReplayMemory):
    def __init__(self, model, capacity=1000000):
        self.model = model
        super(DeepQReplayMemory, self).__init__(capacity=capacity, names=[ 
                                                                self.model.get_inputs()['state'],
                                                                self.model.get_inputs()['action'],
                                                                self.model.get_inputs()['reward'],
                                                                self.model.get_inputs()['next_state'],
                                                                self.model.get_inputs()['done'],
                                                                self.model.get_inputs()['opponent_action']]) 
