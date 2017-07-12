import numpy as np
import tensorflow as tf
import argparse, logging
import os, sys

from tqdm import *

from model import DeepQNetwork
from learn import DeepQLearner
from agent import DeepQAgent

from dqn import DeepQTrainer, DeepQReplayMemory, DeepQStateBuilder

from soccer_player import SoccerPlayer

from util import get_config

def main(args):
    learning_rate = 4e-4
    gamma = 0.99
    eps = 0.99
    eps_decay = 0.9
    min_eps = 0.1
    max_timestep = 5000000
    replay_mem_capacity = 1000000
    min_replay_mem_size = 50000
    dqn_state_shape = [84, 84, 4]
    env_image_shape = [192, 288, 3]
    batch_size = 64
    num_action = 5
    report_stat_period = 2500

    logging.info('Initialize model with [state_shape = %s, num_action = %d] ... ' % (dqn_state_shape, num_action))
    model = DeepQNetwork(name='DQN', reuse=False, state_shape=dqn_state_shape, num_action=num_action)
 
    logging.info('Initialize environment with [image shape = %s] ' % (env_image_shape))
    state_builder = DeepQStateBuilder(image_shape=env_image_shape)
    env = SoccerPlayer(state_builder=state_builder, viz=True)
    
    import random

    env.reset()
    for t in range(10):
        next_state, reward, done, _ = env.step(random.randint(0, 4))
    
    import cv2
    for i in range(0, 4):
        cv2.imwrite('test-state-%d.png' % i, next_state[:, :, i])
    logging.info('Test state finished')
   
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args) 
