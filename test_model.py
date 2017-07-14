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
    learning_rate = 0.001
    gamma = 0.99

    eps = 1.0
    eps_decay = 0.99
    eps_decay_period = 10000
    min_eps = 0.1

    max_timestep = 5000000
    replay_mem_capacity = 1000000
    min_replay_mem_size = 50000
    dqn_state_shape = [84, 84, 4]
    env_image_shape = [192, 288, 3]
    batch_size = 64
    num_action = 5
    report_stat_period = 5000

    logging.info('Initialize model with [state_shape = %s, num_action = %d] ... ' % (dqn_state_shape, num_action))
    model = DeepQNetwork(name='DQN', reuse=False, state_shape=dqn_state_shape, num_action=num_action)

    logging.info('Initialize learner with [lr = %f, gamma = %f] ... ' % (learning_rate, gamma))
    learner = DeepQLearner(model, lr=learning_rate, gamma=gamma)

    assert (len(model.get_policy_network_vars()) == len(model.get_target_network_vars()))
    for p, t in zip(model.get_policy_network_vars(), model.get_target_network_vars()):
        print (p, t)   

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='checkpoints', type=str, default=None)
    parser.add_argument('--opponent', help='oppoenent strategy', type=str, default=None)
    parser.add_argument('--viz', help='visualize', type=bool, default=False)
    args = parser.parse_args()
    main(args) 
