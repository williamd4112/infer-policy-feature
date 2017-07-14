import tensorflow as tf
import argparse, logging
import os, sys

from tqdm import *

from model import DeepQNetwork
from learn import DeepQLearner
from agent import DeepQAgent
from dqn import DeepQTrainer, DeepQReplayMemory, DeepQStateBuilder 
from wrapper import train_dqn

from soccer_player import SoccerPlayer
from util import get_config

def main(args):
    learning_rate = 0.001
    gamma = 0.99

    dqn_state_shape = [84, 84, 4]
    env_image_shape = [192, 288, 3]
    num_action = 5

    logging.info('Initialize model with [state_shape = %s, num_action = %d] ... ' % (dqn_state_shape, num_action))
    model = DeepQNetwork(name='DQN', reuse=False, state_shape=dqn_state_shape, num_action=num_action)

    logging.info('Initialize learner with [lr = %f, gamma = %f] ... ' % (learning_rate, gamma))
    learner = DeepQLearner(model, lr=learning_rate, gamma=gamma)
  
    logging.info('Initialize environment with [image shape = %s], [opponent = %s] ' % (env_image_shape, args.opponent))
    state_builder = DeepQStateBuilder(image_shape=env_image_shape)
    env = SoccerPlayer(state_builder=state_builder, frame_skip=4, mode=args.opponent, viz=args.viz)
    
    train_dqn(env=env, model=model, learner=learner, num_action=num_action, load=args.load)
    
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='checkpoints', type=str, default=None)
    parser.add_argument('--opponent', help='oppoenent strategy', type=str, default=None)
    parser.add_argument('--viz', help='visualize', type=bool, default=False)
    args = parser.parse_args()
    main(args) 
