import tensorflow as tf
import argparse, logger
import os, sys

from tqdm import *

from model import DeepQNetwork
from agent import DeepQAgent
from dqn import DeepQStateBuilder
from wrapper import play_model
 
from soccer_player import SoccerPlayer

from util import get_config, load_model

def main(args):
    dqn_state_shape = [84, 84, 4]
    env_image_shape = [192, 288, 3]
    num_action = 5

    logger.info('Initialize environment with [image shape = %s] ' % (env_image_shape))
    state_builder = DeepQStateBuilder(image_shape=env_image_shape)
    env = SoccerPlayer(state_builder=state_builder, mode=args.opponent, viz=args.viz)

    logger.info('Initialize model with [state_shape = %s, num_action = %d] ... ' % (dqn_state_shape, num_action))
    model = DeepQNetwork(name='DQN', 
                    reuse=False, 
                    state_shape=dqn_state_shape, 
                    num_action=num_action, 
                    with_target_network=False)

    play_model(env=env, model=model, load=args.load, max_episode=args.episode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='checkpoints', type=str, default=None)
    parser.add_argument('--episode', help='number of episode to run', type=int, default=100)
    parser.add_argument('--opponent', help='oppoenent strategy', type=str, default=None)
    parser.add_argument('--viz', help='visualize', type=bool, default=True)
    args = parser.parse_args()
    main(args) 
