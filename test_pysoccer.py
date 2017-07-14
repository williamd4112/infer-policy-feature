import tensorflow as tf
import argparse, logging
import os, sys

from tqdm import *

from model import DeepQNetwork
from agent import DeepQAgent
from dqn import DeepQStateBuilder
 
from soccer_player import SoccerPlayer

from util import get_config, load_model

def main(args):
    dqn_state_shape = [84, 84, 4]
    env_image_shape = [192, 288, 3]
    num_action = 5

    logging.info('Initialize environment with [image shape = %s] ' % (env_image_shape))
    state_builder = DeepQStateBuilder(image_shape=env_image_shape)
    env = SoccerPlayer(state_builder=state_builder, mode=args.opponent, viz=args.viz)

    logging.info('Initialize model with [state_shape = %s, num_action = %d] ... ' % (dqn_state_shape, num_action))
    model = DeepQNetwork(name='DQN', 
                    reuse=False, 
                    state_shape=dqn_state_shape, 
                    num_action=num_action, 
                    with_target_network=False)
    
    with tf.Session(config=get_config()) as sess:
        load_model(sess, args.load)
        agent = DeepQAgent(sess=sess, 
                        model=model, 
                        num_action=num_action, 
                        eps=0.1, 
                        decay=1.0,
                        decay_period=1,
                        min_eps=0.1)

        logging.info('Initialize tf variables ... ')
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        n_episode = 0
        max_episode = args.episode
        state = env.reset()
        while n_episode < max_episode:
            action = agent.act([state])[0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            
            if done:
                logging.info('Episode %d = %f: %s, eps = %f' % (n_episode, reward, env.stat(), agent.eps))
                n_episode += 1

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='checkpoints', type=str, default=None)
    parser.add_argument('--episode', help='number of episode to run', type=int, default=100)
    parser.add_argument('--opponent', help='oppoenent strategy', type=str, default=None)
    parser.add_argument('--viz', help='visualize', type=bool, default=True)
    args = parser.parse_args()
    main(args) 
