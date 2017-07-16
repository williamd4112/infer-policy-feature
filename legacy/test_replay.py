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
    eps_decay = 1.0
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

    logging.info('Initialize replay memory with [capacity = %d]' % (replay_mem_capacity))
    replay = DeepQReplayMemory(model=model, capacity=replay_mem_capacity)
  
    logging.info('Initialize environment with [image shape = %s], [opponent = %s] ' % (env_image_shape, args.opponent))
    state_builder = DeepQStateBuilder(image_shape=env_image_shape)
    env = SoccerPlayer(state_builder=state_builder, frame_skip=1, mode=args.opponent, viz=args.viz)
   
    with tf.Session(config=get_config()) as sess: 
        logging.info('Initialize agent with [eps = %f, eps_decay = %f, min_eps = %f] ... ' % (eps, eps_decay, min_eps))
        agent = DeepQAgent(sess=sess, 
                        model=model, 
                        num_action=num_action, 
                        eps=eps, 
                        decay=eps_decay,
                        decay_period=eps_decay_period,
                        min_eps=min_eps)

        logging.info('Initialize tf variables ... ')
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        begin_step = 0
        end_step = 1000
        env.reset()
        for timestep in tqdm(range(begin_step, end_step)):
            state = env.state().copy()
            action = agent.act([state])[0]
            action = action % 4
            next_state, reward, done, _ = env.step(action)
            replay.append((state, action, reward, next_state.copy(), done))

        import cv2
        import numpy as np
        # Sample a batch and dump
        batch = np.asarray(replay.buffer)[:batch_size]
        for t in range(batch_size):
            s = batch[t][0]
            next_s = batch[t][3]
            img_current = np.hstack((s[:,:,0], s[:,:,1], s[:,:,2], s[:,:,3]))
            img_next = np.hstack((next_s[:,:,0], next_s[:,:,1], next_s[:,:,2], next_s[:,:,3]))
            img = np.vstack((img_current, img_next))
            same = np.all(np.equal(img_current, img_next))
            cv2.imwrite('replay_s_t-%04d-%d.png' % (t, same), img)
           

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='checkpoints', type=str, default=None)
    parser.add_argument('--opponent', help='oppoenent strategy', type=str, default=None)
    parser.add_argument('--viz', help='visualize', type=bool, default=False)
    args = parser.parse_args()
    main(args) 
