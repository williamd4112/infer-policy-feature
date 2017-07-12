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
    dqn_state_shape = [84, 84, 12]
    env_image_shape = [192, 288, 3]
    batch_size = 64
    num_action = 5
    report_stat_period = 2500

    logging.info('Initialize model with [state_shape = %s, num_action = %d] ... ' % (dqn_state_shape, num_action))
    model = DeepQNetwork(name='DQN', reuse=False, state_shape=dqn_state_shape, num_action=num_action)

    logging.info('Initialize learner with [lr = %f, gamma = %f] ... ' % (learning_rate, gamma))
    learner = DeepQLearner(model, lr=learning_rate, gamma=gamma)

    logging.info('Initialize replay memory with [capacity = %d]' % (replay_mem_capacity))
    replay = DeepQReplayMemory(model=model, capacity=replay_mem_capacity)
  
    logging.info('Initialize environment with [image shape = %s] ' % (env_image_shape))
    state_builder = DeepQStateBuilder(image_shape=env_image_shape)
    env = SoccerPlayer(state_builder=state_builder, viz=args.viz)
   
    with tf.Session(config=get_config()) as sess:
        logging.info('Initialize trainer ... ')
        trainer = DeepQTrainer(sess=sess, 
                            model=model, 
                            learner=learner,
                            load=args.load)
        
        logging.info('Initialize agent with [eps = %f, eps_decay = %f, min_eps = %f] ... ' % (eps, eps_decay, min_eps))
        agent = DeepQAgent(sess=sess, 
                        model=model, 
                        num_action=num_action, 
                        eps=eps, 
                        decay=eps_decay, 
                        min_eps=min_eps)

        logging.info('Initialize tf variables ... ')
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        global_step = int(sess.run([learner.get_global_step()])[0])
        logging.info('Start from global_step = %d ' % (global_step))

        try:
            begin_step = int(global_step)
            end_step = max_timestep
            state = env.reset()
            for timestep in tqdm(range(begin_step, end_step)):
                action = agent.act([state])[0]
                next_state, reward, done, _ = env.step(action)

                replay.append((state.copy(), action, reward, next_state.copy(), done))
                if len(replay) >= min_replay_mem_size:
                    trainer.train(replay.sample_batch(batch_size))
                state = next_state
                
                if timestep % report_stat_period == 0 and done:
                    logging.info('Statistics in recent 100 episodes: %s' % (env.stat()))

        except KeyboardInterrupt:
            logging.info ('Saving the model ...')
            trainer.save()

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='checkpoints', type=str, default=None)
    parser.add_argument('--viz', help='visualize', type=bool, default=False)
    args = parser.parse_args()
    main(args) 
