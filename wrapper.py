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

def train_dqn(env, model, learner, num_action, load):
    # Agent hyper parameter
    eps = 1.0
    eps_decay = 0.8
    eps_decay_period = 25000
    min_eps = 0.1
    
    # Training hyper parameter
    max_timestep = 5000000
    batch_size = 64
    report_stat_period = 25000
    
    # Replay memory hyper paramter
    replay_mem_capacity = 1000000
    min_replay_mem_size = 50000
    
    logging.info('Initialize replay memory with [capacity = %d]' % (replay_mem_capacity))
    replay = DeepQReplayMemory(model=model, capacity=replay_mem_capacity)

    with tf.Session(config=get_config()) as sess:
        logging.info('Initialize trainer ... ')
        trainer = DeepQTrainer(sess=sess, 
                            model=model, 
                            learner=learner,
                            load=load)
        
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

        global_step = trainer.get_global_step()
        logging.info('Start from global_step = %d ' % (global_step))

        try:
            begin_step = int(global_step)
            end_step = max_timestep
            env.reset()
            for timestep in tqdm(range(begin_step, end_step)):
                state = env.state()
                action = agent.act([state])[0]
                next_state, reward, done, _ = env.step(action)

                replay.append((state.copy(), action, reward, next_state.copy(), done))
                if len(replay) >= min_replay_mem_size:
                    loss, global_step = trainer.train(replay.sample_batch(batch_size))
                    if (timestep + 1) % report_stat_period == 0:
                        logging.info('global_step = %d, loss = %f' % (global_step, loss))
                
                if (timestep + 1) % report_stat_period == 0:
                    global_step = trainer.get_global_step()
                    logging.info('global_step = %d, %s, eps = %f' % (global_step, env.stat(), agent.eps))

        except KeyboardInterrupt:
            logging.info ('Saving the model ...')
            trainer.save()
