import tensorflow as tf
import argparse, logger
import os, sys

from tqdm import *

from model import DeepQNetwork
from learn import DeepQLearner
from agent import DeepQAgent

from dqn import DeepQTrainer, DeepQReplayMemory, DeepQStateBuilder
from dqn_policy_infer import DeepPolicyInferQReplayMemory
 
from soccer_player import SoccerPlayer

from util import get_config, load_model
from util import LinearInterpolation

MAX_REPLAY_MEMORY_SIZE = 1000000
MIN_REPLAY_MEMORY_SIZE = 50000
TRAIN_FREQ = 4
PER_EPOCH_STEP = 10000 // TRAIN_FREQ * 10

def _get_deep_q_agent(sess, env, model):
    INIT_STEP = MIN_REPLAY_MEMORY_SIZE
    INIT_EPS = 1.0
    STOP_EPS = 0.1
    STOP_STEP = INIT_STEP + PER_EPOCH_STEP * 10

    num_action = env.get_num_action()

    logger.info('Initialize agent with [(%d, %f) -> (%d, %f)] ... ' % (INIT_STEP, INIT_EPS, STOP_STEP, STOP_EPS))
    agent = DeepQAgent(sess=sess, 
                        model=model, 
                        num_action=num_action, 
                        eps=LinearInterpolation((INIT_STEP, INIT_EPS), (STOP_STEP, STOP_EPS)))
    return agent

def play_model(env, model, load, max_episode): 
    num_action = env.get_num_action()

    with tf.Session(config=get_config()) as sess:
        load_model(sess, load)
        agent = DeepQAgent(sess=sess, 
                        model=model, 
                        num_action=num_action, 
                        eps=0.1, 
                        decay=1.0,
                        decay_period=1,
                        min_eps=0.1)

        logger.info('Initialize tf variables ... ')
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        n_episode = 0
        env.reset()
        while n_episode < max_episode:
            state = env.state()
            action = agent.act([state])[0]
            next_state, reward, done, _ = env.step(action)
            
            if done:
                logger.info('Episode %d = %f: %s, eps = %f' % (n_episode, reward, env.stat(), agent.eps))
                n_episode += 1


def train_dqn_policy_infer(env, model, learner, num_action, init_eps, load):
    # Agent hyper parameter
    eps = init_eps
    eps_decay = 0.9
    eps_decay_period = 25000
    min_eps = 0.1
    
    # Training hyper parameter
    max_timestep = 5000000
    batch_size = 64
    report_stat_period = 25000
    
    # Replay memory hyper paramter
    replay_mem_capacity = 1000000
    min_replay_mem_size = 50000
    
    logger.info('Initialize replay memory with [capacity = %d]' % (replay_mem_capacity))
    replay = DeepPolicyInferQReplayMemory(model=model, capacity=replay_mem_capacity)

    with tf.Session(config=get_config()) as sess:
        logger.info('Initialize trainer ... ')
        trainer = DeepQTrainer(sess=sess, 
                            model=model, 
                            learner=learner,
                            load=load)

        agent = _get_deep_q_agent(sess, model)    

        
        logger.info('Initialize tf variables ... ')
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        logger.info('Check available checkpoints ...')
        assert trainer is not None
        trainer.restore(load)

        global_step = trainer.get_global_step()
        logger.info('Start from global_step = %d ' % (global_step))

        try:
            begin_step = int(global_step)
            end_step = max_timestep
            env.reset()
            for timestep in tqdm(range(begin_step, end_step)):
                state = env.state()
                action = agent.act([state])[0]
                next_state, reward, done, opponent_action = env.step(action)

                replay.append((state.copy(), action, reward, next_state.copy(), done, opponent_action))
                if len(replay) >= min_replay_mem_size:
                    loss, global_step = trainer.train(replay.sample_batch(batch_size))
                    if (timestep + 1) % report_stat_period == 0:
                        logger.info('global_step = %d, loss = %f' % (global_step, loss))
                
                if (timestep + 1) % report_stat_period == 0:
                    global_step = trainer.get_global_step()
                    logger.info('global_step = %d, %s, eps = %f' % (global_step, env.stat(), agent.eps))

        except KeyboardInterrupt:
            logger.info ('Saving the model ...')
            trainer.save()

def train_dqn(env, model, learner, num_action, load): 
    # Training hyper parameter
    max_timestep = 5000000
    batch_size = 64
    report_stat_period = PER_EPOCH_STEP
     
    logger.info('Initialize replay memory with [capacity = %d]' % (MAX_REPLAY_MEMORY_SIZE))
    replay = DeepQReplayMemory(model=model, capacity=MAX_REPLAY_MEMORY_SIZE)

    with tf.Session(config=get_config()) as sess:
        logger.info('Initialize trainer ... ')
        trainer = DeepQTrainer(sess=sess, 
                            model=model, 
                            learner=learner,
                            load=load)
        
        agent = _get_deep_q_agent(sess, env, model)

        logger.info('Initialize tf variables ... ')
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        logger.info('Check available checkpoints ...')
        trainer.restore(load)

        global_step = trainer.get_global_step()
        logger.info('Start from global_step = %d ' % (global_step))

        try:
            begin_step = int(global_step)
            end_step = max_timestep
            env.reset()
            for timestep in tqdm(range(begin_step, end_step)):
                state = env.state()
                action = agent.act([state])[0]
                next_state, reward, done, _ = env.step(action)
                replay.append((state.copy(), action, reward, next_state.copy(), done))

                if len(replay) >= MIN_REPLAY_MEMORY_SIZE:
                    if (timestep) % TRAIN_FREQ == 0:
                        loss, global_step = trainer.train(replay.sample_batch(batch_size))
                        if (timestep) % report_stat_period == 0:
                            logger.info('global_step = %d, loss = %f' % (global_step, loss))
                
                if (timestep) % report_stat_period == 0:
                    global_step = trainer.get_global_step()
                    logger.info('global_step = %d, %s, eps = %f' % (global_step, env.stat(), agent.get_eps()))

        except KeyboardInterrupt:
            logger.info ('Saving the model ...')
            trainer.save()
