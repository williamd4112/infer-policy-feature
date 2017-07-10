import numpy as np
import tensorflow as tf
import argparse, logging
import os, sys

import soccer

from tqdm import *

from model import DeepQNetwork
from learn import DeepQLearner
from agent import DeepQAgent
from train import FeedDictTrainer, PeriodicCallback

from data import FrameStateBuilder, ResizeFrameStateBuilder, StackedFrameStateBuilder, NamedReplayMemory
from util import get_config

def main(args):
    env = soccer.SoccerEnvironment(renderer_max_fps=60)

    model = DeepQNetwork(name='DQN', reuse=False, state_shape=[84, 84, 12], num_action=5)
    learner = DeepQLearner(model, lr=1e-4, gamma=0.99)
    agent = DeepQAgent(model=model)
    trainer = FeedDictTrainer(learner, callbacks=[PeriodicCallback(lambda sess : sess.run([learner.get_update_target_network_op()]), 10000)])
    
    state_builder = FrameStateBuilder([210, 160, 3], np.uint8)
    state_builder = ResizeFrameStateBuilder(state_builder, (84, 84))
    state_builder = StackedFrameStateBuilder(state_builder, 4)
    
    replay_mem = NamedReplayMemory(capacity=100000, names=[ model.get_inputs()['state'],
                                                            model.get_inputs()['action'],
                                                            model.get_inputs()['reward'],
                                                            model.get_inputs()['next_state'],
                                                            model.get_inputs()['done'] ])
    min_replay_mem_size = 10000
    batch_size = 32

    def observe(env):
        env.render()
        obs = env.renderer.get_screenshot()
        obs = np.fromstring(obs, dtype=np.uint8)
        obs = np.reshape(obs, [192, 288, 4])
        obs = obs[:,:,:3]
        return obs

    with tf.Session(config=get_config()) as sess:
        # Create initializer
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        
        # Train agent
        print ('Training ...')
        saver = tf.train.Saver()
        if args.load is not None:
            print ('Loading ...')
            saver = tf.train.import_meta_graph('Soccer.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./'))

        global_step = sess.run([learner.get_global_step()])[0]
        try:
            begin_step = int(global_step)
            end_step = 5000000 + min_replay_mem_size
            done = False

            env.reset()
            state_builder.reset()
            state_builder.set_state(observe(env))

            for timestep in tqdm(range(begin_step, end_step)):
                state = state_builder.get_state(copy=True)
                action = agent.act(sess, [state])

                #next_observation, reward, done, _ = env.step(action)
                response = env.take_action(action)
                reward = response.reward
                action = response.action
                done = env.state.is_terminal()

                state_builder.set_state(observe(env))
                next_state = state_builder.get_state(copy=True)

                replay_mem.append((state, action, reward, next_state, done))

                if replay_mem.size < min_replay_mem_size:
                    global_step = trainer.train(sess, replay_mem.sample_batch(batch_size))

                if done:
                    done = False
                    state_builder.reset()
                    state_builder.set_state(env.reset())

        except KeyboardInterrupt:
            print ('Saving the model ...')
            saver.save(sess, 'Soccer', global_step=global_step)       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='checkpoints', type=str, default=None)
    args = parser.parse_args()
    main(args) 
