import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.common.atari_wrappers_deprecated import ScaledFloatFrame

# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from soccer_env import SoccerEnv, wrap_dqn_for_soccer
from model import model

# import for policy infer
# TODO: implement AugmentReplayMemory for policy infer
#from data.augment_replay import AugmentReplayBuffer
#from dqn_policy_infer.build_graph import build_train, build_act

def parse_args():
    parser = argparse.ArgumentParser("Soccer environment")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000, help="number of iterations between every target network update")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None, help="directory in which training state and model should be saved.")
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()

def make_env():
    env = SoccerEnv(frameskip=1)
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn_for_soccer(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env

def maybe_save_model(savedir, container, state):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    args = parse_args()
    
    # Parse savedir and azure container.
    savedir = args.save_dir
    container = None

    # Create and seed the env.
    env, monitored_env = make_env()

    with U.make_session(4) as sess:
        # Create training graph and replay buffer
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph = lambda name: U.Uint8Input(env.observation_space.shape, name=name),  
            q_func = model,
            num_actions = env.action_space.n,
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma = 0.99,
            grad_norm_clipping = 10.0,
            double_q=False
        )

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 50, 0.1),
            (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)

        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, None)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()

        # Main trianing loop
        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            action = act(np.array(obs)[None], update_eps=exploration.value(num_iters))[0]
            new_obs, rew, done, info = env.step(action)

            # TODO: info can be opponent action
            opponent_action = info['computer_action']
            print ('Opponent action %d' % opponent_action)          
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            if done:
                obs = env.reset()

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                weights = np.ones_like(rewards)

                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env.get_state()
                })

            if info["steps"] > args.num_steps:
                break

            if done:
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 1)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                logger.record_tabular("exploration", exploration.value(num_iters))
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
