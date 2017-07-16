import tensorflow as tf
import numpy as np
import random

from baselines.deepq.replay_buffer import ReplayBuffer

class AugmentReplayBuffer(ReplayBuffer):
    '''
        AugmentReplayBuffer store action as tuple
        (self action, computer action)
    '''
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, opponent_actions = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]

            # NOTE: action is (self_action, computer_action)
            obs_t, action, reward, obs_tp1, done = data
            self_action = action[0]
            opponent_action = action[1]

            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(self_action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            opponent_actions.append(np.array(opponent_action, copy=False))

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(opponent_actions)
 
