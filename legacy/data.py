from collections import deque
import random
import numpy as np

import cv2

class StateBuilderProxy(object):
    def __init__(self, state_builder):
        self.state_builder = state_builder

    def reset(self):
        self.state_builder.reset()

    def set_state(self, obs):
        self.state_builder.set_state(obs)

    def get_state(self, copy=False):
        return self.state_builder.get_state(copy=copy)

    def __call__(self, obs):
        self.set_state(obs)
        return self.get_state(copy=False)
       
class FrameStateBuilder(object):
    def __init__(self, state_shape, state_dtype):
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.state = np.zeros(self.state_shape, dtype=self.state_dtype)

    def reset(self):
        self.state.fill(0)    

    def set_state(self, observation):
        self.state = observation

    def get_state(self, copy=False):
        return self.state.copy() if copy else self.state

class GrayscaleFrameStateBuilder(StateBuilderProxy):
    def __init__(self, state_builder):
        super(GrayscaleFrameStateBuilder, self).__init__(state_builder)

    def get_state(self, copy=False):
        return cv2.cvtColor(self.state_builder.get_state(copy=copy), cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]

class ResizeFrameStateBuilder(StateBuilderProxy):
    def __init__(self, state_builder, shape):
        super(ResizeFrameStateBuilder, self).__init__(state_builder)
        self.shape = shape

    def get_state(self, copy=False):
        resized_img = cv2.resize(self.state_builder.get_state(copy=copy), self.shape)
        if len(resized_img.shape) == 2:
            resized_img = resized_img[:, :, np.newaxis]
        return resized_img
        
class StackedFrameStateBuilder(StateBuilderProxy):
    def __init__(self, state_builder, size):
        super(StackedFrameStateBuilder, self).__init__(state_builder)
        self.size = size
        
        state_shape = self.state_builder.get_state().shape
        state_dtype = self.state_builder.get_state().dtype

        self.n_channel = state_shape[-1]
        self.state_buffer = np.zeros(state_shape[:-1] + (state_shape[-1] * self.size,), dtype=state_dtype)

    def reset(self):
        self.state_builder.reset()
        self.state_buffer.fill(0) 

    def set_state(self, observation):
        self.state_builder.set_state(observation)
        state = self.state_builder.get_state(copy=False)
        self._append_state(state)

    def get_state(self, copy=False):
        return self.state_buffer.copy() if copy else self.state_buffer
 
    def _append_state(self, state):
        self.state_buffer[:, :, :-self.n_channel] = self.state_buffer[:, :, self.n_channel:]
        self.state_buffer[:, :, -self.n_channel:] = state 

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.buffer = deque(maxlen=self.capacity)

    def append(self, exp):
        self.buffer.append(exp)
        if self.size < self.capacity:
            self.size += 1

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size) 
        return batch

    def __len__(self):
        return self.size

class NamedReplayMemory(ReplayMemory):
    def __init__(self, capacity, names):
        super(NamedReplayMemory, self).__init__(capacity)
        self.names = names

    def sample_batch(self, batch_size):
        batch = super(NamedReplayMemory, self).sample_batch(batch_size)
        batch_dict = {}
        
        for id, name in enumerate(self.names):
            batch_dict[name] = np.array([ data[id] for data in batch ])

        return batch_dict        

if __name__ == '__main__':
    import gym
    env = gym.make('Pong-v0')
    done = False

    state_builder = FrameStateBuilder([210, 160, 3], np.uint8)
    state_builder = ResizeFrameStateBuilder(state_builder, (84, 84))
    state_builder = StackedFrameStateBuilder(state_builder, 4)
    state_builder.set_state(env.reset())

    replay_mem = NamedReplayMemory(capacity=10000, names=['state', 'action', 'reward', 'next_state', 'done'])

    while not done:
        state = state_builder.get_state(copy=True)
        action = env.action_space.sample()
        next_observation, reward, done, _ = env.step(action)
        state_builder.set_state(next_observation)
        next_state = state_builder.get_state(copy=True)

        replay_mem.append((state, action, reward, next_state, done))
        
    replay_mem.sample_batch(32)
