import numpy as np
import tensorflow as tf

from model import DeepQNetwork
from learn import DeepQLearner
from agent import DeepQAgent

from data import (FrameStateBuilder, GrayscaleFrameStateBuilder, \
        ResizeFrameStateBuilder, StackedFrameStateBuilder, 
        NamedReplayMemory, StateBuilderProxy)

from util import load_model

class DeepQReplayMemory(NamedReplayMemory):
    def __init__(self, model, capacity=1000000):
        self.model = model
        super(DeepQReplayMemory, self).__init__(capacity=capacity, names=[ 
                                                                self.model.get_inputs()['state'],
                                                                self.model.get_inputs()['action'],
                                                                self.model.get_inputs()['reward'],
                                                                self.model.get_inputs()['next_state'],
                                                                self.model.get_inputs()['done']])

class DeepQStateBuilder(StateBuilderProxy):
    def __init__(self, image_shape=[192, 288, 3]):
        state_builder = FrameStateBuilder(image_shape, np.uint8)
        state_builder = GrayscaleFrameStateBuilder(state_builder)
        state_builder = ResizeFrameStateBuilder(state_builder, (84, 84))
        state_builder = StackedFrameStateBuilder(state_builder, 4)
        super(DeepQStateBuilder, self).__init__(state_builder) 


class DeepQTrainer(object):
    def __init__(self, sess, model, learner, 
            clip_val=10.0,
            update_freq=250, 
            save_freq=10000, 
            load=None, name='dqn-model'):
        self.sess = sess
        self.clip_val = clip_val
        self.update_freq = update_freq
        self.save_freq = save_freq
        self.load = load
        self.name = name

        # Get neccessary variables from learner
        optimizer = learner.get_optimizer()
        grads_vars = learner.get_grads_vars()
        
        # Preprocess gradient and vars
        grads = [k[0] for k in grads_vars]
        vars = [k[1] for k in grads_vars]
        grads, _ = tf.clip_by_global_norm(grads, self.clip_val)
        clipped_grads_vars = list(zip(grads, vars))

        # Build train operation
        self.train_op = optimizer.apply_gradients(clipped_grads_vars)
        
        # Build update operation
        self.update_op = learner.get_update_target_network_op()

        # Get global_step operation
        self.global_step_var  = learner.get_global_step()
        self.global_step_op = tf.assign(self.global_step_var, self.global_step_var + 1)
        
        # Build model saver
        self.saver = tf.train.Saver()
        if load is not None:
            self.saver = load_model(self.sess, load)

    def train(self, batch):
        _, self.global_step = self.sess.run([self.train_op, self.global_step_op], feed_dict=batch)
        self.global_step = int(self.global_step)

        if self.global_step % self.update_freq == 0:
            self.sess.run([self.update_op]) 

        if self.global_step % self.save_freq == 0:
            self.save()

        return self.global_step

    def get_global_step(self):
        return int(self.sess.run(self.global_step_var))
   
    def save(self):
        self.saver.save(self.sess, self.name, global_step=self.global_step_var)

