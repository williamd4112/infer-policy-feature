import numpy as np
import tensorflow as tf

from model import DeepQNetwork
from learn import DeepQLearner

from data import (FrameStateBuilder, GrayscaleFrameStateBuilder, \
        ResizeFrameStateBuilder, StackedFrameStateBuilder, 
        NamedReplayMemory, StateBuilderProxy)

from util import load_model

class DeepRecurrentQReplayMemory(NamedReplayMemory):
    def __init__(self, model, capacity=1000000):
        self.model = model
        super(DeepRecurrentQReplayMemory, self).__init__(capacity=capacity, names=[ 
                                                                self.model.get_inputs()['state'],
                                                                self.model.get_inputs()['action'],
                                                                self.model.get_inputs()['reward'],
                                                                self.model.get_inputs()['next_state'],
                                                                self.model.get_inputs()['done']])

class DeepRecurrentQStateBuilder(StateBuilderProxy):
    def __init__(self, image_shape=[192, 288, 3]):
        state_builder = FrameStateBuilder(image_shape, np.uint8)
        state_builder = GrayscaleFrameStateBuilder(state_builder)
        state_builder = ResizeFrameStateBuilder(state_builder, (84, 84))
        state_builder = StackedFrameStateBuilder(state_builder, 4)
        super(DeepRecurrentQStateBuilder, self).__init__(state_builder) 

class DeepRecurrentQNetwork(DeepQNetwork):
    def _init_inputs(self):
        self.inputs = { 'state': tf.placeholder(dtype=tf.uint8, shape=[None,] + self.state_shape, name='state'),
                        'action': tf.placeholder(dtype=tf.int32, shape=[None,], name='action'),
                        'reward': tf.placeholder(dtype=tf.float32, shape=[None,], name='reward'),
                        'next_state': tf.placeholder(dtype=tf.uint8, shape=[None,] + self.state_shape, name='next_state'),
                        'done': tf.placeholder(dtype=tf.bool, shape=[None,], name='done'),
                        'episode_len': tf.placeholder(dtype=tf.int32, shape=[None,], name='episode_len')}
        
    def _build_q_network(self, state):
        l = state
        l = Conv2D(l, [8, 8], 32, 4, 'VALID', 'conv0', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu0') 
        l = Conv2D(l, [4, 4], 64, 2, 'VALID', 'conv1', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu1')
        l = Conv2D(l, [3, 3], 64, 1, 'VALID', 'conv2', reuse=self.reuse)
        l = PReLu(l, 0.001, 'relu2')
        l = FC(l, 512, 'fc0')
        l = LeakyReLu(l, 0.01, 'relu3')
        l = FC(l, self.num_action, 'fc1')
        return l

class DeepRecurrentQLearner(DeepQLearner): 
    def _build_optimize_op(self):
        # Model arguments
        num_action = self.model.num_action

        # Inputs
        state = self.model.get_inputs()['state']
        action = self.model.get_inputs()['action']
        reward = self.model.get_inputs()['reward']
        next_state = self.model.get_inputs()['next_state']
        done = self.model.get_inputs()['done']

        # Outputs
        policy_q = self.model.get_outputs()['policy_q']
        target_q = self.model.get_outputs()['target_q']

        # Loss
        action_one_hot = tf.one_hot(action, num_action, 1.0, 0.0)
        pred = tf.reduce_sum(policy_q * action_one_hot, 1)

        target_q_max = tf.reduce_max(target_q, 1)
        target = reward + (1.0 - tf.cast(done, tf.float32)) * self.gamma * tf.stop_gradient(target_q_max)
        loss = tf.reduce_mean(HuberLoss(target - pred), name='loss')
        self.loss = loss

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-3)
        self.grads_vars = self.optimizer.compute_gradients(loss)
        self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0.0), trainable=False) 

class DeepRecurrentQTrainer(object):
    def __init__(self, sess, model, learner, 
            clip_val=10.0,
            update_freq=2500, 
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
        self.loss = learner.get_loss()
        
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
        self.global_step = 0
        
        # Build model saver
        self.saver = tf.train.Saver()
        if load is not None:
            self.saver = load_model(self.sess, load)

    def train(self, batch):
        if self.global_step % self.update_freq == 0:
            self.sess.run([self.update_op]) 

        _, loss, self.global_step = self.sess.run([self.train_op, self.loss, self.global_step_op], feed_dict=batch)
        self.global_step = int(self.global_step)

        if self.global_step % self.save_freq == 0:
            self.save()

        return loss, self.global_step

    def get_global_step(self):
        return int(self.sess.run(self.global_step_var))
   
    def save(self):
        self.saver.save(self.sess, self.name, global_step=self.global_step_var)

