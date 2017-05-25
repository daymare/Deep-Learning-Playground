from __future__ import division

import gym
import numpy as np
import random

import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import scipy.misc
import os

import params


# experience replay class for storing samples and building minibatches
class ExperienceBuffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = [] # clear earlier elements until we have enough to fit the new elements
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size,5])




def processState(states):
    return np.reshape(states, [21168])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx + total_vars//2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# neural network approximator
class QNetwork:
    def __init__(self, h_size, env):

        self.env = env

        # convolutional layers
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageInput = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])

        self.conv1 = slim.conv2d(inputs=self.imageInput, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)

        # advantage and value layers
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.advantageWeights = tf.Variable(xavier_init([h_size//2, env.actions]))
        self.valueWeights = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.advantageWeights)
        self.Value = tf.matmul(self.streamV, self.valueWeights)

        # combine advantage and value layers
        self.output = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.output, 1)

        # obtain loss and build trainer
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

