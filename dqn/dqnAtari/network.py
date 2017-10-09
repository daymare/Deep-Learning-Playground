
from __future__ import division
import numpy as np

import random

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys

import params



# experience replay class for storing samples and building minibatches
class ExperienceBuffer():
    def __init__(self, buffer_size = 30000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.current_size = 0

    def add(self, experience):
        self.current_size = min(self.current_size + len(experience), self.buffer_size)
        
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = [] # clear earlier elements until we have enough to fit the new elements
        self.buffer.extend(experience)

    # TODO review this and make sure it is correct
    def sample(self, size):
        # ensure there is enough history to sample
        assert self.current_size > params.history_per_state

        samples = []
        numSamples = 0

        for numSamples in range(size):
            # get a random index
            index = random.randint(params.history_per_state, self.current_size-1)

            # get SARS from that index
            SARS = self.buffer[index]

            # get the history of the last few states
            stateWithHistory = []
            for i in range(params.history_per_state):
                stateWithHistory.insert(0, self.buffer[index-i][0])

            # get the resulting state with history
            rStateWithHistory = []
            for i in range(params.history_per_state):
                rStateWithHistory.insert(0, self.buffer[index-i][3])

            # convert state to np array and reshape
            npStateWithHistory = np.array(stateWithHistory)
            npStateWithHistory = npStateWithHistory.reshape(210, 160, -1)
            npRStateWithHistory = np.array(rStateWithHistory)
            npRStateWithHistory = npRStateWithHistory.reshape(210, 160, -1)

            # repackage new SARS information into a new np array
            sample = []
            sample.append(npStateWithHistory)
            sample.append(SARS[1])
            sample.append(SARS[2])
            sample.append(npRStateWithHistory)
            sample.append(SARS[4])

            npSample = np.array(sample)

            # add sample to batch
            samples.append(npSample)

        # convert batch to np array and return
        npSamples = np.array(samples)

        return npSamples



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
        self.imageInput = tf.placeholder(shape=[None, None, None, params.history_per_state], dtype=tf.float32)
        self.processedImage = tf.image.resize_images(self.imageInput, [84, 84])
        print 'image input: ', self.imageInput
        print 'processed image: ', self.processedImage

        self.conv1 = slim.conv2d(inputs=self.processedImage, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None)
        print 'conv1: ', self.conv1

        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None)
        print 'conv2: ', self.conv2

        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        print 'conv3: ', self.conv3

        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)
        print 'conv4: ', self.conv4

        # advantage and value layers
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        print 'streamAC: ', self.streamAC
        print 'streamVC: ', self.streamVC

        self.streamA = slim.flatten(self.streamAC)
        print 'streamA: ', self.streamA

        self.streamV = slim.flatten(self.streamVC)
        print 'streamV: ', self.streamV

        xavier_init = tf.contrib.layers.xavier_initializer()
        self.advantageWeights = tf.Variable(xavier_init([h_size//2, env.action_space.n]))
        print 'advantageWeights: ', self.advantageWeights

        self.valueWeights = tf.Variable(xavier_init([h_size//2, 1]))
        print 'valueWeights: ', self.valueWeights

        self.Advantage = tf.matmul(self.streamA, self.advantageWeights)
        self.Value = tf.matmul(self.streamV, self.valueWeights)

        # combine advantage and value layers
        self.output = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.output, 1)

        # obtain loss and build trainer
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.env.action_space.n, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
        self.updateModel = self.trainer.minimize(self.loss)

