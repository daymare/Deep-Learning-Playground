
from __future__ import division
import numpy as np


# tensorflow imports
import tensorflow as tf
import tensorflow.contrib.slim as slim

# keras imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense

# user defined imports
import params


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
        model = Sequential()
        
        # layer 1, 32 filters, 8x8 stride 4 with relu relu activation.
        model.add(Conv2D(
            filters=32, 
            activation='relu', 
            kernel_size=(8,8),
            input_shape=(84, 84, 4),
            strides=4))

        # layer 2, 64 filters, 4x4 stride 2 with relu activation.
        model.add(Conv2D(
            filters=64,
            activation='relu',
            kernal_size=(4,4),
            strides=2))

        # layer 3, 64 filters, 3x3, stride 1 with relu activation.
        model.add(Conv2D(
            filters=64,
            activation='relu',
            kernal_size=(3,3),
            strides=1))

        # final hidden layer, 512 fully connected units
        model.add(Dense(
            units=params.h_size,
            activation='relu'))

        # output layer, fully connected linear layer with an output for each action
        model.add(Dense(
            units=env.action_space.n,
            activation='softmax'))

        # compile loss and optimizer
        model.compile(
            loss=keras.losses.categorial_crossentropy,
            optimizer=keras.optimizers.SGD(lr=params.learning_rate, momentum=params.momentum),
            metrics=['accuracy'])
            







