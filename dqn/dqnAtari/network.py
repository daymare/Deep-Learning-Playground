
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


# neural network approximator
class QNetwork:
    def __init__(self, h_size, env):
        self.env = env
        self.num_actions = env.action_space.n

        self.buildNetwork(h_size, env)


    def build_training_ops(self):
        # trainable weights
        network_weights = self.model.trainable_weights

        # action placeholder
        a = tf.placeholder(tf.int64, [None])
        # ??? what is y
        y = tf.placeholder(tf.float32, [None])

        # convert action to action onehot
        a_one_hot = tf.one_hot(a, self.num_actions)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # clip the error
        # loss is quadratic when error is within (-1, 1) and linear outside that region
        error = tf.abs(y - q_value)
        quadratic_error_part = tf.clib_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        # build optimizer and update function
        optimizer = tf.train.RMSPropOptimizer(params.learning_rate, momentum=params.momentum, epsilon = params.minimum_gradient)
        gradiant_update = optimizer.minimize(loss, var_list=network_weights)

        return a, y, loss, gradiant_update


    def buildNetwork(self, h_size, env):
        # build model
        self.model = Sequential()
        
        # layer 1, 32 filters, 8x8 stride 4 with relu relu activation.
        self.model.add(Conv2D(
            filters=32, 
            activation='relu', 
            kernel_size=(8,8),
            input_shape=(params.image_width, params.image_height, params.history_per_state),
            strides=4))

        # layer 2, 64 filters, 4x4 stride 2 with relu activation.
        self.model.add(Conv2D(
            filters=64,
            activation='relu',
            kernal_size=(4,4),
            strides=2))

        # layer 3, 64 filters, 3x3, stride 1 with relu activation.
        self.model.add(Conv2D(
            filters=64,
            activation='relu',
            kernal_size=(3,3),
            strides=1))

        # final hidden layer, 512 fully connected units
        self.model.add(Dense(
            units=params.h_size,
            activation='relu'))

        # output layer, fully connected linear layer with an output for each action
        self.model.add(Dense(
            units=self.num_actions
            activation='softmax'))

        # compile loss and optimizer
        self.model.compile(
            loss=keras.losses.categorial_crossentropy,
            optimizer=keras.optimizers.SGD(lr=params.learning_rate, momentum=params.momentum),
            metrics=['accuracy'])

        # setup state placeholder?
        self.s = tf.placeholder(tf.float32, [None, params.image_width, params.image_height, params.history_per_state])
        self.q_values = model(s)







