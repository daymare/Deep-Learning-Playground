#!/usr/bin/python2.7

import gym
import time
import sys
sys.path.append("../")

import scipy
import scipy.stats
import numpy as np
import random as rand
import matplotlib.pyplot as plt

import tensorflow as tf


# q learning with a neural network approximator



# neural network approximator
class ApproxQ:
    def __init__(self, actionSpace, stateSpace):
        # setup tensorflow graph
        tf.reset_default_graph()

        # session and init
        self.session = tf.Session()

        # space constraints
        self.numActions = actionSpace.n
        self.numStates = stateSpace.n

        # network architecture for q function approximator
        # assume actions are discrete
        # NOTE output are action q values
        self.inputs = tf.placeholder(shape=[1, self.numStates], dtype=tf.float32)
        self.weights = tf.Variable(tf.random_uniform([self.numStates, self.numActions], 0, 0.01))
        self.output = tf.matmul(self.inputs, self.weights.initialized_value())
        self.actionChoice = tf.argmax(self.output, 1)

        # netowrk loss and training
        self.nextQ = tf.placeholder(shape=[1, self.numActions], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.output))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.update = self.trainer.minimize(self.loss)

        # initialize the session
        self.session.run(tf.global_variables_initializer())


    def updateQ(self, state, action, reward, rState, alpha, gamma, actionSpace):
        # obtain the current q value
        curQvec = self.session.run(self.output, feed_dict={self.inputs:np.identity(self.numStates)[state:state+1]})
        curQ = curQvec[0][action]

        # obtain the resulting state q values
        rQ = self.session.run(self.output, feed_dict={self.inputs:np.identity(self.numStates)[rState:rState+1]})
        maxRQ = np.max(rQ)

        # calculate our target q value
        targetQ = curQvec
        targetQ[0, action] = (1-alpha)*curQ + alpha * (reward + gamma*maxRQ)

        # train the network using target and predicted q values
        _, W1 = self.session.run([self.update, self.weights], feed_dict={self.inputs:np.identity(self.numStates)[state:state+1], self.nextQ:targetQ})

    # get the max value action in a given state
    def getArgmax(self, state, actionSpace):
        maxAction = self.session.run(self.actionChoice, feed_dict={self.inputs:np.identity(self.numStates)[state:state+1]})
        return maxAction[0]

    def getQ(self, state, action):
        # get the q vector for all q values in this state
        QValues = self.session.run(self.output, feed_dict={self.inputs:np.identity(self.numStates)[state:state+1]})
        return QValues[0][action]



class QLearning:
    def __init__(self):
        # environments
        #self.env = gym.make('Taxi-v2')
        self.env = gym.make('FrozenLake-v0')

        self.Q = ApproxQ(self.env.action_space, self.env.observation_space)
        self.epsilon = 0.1
        self.alpha = 1.0
        self.gamma = 0.9

        self.episodes = 0
        return

    def reinitialize(self):
        self.Q = ApproxQ(self.env.action_space, self.env.observation_space)
        return

    def train(self, numEpisodes):
        for i in range(numEpisodes):
            self.runEpisode()

    def runEpisode(self, maxSteps=100, training=False, display=False):
        total_reward = 0
        state = self.env.reset()

        for i in range(maxSteps):
            # perform alpha decay
            self.alpha = self.alpha / 1.00001

            # get action
            random_num = rand.random()
            if random_num < self.epsilon and training == True:
                action = self.env.action_space.sample()
            else:
                action = self.eGreedy(state)

            # display the environment
            if display == True:
                print(chr(27) + "[2J]")
                print("episodes: " + str(self.episodes) + "\n")
                print("learning rate: " + str(self.alpha) + "\n")
                qValue = self.Q.getQ(state, action)
                print("chosen action Q-Value: " + str(qValue) + "\n")
                self.env.render()
                time.sleep(0.10)

            # update environment and QFunction
            next_state, reward, terminal, _ = self.env.step(action)
            if training == True:
                self.Q.updateQ(state, action, reward, next_state, self.alpha, self.gamma, self.env.action_space)

            state = next_state
            total_reward += reward

            if terminal:
                break

        return total_reward

    def eGreedy(self, state):
        # find the max reward state
        maxAction = self.Q.getArgmax(state, self.env.action_space)
        return maxAction

    
    def test(self, numEpisodes=3000, viewDelay=2000, display=False):
        rewards = []
        legend = []
        averagedRewards = []

        plt.title("QLearning")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")

        for i in range(numEpisodes):
            if i % viewDelay == 0:
                print ("episode ", i)
            
            self.episodes = i
            if display:
                self.runEpisode(100, True, i%viewDelay==0)
            else:
                self.runEpisode(100, True, False)
            rewards.append(self.runEpisode(100, False))

        averages = range(numEpisodes-100)

        # create averaged rewards
        summation = 0
        for i in range(numEpisodes):
            if i // 100 == 0:
                summation += rewards[i]
            else:
                summation += rewards[i]
                summation -= rewards[i-100]
                averagedRewards.append(summation/100)


        plt.plot(averages, averagedRewards)
        rewards = []
        legend = []
        self.reinitialize()

        plt.show()
        return

def main():
    q = QLearning()

    q.test(8000, 200, False)
    #q.test(10000, 9999999999999999)
    return

if __name__ == '__main__':
    main()






