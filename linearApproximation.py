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

from sklearn import linear_model

# NOTE algorithm does not learn. I think because there are not enough features for the linear model to have the power to capture our Q function.


# linear approximated q learning
# will update the qfunction with values once in a given number of examples
class ApproxQ:
    def __init__(self):
        # training data
        self.inputData = []
        self.desiredOutputs = []

        # model update delay
        self.modelUpdateTime = 1 # update the model every sample for now
        self.numSamples = 0

        # model
        self.model = linear_model.LinearRegression()

        x = np.zeros((2, 1), dtype=np.int32)
        y = np.zeros((1, 1), dtype=np.int32)

        y = y.reshape(-1, 1)
        x = x.reshape(-1, 2)

        self.model.fit(x, y)


    def setQ(self, state, action, value):
        # add state and action information to our model
        self.inputData.append((state, action))
        self.desiredOutputs.append(value)

        # check if we need to update the approximation
        if self.numSamples % self.modelUpdateTime == 0:
            # update the model
            x = np.array(self.inputData)
            y = np.array(self.desiredOutputs)

            x = x.reshape(-1, 2)
            y = y.reshape(-1, 1)

            self.model.fit(x, y)

    def updateQ(self, state, action, reward, rState, alpha, gamma, actionSpace):
        _, maxQValue = self.getArgmax(state, actionSpace)

        currentQ = self.getQ(state, action)
        newInfo = reward + gamma * maxQValue

        newQValue = (1-alpha)*currentQ + alpha * newInfo
        self.setQ(state, action, newQValue)


    def getQ(self, state, action):
        # get the model to spit out which action we want
        x = np.array([state, action])
        x = x.reshape(1, -1)

        prediction = self.model.predict(x)
        return prediction

    # get the max value action in a given state
    # only works with descrete actions!
    def getArgmax(self, state, actionSpace):

        # default is random action if there is nothing better
        best = -1
        bestAction = actionSpace.sample()

        # find the argmax
        for action in range(actionSpace.n):
            value = self.getQ(state, action)

            if value > best:
                best = value
                bestAction = action

        return bestAction, best



class QLearning:
    def __init__(self):
        # environments
        self.env = gym.make('Taxi-v2')
        #self.env = gym.make('FrozenLake-v0')

        self.Q = ApproxQ()
        self.epsilon = 0.1
        self.alpha = 1.0
        self.gamma = 0.9

        self.episodes = 0
        return

    def reinitialize(self):
        self.Q = ApproxQ()
        return

    def train(self, numEpisodes):
        for i in range(numEpisodes):
            self.runEpisode

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
                time.sleep(0.25)

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
        maxAction, _ = self.Q.getArgmax(state, self.env.action_space)
        return maxAction

    
    def test(self, numEpisodes=3000, viewDelay=2000):
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
            self.runEpisode(100, True, i%viewDelay==0)
            #self.runEpisode(100, True, False)
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

    q.test(10000, 1)
    return

if __name__ == '__main__':
    main()






