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


# so we need to keep an array of samples and then train our classifier on those samples every once and a while

class ApproxQ:
    def __init__(self):
        self.inputData = {}
        self.desiredOutputs = {}


class QLearning:
    def __init__(self):
        # environments
        self.env = gym.make('Taxi-v2')
        #self.env = gym.make('FrozenLake-v0')

        self.Q = np.tile(1/float(self.env.action_space.n), (self.env.observation_space.n, self.env.action_space.n))
        self.epsilon = 0.1
        self.alpha = 1.0
        self.gamma = 0.9

        self.displayTime = 0
        self.numTimesteps = 0
        return

    def reinitialize(self):
        self.Q = np.tile(1/float(self.env.action_space.n), (self.env.observation_space.n, self.env.action_space.n))
        return

    def train(self, numEpisodes):
        for i in range(numEpisodes):
            self.runEpisode

    def runEpisode(self, maxSteps=100, training=False):
        total_reward = 0
        state = self.env.reset()

        for i in range(maxSteps):
            # display or do not display the environment
            if self.displayTime > 0:
                self.displayTime -= 1
                print('\n\n\n\n')
                print('timesteps: ', self.numTimesteps)
                self.env.render()
                time.sleep(0.25)

            if self.numTimesteps % 20000 == 0:
                self.displayTime = 20

            self.numTimesteps += 1

            # perform alpha decay
            self.alpha = self.alpha / 1.00001

            # get action
            random_num = rand.random()
            if random_num < self.epsilon and training == True:
                action = self.env.action_space.sample()
            else:
                action = self.eGreedy(state)

            # update environment and QFunction
            next_state, reward, terminal, _ = self.env.step(action)
            if training == True:
                self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + (self.alpha) * (reward + self.gamma * self.Q[next_state].max())

            state = next_state
            total_reward += reward

            if terminal:
                break

        return total_reward

    def eGreedy(self, state):
        return self.Q[state].argmax()

    
    def test(self, numEpisodes=3000):
        rewards = []
        legend = []
        averagedRewards = []

        plt.title("QLearning")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")

        for i in range(numEpisodes):
            self.runEpisode(100, True)
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

    q.test(10000)
    return

if __name__ == '__main__':
    main()






