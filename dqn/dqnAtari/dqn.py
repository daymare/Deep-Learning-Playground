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

from QValues import QValues
import params


# q learning with a neural network approximator
# modeled from https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb 


class QLearning:
    def __init__(self):
        # environments
        #self.env = gym.make('VideoPinball-v0')
        self.env = gym.make('Breakout-v0')

        # q network
        self.Q = QValues(self.env)

        # load the q network
        if params.load_model == True:
            self.Q.loadQ()

        # load the saved parameters
        if params.load_parameters == True:
            params.loadOnlineParameters()

    def runEpisode(self, maxSteps=100, training=False, display=False):
        total_reward = 0
        state = self.env.reset()

        for i in range(maxSteps):
            # perform epsilon decay
            if params.total_steps > params.pre_train_steps and params.total_steps < params.pre_train_steps + params.annealing_steps:
                params.epsilon -= params.epsilonStepDrop

            if params.total_steps == params.pre_train_steps:
                print 'entering training mode!'

            # get action
            random_num = rand.random()
            if (random_num < params.epsilon or params.total_steps < params.pre_train_steps) and training == True:
                action = self.env.action_space.sample()
            else:
                action = self.eGreedy(state)

            # display the environment
            if display == True:
                print(chr(27) + "[2J]")
                print("episodes: " + str(params.total_episodes) + "\n")
                print "timesteps: ", params.total_steps
                print("epsilon: " + str(params.epsilon) + "\n")
                qValue = self.Q.getQ(state, action)
                print("chosen action Q-Value: " + str(qValue) + "\n")
                self.env.render()

            # update environment and QFunction
            next_state, reward, terminal, _ = self.env.step(action)

            if training == True:
                self.Q.updateQ(state, action, reward, next_state, terminal)
                params.total_steps += 1

            state = next_state
            total_reward += reward

            if terminal:
                break

        if training == True:
            params.total_episodes += 1

        return total_reward

    def eGreedy(self, state):
        # find the max reward state
        maxAction = self.Q.getArgmax(state, self.env.action_space)
        return maxAction

    
    def test(self, viewDelay=2000, display=False):
        legend = []
        averagedRewards = []

        plt.title("QLearning")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")

        for i in range(params.num_episodes - params.total_episodes):

            reward = 0
            
            # train the model
            if display:
                if i % viewDelay == 0 and params.total_steps > params.pre_train_steps:
                    raw_input("Press Enter to continue...")
                    reward = self.runEpisode(params.max_episode_length, True, True)
                else:
                    reward = self.runEpisode(params.max_episode_length, True, False)
            else:
                reward = self.runEpisode(params.max_episode_length, True, False)

            params.rewards.append(reward)

            if i % params.print_delay == 0:
                print "episode: ", params.total_episodes, 'timesteps: ', params.total_steps, 'reward: ', reward

            # save the model
            if i % params.save_delay == 0 and i != 0:
                self.Q.saveQ()
                params.saveOnlineParameters()

        averages = range(params.total_episodes-100)

        # create averaged rewards
        summation = 0
        for i in range(params.total_episodes):
            if i // 100 == 0:
                summation += params.rewards[i]
            else:
                summation += params.rewards[i]
                summation -= params.rewards[i-100]
                averagedRewards.append(summation/100)


        plt.plot(averages, averagedRewards)
        legend = []

        plt.show()


def main():
    q = QLearning()

    q.test(params.episode_view_delay, params.view_toggle)



if __name__ == '__main__':
    main()






