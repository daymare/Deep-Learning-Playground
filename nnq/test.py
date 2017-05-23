

import gym
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')


print env.action_space.n
print env.observation_space.n


