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

from gridworld import gameEnv

from params import *
from approximator import *

# q learning with a deep neural network function approximator


env = gameEnv(partial=False, size=5)



tf.reset_default_graph()
mainQN = QNetwork(h_size, env)
targetQN = QNetwork(h_size, env)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOperations = updateTargetGraph(trainables, tau)

myBuffer = ExperienceBuffer()

# epsilon action decrease
epsilon = startEpsilon
stepDrop = (startEpsilon - endEpsilon) / annealing_steps

# lists for total rewards and steps per episode
jList = []
rList = []
total_steps = 0


# ensure we have a path to save to
if not os.path.exists(path):
    os.makedirs(path)


with tf.Session() as sess:
    sess.run(init)

    # load the model
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, chkpt.model_checkpoint_path)

    updateTarget(targetOperations, sess) # set the target network to be equal to the primary network

    for i in range(num_episodes):
        episodeBuffer = ExperienceBuffer()

        # reset environment and get first observation
        state = env.reset()
        state = processState(state)
        done = False
        totalReward = 0
        j = 0

        while j < max_episode_length:
            j+=1

            # get a decision either randomly or through the network
            if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
                action = np.random.randint(0, 4)
            else:
                action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[state]})[0]


            # tell environment to take action
            rState, reward, done = env.step(action)
            rState = processState(rState)
            total_steps += 1

            #plt.show(env.renderEnv)
            #time.sleep(1)


            # add to episode buffer
            episodeBuffer.add(np.reshape(np.array([state, action, reward, rState, done]), [1,5]))

            
            if total_steps > pre_train_steps:
                # apply epsilon decay
                if epsilon > endEpsilon:
                    epsilon -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)

                    mainQ = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    targetQ = sess.run(targetQN.output, feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})

                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = targetQ[range(batch_size), mainQ]
                    newTargetQ = trainBatch[:,2] + (gamma * doubleQ * end_multiplier)

                    _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:newTargetQ, mainQN.actions:trainBatch[:,1]})

                    updateTarget(targetOperations, sess)


            totalReward += reward
            state = rState

            if done == True:
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(totalReward)

        # periodically save the model
        if i % 1000 == 0:
            saver.save(sess, path+'/model-' + str(i) + '.cptk')
            print 'Saved Model'

        if len(rList) % 10 == 0:
            print(len(rList), total_steps, np.mean(rList[-10:]), epsilon)

    saver.save(sess, path+'/model-' + str(i) + '.cptk')

print "Percent of successful episodes: " + str(sum(rList) / num_episodes) + "%"

# reward over time
rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
rMean = np.average(rMat, 1)

plt.gcf().clear()

print rMean
plt.plot(rMean)
plt.show()










