
import tensorflow as tf
import os

from network import *

import random

from network import *

import params



class QValues:
    def __init__(self, env):
        tf.reset_default_graph()

        ### variables
        # network approximators
        self.mainQN = QNetwork(params.h_size, env)
        self.targetQN = QNetwork(params.h_size, env)

        # utility 
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session = tf.Session()

        # target operations
        self.trainables = tf.trainable_variables()
        self.targetOperations = updateTargetGraph(self.trainables, params.tau)

        # experience buffer
        self.expBuffer = ExperienceBuffer() # mmmmm exp

        ### setup for training
        # initialize the session
        self.session.run(self.init)
        # set the target network to be equal to the primary network
        updateTarget(self.targetOperations, self.session)

    def unpackStateBatch(self, stateArray):
        listResult = []
        
        for image in stateArray:
            listResult.append(image)

        npResult = np.array(listResult)

        return npResult


    # update the network q values
    def updateQ(self, state, action, reward, rState, isTerminal):
        self.expBuffer.add(np.reshape(np.array([state, action, reward, rState, isTerminal]), [1,5]))

        if params.total_steps > params.pre_train_steps:

            if params.total_steps % (params.update_freq) == 0:
                trainBatch = self.expBuffer.sample(params.batch_size)

                stateBatch = self.unpackStateBatch(trainBatch[:, 0])
                resultStateBatch = self.unpackStateBatch(trainBatch[:, 3])

                mainQ = self.session.run(self.mainQN.predict, feed_dict={self.mainQN.imageInput:resultStateBatch})
                targetQ = self.session.run(self.targetQN.output, feed_dict={self.targetQN.imageInput:resultStateBatch})
                end_multiplier = -(trainBatch[:,4] - 1)

                doubleQ = targetQ[range(params.batch_size), mainQ]
                #newTargetQ = trainBatch[:,2] + (gamma * doubleQ * end_multiplier)
                newTargetQ = trainBatch[:,2] + (params.gamma * doubleQ) # try removing end multiplier because it does not make sense to me

                _ = self.session.run(self.mainQN.updateModel, feed_dict={self.mainQN.imageInput:stateBatch, self.mainQN.targetQ:newTargetQ, self.mainQN.actions:trainBatch[:,1]})

                updateTarget(self.targetOperations, self.session)

    # get the max value action in a given state
    def getArgmax(self, state, actionSpace):
        return self.session.run(self.mainQN.predict, feed_dict={self.mainQN.imageInput:[state]})[0]

    # get the q value of a particular state and action
    def getQ(self, state, action):
        listState = []
        listState.append(state)
        processedState = np.array(listState)

        qvalues = self.session.run(self.mainQN.output, feed_dict={self.mainQN.imageInput:processedState})
        return qvalues

    # save the model into the default path 
    def saveQ(self):
        # ensure we have a path to save to
        if not os.path.exists(params.path):
            os.makedirs(params.path)

        # save the network
        self.saver.save(self.session, params.path+'/model-' + str(params.total_steps) + '.cptk')
        print 'Saved Model'
        

    # load the most recent model saved to the default path
    def loadQ(self):
        print 'Loading Model...'
        chkpt = tf.train.get_checkpoint_state(params.path)
        self.saver.restore(self.session, chkpt.model_checkpoint_path)


