
import tensorflow as tf
import os

from network import *

import random

from network import *

import params



class QValues:
    def __init__(self, env):
        tf.reset_default_graph()

        # experience buffer
        self.expBuffer = ExperienceBuffer()

        # network approximators
        self.mainNetwork = QNetwork(params.h_size, env)
        self.targetNetwork = QNetwork(params.h_size, env)

        # define target network update
        target_weights = self.targetNetwork.model.trainable_weights
        self.update_target_network = [target_weights[i].assign(q_network_weights[i]) for i in range(len(target_wegiths))]

        # define loss and gradient update operation
        self.a, self.y, self.loss, self.gradient_update = self.mainNetwork.build_training_ops()

        # create session, saver, and summary placeholders
        main_weights = self.mainNetwork.model.trainable_weights
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(main_weigts)
        self.summary_placeholders, self.update_operations, self.summary_operation = self.setup_summary()
        self.summary_writer = tf.train.SummaryWriter(params.summaryPath, self.sess.graph)

    def unpackStateBatch(self, stateArray):
        listResult = []
        
        for image in stateArray:
            listResult.append(image)

        npResult = np.array(listResult)

        return npResult

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray


    # update the network q values
    # TODO update
    def updateQ(self, state, action, reward, rState, isTerminal):
        self.expBuffer.add(np.reshape(np.array([state, action, reward, rState, isTerminal]), [1,5]))

        if params.pre_train_override > 0:
            params.pre_train_override -= 1
            return

        if params.pre_train_override == 0:
            print 'exiting pre-train override mode!'
            params.pre_train_override -= 1

        if params.total_steps > params.pre_train_steps:
            if params.total_steps % (params.update_freq) == 0:
                trainBatch = self.expBuffer.sample(params.batch_size)

                stateBatch = self.unpackStateBatch(trainBatch[:, 0])
                resultStateBatch = self.unpackStateBatch(trainBatch[:, 3])

                mainQ = self.session.run(self.mainQN.predict, feed_dict={self.mainQN.imageInput:resultStateBatch})
                #mainQValue = self.session.run(self.mainQN.output, feed_dict={self.mainQN.imageInput:resultStateBatch})

                end_multiplier = -(trainBatch[:,4] - 1)
                targetQ = self.session.run(self.targetQN.output, feed_dict={self.targetQN.imageInput:resultStateBatch})
                doubleQ = targetQ[range(params.batch_size), mainQ]
                newTargetQ = trainBatch[:,2] + (params.gamma * doubleQ * end_multiplier)


                _ = self.session.run(self.mainQN.updateModel, feed_dict={self.mainQN.imageInput:stateBatch, self.mainQN.targetQ:newTargetQ, self.mainQN.actions:trainBatch[:,1]})
                updateTarget(self.targetOperations, self.session)


                """
                debug printouts
                newnewmainQValue = self.session.run(self.mainQN.output, feed_dict={self.mainQN.imageInput:resultStateBatch})
                newnewtargetQ = self.session.run(self.targetQN.output, feed_dict={self.targetQN.imageInput:resultStateBatch})

                print 'main q: ', mainQ[0]
                print 'main q value: ', mainQValue[0]
                print 'target q: ', targetQ[0]

                print 'double q: ', doubleQ[0]

                print 'reward: ', trainBatch[:,2][0]

                print 'newTargetQ: ', newTargetQ[0]

                print 'result main q:', newnewmainQValue[0]
                print 'result target q:', newnewtargetQ[0]
                raw_input()
                """

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
        if not os.path.exists(params.savePath):
            os.makedirs(params.savePath)

        # save the network
        self.saver.save(self.session, params.savePath+'/model-' + str(params.total_steps) + '.cptk')
        print 'Saved Model'
        

    # load the most recent model saved to the default path
    def loadQ(self):
        print 'Loading Model...'
        chkpt = tf.train.get_checkpoint_state(params.savePath) self.saver.restore(self.session, chkpt.model_checkpoint_path) 
