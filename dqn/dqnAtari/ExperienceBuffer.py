

# system imports
import random
import sys
import copy

# user defined imports
import params


# experience replay class for storing samples and building minibatches
class ExperienceBuffer():
    def __init__(self, buffer_size = 10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.current_size = 0

    # experience: state, action, reward, isTerminal
    def add(self, experience):

        # remove r_state, redundant information
        t_experience = []
        for sample in experience:
            t_sample = (copy.copy(sample[0]), copy.copy(sample[1]), copy.copy(sample[2]), copy.copy(sample[4]))
            t_experience.append(t_sample)
        

        self.current_size = min(self.current_size + len(t_experience), self.buffer_size)
        
        if len(self.buffer) + len(t_experience) >= self.buffer_size:
            self.buffer[0:(len(t_experience)+len(self.buffer))-self.buffer_size] = [] # clear earlier elements until we have enough to fit the new elements

        print(asizeof(self.buffer))
        self.buffer.extend(t_experience)
        print(asizeof(t_experience))
        print(asizeof(experience))
        print(asizeof(self.buffer))
        raw_input()

    def sample(self, size):
        # ensure there is enough history to sample
        assert self.current_size > params.history_per_state

        samples = []
        numSamples = 0

        for numSamples in range(size):
            # get a random index
            index = random.randint(params.history_per_state, self.current_size-2)

            # get SARS from that index
            SARS = self.buffer[index]

            # get the history of the last few states
            stateWithHistory = []
            for i in range(params.history_per_state):
                stateWithHistory.insert(0, self.buffer[index-i][0])

            # get the resulting state with history
            rStateWithHistory = []
            for i in range(params.history_per_state):
                rStateWithHistory.insert(0, self.buffer[index-i+1][0])

            # convert state to np array and reshape
            npStateWithHistory = np.array(stateWithHistory)
            npStateWithHistory = npStateWithHistory.reshape(210, 160, -1)
            npRStateWithHistory = np.array(rStateWithHistory)
            npRStateWithHistory = npRStateWithHistory.reshape(210, 160, -1)

            # repackage new SARS information into a new np array
            sample = []
            sample.append(npStateWithHistory)
            sample.append(SARS[1])
            sample.append(SARS[2])
            sample.append(npRStateWithHistory)
            sample.append(SARS[3])

            npSample = np.array(sample)

            # add sample to batch
            samples.append(npSample)

        # convert batch to np array and return
        npSamples = np.array(samples)

        return npSamples

