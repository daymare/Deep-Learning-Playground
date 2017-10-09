import pickle


class OnlineParameters:
    def __init__(self):
        self.total_steps = 0
        self.total_episodes = 0
        self.epsilon = 0.
        self.rewards = []


# parameter saving/loading
def saveOnlineParameters():
    # initialize online parameters
    op = OnlineParameters()
    op.total_steps = total_steps
    op.total_episodes = total_episodes
    op.epsilon = epsilon
    op.rewards = rewards

    print 'saved values:'
    print 'total steps: ', total_steps
    print 'total episodes: ', total_episodes
    print 'epsilon: ', epsilon
    print 'rewards: ', rewards
    
    # open pickle file for write
    f = open(paramPath, 'w')

    # pickle op
    pickle.dump(op, f)

def loadOnlineParameters():
    # open pickle file for read
    f = open(paramPath, 'r')

    # unpickle op
    op = pickle.load(f)

    print 'loaded values: '
    print 'total steps: ', op.total_steps
    print 'total episodes: ', op.total_episodes
    print 'epsilon: ', op.epsilon
    print 'rewards: ', op.rewards

    # unload op
    global total_steps 
    global total_episodes 
    global epsilon 
    global rewards 
    global pre_train_override

    pre_train_override = 2000

    total_steps = op.total_steps
    total_episodes = op.total_episodes
    epsilon = op.epsilon
    rewards = op.rewards

    print 'Loaded online parameters!'



# global hyperparameters
batch_size = 32 # how many experiences to use for each minibatch
update_freq = 4 # how often to perform a minibatch in our training
gamma = .99

history_per_state = 4

startEpsilon = 1
endEpsilon = 0.1
annealing_steps = 1000000. # how many steps of training to reduce epsilon over
epsilonStepDrop = (startEpsilon - endEpsilon) / annealing_steps
pre_train_steps = 50000 # how many steps of random actions to take before training begins
pre_train_steps = 5000
pre_train_override = 0

max_episode_length = 10000
num_episodes = 100000

view_toggle = False
episode_view_delay = 1
print_delay = 1

load_model = True # whether to load a saved model
load_parameters = True # whether to load saved learning parameters
save_delay = 500 # how many episodes to wait between saves for the model
path = "./save" # path to save our model to
paramPath = "./onlineParameters" # path to save our online parameters to

h_size = 512
tau = 0.0001 # rate to update target network towards primary network

# online parameters
total_steps = 0
total_episodes = 0
epsilon = startEpsilon
rewards = []



