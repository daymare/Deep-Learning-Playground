

# global hyperparameters
batch_size = 32 # how many experiences to use for each minibatch
update_freq = 4 # how often to perform a minibatch in our training
gamma = .99

startEpsilon = 1
endEpsilon = 0.1

annealing_steps = 10000. # how many steps of training to reduce epsiolon over

num_episodes = 220

pre_train_steps = 10000 # how many steps of random actions to take before training begins

max_episode_length = 50

load_model = False # whether to load a saved model
path = "./dqn" # path to save our model to

h_size = 512
tau = 0.001 # rate to update target network towards primary network



