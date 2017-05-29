
import params

import matplotlib.pyplot as plt


params.loadOnlineParameters()


legend = []
averagedRewards = []

plt.title("QLearning")
plt.xlabel("Time Step")
plt.ylabel("Reward")


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

