import numpy as np
import matplotlib.pyplot as plt

total_reward_episode = []
file = open('total_reward_episode_v0d_600k.txt', 'r')
for line in file:
    line = line.strip()
    reward = int(float(line))
    total_reward_episode.append(reward)
# total_reward_episode = file.readlines()
# total_reward_episode = [np.rint(total_reward_episode[episode]) for episode in range(600000)]

plt.figure()
plt.plot(total_reward_episode[:600000])
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()