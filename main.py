import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import connect_four

env = gym.make("connect_four/ConnectFour-v0", render_mode="human")
num_steps = 42

obs = env.reset()
is_done = False
found = False

print("The initial observation is {}".format(obs))

for step in range(num_steps):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print("The  observation is {}".format(obs))
    print("The reward is {}".format(reward))
    env.render()
    time.sleep(0.1)
    if done:
        time.sleep(5)
        break
        # env.reset()
# while not found:
#     while not is_done:
#         action = env.action_space.sample()
#         obs, reward, is_done, _, info = env.step(action)
#         print(obs)
#         print("The reward is {}".format(reward))
#         env.render()
#         time.sleep(0.001)
#         if np.abs(reward) == 10:
#             found = True
#             break

#         if is_done:
#             env.reset()
#             is_done = False

env.close()