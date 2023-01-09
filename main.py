import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import connect_four
import random
import torch
from dqn import DQN

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def q_learning(env, estimator, n_episode, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    """
    Deep Q-Learning using DQN
    @param env: Gym environment
    @param estimator: Estimator object
    @param n_episode: number of episodes
    @param gamma: the discount factor
    @param epsilon: parameter for epsilon_greedy
    @param epsilon_greedy: epsilon decreasing factor
    """
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state, info = env.reset()
        is_done = False

        while not is_done:

            # Action of random agent
            action = env.action_space.sample()
            next_state, reward, is_done, _, info = env.step(action)
            total_reward_episode[episode] += reward
            env.render()
            time.sleep(0.001)

            # if not is_done:
            #     # Action of random agent
            #     action = env.action_space.sample()
            #     next_state, reward, is_done, _, info = env.step(action)
            #     total_reward_episode[episode] += reward
            #     env.render()
            #     time.sleep(0.1)

            if not is_done:
                # Action of learning agent
                action = policy(next_state.flatten())
                next_state, reward, is_done, _, info = env.step(action)
                total_reward_episode[episode] += reward

                modified_reward = 0 #next_state[0] + 0.5
                # if next_state[0] >= 0.5:
                #     modified_reward += 100
                # elif next_state[0] >= 0.25:
                #     modified_reward += 20
                # elif next_state[0] >= 0.1:
                #     modified_reward += 10
                # elif next_state[0] >= 0:
                #     modified_reward += 5
                q_values = estimator.predict(state.flatten()).tolist()
                env.render()
                time.sleep(0.001)
                
                if is_done:
                    q_values[action] = modified_reward
                    estimator.update(state.flatten(), q_values)
                    break
                q_values_next = estimator.predict(next_state.flatten())
                # print('q_values: {}'.format(q_values))
                # print('modified_reward: {}'.format(modified_reward))
                # print('q_values_next: {}'.format(q_values_next))
                # print('result: {}'.format(gamma*torch.max(q_values_next).item()))
                q_values[action] = modified_reward + gamma*torch.max(q_values_next).item()
                estimator.update(state.flatten(), q_values)

            state = next_state
        
        print('Episode: {}, total reward: {}, epsilon: {}'.format(
            episode, total_reward_episode[episode], epsilon
        ))
        epsilon = max(epsilon * epsilon_decay, 0.01)

env = gym.make("connect_four/ConnectFour-v0", render_mode="human")
num_steps = int(6.0*7.0/2.0)

n_state = np.prod(env.observation_space.shape)
n_action = env.action_space.n
n_hidden = 50
lr = 0.001
dqn = DQN(n_state, n_action, n_hidden, lr)

n_episode = 50
total_reward_episode = [0] * n_episode

q_learning(env, dqn, n_episode, gamma=0.99, epsilon=0.3)

# obs = env.reset()
# is_done = False
# found = False

# print("The initial observation is {}".format(obs))

# for step in range(num_steps):
#     action = env.action_space.sample()
#     obs, reward, done, _, info = env.step(action)
#     print("The  observation is {}".format(obs))
#     print("The reward is {}".format(reward))
#     env.render()
#     time.sleep(0.1)
#     if done:
#         time.sleep(5)
#         break
#         # env.reset()

env.close()

# Plot the results
episodes = np.linspace(start=1,stop=n_episode, num=n_episode)
plt.figure()
plt.plot(episodes, np.array(total_reward_episode))
plt.show()