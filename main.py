import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import connect_four
import random
import torch
from dqn import DQN
from collections import deque

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def q_learning(env, estimator, n_episode, replay_size, target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    """
    Deep Q-Learning using DQN
    @param env: Gym environment
    @param estimator: Estimator object
    @param replay_size: the number of samples we use to update the model each time
    @param target_update: number of episodes before updating the target network
    @param n_episode: number of episodes
    @param gamma: the discount factor
    @param epsilon: parameter for epsilon_greedy
    @param epsilon_greedy: epsilon decreasing factor
    """
    for episode in range(n_episode):
        # if episode % target_update == 0:
        #     estimator.copy_target()

        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state, info = env.reset()
        is_done = False
        step = 0

        while not is_done:

            # Action of random agent
            rand_action = env.action_space.sample()
            next_state, reward, is_done, _, _ = env.step(rand_action)
            total_reward_episode[episode] += reward
            memory.append((state.flatten(), rand_action, next_state.flatten(), reward, is_done))

            estimator.replay(memory, replay_size, gamma)

            if is_done:
                break

            # estimator.replay(memory, replay_size, gamma)
            state = next_state
            step += 1

            # Action of learning agent
            action = policy(next_state.flatten())
            next_state, reward, is_done, _, _ = env.step(action)
            total_reward_episode[episode] += reward
            memory.append((state.flatten(), action, next_state.flatten(), reward, is_done))

            estimator.replay(memory, replay_size, gamma)
                
            if is_done:
                break

            # estimator.replay(memory, replay_size, gamma)
            state = next_state
            step += 1
        
        print('Episode: {}, total reward: {}, epsilon: {}, number of steps: {}'.format(
            episode, total_reward_episode[episode], epsilon, step
        ))

        if estimator.save_mode:
            if (episode + 1) % 100000 == 0 and episode > 0:
                episode_nr = int((episode + 1)/100000)
                episode_name = str(episode_nr) + '00k'
                estimator.save(episode_name)

                filename = 'total_reward_episode_v0d_{}.txt'.format(episode_name)
                with open(filename, 'w') as file:
                    for total_reward in total_reward_episode:
                        file.write('{}\n'.format(total_reward))
                    file.close()

        epsilon = max(epsilon * epsilon_decay, 0.1)

env = gym.make("connect_four/ConnectFour-v0") # , render_mode="human"
num_steps = int(6.0*7.0/2.0)

n_state = np.prod(env.observation_space.shape)
n_action = env.action_space.n
n_hidden = 50
lr = 0.005
# dqn = DQN(n_state, n_action, n_hidden, lr)
dqn = DQN(n_state, n_action, n_hidden, lr, save=True)

memory = deque(maxlen=10000)
replay_size = 20
target_update = 10
n_episode = 2000000
total_reward_episode = [0] * n_episode

q_learning(env, dqn, n_episode, replay_size, target_update, gamma=0.9, epsilon=0.99, epsilon_decay=0.999)

env.close()

# # Plot the results
# episodes = np.linspace(start=1,stop=n_episode, num=n_episode)
# plt.figure()
# plt.plot(episodes, np.array(total_reward_episode))
# plt.show()