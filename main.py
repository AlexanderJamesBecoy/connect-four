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

def q_learning(env, estimator, n_episode, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    """
    Deep Q-Learning using DQN
    @param env: Gym environment
    @param estimator: Estimator object
    @param replay_size: the number of samples we use to update the model each time
    @param n_episode: number of episodes
    @param gamma: the discount factor
    @param epsilon: parameter for epsilon_greedy
    @param epsilon_greedy: epsilon decreasing factor
    """
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state, info = env.reset()
        is_done = False
        step = 0

        while not is_done:

            # Action of random agent
            rand_action = env.action_space.sample()
            next_state, reward_first, is_done, _, _ = env.step(rand_action)
            step += 1
            env.render()

            if not is_done:
                # Action of learning agent
                state = next_state
                action = policy(next_state.flatten())
                next_state, reward_second, is_done, _, info = env.step(action)

                modified_reward = reward_first + reward_second
                total_reward_episode[episode] += modified_reward
                memory.append((state.flatten(), action, next_state.flatten(), modified_reward, is_done))

                image = env.render()
                # print(type(image))
                # plt.imshow(image)
                # plt.show()
                
                if is_done:
                    break

                estimator.replay(memory, replay_size, gamma)


                state = next_state
            else:
                modified_reward = reward_first
                total_reward_episode[episode] += modified_reward
            
            step += 1
        
        print('Episode: {}, total reward: {}, epsilon: {}, number of steps: {}'.format(
            episode, total_reward_episode[episode], epsilon, step
        ))

        if not estimator.is_loaded:
            if (episode + 1) % 1000 == 0 and episode > 0:
                episode_nr = int((episode + 1)/1000)
                episode_name = str(episode_nr) + 'k'
                estimator.save(episode_name)

                filename = 'total_reward_episode_{}.txt'.format(episode_name)
                with open(filename, 'w') as file:
                    for total_reward in total_reward_episode:
                        file.write('{}\n'.format(total_reward))
                    file.close()

        epsilon = max(epsilon * epsilon_decay, 0.01)

env = gym.make("connect_four/ConnectFour-v0") # , render_mode="human"
num_steps = int(6.0*7.0/2.0)

# n_state = int(np.prod(env.observation_space.shape)*3)
n_state = np.prod(env.observation_space.shape)
n_action = env.action_space.n
n_hidden = 64
lr = 0.005
# dqn = DQN(n_state, n_action, n_hidden, lr)
dqn = DQN(n_state, n_action, n_hidden, lr)

memory = deque(maxlen=1000)
replay_size = 200
n_episode = 10000
total_reward_episode = [0] * n_episode

q_learning(env, dqn, n_episode, replay_size, gamma=0.99, epsilon=0.9, epsilon_decay=0.99)

env.close()

# Plot the results
episodes = np.linspace(start=1,stop=n_episode, num=n_episode)
plt.figure()
plt.plot(episodes, np.array(total_reward_episode))
plt.show()