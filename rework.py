import gym
import connect_four
import numpy as np
from model import Model

env = gym.make("connect_four/ConnectFour-v0", render_mode="human")

def q_learning(env, estimator, n_episode, gamma=1.0):
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
    rolling_reward = 0
    full_columns = 0

    policy = estimator.epsilon_greedy_policy(epsilon=0.9, epsilon_decay=0.95, min_epsilon=0.01)
    
    for episode in range(n_episode):
        estimator.update_target(episode)
        state, _ = env.reset()
        is_done = False
        step = 0

        while not is_done:

            # Action of random agent
            rand_action = env.action_space.sample()
            next_state, rewards, is_done, _, _ = env.step(rand_action)
            total_reward_episode[episode] += rewards[1]

            if is_done:
                break
            
            state = next_state
            step += 1

            # Action of learning agent
            action = policy(next_state.flatten())
            next_state, rewards, is_done, _, _ = env.step(action)
            total_reward_episode[episode] += rewards[0]
                
            # estimator.update(state, action, reward, next_state, is_done)
            estimator.remember(state.flatten(), action, next_state.flatten(), rewards[0], is_done)

            if is_done:
                break
                            
            estimator.replay()

            state = next_state
            step += 1
        
        rolling_reward = rolling_reward *0.9 + total_reward_episode[episode]*0.1
        print('Episode: {}, rolling_reward {:.2f}, total reward: {:.2f}, number of steps: {}, full columns: {}'.format(
            episode, rolling_reward,total_reward_episode[episode], step, full_columns
        ))
        full_columns = 0

        # if estimator.save_mode:
        #     if (episode + 1) % 1000000 == 0:
        #         episode_nr = int((episode + 1)/1000000)
        #         episode_name = str(episode_nr) + 'M'
        #         estimator.save_model(episode_name)

        #         filename = 'total_reward_episode_{}_{}.txt'.format(estimator.name, episode_name)
        #         with open(filename, 'w') as file:
        #             for i in range(episode + 1):
        #                 file.write('{}\n'.format(total_reward_episode[i]))
        #             file.close()

        # epsilon = max(epsilon * epsilon_decay, 0.05)
        if total_reward_episode[episode] < -1.5:
            full_columns += 1

n_state = np.prod(env.observation_space.shape)
n_action = env.action_space.n
n_episode = 100
total_reward_episode = [0] * n_episode
n_hidden = 256
lr = 3.0e-4
dqn = Model(None, n_state, n_action, n_hidden, lr, gamma=0.95, device='cpu', save=False, debug=True)

q_learning(env, dqn, n_episode)

env.close()