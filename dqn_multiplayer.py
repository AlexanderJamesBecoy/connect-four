import gym
import connect_four
import numpy as np
from model import Model

env = gym.make("connect_four/ConnectFour-v0", render_mode="human")

def q_learning(env, estimators, n_episode):
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
    rolling_rewards = [0, 0]
    full_columns = [0, 0]

    policies = [
        estimators[0].epsilon_greedy_policy(epsilon=0.9, epsilon_decay=0.95, min_epsilon=0.01),
        estimators[1].epsilon_greedy_policy(epsilon=0.9, epsilon_decay=0.95, min_epsilon=0.01)
    ]
    
    for episode in range(n_episode):
        rewards = [0, 0]
        state, _ = env.reset()
        is_done = False
        step = 0
        i = episode % 2

        while not is_done:
            for j in range(no_players):
                # Action of learning agent
                k = i + j
                k = 0 if k == 2 else k
                l = int(not k)
                bin_state = env.state_to_binary(state)
                action = policies[k](bin_state.flatten())
                next_state, rewards, is_done, _, _ = env.step(action)
                total_rewards_episode[k][episode] += rewards[0]
                total_rewards_episode[l][episode] += rewards[1]

                next_bin_state = env.state_to_binary(next_state)
                estimators[k].remember(bin_state.flatten(), action, next_bin_state.flatten(), rewards[0], is_done)
                estimators[l].remember(bin_state.flatten(), action, next_bin_state.flatten(), rewards[1], is_done)

                if is_done:
                    break

                estimators[k].replay()

                state = next_state
                step += 1
        
        rolling_rewards[0] = rolling_rewards[0] *0.9 + total_rewards_episode[0][episode]*0.1
        rolling_rewards[1] = rolling_rewards[1] *0.9 + total_rewards_episode[1][episode]*0.1
        print('Episode: {}, rolling_reward {}, number of steps: {}, full columns: {}'.format(
            episode, rolling_rewards, step, full_columns
        ))
        full_columns = [0,0]

        if estimators[0].save_mode or estimators[1].save_mode:
            if (episode + 1) % 1000000 == 0:
                episode_nr = int((episode + 1)/1000000)
                episode_name = str(episode_nr) + 'M'

                for i in range(no_players):
                    estimators[i].save_model(episode_name)

                    filename = 'total_reward_episode_{}_{}.txt'.format(estimator.name, episode_name)
                    with open(filename, 'w') as file:
                        for j in range(episode + 1):
                            file.write('{}\n'.format(total_rewards_episodes[i][j]))
                        file.close()

        if total_rewards_episode[0][episode] < -1.5:
            full_columns[0] += int(total_rewards_episode[0][episode] / (-1*5.0))
        if total_rewards_episode[1][episode] < -1.5:
            full_columns[1] += int(total_rewards_episode[1][episode] / (-1*5.0))

n_state = np.prod(env.observation_space.shape) * 2
n_action = env.action_space.n
n_episode = 100
total_rewards_episode = [
    [0] * n_episode,
    [0] * n_episode,
]
n_hidden = 256
lr = 3.0e-4

dqns = [
    Model('expDQN_p1', None, n_state, n_action, n_hidden, lr, gamma=0.85, device='cuda', save=False, debug=True),
    Model('expDQN_p2', None, n_state, n_action, n_hidden, lr, gamma=0.95, device='cuda', save=False, debug=True),
]
no_players = len(dqns)
q_learning(env, dqns, n_episode)

env.close()