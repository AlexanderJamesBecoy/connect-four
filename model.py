from dqn import DQN, ExpDQN, DoubleDQN
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class Model(DoubleDQN):
    def __init__(self, weights, n_state, n_action, n_hidden, alpha, gamma, device='cpu', save=False, debug=False):
        self.name = 'DQN_v0'
        self.criterion = nn.MSELoss()
        self.n_state = n_state
        self.n_action = n_action
        self.n_hidden = n_hidden

        if weights is None:
            self.model = nn.Sequential(
                nn.Linear(n_state, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_action),
            )
        else:
            self.model = torch.load(weights)

        super().__init__(
            name=self.name, 
            model=self.model, 
            n_action=self.n_action, 
            criterion=self.criterion, 
            alpha=alpha, 
            gamma=gamma,
            device=device, 
            save=save, 
            debug=debug,
        )

    def epsilon_greedy_policy(self, epsilon=0.95, epsilon_decay=0.95, min_epsilon=0.01):
        """
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        def policy_function(state):
            if random.random() < self.epsilon:
                action = random.randint(0, self.n_action-1)
            else:
                q_values = self.predict(state)
                action = torch.argmax(q_values).item()
                if self.debug_mode: # Display Q-values
                    self.display_qvalues(q_values)
                    # print("{} epsilon: {}".format(self.name, self.epsilon))

            self.epsilon = max(self.epsilon * self.epsilon_decay, min_epsilon)

            return action

        return policy_function
    
    def update_target(self, episode):
        """
        """
        if episode % self.target_update == 0:
            self.copy_target()
            if self.debug_mode:
                print("Target model updated at episode {}.".format(episode))