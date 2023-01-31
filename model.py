import dqn
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class Model_v0(dqn.DQN):
    def __init__(self, weights, n_state, n_action, n_hidden, alpha, gamma, epsilon, epsilon_decay, device='cpu', save=False, debug=False):
        self.name = 'DQN_v0'
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.n_state = n_state
        self.n_action = n_action
        self.n_hidden = n_hidden
        self.debug_mode = debug

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

        super().__init__(self.name, self.model, self.n_action, self.criterion, alpha, gamma, self.epsilon, self.epsilon_decay, device, save, debug)

    def epsilon_greedy_policy(self):
        """
        """
        def policy_function(state):

            if random.random() < self.epsilon:
                return random.randint(0, self.n_action - 1)
            else:
                q_values = self.predict(state)

                # Display Q-values
                if self.debug_mode:
                    self.display_qvalues(q_values)

                return torch.argmax(q_values).item()

        return policy_function