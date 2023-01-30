import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import copy
import numpy as np
from collections import deque

class DQN():
    def __init__(self, name, model, criterion, alpha, gamma, epsilon, epsilon_decay, save=False, debug=False):
        """
        Initialize the deep Q-learning network
        @param name             - the name of this DQN instance
        @param model            - DQN neural network model
        @param criterion        - the loss function of the model
        @param alpha            - learning rate of the model
        @param gamma            - discount factor
        @param epsilon          - exploitation-vs-exploration factor
        @param epsilon_decay    - discount factor in exploitation-vs-exploration over many episodes
        @param save             - enable saving this DQN instance's model
        @param debug            - enable debug displaying plots of rewards and Q-values
        """
        self.name = name
        self.save_mode = save
        self.debug_mode = debug
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Load the model, loss function and optimizer
        self.criterion = criterion
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), alpha)

        # Execute PyTorch tensor calculation on CPU or GPU (CUDA)
        if device == 'cuda':
            assert torch.cuda.is_available(), f'CUDA is not available on this device.'
        self.device = torch.device(device)
        print("PyTorch execution on device: {}".format(self.device))

        # Variables for debugging:
        if debug:
            rolling_reward = 0.0
            history_rewards = deque(maxlen=50)

    def update(self, state, y_target):
        """
        Update the weights of the DQN given a training sample
        @param state    - state
        @param y_target - target value
        """
        y_pred = self.model(torch.Tensor(state).to(self.device))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y_target).to(self.device)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """
        Compute the Q values of the state for all actions using the learning model
        @param state    - input state
        @return         - Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model(torch.Tensor(state).to(self.device))

    def save_model(self, episode_name):
        """
        Save the model weights
        """
        if self.save_mode:
            torch.save(self.model, 'models/model_{}_weights_{}.pth'.format(self.name, episode_name))
        else:
            print("Save mode is not activated. Activate this during __init__() with save=True before using this method.")

class ExpDQN(DQN):
    def __init__(self, name, model, criterion, alpha, gamma, epsilon, epsilon_decay, save=False, debug=False):
        """
        Initialize a double deep Q-learning network
        @param name         - the name of this DQN instance
        @param model        - DQN neural network model
        @param criterion    - the loss function of the model
        @param alpha        - learning rate of the model
        @param save         - enable saving this DQN instance's model
        @param debug        - enable debug displaying plots of rewards and Q-values
        """
        super.__init__(name, model, criterion, alpha, gamma, epsilon, epsilon_decay, save=False, debug=False)
        
        # Declare the variable defining the minimal replay size, and memory of past experience
        self.replay_size = replay_size
        self.memory = deque(maxlen=memory_size)

    def remember(self, state, action, next_state, reward, is_done):
        """
        Gain experience by remembering the state and the taken action, consequence, obtained reward, and completion progress
        @param state        - state
        @param action       - action taken in state
        @param next_state   - the consequence of taking action in state
        @param reward       - reward obtained from taking action in state
        @param is_done      - if action result in completion of episode
        """
        self.memory.append((state, action, next_state, reward, is_done))

    def replay(self):
        """
        Update the model based on the randomly selected past experiences
        """
        if len(self.memory) >= self.replay_size:
            states, actions, next_states, rewards, is_dones = zip(*random.sample(self.memory, self.replay_size))
            states = np.array(states)
            next_states = np.array(next_states)
            actions = list(actions)
            is_dones = list(is_dones)

            q_values = self.predict(states)
            q_values_next = self.predict(next_states)
            q_values[:,actions] = torch.Tensor(rewards).to(self.device)
            for i, is_done in enumerate(is_dones):
                if not is_done:
                    q_values[i,actions[i]] += self.gamma * torch.max(q_values_next[i])

        self.update(states, q_values)

class DoubleDQN(ExpDQN):
    def __init__(self, name, model, criterion, alpha, gamma, epsilon, epsilon_decay, save=False, debug=False):
        """
        Initialize a double deep Q-learning network
        @param name         - the name of this DQN instance
        @param model        - DQN neural network model
        @param criterion    - the loss function of the model
        @param alpha        - learning rate of the model
        @param save         - enable saving this DQN instance's model
        @param debug        - enable debug displaying plots of rewards and Q-values
        """
        super.__init__(name, model, criterion, alpha, gamma, epsilon, epsilon_decay, save=False, debug=False)
        self.model_target = copy.deepcopy(self.model)

    def target_predict(self, state):
        """
        Compute the Q values of the state for all actions using the target network
        @param state    - input state
        @return         - targeted Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model_target(torch.Tensor(state).to(self.device))

    def copy_target(self):
        """
        Synchronize the weights of the target network
        """
        self.model_target.load_state_dict(self.model.state_dict())

    def replay(self):
        """
        Update the model based on the randomly selected past experiences
        """
        if len(self.memory) >= self.replay_size:
            states, actions, next_states, rewards, is_dones = zip(*random.sample(self.memory, self.replay_size))
            states = np.array(states)
            next_states = np.array(next_states)
            actions = list(actions)
            is_dones = list(is_dones)

            q_values = self.predict(states)
            q_values_next = self.target_predict(next_states).detach()
            q_values[:,actions] = torch.Tensor(rewards).to(self.device)
            for i, is_done in enumerate(is_dones):
                if not is_done:
                    q_values[i,actions[i]] += self.gamma * torch.max(q_values_next[i])

        self.update(states, q_values)