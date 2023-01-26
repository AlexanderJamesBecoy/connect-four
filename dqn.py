import torch
import torch.nn as nn
import torch.nn.functional as Fd
from torch.autograd import Variable
import random
import copy
# import torchvision.models as models

class DQN():
    """
    Code provided by Yuxi (Hayden) Liu's PyTorch Reinforcement Learning Cookbook
    """

    def __init__(self, name, n_state, n_action, n_hidden=50, lr=0.05, weights=None, save=True):
        self.name = name
        self.save_mode = save
        self.criterion = nn.MSELoss()

        if weights is None:
            self.model = nn.Sequential(
                nn.Linear(n_state, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_action),
            )
        else:
            self.model = torch.load(weights)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.model_target = copy.deepcopy(self.model)

    def update(self, s, y):
    # def update(self, s, y_predict, y_target):
        """
        Update the weights of the DQN given a training sample
        @param s: state
        @param y: target value
        @return:
        """
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        @param s: input state
        @return: Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay with target network
        @param memory: list of experience
        @param replay_size: the number of samples we use to update the model each time
        @param gamma: the discount factor
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []

            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()

                if is_done:
                    q_values[action] += reward
                else:
                    # q_values_next = self.target_predict(next_state).detach()
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                
                td_targets.append(q_values)

            self.update(states, td_targets)

    def target_predict(self, s):
        """
        Compute the Q values of the state for all actions using the target network
        @param s: input state
        @return: targeted Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def copy_target(self):
        """
        Synchronize the weights of the target network
        """
        self.model_target.load_state_dict(self.model.state_dict())

    def save(self, episode_name):
        """
        Save the model weights
        """
        assert self.save_mode, "Save mode is not activated. Activate this during __init__() with save=True before using this method."
        torch.save(self.model, 'models/model_{}_weights_{}.pth'.format(self.name, episode_name))
