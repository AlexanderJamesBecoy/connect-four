import torch
import torch.nn as nn
import torch.nn.functional as Fd
from torch.autograd import Variable
import random
# import torchvision.models as models

class DQN():
    """
    Code provided by Yuxi (Hayden) Liu's PyTorch Reinforcement Learning Cookbook
    """

    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05, weights=None):
    # def __init__(self, n_action, lr=1e-6):
        self.criterion = nn.MSELoss()
        if weights is None:
            self.model = nn.Sequential(
                nn.Linear(n_state, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_action),
            )
            # self.model = DQNModel(n_action)
            self.is_loaded = False
        else:
            # weights = 'models/v0/' + weights
            self.model = torch.load(weights)
            self.is_loaded = False
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr
        )

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
        # loss = self.criterion(y_predict, y_target)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return loss

    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        @param s: input state
        @return: Q values of the state for all actions
        """
        # with torch.no_grad():
        #     return self.model(torch.Tensor(s))
        return self.model(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay
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
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                    td_targets.append(q_values)

                self.update(states, q_values)

    def save(self, episode_name):
        """
        Save the model weights
        """
        torch.save(self.model, 'models/model_v0a_weights_{}.pth'.format(episode_name))
    # def replay(self, memory, replay_size, gamma):
    #     """
    #     Experience replay
    #     @param memory: a list of experience
    #     @param replay_size: the number of samples we use to update the model each time
    #     @param gamma: the discount factor
    #     @return: the loss
    #     """
    #     if len(memory) >= replay_size:
    #         replay_data = random.sample(memory, replay_size)
    #         state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*replay_data)
    #         state_batch = torch.cat(
    #             tuple(state for state in state_batch)
    #         )
    #         next_state_batch = torch.cat(
    #             tuple(state for state in next_state_batch)
    #         )
    #         q_values_batch = self.predict(state_batch)
    #         q_values_next_batch = self.predict(next_state_batch)
    #         reward_batch = torch.from_numpy(np.array(
    #             reward_batch, dtype=np.float32)[:, None])
    #         action_batch = torch.from_numpy(
    #             np.array([[1,0] if action == 0 else [0,1]
    #             for action in action_batch], dtype=np.float32
    #         ))
    #         q_value = torch.sum(
    #             q_values_batch * action_batch, dim=1
    #         )
    #         td_targets = torch.cat(
    #             tuple(
    #                     reward if terminal else reward + gamma * torch.max(prediction) for reward, 
    #                     terminal,
    #                     prediction in zip(reward_batch, done_batch, q_values_next_batch)
    #             )
    #         )
    #         loss = self.update(q_value, td_targets)
    #         return loss

# class DQNModel(nn.Module):
#     def __init__(self, n_state, n_action=2):
#         super(DQNModel, self).__init__()
#         self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
#         self.fc = nn.Linear(7 * 7 * 64, n_state)
#         self.out = nn.Linear(n_state, n_action)
#         self._create_weights()

#     def _create_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or instance(m, nn.Linear):
#                 nn.init.uniform(m.weight, -0.01, 0.01)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         x = F.relu(self, conv1(x))
#         x = F.relu(self, conv2(x))
#         x = F.relu(self, conv3(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc(x))
#         output = self.out(x)
#         return output