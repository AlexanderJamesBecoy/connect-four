import torch
from torch.autograd import Variable
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque



class DQN():
    def __init__(self, name, model, n_action, criterion, alpha, gamma, device, save, debug):
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
        self.gamma = gamma

        # Load the model, loss function and optimizer
        self.criterion = criterion
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), alpha)

        # Execute PyTorch tensor calculation on CPU or GPU (CUDA)
        if device == 'cuda':
            assert torch.cuda.is_available(), f'CUDA is not available on this device.'
        self.device = torch.device(device)
        print("PyTorch execution on device: {}".format(self.device))

        self.save_mode = save
        if save:
            rolling_reward = 0.0
            history_rewards = deque(maxlen=50)

        self.debug_mode = debug
        if debug:
            self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1)

            self.bars = self.ax1.bar(range(1,n_action+1), np.zeros(n_action))
            self.ax1.set_ylim(bottom=-1.0, top=1.0)
            self.ax1.set_xlabel('Action')
            self.ax1.set_ylabel('Q-value')
            self.ax1.set_title('{}\'s Q-values for the set of actions'.format(self.name))

            self.debug_states = deque(maxlen=50)
            self.debug_qms = [deque(maxlen=50),deque(maxlen=50)]
            self.qmin_plot, = self.ax2.plot(self.debug_states, self.debug_qms[0], color='blue', label='Q-max')
            self.qmax_plot, = self.ax2.plot(self.debug_states, self.debug_qms[1], color='red', label='Q-min')
            self.ax2.set_xlabel('State')
            self.ax2.set_ylabel('Q-value')
            self.ax2.set_title('{}\'s Q-min and Q-max for the last 50 states'.format(self.name))

            plt.tight_layout()

    def update(self, state, action, reward, next_state, is_done):
        """
        Update the weights of the DQN given a training sample
        @param y_target     - target value
        @param state        - state
        @param action       - action taken in state
        @param reward       - reward obtained from action taken in state
        @param next_state   - next_state after taken action
        @param is_done      - if action results into completion
        """
        state = state.flatten()
        next_state = next_state.flatten()
        q_values = self.predict(state)

        if is_done:
            q_values[action] = reward
        else:
            q_values_next = self.predict(next_state)
            q_values[action] = reward + self.gamma * torch.max(q_values_next)

        y_pred = self.model(torch.Tensor(state).to(self.device))
        loss = self.criterion(y_pred, Variable(torch.Tensor(q_values).to(self.device)))
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
        # else:
        #     print("Save mode is not activated. Activate this during __init__() with save=True before using this method.")

    def display_qvalues(self, q_values):
        """
        Display the set of q-values of a given state as a bar graph, and the Q-min and Q-max value for the last 50 states.
        @q_values   - set of q_values
        """
        if len(self.debug_states) > 0:
            state_idx = self.debug_states[-1] + 1
            self.debug_states.append(state_idx)
        else:
            self.debug_states.append(0)
        q_min = min(q_values)
        q_max = max(q_values)
        self.debug_qms[0].append(q_min)
        self.debug_qms[1].append(q_max)

        self.qmin_plot.set_xdata(self.debug_states)
        self.qmin_plot.set_ydata(self.debug_qms[0])
        self.qmax_plot.set_xdata(self.debug_states)
        self.qmax_plot.set_ydata(self.debug_qms[1])
        self.ax2.relim()
        self.ax2.autoscale_view()

        for bar, q_value in zip(self.bars, q_values):
            bar.set_height(q_value)
            bar.set_color('gray')
            if q_value == q_max:
                bar.set_color('red')
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(1.e-9)

class ExpDQN(DQN):
    def __init__(self, name, model, n_action, criterion, alpha, gamma, device, replay_size=20, memory_size=10000, save=False, debug=False):
        """
        Initialize a double deep Q-learning network
        @param name         - the name of this DQN instance
        @param model        - DQN neural network model
        @param criterion    - the loss function of the model
        @param alpha        - learning rate of the model
        @param save         - enable saving this DQN instance's model
        @param debug        - enable debug displaying plots of rewards and Q-values
        """
        # Declare the variable defining the minimal replay size, and memory of past experience
        self.replay_size = replay_size
        self.memory = deque(maxlen=memory_size)
        
        super().__init__(name, model, n_action, criterion, alpha, gamma, device, save, debug)

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
    def __init__(self, name, model, n_action, criterion, alpha, gamma, device, replay_size=20, memory_size=10000, target_update=10, save=False, debug=False):
        """
        Initialize a double deep Q-learning network
        @param name         - the name of this DQN instance
        @param model        - DQN neural network model
        @param criterion    - the loss function of the model
        @param alpha        - learning rate of the model
        @param save         - enable saving this DQN instance's model
        @param debug        - enable debug displaying plots of rewards and Q-values
        """
        super().__init__(name, model, n_action, criterion, alpha, gamma, device, replay_size, memory_size, save, debug)
        self.target_update = target_update
        self.model_target = copy.deepcopy(self.model)

        if debug:
            self.debug_double, (self.debug_double_ax1, self.debug_double_ax2) = plt.subplots(2, 1)

            self.debug_double_bars_model = self.ax1.bar(range(1,n_action+1), np.zeros(n_action), label='model')
            self.debug_double_bars_target = self.ax1.bar(range(1,n_action+1), np.zeros(n_action), label='target')
            self.debug_double_ax1.set_ylim(bottom=-1.0, top=1.0)
            self.debug_double_ax1.set_xlabel('Action')
            self.debug_double_ax1.set_ylabel('Q-value')
            self.debug_double_ax1.legend()
            self.debug_double_ax1.set_title('Q-values for the set of actions')

            self.debug_double_states = deque(maxlen=50)
            self.debug_double_qms = [deque(maxlen=50),deque(maxlen=50)]
            self.debug_double_qmin_plot, = self.ax2.plot(self.debug_states, self.debug_qms[0], color='blue', label='Q-max')
            self.debug_double_qmax_plot, = self.ax2.plot(self.debug_states, self.debug_qms[1], color='red', label='Q-min')
            self.debug_double_ax2.set_xlabel('State')
            self.debug_double_ax2.set_ylabel('Q-value')
            self.debug_double_ax2.set_title('Q-min and Q-max for the last 50 states')

            plt.tight_layout()

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

            # if self.debug_mode:
            #     self.display_double_qvalues(q_values, q_values_next)

            self.update(states, q_values)

    def display_double_qvalues(self, q_values_model, q_values_target):
        """
        Display the set of q-values of a given state as a bar graph, and the Q-min and Q-max value for the last 50 states.
        @q_values   - set of q_values
        """
        if len(self.debug_double_states) > 0:
            state_idx = self.debug_double_states[-1] + 1
            self.debug_double_states.append(state_idx)
        else:
            self.debug_double_states.append(0)
        q_means_model = torch.mean(copy.deepcopy(q_values_model), 0)
        q_means_target = torch.mean(copy.deepcopy(q_values_target), 0)
        q_min_model = min(q_means_model)
        q_max_model = max(q_means_model)
        q_min_target = min(q_means_target)
        q_max_target = max(q_means_target)
        self.debug_double_qms[0].append(q_min_model)
        self.debug_double_qms[1].append(q_max_model)

        self.debug_double_qmin_plot.set_xdata(self.debug_states)
        self.debug_double_qmin_plot.set_ydata(self.debug_qms[0])
        self.debug_double_qmax_plot.set_xdata(self.debug_states)
        self.debug_double_qmax_plot.set_ydata(self.debug_qms[1])
        self.debug_double_ax2.relim()
        self.debug_double_ax2.autoscale_view()

        for bar_model, bar_target, q_value_model, q_value_target in zip(self.debug_double_bars_model, self.debug_double_bars_target, q_means_model, q_means_target):
            bar_model.set_height(q_value_model)
            bar_model.set_color('gray')
            if q_value_model == q_max_model:
                bar_model.set_color('red')
            bar_target.set_height(q_value_target)
            bar_target.set_color('blue')
        
        self.debug_double.canvas.draw()
        self.debug_double.canvas.flush_events()
        plt.pause(1.e-9)