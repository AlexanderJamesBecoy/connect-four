import gym
from gym import spaces
import pygame
import numpy as np

class ConnectFour(gym.Env):
    """
    This is a OpenAI gym environment that simulates the game Connect Four for which the reinforcement learning model will be trained on.
    This class is based on the structure of gymlibrarydev's GridWorld: https://www.gymlibrary.dev/content/environment_creation/
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self._WIDTH = 7
        self._HEIGHT = 6
        self._NUMBER_OF_PLAYERS = 2
        self.window_size = 512 # The size of the PyGame window

        # Create an observation corresponding to a Connect Four 6x7 board
        self.observation_space = spaces.Box(
            low=0, high=self._NUMBER_OF_PLAYERS,
            shape=(self._HEIGHT, self._WIDTH), dtype=np.uint8 # TODO to Torch torch.uint8
        )

        # We have 7 actions corresponding to horizontal slots, and 2 players.
        self.action_space = spaces.Discrete(self._WIDTH)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.winow` will be a reference
        to the winow that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._board

    def _get_info(self):
        no_red = np.count_nonzero(self._board == 1)
        no_yellow = np.count_nonzero(self._board == 2)
        return {"red": no_red, "yellow": no_yellow}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the Connect Four board into an empty board, hence containing only zeros.
        self._board = np.zeros(shape=(self._HEIGHT,self._WIDTH),dtype=np.uint8) # TODO
        # self._board = self.observation_space.sample()
        self._turn = 1 # Player 1 starts

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Move the inserted disk to the bottom due to gravity, start at bottom and check if it is empty.
        last_row = 0
        for i in range(self._HEIGHT, 0, -1):
            if self._board[i-1, action] == 0:
                # Replace the value with the player's value depending on `self._turn`
                self._board[i-1, action] = self._turn
                last_row = i-1
                break
        
        # Check whether a connect four is created or board is full.
        reward = 0
        found = False
        connect_found = False
        for i in range(self._WIDTH-3):  # Check if the last insertion resulted in a horizontal connect four.
            if np.count_nonzero(self._board[last_row,i:i+4] == self._turn) == 4:
                connect_found = True
                break
        if not connect_found:
            for i in range(self._HEIGHT-2): # Check if the last insertion resulted in a vertical connect four.
                if np.count_nonzero(self._board[i:i+4,action] == self._turn) == 4:
                    connect_found = True
                    break
        if not connect_found: # Check if the last insertion resulted in a diagonal top-left to bottom-right connect four
            rows = range(last_row-3,last_row+4)
            cols = range(action-3,action+4)
            connection = 0
            for i in range(self._WIDTH):
                if rows[i] >= 0 and cols[i] >= 0 and rows[i] < self._HEIGHT and cols[i] < self._WIDTH:
                    if self._board[rows[i], cols[i]] == self._turn:
                        connection += 1
                    else:
                        connection = 0
                else:
                    connection = 0

                if connection >= 4:
                    connect_found = True
                    print("Hello")
                    print(rows[i])
                    print(cols[i])
                    found = True
                    break
        if not connect_found: # Check if the last insertion resulted in a diagonal top-right to bottom-left connect four
            rows = range(last_row-3,last_row+4)
            cols = range(action+3,action-4,-1)
            connection = 0
            for i in range(self._WIDTH):
                if rows[i] >= 0 and cols[i] >= 0 and rows[i] < self._HEIGHT and cols[i] < self._WIDTH:
                    if self._board[rows[i], cols[i]] == self._turn:
                        connection += 1
                    else:
                        connection = 0
                else:
                    connection = 0

                if connection >= 4:
                    connect_found = True
                    print("Hello")
                    print(rows[i])
                    print(cols[i])
                    found = True
                    break
        # TODO: Optimization



        is_done = False
        if connect_found:
            if self._turn == 2:
                reward = 1
            else:
                reward = -1
            is_done = True
        if found:
            if self._turn == 2:
                reward = 10
            else:
                reward = -10
        elif np.count_nonzero(self._board == 0) == 0:
            is_done = True

        # Switch player 1 and 2's turn
        if self._turn == 1:
            self._turn = 2
        else:
            self._turn = 1

        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, is_done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()