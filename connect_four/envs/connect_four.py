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
        self._connect_found = False # Boolean to determine whether a connect four is created.
        self._connect_four = [] # A list to store the found connect four.

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Move the inserted disk to the bottom due to gravity, start at bottom and check if it is empty.
        # TODO: check if action is invalid
        last_row = 0
        for i in range(self._HEIGHT, 0, -1):
            if self._board[i-1, action] == 0:
                # Replace the value with the player's value depending on `self._turn`
                self._board[i-1, action] = self._turn
                last_row = i-1
                break
        
        # Check whether a connect four is created or board is full.
        self._connect_four = []
        for i in range(self._WIDTH-3):  # Check if the last insertion resulted in a horizontal connect four.
            self._connect_four = []
            if np.count_nonzero(self._board[last_row,i:i+4] == self._turn) == 4:
                self._connect_found = True
                for j in range(4):
                    location = np.array([i+j,last_row])
                    self._connect_four.append(location)
                break
        if not self._connect_found:
            for i in range(self._HEIGHT-2): # Check if the last insertion resulted in a vertical connect four.
                if np.count_nonzero(self._board[i:i+4,action] == self._turn) == 4:
                    self._connect_found = True
                    for j in range(4):
                        location = np.array([action,i+j])
                        self._connect_four.append(location)
                    break
        if not self._connect_found: # Check if the last insertion resulted in a diagonal top-left to bottom-right connect four
            rows = range(last_row-3,last_row+4)
            cols = range(action-3,action+4)
            connection = 0
            self._connect_four = []
            for i in range(self._WIDTH):
                if rows[i] >= 0 and cols[i] >= 0 and rows[i] < self._HEIGHT and cols[i] < self._WIDTH:
                    if self._board[rows[i], cols[i]] == self._turn:
                        connection += 1
                        location = np.array([cols[i],rows[i]])
                        self._connect_four.append(location)
                    else:
                        connection = 0
                        self._connect_four = []
                else:
                    connection = 0
                    self._connect_four = []

                if connection >= 4:
                    self._connect_found = True
                    break
        if not self._connect_found: # Check if the last insertion resulted in a diagonal top-right to bottom-left connect four
            rows = range(last_row-3,last_row+4)
            cols = range(action+3,action-4,-1)
            connection = 0
            self._connect_four = []
            for i in range(self._WIDTH):
                if rows[i] >= 0 and cols[i] >= 0 and rows[i] < self._HEIGHT and cols[i] < self._WIDTH:
                    if self._board[rows[i], cols[i]] == self._turn:
                        connection += 1
                        location = np.array([cols[i],rows[i]])
                        self._connect_four.append(location)
                    else:
                        connection = 0
                        self._connect_four = []
                else:
                    connection = 0
                    self._connect_four = []

                if connection >= 4:
                    self._connect_found = True
                    break
        # TODO: Optimization

        is_done = False
        reward = 0
        if self._connect_found:
            if self._turn == 2:
                reward = 10
            else:
                reward = -10
            is_done = True
        elif np.count_nonzero(self._board == 0) == 0:
            print("Hello")
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
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 255))
        pix_square_size = (
            self.window_size / self._WIDTH
        ) # The size of a single grid square in pixels

        # Draw the disks
        for i in range(self._board.shape[0]):
            for j in range(self._board.shape[1]):
                loc = np.array([j,i+1])

                color = [255,255,255]
                if self._board[i,j] == 1:
                    color = [255,0,0]
                elif self._board[i,j] == 2:
                    color = [255,255,0]

                pygame.draw.circle(
                    canvas,
                    color,
                    (loc + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )
        
        # If connect four is created, display it
        if self._connect_found:
            for i in range(len(self._connect_four)):
                pygame.draw.rect(
                    canvas,
                    (0, 255, 0),
                    pygame.Rect(
                        pix_square_size * (self._connect_four[i] + np.array([0,1])),
                        (pix_square_size, pix_square_size),
                    ),
                    width=5
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:   # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()