import gym
from gym import spaces
import pygame
import numpy as np

connect = {
    'three': {
        'width': 5,
        'height': 4,
        'combo': 3,
    },
    'four': {
        'width': 7,
        'height': 6,
        'combo': 4,
    },
}

reward_system = {
    'win': 1.0,
    'lose': -1.0,
    'overflow': -5.0,
}

class ConnectFour(gym.Env):
    """
    This is a OpenAI gym environment that simulates the game Connect Four for which the reinforcement learning model will be trained on.
    This class is based on the structure of gymlibrarydev's GridWorld: https://www.gymlibrary.dev/content/environment_creation/
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self._WIDTH = connect['four']['width']
        self._HEIGHT = connect['four']['height']
        self._WIN_COMBO = connect['four']['combo']
        self._NUMBER_OF_PLAYERS = 2
        self.window_size = 512 # The size of the PyGame window

        # Create an observation corresponding to a Connect Four 6x7 board
        self.observation_space = spaces.Box(
            low=0, high=self._NUMBER_OF_PLAYERS,
            shape=(self._HEIGHT, self._WIDTH), dtype=np.uint8 # TODO to Torch torch.uint8
        )

        # We have 7 actions corresponding to horizontal slots, and 2 players.
        self.action_space = spaces.Discrete(self._WIDTH)

        # Player 1 starts at first game.
        self._player_1_start = True

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
        self._board = np.zeros(shape=(self._HEIGHT,self._WIDTH),dtype=np.uint8)
        
        self._connect_found = False # Boolean to determine whether a connect four is created.
        self._connect_four = [] # A list to store the found connect four.

        # Every new round, every other player starts
        if self._player_1_start:
            self._turn = 1
        else:
            self._turn = 2
        self._player_1_start = not self._player_1_start # Other player starts next round

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        rewards = [0.0, 0.0]
        is_done = False

        # Punish for choosing a full column
        if np.count_nonzero(self._board[:,action] == 0) == 0:
            rewards = (reward_system['overflow'], 0.0)
            observation = self._get_obs()
            info = self._get_info()
            return observation, rewards, False, False, info
        
        # Get new position of newly-placed token
        new_token = self.set_token(action, self._turn)

        # Extract lines
        row = self.extract_line(new_token, axis='row')
        col = self.extract_line(new_token, axis='col')
        ldiag = self.extract_line(new_token, axis='ldiag')
        rdiag = self.extract_line(new_token, axis='rdiag')
        lines = {'row': row, 'col': col, 'ldiag': ldiag, 'rdiag': rdiag}

        # Get combos
        combos = []
        for line in lines:
            combo, combo_idxs = self.find_combo(lines[line], self._turn)
            if combo >= self._WIN_COMBO and not is_done:
                is_done = True
                rewards[0] += reward_system['win']
                rewards[1] += reward_system['lose']
            combos.append(combo)

        # # Modify reward
        # mod_reward = np.sum(np.array(combos) - 1.0)*1.0e-3
        # if self._turn == 1:
        #     mod_reward = -1*mod_reward
        # rewards[0] += mod_reward
        # rewards[1] -= mod_reward

        # Check if the board is full
        if np.count_nonzero(self._board == 0) == 0:
            is_done = True

        # Switch players and finish
        if not is_done:
            if self._turn == 1:
                self._turn = 2
            else:
                self._turn = 1

        # Obtain the observation and information of new state
        observation = self._get_obs()
        info = self._get_info()

        # Render PyGame if applied
        if self.render_mode == "human":
            self._render_frame()

        return observation, rewards, is_done, False, info

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

    def extract_line(self, new_token, axis):
        """
        Extract a line that is either horizontal, vertical or diagonal where the new token is placed.
        @param new_token: (y,x) position of a newly-placed token
        @param axis: type of line; 'row', 'col', 'ldiag', 'rdiag'
        @return: a list of of number ranging 0-2.
        """
        if axis == 'row':
            return self._board[new_token[0]]
        elif axis == 'col':
            return self._board[:,new_token[1]]
        elif axis == 'ldiag':
            offset = new_token[1] - new_token[0]
            line = np.diagonal(self._board, offset)
            return line.tolist()
        else: # axis == 'rdiag':
            offset = -new_token[1] + self._HEIGHT - new_token[0]
            line = np.fliplr(self._board.copy()).diagonal(offset)
            return line.tolist()

    def find_combo(self, tokens, color):
        """
        Obtain the maximum combo of tokens that a player made in a given line in the board.
        @param tokens: list of colored tokens extracted from board
        @param color: player's color
        @return: maximum number of combo, list of indices of the largest combo
        """
        max_tokens_combo = []
        tokens_combo = []
        max_token_combo = 0
        token_combo = 0

        for idx, token in enumerate(tokens):
            if token == color:
                token_combo += 1
                max_token_combo = max(token_combo, max_token_combo)
                tokens_combo.append(idx)
            else:
                token_combo = 0
                if len(tokens_combo) > len(max_tokens_combo):
                    max_tokens_combo = tokens_combo
                tokens_combo = []

        
        return max_token_combo, max_tokens_combo

    def set_token(self, action, color):
        """
        Move the inserted disk to the bottom due to gravity, start at bottom and check if it is empty.
        @param action: the chosen column of newly-placed token
        @param color: player's color
        @return: new token's position
        """
        row = 0
        board = np.zeros((self._HEIGHT+1, self._WIDTH))
        board[:self._HEIGHT, :self._WIDTH] = self._board
        while(board[row+1, action] == 0 and row < self._HEIGHT - 1):
            row += 1
        self._board[row, action] = color
        return (row, action)

    def state_to_binary(self, state):
        """
        Add binary channels to the state each describing whether or not for player 1 or 2's token in each
        grid exists, respectively. E.g. [0,0] = None, [1,0] = Player 1, [0,1] = Player 2, [1,1] = Illegal
        @param state            - current game state in 7x6
        @return converted_state - current game state in 7x6x2
        """
        converted_state = np.zeros((self._HEIGHT, self._WIDTH, self._NUMBER_OF_PLAYERS))
        for i in range(self._HEIGHT):
            for j in range(self._WIDTH):
                if state[i,j] == 0:
                    continue
                player = state[i,j]-1
                converted_state[i,j,player] = 1
        return converted_state