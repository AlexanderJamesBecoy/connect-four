# Connect 4 Robot
This project aims to design a robot to play connect four against a human player in reality. It consists of two main parts:

 - Decision-making
 - Mechatronics

The decision-making aspect should enable the robot to make smart and tactical decisions so that it can plan, block and eventually win the game. The mechatronics aspect should enable to make the robot able to see the current state of the game with a camera sensor, and insert a token in the correct slot every turn mechanically. This project is started to learn and practice my skills and knowledge in mechatronics and reinforcement learning. If there is enough time and motivation, I could broaden this project with additional features such as automatic sorting and more advanced reinforcement learning method.

## TODO - Decision-making
- [x] Create a PyGame environment for simulation
    - [x] New game, switch player start
- [x] Abstract class of DQN
    - [x] Debug plots displaying Q-values
- [ ] Apply Monte Carlo Tree Search / Minimax Alpha-Beta Pruning
- [ ] Train AI in simple setting (Connect 3)
- [ ] Two RL agents competing each other

## TODO - Mechatronics
- [x] 3D model of the game board
- [x] Design token dropper
- [ ] Design axis movement