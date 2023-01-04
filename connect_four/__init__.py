from gym.envs.registration import register

register(
    id="connect_four/GridWorld-v0",
    entry_point='connect_four.envs:GridWorldEnv',
    max_episode_steps=300,
)