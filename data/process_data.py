import minari
import gymnasium as gym
import gymnasium_robotics
import importlib
import numpy as np

dataset = minari.load_dataset('D4RL/relocate/human-v2')
env = dataset.recover_environment(render_mode='human', max_episode_steps=1000)

episode = dataset[0]
env.reset(seed=episode.id.item())

for i in range(300):
    obs, rew, terminated, truncated, info = env.step(episode.actions[i])
    if terminated or truncated:
        env.close()
env.close()