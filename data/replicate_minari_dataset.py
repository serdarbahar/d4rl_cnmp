import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp_ import CNMP, generate_trajectory # Assuming these are your custom modules
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import pickle
from scipy.interpolate import interp1d

dataset = minari.load_dataset('D4RL/relocate/human-v2')
dataset_absolute = pickle.load(open('data/relocate.pickle', 'rb'))

training_region = np.zeros((25, 9), dtype=np.float64)
for i in range(25):
    episode = dataset[i]
    obs = episode.observations
    training_region[i, :] = obs[0, -9:]

# --- Corrected Evaluation Loop with Correct Joint Names and API ---
success_count = 0.0

observations = []
actions = []

for i in range(25):

    observations.append([])
    actions.append([])

    training_obs = training_region[i, :]
    training_episode = dataset[i]
    absolute_episode = dataset_absolute[i]

    env = dataset.recover_environment(max_episode_steps=550)
    obs, _ = env.reset()

    env = env.unwrapped

    initial_state_dict = env.get_env_state()

    qpos = initial_state_dict['qpos']
    qvel = initial_state_dict['qvel']
    absolute_obj_pos = initial_state_dict['obj_pos']
    absolute_target_pos = initial_state_dict['target_pos']
    absolute_palm_pos = initial_state_dict['palm_pos']

    training_palm_to_ball = training_obs[0:3]
    training_palm_to_target = training_obs[3:6]
    training_ball_to_target = training_obs[6:9]

    new_absolute_obj_pos = absolute_palm_pos - training_palm_to_ball
    new_absolute_target_pos = absolute_palm_pos - training_palm_to_target

    new_state_dict = {
        'qpos': qpos,
        'qvel': qvel,
        'obj_pos': new_absolute_obj_pos,
        'target_pos': new_absolute_target_pos
    }
    env.set_env_state(new_state_dict)

    for step_num in range(450):
        if step_num < len(training_episode.actions):
            action = training_episode.actions[step_num]
        else:
            action = training_episode.actions[-1]
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
    
    env.close()