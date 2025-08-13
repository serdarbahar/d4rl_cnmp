import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
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

observations = np.zeros((25, 500, 48))
actions = np.zeros((25, 500, 30))
time_lengths = np.zeros((25, 1))

for i in range(25):

    training_obs = training_region[i, :]
    training_episode = dataset[i]
    absolute_episode = dataset_absolute[i]

    env = dataset.recover_environment(max_episode_steps=550, render_mode='human')
    obs, _ = env.reset()

    env = env.unwrapped

    initial_state_dict = env.get_env_state()

    qpos = initial_state_dict['qpos']
    qvel = initial_state_dict['qvel']
    absolute_obj_pos = initial_state_dict['obj_pos']
    absolute_target_pos = initial_state_dict['target_pos']
    absolute_palm_pos = initial_state_dict['palm_pos']

    new_state_dict = {
        'qpos': qpos,
        'qvel': qvel,
        'obj_pos': absolute_episode['init_state_dict']['obj_pos'],
        'target_pos': absolute_episode['init_state_dict']['target_pos'],
    }

    env.set_env_state(new_state_dict)

    obs = env._get_obs()

    env_state = env.get_env_state()
    absolute_ball = env_state['obj_pos']
    absolute_target = env_state['target_pos']
    absolute_palm_pos = env.data.site_xpos[env.S_grasp_site_id].ravel()

    full_obs = np.concatenate([
        obs,
        absolute_palm_pos,
        absolute_ball,
        absolute_target
    ])

    observations[i, 0] = full_obs

    for step_num in range(450):
        if step_num < len(training_episode.actions):
            action = training_episode.actions[step_num]
        else:
            action = training_episode.actions[-1]
        obs, rew, terminated, truncated, info = env.step(action)

        env_state = env.get_env_state()
        absolute_ball = env_state['obj_pos']
        absolute_target = env_state['target_pos']
        absolute_palm = env.data.site_xpos[env.S_grasp_site_id].ravel()

        full_obs = np.concatenate([
            obs,
            absolute_palm,
            absolute_ball,
            absolute_target
        ])



        actions[i, step_num] = action
        observations[i, step_num+1] = full_obs

        if terminated or truncated:
            break

    
    time_lengths[i, 0] = step_num + 1
    
    env.close()

new_observations = np.zeros((24, 500, 48))
new_actions = np.zeros((24, 500, 30))
new_time_lengths = np.zeros((24, 1))

new_observations[:13] = observations[:13]
new_observations[13:] = observations[14:]
new_actions[:13] = actions[:13]
new_actions[13:] = actions[14:]
new_time_lengths[:13] = time_lengths[:13]
new_time_lengths[13:] = time_lengths[14:]


print(f"Episode {i+1} actions shape: {actions.shape}")
print(f"Episode {i+1} observations shape: {observations.shape}")

#np.save(f'data/mixed_observations.npy', new_observations)
#np.save(f'data/mixed_actions.npy', new_actions)
#np.save(f'data/mixed_time_lengths.npy', new_time_lengths)

print("Evaluation loop completed successfully.")