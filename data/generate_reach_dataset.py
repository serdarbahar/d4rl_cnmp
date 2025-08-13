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

    action = np.zeros((30,))

    for step_num in range(450):
        if step_num < len(training_episode.actions):
            action[:5] = training_episode.actions[step_num][:5]
        else:
            action[:5] = training_episode.actions[-1][:5]
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

        #print("Relative palm-ball position:", obs[30:33])

        print(np.linalg.norm(obs[30:33]), obs[32])

        if terminated or truncated or (np.linalg.norm(obs[30:31]) < 0.055 and obs[32] < 0.025):
            break
    
    print(f"Episode {i+1} finished after {step_num + 1} steps.")
    time_lengths[i, 0] = step_num + 1
    
    env.close()

new_observations = np.zeros((24, 450, 48))
new_actions = np.zeros((24, 450, 30))

skip_index = 13
for i in range(time_lengths.shape[0]):

    if i < skip_index:
        act_interp = interp1d(np.arange(time_lengths[i, 0]), actions[i, :int(time_lengths[i, 0])], axis=0)
        obs_interp = interp1d(np.arange(time_lengths[i, 0]), observations[i, :int(time_lengths[i, 0])], axis=0)
        new_actions[i] = act_interp(np.linspace(0, time_lengths[i, 0] - 1, 450))
        new_observations[i] = obs_interp(np.linspace(0, time_lengths[i, 0] - 1, 450))
    elif i > skip_index:
        act_interp = interp1d(np.arange(time_lengths[i, 0]), actions[i, :int(time_lengths[i, 0])], axis=0)
        obs_interp = interp1d(np.arange(time_lengths[i, 0]), observations[i, :int(time_lengths[i, 0])], axis=0)
        new_actions[i - 1] = act_interp(np.linspace(0, time_lengths[i, 0] - 1, 450))
        new_observations[i - 1] = obs_interp(np.linspace(0, time_lengths[i, 0] - 1, 450))


print(f"Episode {i+1} actions shape: {actions.shape}")
print(f"Episode {i+1} observations shape: {observations.shape}")

#np.save(f'data/long_reach_observations.npy', new_observations)
#np.save(f'data/long_reach_actions.npy', new_actions)

print("Evaluation loop completed successfully.")