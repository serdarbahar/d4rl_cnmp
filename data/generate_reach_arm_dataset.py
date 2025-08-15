import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import pickle
from scipy.interpolate import interp1d
from gymnasium.wrappers import RecordVideo

video_folder = "reach_arm_training_set"

# Load the datasets
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

#skip_indices = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 20, 21, 22, 23, 24]
skip_indices = [13]

for i in range(25):
    if i in skip_indices:
        continue

    temp_name_prefix = f"reach_arm_training_{i}"

    training_obs = training_region[i, :]
    training_episode = dataset[i]
    absolute_episode = dataset_absolute[i]

    # Recover the environment. The 'env' object may be wrapped.
    env = dataset.recover_environment(max_episode_steps=550, render_mode='rgb_array')

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=temp_name_prefix,
        episode_trigger=lambda x: True # Record this one episode
    )

    obs, _ = env.reset()
    
    # DO NOT reassign env to env.unwrapped.
    # Instead, access the base environment's methods and attributes via `env.unwrapped`.
    # This preserves the wrapper functionality (like video recording).

    # Access the base environment's state getting method
    initial_state_dict = env.unwrapped.get_env_state()

    qpos = initial_state_dict['qpos']
    qvel = initial_state_dict['qvel']

    new_state_dict = {
        'qpos': qpos,
        'qvel': qvel,
        'obj_pos': absolute_episode['init_state_dict']['obj_pos'],
        'target_pos': absolute_episode['init_state_dict']['target_pos'],
    }

    # Set the state of the base environment
    env.unwrapped.set_env_state(new_state_dict)

    # Get the observation from the base environment
    obs = env.unwrapped._get_obs()

    # Get the state from the base environment
    env_state = env.unwrapped.get_env_state()
    absolute_ball = env_state['obj_pos']
    absolute_target = env_state['target_pos']
    # Access MuJoCo data through the unwrapped environment
    absolute_palm_pos = env.unwrapped.data.site_xpos[env.unwrapped.S_grasp_site_id].ravel()


    full_obs = np.concatenate([
        obs,
        absolute_palm_pos,
        absolute_ball,
        absolute_target
    ])

    observations[i, 0] = full_obs

    action = np.zeros((30,))

    #_, __ = env.reset(options={'init_state_dict': new_state_dict})

    for step_num in range(450):
        if step_num < len(training_episode.actions):
            action[:6] = training_episode.actions[step_num][:6]
        else:
            action[:6] = training_episode.actions[-1][:6]
        
        # Use the wrapped 'env' for stepping. This allows wrappers to function correctly.
        obs, rew, terminated, truncated, info = env.step(action)

        # Access the base environment's state and data
        env_state = env.unwrapped.get_env_state()
        absolute_ball = env_state['obj_pos']
        absolute_target = env_state['target_pos']
        absolute_palm = env.unwrapped.data.site_xpos[env.unwrapped.S_grasp_site_id].ravel()

        full_obs = np.concatenate([
            obs,
            absolute_palm,
            absolute_ball,
            absolute_target
        ])

        actions[i, step_num] = action
        observations[i, step_num+1] = full_obs

        if terminated or truncated or ((np.linalg.norm(obs[30:33] + [0, 0.049, 0]) < 0.0725) and step_num > 50):
            for _ in range(step_num + 1, 450):
                env.step(action)
            break
    
    print(f"Episode {i+1} finished after {step_num + 1} steps.")
    time_lengths[i, 0] = step_num + 1
    
    # Close the environment
    env.close()

new_observations = np.zeros((25 - len(skip_indices), 450, 48))
new_actions = np.zeros((25 - len(skip_indices), 450, 6))
skip_count = 0
for i in range(time_lengths.shape[0]):

    if i in skip_indices:
        skip_count += 1
        continue

    curr_time_len = int(time_lengths[i, 0])

    act_interp = interp1d(np.arange(curr_time_len), actions[i, :curr_time_len, :6], axis=0)
    obs_interp = interp1d(np.arange(curr_time_len), observations[i, :curr_time_len, :], axis=0)
    new_actions[i - skip_count, :, :] = act_interp(np.linspace(0, curr_time_len - 1, 450))
    new_observations[i - skip_count, :, :] = obs_interp(np.linspace(0, curr_time_len - 1, 450))

print(f"Episode {i+1} actions shape: {new_actions.shape}")
print(f"Episode {i+1} observations shape: {new_observations.shape}")

#np.save(f'data/reach_arm_observations.npy', new_observations)
#np.save(f'data/reach_arm_actions.npy', new_actions)

print("Evaluation loop completed successfully.")