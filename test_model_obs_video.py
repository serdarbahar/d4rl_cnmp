import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp import CNMP, CNMP_v2, generate_trajectory # Assuming these are your custom modules
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import os

# --- Video Directory Setup ---
video_folder = "evaluation_videos_obs"
os.makedirs(video_folder, exist_ok=True)

# --- Model and Data Loading (Unchanged) ---
model = CNMP_v2(d_x=1, d_y=69, d_SM=69).double()
model.load_state_dict(torch.load("best_model_human_obs.pth"))
model.eval()

dataset = minari.load_dataset('D4RL/relocate/human-v2')

# 3 subplots
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

training_region = np.zeros((25, 6), dtype=np.float64)
for i in range(25):
    episode = dataset[i]
    obs = episode.observations
    training_region[i, :] = obs[0, -9:-3]
    axs[0].scatter(obs[0, -9], obs[0, -8], c='black', label='Hand to Ball')
    axs[1].scatter(obs[0, -7], obs[0, -6], c='black', label='Hand to Target')
    axs[2].scatter(obs[0, -5], obs[0, -4], c='black', label='Ball to Target')

# --- Corrected Evaluation Loop with Correct Joint Names and API ---
success_count = 0.0
traj_length = 400
for i in tqdm(range(100), desc="Evaluation Progress"):
    temp_name_prefix = f"test-run-{i}"

    env = dataset.recover_environment(max_episode_steps=450, render_mode='rgb_array')

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=temp_name_prefix,
        episode_trigger=lambda x: True # Record this one episode
    )

    obs, _ = env.reset()
    action = np.zeros(30)   
    
    first_cnmp_obs = np.concat((action, obs), axis=-1)

    cnmp_obs_list_all = [first_cnmp_obs]
    
    for j in range(traj_length):
        cnmp_obs_list = cnmp_obs_list_all[-3:]
        cnmp_obs = np.zeros((len(cnmp_obs_list) + 1, 70))
        for k in range(cnmp_obs.shape[0] - 1):
            time = float(j - (cnmp_obs.shape[0]) + 1 + k) / float((traj_length))
            cnmp_obs[k, :] = np.concat((np.array([time]), cnmp_obs_list[k]), axis=-1)
        cnmp_obs[-1, :] = np.concat((np.array([0.0]), first_cnmp_obs), axis=-1)

        context = obs[30:39]
        traj_length = 400
        action_trajectory = generate_trajectory(model, cnmp_obs, context).squeeze(0)
        action_trajectory = action_trajectory.detach().numpy()
        action_trajectory = interp1d(np.linspace(0, 1, len(action_trajectory)), action_trajectory, axis=0)(np.linspace(0, 1, traj_length))

        action = action_trajectory[j][:30]

        obs, rew, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

        if j == 0: cnmp_obs_list_all = []
        cnmp_obs_list_all.append(np.concat((action, obs), axis=-1))

    if info['success']:
        axs[0].scatter(obs[-9], obs[-8], c='red', label='Hand to Ball')
        axs[1].scatter(obs[-7], obs[-6], c='red', label='Hand to Target')
        axs[2].scatter(obs[-5], obs[-4], c='red', label='Successful Test')
        success_count += 1.0
        #print(context)
    else:
        axs[0].scatter(obs[-9], obs[-8], c='blue', label='Hand to Ball')
        axs[1].scatter(obs[-7], obs[-6], c='blue', label='Hand to Target')
        axs[2].scatter(obs[-5], obs[-4], c='blue', label='Failed Test')
    
    is_success = info.get('success', False)

    env.close()
    
    # Since we create a new environment for each test, the episode count inside the
    # wrapper will always be 0. So we know the exact temporary filename.
    temp_video_filename = f"{temp_name_prefix}-episode-0.mp4"
    temp_video_path = os.path.join(video_folder, temp_video_filename)

    # Define the final desired filename
    final_video_filename = f"{i}_{is_success}.mp4"
    final_video_path = os.path.join(video_folder, final_video_filename)

    # Rename the file from its temporary name to the final name
    os.rename(temp_video_path, final_video_path)

    if i % 10 == 0 and i > 0:
        #print(f"Completed {i} episodes, success rate: {(success_count / i) * 100 :.2f}%")
        tqdm.write(f"Completed {i} episodes, success rate: {(success_count / i) * 100 :.2f}%")
    
for i in range(25):
    episode = dataset[i]
    obs = episode.observations
    axs[0].scatter(obs[0, -9], obs[0, -8], c='black', label='Hand to Ball')
    axs[1].scatter(obs[0, -7], obs[0, -6], c='black', label='Hand to Target')
    axs[2].scatter(obs[0, -5], obs[0, -4], c='black', label='Training Data')

axs[0].grid()
axs[1].grid()
axs[2].grid()

plt.savefig('evaluation_results_different_seed.png')

print(f"Success rate: {(success_count / 1000.0) * 100:.2f}%")

print("Evaluation loop completed successfully.")