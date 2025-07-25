import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp import CNMP, CNMP_H, generate_trajectory # Assuming these are your custom modules
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
from gymnasium.wrappers import RecordVideo

# --- Video Directory Setup ---
video_folder = "evaluation_videos"
os.makedirs(video_folder, exist_ok=True)

# --- Model and Data Loading (Unchanged) ---
model = CNMP_H(d_x=1, d_y=30, d_SM=30).double()
model.load_state_dict(torch.load("save/best_model_newdataset_diff_hierarch.pth"))
model.eval()

dataset = minari.load_dataset('D4RL/relocate/human-v2')

# --- Plotting Setup (Unchanged) ---
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

training_region = np.zeros((25, 6), dtype=np.float64)
for i in range(25):
    episode = dataset[i]
    obs = episode.observations
    training_region[i, :] = obs[0, -9:-3]
    axs[0].scatter(obs[0, -9], obs[0, -8], c='black', label='Hand to Ball')
    axs[1].scatter(obs[0, -7], obs[0, -6], c='black', label='Hand to Target')
    axs[2].scatter(obs[0, -5], obs[0, -4], c='black', label='Ball to Target')

# --- Corrected Evaluation Loop with Recording and Renaming ---
success_count = 0.0
num_tests = 100

for i in range(num_tests):
    # Set a unique, temporary prefix for this test's video
    # This avoids overwriting and allows us to find the file later
    temp_name_prefix = f"test-run-{i}"

    env = dataset.recover_environment(max_episode_steps=450, render_mode='rgb_array')

    # Wrap the environment for recording. The video will be saved with the temp prefix.
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=temp_name_prefix,
        episode_trigger=lambda x: True # Record this one episode
    )

    obs, _ = env.reset(seed=420+i)

    # --- Trajectory Generation and Execution (Unchanged) ---
    cnmp_obs = np.array([np.concatenate((np.array([0.0]), obs[:30]), axis=-1)])
    
    context = np.zeros((12))  # Adjusted context size to match the model's expectation
    context[0:6] = obs[30:36]
    init_state_dict = env.unwrapped.get_env_state()
    context[6:9] = init_state_dict['obj_pos']
    context[9:12] = init_state_dict['target_pos']

    traj_length = 400
    action_trajectory = generate_trajectory(model, cnmp_obs, context).squeeze(0)
    action_trajectory = action_trajectory.detach().numpy()
    action_trajectory = interp1d(np.linspace(0, 1, len(action_trajectory)), action_trajectory, axis=0)(np.linspace(0, 1, traj_length))

    for step_num in range(450):
        if step_num < traj_length:
            action = action_trajectory[step_num]
        else:
            action = action_trajectory[-1]
        obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    is_success = info.get('success', False)

    # Plotting logic (Unchanged)
    if is_success:
        axs[0].scatter(obs[-9], obs[-8], c='red', label='Hand to Ball')
        axs[1].scatter(obs[-7], obs[-6], c='red', label='Hand to Target')
        axs[2].scatter(obs[-5], obs[-4], c='red', label='Successful Test')
        success_count += 1.0
    else:
        axs[0].scatter(obs[-9], obs[-8], c='blue', label='Hand to Ball')
        axs[1].scatter(obs[-7], obs[-6], c='blue', label='Hand to Target')
        axs[2].scatter(obs[-5], obs[-4], c='blue', label='Failed Test')

    # MODIFICATION: The file is saved on env.close(). We rename it immediately after.
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

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1} episodes, success rate: {(success_count / (i + 1)) * 100 :.2f}%")

# --- Final plotting and printing (Unchanged) ---
for k in range(25):
    episode = dataset[k]
    obs = episode.observations
    axs[0].scatter(obs[0, -9], obs[0, -8], c='black', label='Hand to Ball')
    axs[1].scatter(obs[0, -7], obs[0, -6], c='black', label='Hand to Target')
    axs[2].scatter(obs[0, -5], obs[0, -4], c='black', label='Training Data')

axs[0].grid()
axs[1].grid()
axs[2].grid()

plt.savefig('evaluation_results.png')

print(f"\nFinal success rate over {num_tests} tests: {(success_count / num_tests) * 100:.2f}%")
print(f"Evaluation loop completed successfully. Videos are saved in the '{video_folder}' folder.")