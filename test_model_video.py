import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp_ import CNMP_H, generate_trajectory # Assuming these are your custom modules
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
model.load_state_dict(torch.load("save/best_models_v2/cnmp_model_22400.pth"))
model.eval()

dataset = minari.load_dataset('D4RL/relocate/human-v2')
normalization_values = {"actions_min": [],
                        "actions_max": [],
                        "context_min": [],
                        "context_max": []}

time_len = 451
X = np.tile(np.linspace(0, 1, time_len).reshape((1, time_len, 1)), (23, 1, 1))  # 23 trajectories
action_data = np.load('data/long_actions.npy')[:] # Use only the first 23
observation_data = np.load('data/long_observations.npy')[:] # Use only the first 23

print('Action data shape:', action_data.shape)

Y = np.zeros((24, time_len, 30))
Y[:, 1:] = action_data
C = np.zeros((24, 15))
for i in range(24):
    C[i, :9] = observation_data[i, 0, 30:39]
    C[i, 9:] = observation_data[i, 0, 42:]

# --- Normalization (Unchanged) ---
for dim in range(Y.shape[-1]):
    Y_min = np.min(Y[:, :, dim], axis=(0, 1), keepdims=True)
    Y_max = np.max(Y[:, :, dim], axis=(0, 1), keepdims=True)
    normalization_values['actions_min'].append(Y_min)
    normalization_values['actions_max'].append(Y_max)
    Y[:, :, dim] = (Y[:, :, dim] - Y_min) / (Y_max - Y_min + 1e-8)

for dim in range(C.shape[-1]):
    C_min = np.min(C[:, dim], axis=0, keepdims=True)
    C_max = np.max(C[:, dim], axis=0, keepdims=True)
    normalization_values['context_min'].append(C_min)
    normalization_values['context_max'].append(C_max)
    C[:, dim] = (C[:, dim] - C_min) / (C_max - C_min + 1e-8)

# --- Corrected Evaluation Loop with Recording and Renaming ---
success_count = 0.0
num_tests = 1
test_contexts = []
success_list = []
seed = 455

for i in range(num_tests):
    # Set a unique, temporary prefix for this test's video
    # This avoids overwriting and allows us to find the file later
    temp_name_prefix = f"{seed}"

    env = dataset.recover_environment(max_episode_steps=450, render_mode='rgb_array')

    # Wrap the environment for recording. The video will be saved with the temp prefix.
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=temp_name_prefix,
        episode_trigger=lambda x: True # Record this one episode
    )


    obs_, _ = env.reset(seed=seed)

    # --- Trajectory Generation and Execution (Unchanged) ---
    cnmp_obs = np.array([np.concatenate((np.array([0.0]), Y[0, 0, :]), axis=-1)])

    # Re-calculate the CNMP context based on the new state
    current_state = env.unwrapped.get_env_state()
    context_for_cnmp = np.zeros(15)
    context_for_cnmp[:9] = obs_[30:39]
    context_for_cnmp[9:12] = current_state['obj_pos']
    context_for_cnmp[12:15] = current_state['target_pos']

    # Normalize the CNMP context
    for dim in range(15):
        context_for_cnmp[dim] = (context_for_cnmp[dim] - normalization_values['context_min'][dim]) / \
                                (normalization_values['context_max'][dim] - normalization_values['context_min'][dim] + 1e-8)

    test_contexts.append(context_for_cnmp)

    traj_length = 450
    action_trajectory = generate_trajectory(model, cnmp_obs, context_for_cnmp).squeeze(0)
    action_trajectory = action_trajectory.detach().numpy()
    action_trajectory = interp1d(np.linspace(0, 1, len(action_trajectory)), action_trajectory, axis=0)(np.linspace(0, 1, traj_length))

    for dim in range(action_trajectory.shape[-1]):
        action_trajectory[:, dim] = (action_trajectory[:, dim] *
                                     (normalization_values['actions_max'][dim] - normalization_values['actions_min'][dim]) +
                                     normalization_values['actions_min'][dim])

    for step_num in range(450):
        action = action_trajectory[step_num]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    success_list.append(float(info.get('success', False)))

    env.close()

    if (i + 1) % 100 == 0:
        print(f"Completed {i+1} episodes, current success rate: {(sum(success_list) / len(success_list)) * 100 :.2f}%")

print(f"\nFinal Success rate: {(sum(success_list) / len(success_list)) * 100:.2f}%")

# --- Plotting (Unchanged) ---
training_contexts = C.copy()
test_contexts = np.array(test_contexts)

# Separate successful and failed test contexts for plotting
success_mask = np.array(success_list, dtype=bool)
failed_mask = ~success_mask

dims = {0: (9, 10), 1: (12, 13)}
dim_labels = {0: {'xlabel': 'Ball X Position (Absolute)', 'ylabel': 'Ball Y Position (Absolute)'},
                1: {'xlabel': 'Target X Position (Absolute)', 'ylabel': 'Target Y Position (Absolute)'}}
plt.figure(figsize=(10, 5))
for i in range(2):

    dim1, dim2 = dims[i]
    plt.subplot(1, 2, i + 1)
    plt.scatter(training_contexts[:, dim1], training_contexts[:, dim2], color='black', label='Training Points')

    plt.scatter(test_contexts[failed_mask, dim1], test_contexts[failed_mask, dim2], color='red', alpha=0.2, label='Fail')
    plt.scatter(test_contexts[success_mask, dim1], test_contexts[success_mask, dim2], color='green', alpha=0.2, label='Success')
    
    plt.xlabel(dim_labels[i]['xlabel'], fontsize=12)
    plt.ylabel(dim_labels[i]['ylabel'], fontsize=12)

    plt.grid(alpha=0.3)

    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(f"video_test_{seed}.png")
plt.show()


print("Evaluation loop completed successfully.")

print("\nEvaluation loop completed successfully.")