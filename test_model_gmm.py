import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp_ import CNMP_H, generate_trajectory # Assuming these are your custom modules
from scipy.interpolate import interp1d
import joblib  # Import joblib for loading the GMM model and scaler
import matplotlib.pyplot as plt


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

# --- GMM Loading and Sampling (Unchanged) ---
gmm_model = joblib.load('gmm_model.joblib')
gmm_model.random_state = None
scaler = joblib.load('scaler.joblib')

sample_size = 1000
random_samples = []

# Generate random samples from the GMM with rejection sampling
while len(random_samples) < sample_size:
    sample = gmm_model.sample(1)[0]
    sample = scaler.inverse_transform(sample)  # Unscale the sample
    # Unnormalize the sample
    if np.all((sample >= 0) & (sample <= 1)):

        # calculate distance from the training context
        distances = np.linalg.norm(C[:, -6:] - sample, axis=1)
        if np.min(distances) <= 0.225:  # Adjust threshold as needed
            random_samples.append(sample[0])

random_samples = np.array(random_samples)

# unnormalize the random samples
for dim in range(random_samples.shape[-1]):
    random_samples[:, dim] = (random_samples[:, dim] *
                              (normalization_values['context_max'][dim+9] - normalization_values['context_min'][dim+9]) +
                              normalization_values['context_min'][dim+9])
print(f"Generated {sample_size} random samples from the GMM.")

# --- Corrected Evaluation Loop ---
test_contexts = []
success_list = []
for i in range(random_samples.shape[0]):
    env = dataset.recover_environment(max_episode_steps=500)
    init_obs, _ = env.reset(seed=420+i)
    env = env.unwrapped

    # Get the GMM-sampled context (obj_pos and target_pos)
    context_from_gmm = random_samples[i]

    # Get the complete initial state from the environment
    full_init_state = env.get_env_state()

    # Create a NEW dictionary with ONLY the keys expected by set_env_state
    new_state_to_set = {
        'qpos': full_init_state['qpos'],
        'qvel': full_init_state['qvel'],
        'obj_pos': context_from_gmm[:3],      # Set object position from GMM
        'target_pos': context_from_gmm[3:6]   # Set target position from GMM
    }

    # Set the environment state using the correctly formatted dictionary
    env.set_env_state(new_state_to_set)

    obs_ = env._get_obs()

    # --- Trajectory Generation and Execution (Unchanged) ---
    cnmp_obs = np.array([np.concatenate((np.array([0.0]), Y[0, 0, :]), axis=-1)])

    # Re-calculate the CNMP context based on the new state
    current_state = env.get_env_state()
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
plt.savefig('gmm_test_.png')
plt.show()

print("Evaluation loop completed successfully.")

absolute_obj_positions = random_samples[:, :3]
absolute_target_positions = random_samples[:, 3:6]

print("Absolute object positions shape:", absolute_obj_positions.shape)
print("Absolute target positions shape:", absolute_target_positions.shape)

# Save the absolute positions for further analysis
np.save('absolute_obj_positions.npy', absolute_obj_positions)
np.save('absolute_target_positions.npy', absolute_target_positions)
