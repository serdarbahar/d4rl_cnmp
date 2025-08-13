import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp_ import CNMP_H, generate_trajectory # Assuming these are your custom modules
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

# --- Model and Data Loading (Unchanged) ---
model = CNMP_H(d_x=1, d_y=30, d_SM=30).double()
model.load_state_dict(torch.load("save/best_models_reach/model_34600.pth"))
model.eval()

dataset = minari.load_dataset('D4RL/relocate/human-v2')
normalization_values = {"actions_min": [], 
                        "actions_max": [],
                        "context_min": [],
                        "context_max": []}

time_len = 451
X = np.tile(np.linspace(0, 1, time_len).reshape((1, time_len, 1)), (23, 1, 1))  # 25 trajectories
action_data = np.load('data/long_reach_actions.npy')  # shape (25, 451, 30)
observation_data = np.load('data/long_reach_observations.npy')  # shape (25, 451, 42)
print('Action data shape:', action_data.shape)

Y = np.zeros((23, time_len, 30))
Y[:, 1:] = action_data[:23]
C = np.zeros((23, 15))
for i in range(23):
    C[i, :9] = observation_data[i, 0, 30:39]
    C[i, 9:] = observation_data[i, 0, 42:]  # add the first observation as context

# normalize Y and C by dimensions
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


test_contexts = []
# --- Corrected Evaluation Loop with Correct Joint Names and API ---
success_list = []
for i in range(100):
    env = dataset.recover_environment(max_episode_steps=500)# render_mode='human')
    init_obs, _ = env.reset(seed=420+i)

    obs_ = env.env.env.env._get_obs()

    # --- Trajectory Generation and Execution (Unchanged) ---
    cnmp_obs = np.array([np.concatenate((np.array([0.0]), Y[0, 0, :]), axis=-1)])

    context = np.zeros((15))  # Adjusted context size to match the model's expectation
    context[0:9] = init_obs[30:39]
    init_state_dict = env.env.env.env.get_env_state()
    context[9:12] = init_state_dict['obj_pos']
    context[12:15] = init_state_dict['target_pos']

    for dim in range(15):
        context[dim] = (context[dim] - normalization_values['context_min'][dim]) / \
                       (normalization_values['context_max'][dim] - normalization_values['context_min'][dim] + 1e-8)
        #print("%.3f" % context[dim].item(), end=' ')

    test_contexts.append(context)

    traj_length = 450
    action_trajectory = generate_trajectory(model, cnmp_obs, context).squeeze(0)
    action_trajectory = action_trajectory.detach().numpy()
    action_trajectory = interp1d(np.linspace(0, 1, len(action_trajectory)), action_trajectory, axis=0)(np.linspace(0, 1, traj_length))

    for dim in range(action_trajectory.shape[-1]):
        action_trajectory[:, dim] = (action_trajectory[:, dim] * 
                                     (normalization_values['actions_max'][dim] - normalization_values['actions_min'][dim]) + 
                                     normalization_values['actions_min'][dim])

    for step_num in range(450):
        action = action_trajectory[step_num] if step_num < traj_length else action_trajectory[-1]
        obs, reward, terminated, truncated, info = env.step(action)
        info['success'] = True if (np.linalg.norm(obs[30:31]) < 0.05 and obs[32] < 0.025) else False
        if terminated or truncated:
            break

    if info['success']:
        #print(f"Episode {i+1}: Success!")
        success_list.append(1.0)
    else:
        #print(f"Episode {i+1}: Failed.")
        success_list.append(0.0)

    #print()

    env.close()

    if i % 100 == 0:
        print(f"Completed {i} episodes, success rate: {(sum(success_list) / (i+1)) * 100 :.2f}%")

print(f"Success rate: {(sum(success_list) / len(success_list)) * 100:.2f}%")

training_contexts = C.copy()
test_contexts = np.array(test_contexts)

# Separate successful and failed test contexts for plotting
success_mask = np.array(success_list, dtype=bool)
failed_mask = ~success_mask

# plot all dimensions of training and test contexts, each 2 dimensions in a scatter plot. there will be 7 plots in total
import matplotlib.pyplot as plt

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
plt.savefig('reach_success.png')
plt.show()

print("Evaluation loop completed successfully.")