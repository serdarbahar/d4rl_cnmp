import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

test_errors_8 = np.load("errors_reach_arm_8_full.npy") # shape (1000,)
training_errors_8 = np.load("training_errors_reach_arm_8_full.npy") # shape (25,)
test_errors = np.load("errors_reach_arm_full.npy") # shape (1000,)
training_errors = np.load("training_errors_reach_arm_full.npy") # shape (25,)

print(f"Test Errors 8D Actions: Mean: {np.mean(test_errors_8):.4f}, Std: {np.std(test_errors_8):.4f}")
print(f"Training Errors 8D Actions: Mean: {np.mean(training_errors_8):.4f}, Std: {np.std(training_errors_8):.4f}")
print(f"Test Errors 6D Actions: Mean: {np.mean(test_errors):.4f}, Std: {np.std(test_errors):.4f}")
print(f"Training Errors 6D Actions: Mean: {np.mean(training_errors):.4f}, Std: {np.std(training_errors):.4f}")   

# make box plot
data = [test_errors_8, test_errors, training_errors_8, training_errors]
plt.figure(figsize=(6, 4))
sns.boxplot(data=data, color='lightblue')
plt.ylabel('Final Position Error (m)', fontsize=14)
plt.axhline(0.076, color='r', linestyle='dashed', label='Grasp Threshold (0.076 m)')
plt.xticks([0, 1, 2, 3], ['Trained Policy (8 dim)', 'Trained Policy (6 dim)', 'Dataset (8 dim)', 'Dataset (6 dim)'], fontsize=12)
plt.title('Reach Arm Task - Grasp Errors', fontsize=16)
plt.grid(axis='y')
plt.show()
