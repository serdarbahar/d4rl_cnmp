import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib # Import the joblib library

# --- 1. Load and Prepare Your Data ---
# Using the same sample data as before for this example.
# Replace this with your actual data loading.

observation_data = np.load('data/long_observations.npy')  # shape (25, 451, 42)

training_context = np.zeros((24, 6))
for i in range(24):
    training_context[i, :] = observation_data[i, 0, 42:]  # Extract the first observation as context

for dim in range(training_context.shape[-1]):
    C_min = np.min(training_context[:, dim], axis=0, keepdims=True)
    C_max = np.max(training_context[:, dim], axis=0, keepdims=True)
    training_context[:, dim] = (training_context[:, dim] - C_min) / (C_max - C_min + 1e-8)

# --- 2. Scale the Data ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(training_context)

# --- 3. Fit the GMM ---
n_components = 4
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(scaled_data)

print("Model has been trained successfully.")

# --- 4. Save the Model and the Scaler ---
# Use joblib.dump to serialize the objects to files.
# It's common to use the .joblib or .pkl extension.
model_filename = 'gmm_model_4_24.joblib'
scaler_filename = 'scaler.joblib'

joblib.dump(gmm, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"Fitted GMM model saved to: {model_filename}")
print(f"Fitted Scaler saved to: {scaler_filename}")