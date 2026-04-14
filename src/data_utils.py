import numpy as np
import torch

def process_buffer_to_tensor(buffer):
    # 1. Convert to numpy array: (T, 25, 3)
    raw_data = np.array(buffer)
    
    # 2. Transpose to (Channels, Time, Vertices) -> (3, T, 25)
    data = np.transpose(raw_data, (2, 0, 1))
    
    # 3. Keep only X, Y -> (2, T, 25)
    data = data[:2, :, :] 

    # --- SPATIAL NORMALIZATION ---
    if data.shape[2] > 1:
        neck = data[:, :, 1:2] 
        data = data - neck

        if data.shape[2] > 8:
            torso_vect = data[:, :, 1] - data[:, :, 8]
            torso_dist = np.linalg.norm(torso_vect, axis=0)
            avg_torso = np.mean(torso_dist) + 1e-6
            data = data / avg_torso

    # --- FEATURE GENERATION ---
    pos = data 
    vel = np.zeros_like(pos)
    vel[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]

    dist_to_nose = np.linalg.norm(pos - pos[:, :, 0:1], axis=0, keepdims=True)

    # Concatenate -> (5, T, 25)
    final_data = np.concatenate([pos, vel, dist_to_nose], axis=0)

    # Convert to Tensor (1, 5, T, 25)
    tensor = torch.tensor(final_data, dtype=torch.float32).unsqueeze(0)
    return tensor