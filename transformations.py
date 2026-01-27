import numpy as np


def normalize_fluid_data(u, v, p, step=100, eps=1e-12):
    """
    Normalizes u, v, and p arrays at a specific step and step+1.

    Parameters:
    - u, v, p: NumPy arrays (expected shape: [time, samples, height, width])
    - step: The starting time index
    - eps: Small constant to avoid division by zero

    Returns:
    - Tuple of 6 normalized arrays: (u_n, v_n, p_n, u_plus_n, v_plus_n, p_plus_n)
    """

    # Extract slices for the current step and the next step
    iters = {
        'u': u[step],
        'v': v[step],
        'p': p[step],
        'u_plus': u[step + 1],
        'v_plus': v[step + 1],
        'p_plus': p[step + 1]
    }

    normalized_outputs = {}

    for key, data in iters.items():
        # Calculate mean and std along the first axis of the slice (e.g., samples/channels)
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        # Apply normalization formula: (x - mean) / (std + eps)
        normalized_outputs[key] = (data - mean) / (std + eps)

    return (
        normalized_outputs['u'],
        normalized_outputs['v'],
        normalized_outputs['p'],
        normalized_outputs['u_plus'],
        normalized_outputs['v_plus'],
        normalized_outputs['p_plus']
    )