import numpy as np


def normalize_fluid_data(u, v, p, step=100, eps=1e-12):
    # Extract slices for the current step and the next step
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    p = np.asarray(p).reshape(-1)

    u = (u - u.mean()) / (u.std() + eps)
    v = (v - v.mean()) / (v.std() + eps)
    p = (p - p.mean()) / (p.std() + eps)

    return u,v,p

def denormalize_fluid_data(u_pred, v_pred, p_pred, u, v, p, step, eps=1e-12):
    iters = {
        'u': u[step],
        'v': v[step],
        'p': p[step],
    }

    preds = {
        'u': u_pred,
        'v': v_pred,
        'p': p_pred,
    }

    denormalized_preds = {}

    for key, data in iters.items():
        # Calculate mean and std along the first axis of the slice (e.g., samples/channels)
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        # Apply normalization formula: (x - mean) / (std + eps)
        denormalized_preds[key] = (preds[key]*(std + eps) + mean)

    return (
        denormalized_preds['u'],
        denormalized_preds['v'],
        denormalized_preds['p'],
    )

