import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.style.core import library
import numpy as np

print("pyvista version used:",pv.__version__)

def get_fluid_prop(files):
    # %%
    all_U = []
    all_p = []

    # Loop through each file
    for f in files:
        mesh = pv.read(f)
        # Inspect available arrays
        if 'U' in mesh.point_data:
            U = mesh['U']
            all_U.append(U)
        if 'p' in mesh.point_data:
            p = mesh['p']
            all_p.append(p)

    # Convert to numpy arrays for ML
    all_U = np.array(all_U)
    all_p = np.array(all_p)
    u = all_U[:, :, 0]
    v = all_U[:, :, 1]
    return u, v, all_p


