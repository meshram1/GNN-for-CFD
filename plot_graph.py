import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_degree(data):
    N = data.num_nodes
    src = data.edge_index[0]
    deg = torch.bincount(src, minlength=N).float()  # out-degree
    pos = data.x[:, :2].cpu().numpy()
    deg_np = deg.numpy()

    plt.figure()
    plt.scatter(pos[:,0], pos[:,1], c=deg_np, s=2)
    plt.gca().set_aspect('equal', 'box')
    plt.title("Node out-degree (should be ~3 interior, lower near boundaries)")
    plt.colorbar()
    plt.show()