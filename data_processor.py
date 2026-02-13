import glob
import torch
import pyvista as pv
import numpy as np
from get_node_info import get_fluid_prop
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformations import normalize_fluid_data


def sorted_vtk(pattern):
    files = sorted(
        glob.glob(pattern),
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    return files


class GraphDataset(Dataset):
    def __init__(self, cases, xy, edge_index, edge_attr, t_idx, normalize_fn=None):

        self.xy = xy.float()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.t_idx = int(t_idx)
        self.normalize_fn = normalize_fn

        self.bank = []
        for case in cases:
            u_all, v_all, p_all = get_fluid_prop(case["files"])  # [T,N]

            uvp = np.stack([u_all, v_all, p_all], axis=-1)       # [T,N,3]

            self.bank.append({
                "uvp": uvp,
                "g": torch.tensor(case["global_attr"], dtype=torch.float32)
            })

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        sim = self.bank[i]

        uvp_t = sim["uvp"][self.t_idx]      # [N,3]
 ##       print("shape of uvp_t: ", uvp_t.shape)

        u = uvp_t[:,0]
        v = uvp_t[:,1]
        p = uvp_t[:,2]
        g = sim["g"]

        if self.normalize_fn is not None:
            u, v, p = self.normalize_fn(u, v, p, step=None)

        u = np.asarray(u).reshape(-1)
        v = np.asarray(v).reshape(-1)
        p = np.asarray(p).reshape(-1)

        uvp_stack = np.stack([u, v, p], axis=1).astype(np.float32)
        uvp_tensor =  torch.from_numpy(uvp_stack).float()
        print("xy:", type(self.xy), self.xy.shape, self.xy.dtype)
        print("uvp:", type(uvp_tensor), uvp_tensor.shape, uvp_tensor.dtype)
        x = torch.cat([self.xy, uvp_tensor], dim=1)   # [N,5]

        edge_attr = self.edge_attr
        if isinstance(edge_attr, list):
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        elif isinstance(edge_attr, np.ndarray):
            edge_attr = torch.from_numpy(edge_attr).float()
        edge_index = self.edge_index
        if isinstance(edge_index, list):
            edge_index = torch.tensor(edge_index)
        elif isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)

        # If it's [E,2], transpose it:
        if edge_index.dim() == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t()

        edge_index = edge_index.long().contiguous()
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=g
        )


