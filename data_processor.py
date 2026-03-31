import glob
import torch
import numpy as np
import pyvista as pv
from torch.utils.data import Dataset
from torch_geometric.data import Data
from edge_info import read_vtk_2d, edges_from_mesh, build_edge_feature


def sorted_vtk(pattern):
    files = sorted(
        glob.glob(pattern),
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    return files


class GraphDataset(Dataset):
    """
    Graph dataset for 2-D CFD surrogate.

    Each case is a dict with:
        "files"       – list of VTK paths (one per time-step)
        "global_attr" – list/array e.g. [U_mag, AoA]

    The last time-step is used as the target.

    Node features:  [x, y, u, v, p]   shape (N_2d, 5)
    Edge features:  [dx, dy, dist]     shape (E, 3)
    """

    def __init__(self, cases, normalize_fn=None):
        self.normalize_fn = normalize_fn
        self.bank = []

        # Cache *topology* per unique mesh (keyed by raw n_cells before dedup)
        topo_cache = {}  # n_cells_3d → (xy_tensor, edge_index, edge_attr_tensor, inverse)

        for case_idx, case in enumerate(cases):
            # ── 1. Topology (from first file; all files share the same mesh) ──
            ref_file = case["files"][0]
            ref_mesh = pv.read(ref_file)
            n_cells_3d = ref_mesh.n_cells

            if n_cells_3d in topo_cache:
                xy_t, edge_index, edge_attr_t, inverse = topo_cache[n_cells_3d]
            else:
                edge_arr, xy_np, inverse = edges_from_mesh(ref_mesh)

                xy_t       = torch.tensor(xy_np, dtype=torch.float32)
                edge_index = torch.tensor(edge_arr.T, dtype=torch.long)
                edge_attr_t = torch.tensor(
                    build_edge_feature(xy_np, edge_arr), dtype=torch.float32
                )
                topo_cache[n_cells_3d] = (xy_t, edge_index, edge_attr_t, inverse)

            # ── 2. Fluid properties (last time-step) ─────────────────────────
            # read_vtk_2d reads the SAME file object and deduplicates with its
            # OWN inverse, guaranteeing N_2d matches xy_t.
            solution_file = case["files"][-1]
            xy_sol, inv_sol, u_2d, v_2d, p_2d = read_vtk_2d(solution_file)

            # Safety check – if the solution file has a different 3D cell count
            # than the reference, recompute topology from it directly.
            if u_2d.shape[0] != xy_t.shape[0]:
                print(
                    f"  [case {case_idx}] N_2d mismatch: "
                    f"topo={xy_t.shape[0]}, fluid={u_2d.shape[0]}. "
                    f"Recomputing topology from solution file."
                )
                sol_mesh = pv.read(solution_file)
                edge_arr, xy_np, inverse = edges_from_mesh(sol_mesh)
                xy_t       = torch.tensor(xy_np, dtype=torch.float32)
                edge_index = torch.tensor(edge_arr.T, dtype=torch.long)
                edge_attr_t = torch.tensor(
                    build_edge_feature(xy_np, edge_arr), dtype=torch.float32
                )
                # Re-read fluid with the new inverse
                xy_sol, inv_sol, u_2d, v_2d, p_2d = read_vtk_2d(solution_file)

            self.bank.append({
                "u":          u_2d,
                "v":          v_2d,
                "p":          p_2d,
                "xy":         xy_t,
                "edge_index": edge_index,
                "edge_attr":  edge_attr_t,
                "g":          torch.tensor(
                    case["global_attr"], dtype=torch.float32
                ).view(1, -1),
            })

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        sim = self.bank[i]

        u = np.asarray(sim["u"], dtype=np.float32).reshape(-1)
        v = np.asarray(sim["v"], dtype=np.float32).reshape(-1)
        p = np.asarray(sim["p"], dtype=np.float32).reshape(-1)

        if self.normalize_fn is not None:
            u, v, p = self.normalize_fn(u, v, p, step=None)

        uvp_tensor = torch.from_numpy(
            np.stack([u, v, p], axis=1).astype(np.float32)
        ).float()

        xy = sim["xy"].float()

        # ── guard against latent mismatches (should never trigger now) ──
        if xy.shape[0] != uvp_tensor.shape[0]:
            raise RuntimeError(
                f"[item {i}] xy rows={xy.shape[0]}, uvp rows={uvp_tensor.shape[0]}. "
                "This is a bug – both must be N_2d."
            )

        x = torch.cat([xy, uvp_tensor], dim=1)  # [N_2d, 5]

        edge_attr  = sim["edge_attr"]
        edge_index = sim["edge_index"].long().contiguous()

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=sim["g"],
        )