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
        "files"       - list of VTK paths (one per time-step)
        "global_attr" - list/array e.g. [U_mag, AoA]

    The last time-step is used as the target.

    Node features:  [x, y, u, v, p]   shape (N_2d, 5)
    Edge features:  [dx, dy, dist]     shape (E, 3)

    Calling conventions
    -------------------
    Any of these work:

        # Original canonical form
        GraphDataset([{"files": ["f0.vtk", "f1.vtk"], "global_attr": [U, AoA]}, ...])

        # Flat list of strings -> each string = its own single-file case
        GraphDataset(["case0.vtk", "case1.vtk", ...], normalize_fn)

        # Single string -> one case
        GraphDataset("case0.vtk", normalize_fn)
    """

    @staticmethod
    def _coerce_cases(cases):
        """Normalise *cases* to list-of-dicts no matter what was passed in."""
        if isinstance(cases, str):
            return [{"files": [cases], "global_attr": []}]

        # Single case dict passed directly instead of wrapped in a list
        if isinstance(cases, dict):
            return [cases]

        cases = list(cases)
        if not cases:
            return cases

        normalised = []
        for item in cases:
            if isinstance(item, str):
                normalised.append({"files": [item], "global_attr": []})
            elif isinstance(item, dict):
                normalised.append(item)
            else:
                raise TypeError(
                    f"GraphDataset: each case must be a path string or a dict, "
                    f"got {type(item).__name__!r}."
                )
        return normalised

    def __init__(self, cases, normalize_fn=None):
        self.normalize_fn = normalize_fn
        self.bank = []

        cases = self._coerce_cases(cases)

        topo_cache = {}

        for case_idx, case in enumerate(cases):
            ref_file = case["files"][0]
            ref_mesh = pv.read(ref_file)
            n_cells_3d = ref_mesh.n_cells

            if n_cells_3d in topo_cache:
                xy_t, edge_index, edge_attr_t, inverse = topo_cache[n_cells_3d]
            else:
                edge_arr, xy_np, inverse = edges_from_mesh(ref_mesh)

                xy_t        = torch.tensor(xy_np, dtype=torch.float32)
                edge_index  = torch.tensor(edge_arr.T, dtype=torch.long)
                edge_attr_t = torch.tensor(
                    build_edge_feature(xy_np, edge_arr), dtype=torch.float32
                )
                topo_cache[n_cells_3d] = (xy_t, edge_index, edge_attr_t, inverse)

            solution_file = case["files"][-1]
            xy_sol, inv_sol, u_2d, v_2d, p_2d = read_vtk_2d(solution_file)

            if u_2d.shape[0] != xy_t.shape[0]:
                print(
                    f"  [case {case_idx}] N_2d mismatch: "
                    f"topo={xy_t.shape[0]}, fluid={u_2d.shape[0]}. "
                    f"Recomputing topology from solution file."
                )
                sol_mesh = pv.read(solution_file)
                edge_arr, xy_np, inverse = edges_from_mesh(sol_mesh)
                xy_t        = torch.tensor(xy_np, dtype=torch.float32)
                edge_index  = torch.tensor(edge_arr.T, dtype=torch.long)
                edge_attr_t = torch.tensor(
                    build_edge_feature(xy_np, edge_arr), dtype=torch.float32
                )
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

        if xy.shape[0] != uvp_tensor.shape[0]:
            raise RuntimeError(
                f"[item {i}] xy rows={xy.shape[0]}, uvp rows={uvp_tensor.shape[0]}. "
                "This is a bug - both must be N_2d."
            )

        x = torch.cat([xy, uvp_tensor], dim=1)

        edge_attr  = sim["edge_attr"]
        edge_index = sim["edge_index"].long().contiguous()

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=sim["g"],
        )

