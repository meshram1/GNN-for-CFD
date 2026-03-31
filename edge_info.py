import numpy as np
import pyvista as pv


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_inverse(mesh):
    """
    Given a PyVista mesh, return the (xy_unique, inverse) pair that maps
    every 3D cell → its unique 2D (x, y) cell-centre id.

    Works even if the mesh was modified (e.g. point_data_to_cell_data),
    because it always reads mesh.cell_centers() from the passed object.
    """
    centres = mesh.cell_centers().points[:, :2].astype(np.float32)
    xy_rounded = np.round(centres, 6)                 # kill float noise
    _, unique_idx, inverse = np.unique(
        xy_rounded, axis=0, return_index=True, return_inverse=True
    )
    xy_unique = centres[unique_idx]                   # [N_2d, 2]
    return xy_unique, inverse


def _collapse(arr_3d, inverse, n_2d):
    """Average a (n_cells_3d, ...) array down to (n_2d, ...)."""
    shape_out = (n_2d,) + arr_3d.shape[1:]
    out = np.zeros(shape_out, dtype=np.float32)
    cnt = np.zeros(n_2d,      dtype=np.float32)
    np.add.at(out, inverse, arr_3d)
    np.add.at(cnt, inverse, 1.0)
    if out.ndim == 1:
        return out / cnt
    return out / cnt[:, None]


# ──────────────────────────────────────────────────────────────────────────────
# Public API used by data_processor.py
# ──────────────────────────────────────────────────────────────────────────────

def read_vtk_2d(vtk_file):
    """
    Read one VTK file and return everything needed for one time-step,
    all consistently deduplicated to unique 2-D cell positions.

    Returns
    -------
    xy_unique : (N_2d, 2)  float32  – 2-D cell-centre coordinates
    inverse   : (n_3d,)    int64    – maps each 3-D cell → 2-D id
    u         : (N_2d,)    float32
    v         : (N_2d,)    float32
    p         : (N_2d,)    float32
    """
    mesh = pv.read(vtk_file)

    # --- ensure cell-level data -------------------------------------------
    needs_convert = ('U' in mesh.point_data or 'p' in mesh.point_data)
    if needs_convert:
        mesh = mesh.point_data_to_cell_data()

    if 'U' not in mesh.cell_data:
        raise KeyError(f"No 'U' field in {vtk_file}")
    if 'p' not in mesh.cell_data:
        raise KeyError(f"No 'p' field in {vtk_file}")

    U_3d = np.asarray(mesh.cell_data['U'], dtype=np.float32)   # (n_3d, ≥2)
    p_3d = np.asarray(mesh.cell_data['p'], dtype=np.float32)   # (n_3d,)

    # --- deduplicate 3-D cells → unique 2-D positions ---------------------
    xy_unique, inverse = _dedup_inverse(mesh)
    n_2d = xy_unique.shape[0]

    u_2d = _collapse(U_3d[:, 0], inverse, n_2d)
    v_2d = _collapse(U_3d[:, 1], inverse, n_2d)
    p_2d = _collapse(p_3d,       inverse, n_2d)

    return xy_unique, inverse, u_2d, v_2d, p_2d


def edges_from_mesh(mesh):
    """
    Build cell-adjacency edges from shared quad faces of a 3-D extruded mesh.
    Deduplicates 3-D cells to unique 2-D cell positions.

    Parameters
    ----------
    mesh : pyvista.DataSet  (already read)

    Returns
    -------
    edges_2d  : (E, 2)    int64   – directed edges in 2-D id space
    xy_unique : (N_2d, 2) float32 – unique 2-D cell centres
    inverse   : (n_3d,)   int64   – 3-D cell → 2-D id
    """
    xy_unique, inverse = _dedup_inverse(mesh)

    face_owner = {}
    for cell_id in range(mesh.n_cells):
        cell = mesh.get_cell(cell_id)
        for fid in range(cell.n_faces):
            face = cell.get_face(fid)
            if face.n_points != 4:          # quad side-faces only
                continue
            key = tuple(sorted(face.point_ids))
            face_owner.setdefault(key, []).append(cell_id)

    edges_2d = set()
    for owners in face_owner.values():
        if len(owners) == 2:
            a = int(inverse[owners[0]])
            b = int(inverse[owners[1]])
            if a != b:
                edges_2d.add((a, b))
                edges_2d.add((b, a))

    edges_2d = np.array(sorted(edges_2d), dtype=np.int64)
    print(f"    topology: {xy_unique.shape[0]} unique 2D cells, "
          f"{len(edges_2d)//2} undirected edges")

    return edges_2d, xy_unique, inverse


def build_edge_feature(xy_unique, edge_arr):
    """
    Edge features: [dx, dy, dist] from 2-D cell centres.

    Parameters
    ----------
    xy_unique : (N_2d, 2) float32
    edge_arr  : (E, 2)    int64
    """
    src = edge_arr[:, 0]
    dst = edge_arr[:, 1]
    dr   = xy_unique[dst] - xy_unique[src]
    dist = np.linalg.norm(dr, axis=1, keepdims=True) + 1e-10
    return np.hstack([dr, dist]).astype(np.float32)   # (E, 3)