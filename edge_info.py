import numpy as np
import pyvista as pv
from matplotlib.style.core import library

def edges_from_faces(mesh):
    face_owner = {}

    for cell_id in range(mesh.n_cells):
        cell = mesh.get_cell(cell_id)

        for fid in range(cell.n_faces):
            face = cell.get_face(fid)          # this is a Cell (the face)

            # For your extruded-2D wedge: keep ONLY the 3 quad side faces
            if face.n_points != 4:
                continue

            pts = face.point_ids               # vertex IDs (ints)
            key = tuple(sorted(pts))
            face_owner.setdefault(key, []).append(cell_id)

    edges = []
    for owners in face_owner.values():
        if len(owners) == 2:
            i, j = owners
            edges.append((i, j))
            edges.append((j, i))               # directed

    return np.array(edges, dtype=np.int64)
# distance from edge to edge is used as feature

def build_edge_feature(mesh, edge_info):

    centres = mesh.cell_centers().points[:,:2].astype(np.float32)

    src = edge_info[:,0]
    dst = edge_info[:,1]

    dr = centres[src] - centres[dst]
    dist = np.linalg.norm(dr, axis=1, keepdims=True) + 1e-10

    edge_attr = np.hstack([dr, dist]).astype(np.float32)  # (E,3)
    return edge_attr