"""
Inverse-distance weighted interpolation for GNN-predicted flow fields.

Given a mesh with known u, v, p at every node, interpolate to an
arbitrary query point using the K nearest neighbours.
"""

import numpy as np
from scipy.spatial import cKDTree


def interpolate_flow(mesh_xy, uvp, query_x, query_y, k=5, power=2):
    """
    Interpolate u, v, p at an arbitrary (query_x, query_y) point using
    inverse-distance weighting from the *k* nearest mesh nodes.

    Parameters
    ----------
    mesh_xy : ndarray, shape (N, 2)
        The x-y coordinates of every mesh node.
    uvp : ndarray, shape (N, 3)
        The u, v, p values at every mesh node (columns: u, v, p).
    query_x : float
        x-coordinate of the query point.
    query_y : float
        y-coordinate of the query point.
    k : int, optional
        Number of nearest neighbours to use (default 5).
    power : float, optional
        Exponent for inverse-distance weighting (default 2).
        Higher values give more weight to closer neighbours.

    Returns
    -------
    dict
        {'u': float, 'v': float, 'p': float} — interpolated values.
    """
    mesh_xy = np.asarray(mesh_xy, dtype=np.float64)
    if mesh_xy.ndim == 2 and mesh_xy.shape[1] > 2:
        mesh_xy = mesh_xy[:, :2]
    uvp = np.asarray(uvp, dtype=np.float64)

    if mesh_xy.shape[0] != uvp.shape[0]:
        raise ValueError(
            f"mesh_xy has {mesh_xy.shape[0]} nodes but uvp has {uvp.shape[0]} rows"
        )

    tree = cKDTree(mesh_xy)
    query_pt = np.array([[query_x, query_y]], dtype=np.float64)
    distances, indices = tree.query(query_pt, k=k)

    # query returns 1-D arrays when there is a single query point
    distances = distances.flatten()
    indices = indices.flatten()

    # If the query point coincides with a mesh node, return it directly
    if distances[0] < 1e-12:
        vals = uvp[indices[0]]
        return {"u": float(vals[0]), "v": float(vals[1]), "p": float(vals[2])}

    # Inverse-distance weights: w_i = 1 / d_i^power
    weights = 1.0 / np.power(distances, power)
    weights /= weights.sum()  # normalise so they sum to 1

    interpolated = np.dot(weights, uvp[indices])  # weighted average

    return {
        "u": float(interpolated[0]),
        "v": float(interpolated[1]),
        "p": float(interpolated[2]),
    }


# ── convenience wrapper that builds the tree once for many queries ──────

class FlowFieldInterpolator:
    """
    Reusable interpolator — builds the KD-tree once, then answers
    many point queries efficiently.

    Usage
    -----
    >>> interp = FlowFieldInterpolator(mesh_xy, uvp, k=5)
    >>> interp(0.5, 0.01)
    {'u': ..., 'v': ..., 'p': ...}
    >>> interp.query_batch(np.array([[0.5, 0.01], [0.8, -0.02]]))
    array([[u1, v1, p1],
           [u2, v2, p2]])
    """

    def __init__(self, mesh_xy, uvp, k=5, power=2):
        self.mesh_xy = np.asarray(mesh_xy, dtype=np.float64)
        if self.mesh_xy.ndim == 2 and self.mesh_xy.shape[1] > 2:
            self.mesh_xy = self.mesh_xy[:, :2]
        self.uvp = np.asarray(uvp, dtype=np.float64)
        self.k = k
        self.power = power
        self.tree = cKDTree(self.mesh_xy)

    def __call__(self, query_x, query_y):
        """Interpolate at a single point; returns a dict."""
        return self._interpolate(
            np.array([[query_x, query_y]], dtype=np.float64)
        )[0]

    def query_batch(self, query_pts):
        """
        Interpolate at many points at once.

        Parameters
        ----------
        query_pts : ndarray, shape (M, 2)

        Returns
        -------
        ndarray, shape (M, 3) — columns are u, v, p.
        """
        query_pts = np.asarray(query_pts, dtype=np.float64)
        results = self._interpolate(query_pts)
        return np.array([[r["u"], r["v"], r["p"]] for r in results])

    # ── internal ────────────────────────────────────────────────
    def _interpolate(self, query_pts):
        distances, indices = self.tree.query(query_pts, k=self.k)
        out = []
        for d_row, i_row in zip(distances, indices):
            if d_row[0] < 1e-12:
                vals = self.uvp[i_row[0]]
            else:
                w = 1.0 / np.power(d_row, self.power)
                w /= w.sum()
                vals = np.dot(w, self.uvp[i_row])
            out.append({"u": float(vals[0]), "v": float(vals[1]), "p": float(vals[2])})
        return out
