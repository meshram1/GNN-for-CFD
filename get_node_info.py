"""
get_node_info.py  –  legacy shim

The active implementation is now in edge_info.read_vtk_2d, which reads a
VTK file once and returns xy, u, v, p all deduplicated together.

This module is kept for backward compatibility with any notebook cells that
import get_fluid_prop, but data_processor.py no longer calls it.
"""
import numpy as np
import pyvista as pv
from edge_info import read_vtk_2d

print("pyvista version used:", pv.__version__)


def get_fluid_prop(files, inverse=None):
    """
    Legacy wrapper.  Returns u, v, p arrays for the last time-step.
    Shape: [T, N_2d] each.

    NOTE: the `inverse` argument is ignored – deduplication is now handled
    internally by read_vtk_2d which reads and deduplicates each file
    independently, guaranteeing consistency.
    """
    all_u, all_v, all_p = [], [], []
    for f in files:
        _, _, u, v, p = read_vtk_2d(f)
        all_u.append(u)
        all_v.append(v)
        all_p.append(p)

    return (
        np.array(all_u),   # [T, N_2d]
        np.array(all_v),
        np.array(all_p),
    )
