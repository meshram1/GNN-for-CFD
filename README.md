# Using Graph Neural Networks for CFD Flow Prediction with Continuous Spatial Interpolation Across Mesh Resolutions

**Team:** Ksenia Kirsanova, Aditya Meshram

## Overview
This repository contains the implementation of a Graph Neural Network (GNN) designed as a high-efficiency surrogate model for Computational Fluid Dynamics (CFD). The model predicts steady-state velocity components ($u, v$) and pressure distributions ($p$) across 2D unstructured airfoil meshes. By operating directly on graph-based representations of computational meshes, the GNN captures local spatial dependencies and generalizes across varying airfoil geometries.

## Project Objectives
1. **Surrogate Modeling:** Develop a GNN to map airfoil geometry, angle of attack ($\alpha$), and freestream velocity ($U_{in}$) to continuous flow fields, acting as a fast alternative to traditional solvers.
2. **Out-of-Distribution Generalization:** Utilize barycentric coordinate interpolation to predict flow fields on unseen, high-fidelity, refined mesh topologies while maintaining numerical stability.
3. **Scalability via Graph Partitioning:** Implement spatial decomposition (partitioning the mesh into discrete subgraphs with halo regions) to increase computational throughput and accelerate parallel inference.

## Dataset & Simulation Setup
* **Solver:** OpenFOAM (SimpleFOAM) for steady, incompressible flow.
* **Domain:** 2D external flow, extending 7 chord lengths downstream and 2.5 in other directions.
* **Dataset Size:** 770 simulation cases.
* **Parameters:** 11 distinct UIUC airfoil geometries, 10 inlet velocities ($0.1 - 0.7$ m/s), and 7 angles of attack ($0^\circ - 12^\circ$).

## Model Architecture
The GNN operates on a lossless graph representation of the unstructured CFD mesh:
* **Nodes:** Mesh cell centers. Features include spatial coordinates $[x_i, y_i]$.
* **Edges:** Shared faces between adjacent cells. Features include relative spatial displacements and Euclidean distance $[\Delta x, \Delta y, d]$.
* **Global Attributes:** The physical flow conditions $[\alpha, U_{in}]$ broadcast to all edges.
* **Message Passing Framework:** A multi-layer perceptron (MLP) computes messages across edges, aggregates them over local neighborhoods, and updates node embeddings via residual connections. A final decoder maps the embeddings to the target flow quantities $[u, v, p]$.

## Current Progress & Roadmap
* **Completed:** Multi-geometry data generation, lossless graph preprocessing pipeline, and initial model training (770 samples, batch size 8, 100 epochs).
* **Step 1:** Implement graph partitioning, gradient aggregation, and halo region management to optimize computational resources and memory.
* **Step 2:** Execute trial runs on updated architectures and perform inference on finer meshes to evaluate high-resolution spatial interpolation.
* **Step 3:** Optimize hyperparameters, evaluate the surrogate model against mid-fidelity CFD simulations, and benchmark against high-fidelity baselines.
