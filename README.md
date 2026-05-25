# Laplacian Representations for Decision-Time Planning

**ALPS: Augmented Laplacian Planning with Subgoals**

Dikshant Shehmar, Matthew Schlegel, Matthew E. Taylor, Marlos C. Machado

*Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*

[[Paper]](https://arxiv.org/abs/2602.05031) [[Website]](https://dikshuy.github.io/ALPS/)

---

## Overview

Planning with a learned model remains a key challenge in model-based reinforcement learning due to the compounding error problem. In decision-time planning, state representations are critical — they must support local cost computation while preserving long-horizon temporal structure.

We show that the **Laplacian representation** provides an effective latent space for planning by capturing state-space distances at multiple time scales. Specifically, the scaled Laplacian embedding (ψ) is isometric to the **commute-time distance (CTD)** in the data graph, so distances in ψ-space directly reflect how hard it is to navigate between states. This lets ALPS decompose long-horizon problems into subgoals, mitigating the compounding errors that arise over long prediction horizons.

**ALPS** (Augmented Laplacian Planning with Subgoals) is a hierarchical planning algorithm with four main components:

- **ALLO** — Augmented Lagrangian Laplacian Objective for learning the Laplacian representation
- **Forward model** — one-step dynamics model in the original state space
- **Behavior prior** (GCBC) — goal-conditioned behavior cloning policy
- **Planning** — k-means clusters in ψ-space form a high-level subgoal graph; Dijkstra selects waypoints, CEM executes them

---

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

**OGBench** (e.g. AntMaze):
```bash
python main.py --train --test --env-type OGBenchEnv --ogbench-env-type AntMaze \
  --ogbench-task-name antmaze-medium-navigate-v0 --load-offline-dataset \
  --sampling-discount 0.6 --allo-training-steps 1000000 --num-eigenvectors 32 \
  --dynamics-training-steps 1000000 --prior-training-steps 1000000 \
  --num-clusters 64 --show-graph --render --horizon 20 --iterations 5 \
  --samples 500 --seed 14
```

**RoomEnv** (image-based):
```bash
python main.py --train --test --env-type RoomEnv --obs-type image \
  --rooms-env-name hallway --buffer-size 500000 --sampling-discount 0.2 \
  --allo-training-steps 100000 --num-eigenvectors 32 \
  --dynamics-training-steps 200000 --dynamics-hidden-dim 128 --dynamics-num-layers 2 \
  --multistep-horizon 3 --prior-training-steps 0 --prior-max-horizon 1 \
  --prior-hidden-dim 128 --prior-num-layers 2 --batch-size 128 \
  --num-clusters 16 --show-graph --render --no-use-prior-warmstart \
  --horizon 2 --iterations 5 --samples 500 --momentum 0.0 \
  --sigma 0.2 --noise-beta 0.1 --seed 14
```

## Sweeps

Sweeps read directly from a YAML config file:

```bash
python sweep.py --config configs/config.yaml
```

Replace `configs/config.yaml` with any custom sweep file.

## Citation

```bibtex
@inproceedings{shehmar2026alps,
  title     = {Laplacian Representations for Decision-Time Planning},
  author    = {Shehmar, Dikshant and Schlegel, Matthew and Taylor, Matthew E.
               and Machado, Marlos C.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
