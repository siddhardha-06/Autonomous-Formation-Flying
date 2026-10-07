# Vision-Based Pose Estimation for Formation Flying (ANN + Particle Filter)

Re-implementation of *"Vision-Based Precision Pose Estimation For Autonomous Formation
Flying"* (R. Punnoose, Stanford) for the UE24CS352A Machine Learning mini-project.

A follower aircraft estimates the relative pose of a leader from 14 detected structural
features (simulated CNN detector output):

1. **ANN classifier** - 42 inputs -> 100 -> 100 (ReLU) -> 4800-way softmax, Adam. Predicts a
   discrete pose label (10x10x8x6 bins over x, y, z, roll - paper Table I). Pure NumPy.
2. **Particle filter** - 6-DoF state, nonlinear dynamics, synthetic-feature likelihood,
   `floor(aN)` particles resampled by weight, the rest drawn from *ANN label + error distribution*.
3. **Extensions** (beyond the paper): Gaussian likelihood option, Levenberg-Marquardt
   reprojection refinement (the paper's suggested future work), and a known gimbal
   line-of-sight option (`--known-los`) that removes the bearing / attitude ambiguity.

There is **no public dataset** for this paper: all data is generated synthetically (`data_gen.py`).

## Setup
```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib (no PyTorch needed)
```

## Run (in order)
```bash
python data_gen.py --n-train 600000 --n-val 40000   # 1. synthetic dataset -> data/*.npz (~1 min)
python train_ann.py --epochs 20                      # 2. train ANN, Figs 4 & 5 -> models/, results/
python simulate.py                                   # 3. 10 s approach, Figs 6, 9, 10 (10 Monte-Carlo runs)
python simulate.py --likelihood gauss --refine       # 4. extensions
python simulate.py --likelihood gauss --refine --known-los
python visualize.py                                  # 5. example detected / occluded features
```
Training takes about 70 min on a CPU with these settings (~3.5 min/epoch); the best-validation
weights are checkpointed every epoch. For a quick run use `--n-train 100000 --epochs 10`.

The trained model (`models/ann.npz`, `models/error_pool.npy`) is committed, so the **live demo
only needs step 3**; use `python simulate.py --seeds 1` for a fast (~10 s) single run.
`simulate.py` reports mean +- std over `--seeds` runs (fresh detector noise and PF randomness
each run) in `results/summary_*.json`; plots show the first run.

## Files
| File | Purpose |
|---|---|
| `config.py` | camera, Table I bins, dynamics constants |
| `aircraft.py` | 14-feature aircraft model, gimballed pinhole camera, ray-based occlusion |
| `discretize.py` | pose <-> 4800 labels |
| `data_gen.py` | synthetic dataset generator |
| `ann.py`, `train_ann.py` | NumPy MLP + Adam, training and validation figures |
| `dynamics.py` | simplified relative-motion model |
| `particle_filter.py` | ANN-informed particle filter |
| `refine.py` | LM reprojection-error refinement (extension) |
| `simulate.py` | scenario, metrics, plots |
| `visualize.py` | feature visualisation |

## Conventions and simplifications (document these in the write-up)
* Pose = `[x, y, z, phi, theta, psi]`; `(x,y,z) = r_F/L` (follower relative to leader, x=-90 means 90 m behind).
* Camera gimballed at the leader, image "down" aligned with world down. Pinhole f=1000 px, 1280x960, 0.5 px detector noise.
* Features undetected if occluded (fuselage ellipsoid + wing slab ray test) or outside the image.
* Leader pitch/yaw are sampled in +-5 deg and are *not* part of the label (as in the paper).
* Pitch -> vertical velocity and roll -> lateral velocity via `V_NOM*sin(angle)` (our interpretation of the paper's brief dynamics description).
* PF initial/injected particles use the empirical (true pose - classified pose) error pool from the validation set
  (the paper's text cites Fig. 5 but describes this true-vs-classified error, i.e. Fig. 4).
* **Bearing ambiguity.** Because the camera always points at the leader and (as in the paper) the
  gimbal angles are not an input, a change of bearing looks almost like a change of leader
  pitch/yaw. Pose-only refinement therefore plateaus at ~2 m / ~2 deg. With `--known-los`, the
  follower uses its own gimbal pointing (0.1 deg noise): PF particles are moved onto the
  measured bearing and LM gets a bearing residual, which brings refinement down to ~0.3 m.
