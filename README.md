# Fast-Trained, Latent-Adaptive Quadrotor Control

*Combining massively parallel RL with online dynamics/disturbance estimation and an uncertainty-gated safety margin.*

**Author:** Divaksh Arora, IIT Jodhpur
**Status:** 🚧 Project just starting — Phase 1 (single-platform baseline). Results will be added as the project progresses.

---

## Overview

This project builds a low-level reinforcement learning policy for quadrotor control that is meant to work well **across different airframes and disturbances**, not just one drone in one environment. Most existing RL-for-quadrotor papers pick at most two of these four properties: generalizing across platforms/payloads, adapting online to disturbances, training fast, and giving any kind of stability/safety guarantee. This project tries to get all four in one system:

1. **Fast training** — a massively parallel, GPU-vectorized simulator + curriculum (following Eschmann et al., "Learning to Fly in Seconds," RA-L 2024), so a working baseline is achievable in hours, not days.
2. **Cross-platform generalization** — an online-estimated latent embedding of the robot's own dynamics (mass, inertia, motor/thrust constants, drag) and current disturbance (wind, payload, degraded propeller), trained with domain randomization across a *family* of simulated airframes (following Zhang et al., "A Learning-Based Quadcopter Controller with Extreme Adaptation," T-RO 2025).
3. **Safety margin** — an uncertainty-gated clamp on commanded thrust/attitude rates, tightening when the latent-estimation module's confidence is low, in the spirit of Wabersich & Zeilinger's predictive safety filter (Automatica 2021), scaled down to a simple control-invariant set on tilt angle and vertical acceleration.

Full literature review and project rationale: [`RL_Quadrotor_Control_Literature_Cluster.pdf`](./RL_Quadrotor_Control_Literature_Cluster.pdf) / [`.tex`](./RL_Quadrotor_Control_Literature_Cluster.tex).

No existing paper in the reviewed cluster (Hwangbo et al. '17, Lambert et al. '19, Belkhale et al. '21, Eschmann et al. '24, Zhang et al. '25, Wabersich & Zeilinger '21) combines all three of the above in one policy — that's the gap this project targets.

---

## Hypothesis

A policy trained with (a) domain randomization across a family of quadrotor dynamics and disturbances, (b) an online latent-adaptation module, and (c) fast massively-parallel training, will generalize to unseen combinations of airframe + disturbance faster and more robustly than a meta-RL baseline (à la Belkhale) or a fixed-platform fast-trained baseline (à la Eschmann) alone — and adding the uncertainty-gated safety clamp will measurably reduce constraint violations during out-of-distribution disturbances, at a small, quantifiable cost in tracking accuracy.

---

## Ablation conditions

| Condition | Domain randomization | Latent adaptation | Safety clamp |
|---|---|---|---|
| Full system | ✅ | ✅ | ✅ |
| No safety clamp | ✅ | ✅ | ❌ |
| Fixed platform (≈ Eschmann-style) | ❌ | ❌ | ❌ |
| Meta-RL baseline (≈ Belkhale-style) | partial | ✅ (offline meta-adaptation, not the parallel training regime) | ❌ |

---

## Quickstart: Installation & Training

### 1. Prerequisites

- **Python 3.10+** (tested with 3.10–3.12)
- **CUDA-capable GPU** (optional but recommended — CPU-only training works but is slower)
- A display server or virtual framebuffer (for PyBullet GUI rendering — see [Rendering](#rendering) below)

### 2. Create a virtual environment

```bash
# Using conda (recommended)
conda create -n quadrotor python=3.10 -y
conda activate quadrotor

# Or using venv
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch (with your CUDA version)

Install PyTorch **first**, matched to your CUDA toolkit:

```bash
# CUDA 12.4 (check yours with `nvcc --version` or `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU-only (if no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

### 5. Sanity check — verify everything is wired up

```bash
# Headless (fast, just prints output)
python envs/sanity_check.py

# With PyBullet GUI window (random actions — drone crashes a lot, that's expected)
python envs/sanity_check.py --gui

# Calmer GUI demo (small perturbations around hover RPM)
python envs/sanity_check.py --gui --hover
```

---

## Training the TD3 Paper Baseline

The main training script follows the TD3 configuration from Eschmann et al., "Learning to Fly in Seconds" (RA-L 2024), including:
- A **curriculum callback** that anneals reward weights from lenient → strict over the first third of training
- An **exploration noise decay** callback (σ: 0.3 → 0.05)
- **Motor delay** modeled as a first-order low-pass filter (the paper's most critical ablation finding)
- **Automatic weight resumption** — if saved weights exist from a previous session, training resumes from the latest checkpoint instead of starting from scratch

### Train (headless — fastest)

```bash
# Default: 300k steps, no disturbances (auto-resumes if weights exist)
python train/train_td3_paper.py

# With disturbances (random force/torque per episode)
python train/train_td3_paper.py --disturbances

# Custom step count and seed
python train/train_td3_paper.py --total-steps 500000 --seed 42

# Force fresh start (ignore existing weights)
python train/train_td3_paper.py --total-steps 300000 --fresh

# Custom output directory
python train/train_td3_paper.py --total-steps 300000 --log-dir results/my_experiment
```

### Training Memory (Auto-Resume)

Each time you train, the script automatically:
1. **Checks for existing weights** in the log directory
2. **Resumes from the latest checkpoint** if found (loads model + replay buffer)
3. **Saves versioned checkpoints** with timestamps (e.g., `td3_paper_hover_20260904_001500.zip`) alongside the canonical `td3_paper_hover.zip`
4. **Persists the replay buffer** (`td3_replay_buffer.pkl`) so off-policy experience accumulates across sessions
5. **Tracks cumulative timesteps** in `training_metadata.json` so curriculum and exploration schedules continue seamlessly

Use `--fresh` to discard all previous progress and start from scratch.

### All training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--total-steps` | `300000` | Total environment steps for this training session |
| `--disturbances` | off | Enable random per-episode force/torque disturbances |
| `--log-dir` | `results/td3_paper_baseline` | Directory for checkpoints, TensorBoard logs, and monitor CSV |
| `--seed` | `0` | Random seed for reproducibility |
| `--gui` | off | Open PyBullet GUI to watch training live (much slower — see [Rendering](#rendering)) |
| `--fresh` | off | Force a fresh start, ignoring any existing saved weights |

### Monitor training with TensorBoard

```bash
tensorboard --logdir results/td3_paper_baseline
```

Then open `http://localhost:6006` in your browser.

---

## Simulation & Diagnostics

After training, use `simulate_hover.py` to evaluate the learned policy **without further training**. It runs the drone in PyBullet, collects detailed metrics, and generates timestamped diagnostic reports.

```bash
# Default: 10 episodes, headless, auto-detects latest weights
python train/simulate_hover.py

# With PyBullet GUI visualization
python train/simulate_hover.py --gui --episodes 5

# Specific model checkpoint
python train/simulate_hover.py --model results/td3_paper_baseline/td3_paper_hover.zip
```

### What the simulation does

1. **Clean episodes** — runs N episodes without disturbances to measure baseline hover performance
2. **Random-baseline episodes** — runs 3 episodes with random actions for comparison (learning assessment)
3. **Disturbance episodes** — runs N episodes with force/torque disturbances (overfitting detection)

### Report output

Reports are saved to `simulation_reports/` with timestamps:

```
simulation_reports/
├── sim_report_20260904_001500.txt    # human-readable summary
├── sim_report_20260904_001500.json   # machine-readable data
└── ...
```

Each report contains:
- **Per-episode results** — hover time, reward, position error, crash/truncation status
- **Aggregate metrics** — mean ± std for all metrics, crash rate, stability score
- **Learning assessment** — trained policy reward vs. random baseline (has the model learned?)
- **Overfitting check** — clean vs. disturbance performance (>30% degradation flags potential overfitting)
- **Verdict** — overall assessment with recommendations for next steps

### All simulation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | auto-detect latest | Path to saved TD3 `.zip` checkpoint |
| `--episodes` | `10` | Number of evaluation episodes to run |
| `--gui` | off | Open PyBullet GUI for the clean episodes (slower) |
| `--log-dir` | `results/td3_paper_baseline` | Directory to search for saved models |

---

## Rendering

### Option A: Watch training live (`--gui`)

Pass `--gui` to the training script to open a PyBullet GUI window that shows the drone learning in real time:

```bash
python train/train_td3_paper.py --total-steps 300000 --gui
```

> **⚠️ Note:** GUI rendering significantly slows down training (PyBullet syncs rendering with physics steps). Use this for short runs or debugging — not for full 300k-step training.

### Option B: Evaluate a trained model (recommended)

After training, load the saved checkpoint and render the learned policy:

```bash
# Default: loads results/td3_paper_baseline/td3_paper_hover.zip, runs 3 episodes with GUI
python train/eval_td3_paper.py

# Custom model path
python train/eval_td3_paper.py --model results/my_experiment/td3_paper_hover.zip

# More episodes, with disturbances
python train/eval_td3_paper.py --episodes 5 --disturbances

# Headless (print metrics only, no window)
python train/eval_td3_paper.py --no-gui
```

### Option C: Simulation with diagnostics

For detailed evaluation with reports (see [Simulation & Diagnostics](#simulation--diagnostics) above):

```bash
python train/simulate_hover.py --gui --episodes 5
```

### All evaluation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `results/td3_paper_baseline/td3_paper_hover.zip` | Path to saved TD3 `.zip` checkpoint |
| `--episodes` | `3` | Number of episodes to run |
| `--disturbances` | off | Enable random force/torque disturbances during evaluation |
| `--no-gui` | off | Run headless (no PyBullet window) |

### Headless machines (SSH / no display)

If you're on a headless server (e.g. SSH), PyBullet's GUI mode requires a virtual framebuffer:

```bash
# Install xvfb (once)
sudo apt-get install xvfb

# Run training or eval with a virtual display
xvfb-run -a python train/train_td3_paper.py --gui
xvfb-run -a python train/eval_td3_paper.py
```

Alternatively, train headless (no `--gui` flag) and download the checkpoint to a local machine for visualization.

---

## Tooling

- **Simulation:** [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) as the base quadrotor dynamics/sim environment, with GPU-parallel alternatives (IsaacLab / Aerial Gym) as a possible upgrade path for training speed.
- **RL algorithm:** TD3 (Twin Delayed DDPG), via Stable-Baselines3 — the off-policy algorithm used in the paper.
- **Latent-adaptation module:** small encoder over recent state/action history, architecturally similar to prior multimodal-fusion work (VFP-SAC), repurposed to encode dynamics/disturbance identity instead of vision+force.
- **Safety clamp:** ensemble or MC-dropout uncertainty head feeding a simple tilt/thrust-margin clamp.

---

## Repo structure

```
.
├── README.md
├── requirements.txt
├── envs/
│   ├── paper_hover_env.py        # PaperHoverEnv: the paper's reward, motor delay, disturbances, curriculum
│   └── sanity_check.py           # Quick test that gym-pybullet-drones is installed correctly
├── train/
│   ├── train_td3_paper.py        # Main TD3 training script (auto-resume, curriculum, exploration decay)
│   ├── eval_td3_paper.py         # Load a trained model and render it in the GUI
│   └── simulate_hover.py         # Run hover simulation and generate diagnostic reports
├── results/                      # Checkpoints, replay buffers, TensorBoard logs, metadata (gitignored)
│   └── td3_paper_baseline/
│       ├── td3_paper_hover.zip           # Latest (canonical) model weights
│       ├── td3_paper_hover_<timestamp>.zip  # Versioned checkpoints
│       ├── td3_replay_buffer.pkl         # Persisted replay buffer
│       └── training_metadata.json        # Cumulative timesteps & session history
├── simulation_reports/           # Timestamped simulation diagnostic reports (.txt + .json)
├── Papers/                       # Reference PDFs
└── .gitignore
```

---

## Semester timeline

- [ ] **Weeks 1–2:** Stand up `gym-pybullet-drones`; single-platform SAC/PPO hovering baseline (sanity check vs. Hwangbo et al. '17).
- [ ] **Weeks 3–5:** Domain-randomization wrapper (mass/inertia/thrust/drag + wind/payload disturbances); fast massively-parallel training loop.
- [ ] **Weeks 6–9:** Latent-adaptation module; train all four ablation conditions.
- [ ] **Weeks 10–12:** Held-out generalization tests; sample-efficiency/robustness comparisons; uncertainty-gated safety clamp.
- [ ] **Weeks 13–14:** Stretch goal — vision-conditioned latent (if time allows).
- [ ] **Weeks 15–16:** Write-up (targeting RA-L letter format, 6–8 pages).

---

## Evaluation metrics

- Tracking/hover error, seen vs. held-out airframe + disturbance combinations (generalization gap)
- Sample efficiency (steps/episodes to convergence) per ablation condition
- Wall-clock training time
- Constraint-violation rate (excessive tilt, thrust saturation, simulated crash), with vs. without the safety clamp
- *(If stretch goal completed)* same metrics under vision-based vs. ground-truth state estimation

---

## Results

*To be added as the project progresses.*

---

## Core references

1. Hwangbo, Sa, Siegwart, Hutter, "Control of a Quadrotor with Reinforcement Learning," *IEEE RA-L*, 2017.
2. Lambert, Drew, Yaconelli, Levine, Calandra, Pister, "Low-Level Control of a Quadrotor with Deep Model-Based Reinforcement Learning," *IEEE RA-L*, 2019.
3. Belkhale, Li, Kahn, McAllister, Calandra, Levine, "Model-Based Meta-Reinforcement Learning for Flight with Suspended Payloads," *IEEE RA-L*, 2021.
4. Eschmann, Albani, Loianno, "Learning to Fly in Seconds," *IEEE RA-L*, 2024.
5. Wabersich, Zeilinger, "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems," *Automatica*, 2021.
6. Zhang, Loquercio, Tang, Wang, Malik, Mueller, "A Learning-Based Quadcopter Controller with Extreme Adaptation," *IEEE T-RO*, 2025.
7. Loquercio, Kaufmann, Ranftl, Dosovitskiy, Koltun, Scaramuzza, "Deep Drone Racing: From Simulation to Reality with Domain Randomization," *IEEE T-RO*, 2020. *(Phase-2 vision extension.)*

Full list with details in the [literature cluster document](./RL_Quadrotor_Control_Literature_Cluster.pdf).