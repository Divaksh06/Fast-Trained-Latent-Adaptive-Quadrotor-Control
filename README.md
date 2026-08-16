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

## Tooling

- **Simulation:** [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) as the base quadrotor dynamics/sim environment, with GPU-parallel alternatives (IsaacLab / Aerial Gym) as a possible upgrade path for training speed.
- **RL algorithm:** SAC / PPO, via Stable-Baselines3 or custom implementation.
- **Latent-adaptation module:** small encoder over recent state/action history, architecturally similar to prior multimodal-fusion work (VFP-SAC), repurposed to encode dynamics/disturbance identity instead of vision+force.
- **Safety clamp:** ensemble or MC-dropout uncertainty head feeding a simple tilt/thrust-margin clamp.

---

## Repo structure

```
.
├── README.md
├── RL_Quadrotor_Control_Literature_Cluster.tex   # literature review + proposal (LaTeX source)
├── RL_Quadrotor_Control_Literature_Cluster.pdf   # compiled version
├── envs/            # domain-randomized quadrotor env wrapper (TBD)
├── models/          # policy, latent-adaptation module, safety clamp (TBD)
├── train/           # training scripts per ablation condition (TBD)
├── eval/            # generalization / robustness / safety evaluation scripts (TBD)
└── results/         # logs, plots, checkpoints — populated as the project progresses
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