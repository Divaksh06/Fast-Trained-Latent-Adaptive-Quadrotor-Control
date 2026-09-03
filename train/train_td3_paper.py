"""
Train PaperHoverEnv with TD3 (the off-policy algorithm the paper uses,
Section III/IV), with:
  - a curriculum callback that anneals the reward weights from lenient to
    strict over `curriculum_horizon` steps (paper: every 100_000 steps)
  - an exploration-noise-decay callback (paper decays exploration noise on
    the same schedule as the curriculum)
  - AUTOMATIC WEIGHT RESUMPTION: if existing weights are found in the
    log directory, training resumes from the latest checkpoint instead
    of starting from scratch. Use --fresh to force a clean start.

Run:
    python train/train_td3_paper.py --total-steps 300000
    python train/train_td3_paper.py --total-steps 300000 --disturbances
    python train/train_td3_paper.py --total-steps 300000 --gui   # render training live
    python train/train_td3_paper.py --total-steps 300000 --fresh # ignore existing weights

The paper's headline result is ~300_000 steps (~18s wall-clock on their
massively-parallel GPU simulator) for a reliable policy; on a single
CPU-stepped gym-pybullet-drones env this will take much longer wall-clock,
but is the right first target for "does this learn to hover at all".
"""
import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from envs.paper_hover_env import PaperHoverEnv


# ------------------------------------------------------------------ #
# Metadata helpers — track cumulative timesteps across sessions
# ------------------------------------------------------------------ #
METADATA_FILENAME = "training_metadata.json"


def _load_metadata(log_dir: str) -> dict:
    """Load training metadata (cumulative timesteps, session history)."""
    path = os.path.join(log_dir, METADATA_FILENAME)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"cumulative_timesteps": 0, "sessions": []}


def _save_metadata(log_dir: str, metadata: dict):
    """Save training metadata to disk."""
    path = os.path.join(log_dir, METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


# ------------------------------------------------------------------ #
# Model discovery — find the latest saved checkpoint
# ------------------------------------------------------------------ #
def find_latest_model(log_dir: str) -> str | None:
    """Find the most recently saved model .zip in `log_dir`.

    Looks for the canonical 'td3_paper_hover.zip' first, then falls back
    to any timestamped version.  Returns None if no model is found.
    """
    canonical = os.path.join(log_dir, "td3_paper_hover.zip")
    if os.path.exists(canonical):
        return canonical

    # Fallback: find any timestamped checkpoint
    pattern = os.path.join(log_dir, "td3_paper_hover_*.zip")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return candidates[-1]  # latest by filename sort (timestamp-based)

    return None


# ------------------------------------------------------------------ #
# Callbacks
# ------------------------------------------------------------------ #
class CurriculumCallback(BaseCallback):
    """Every `update_freq` steps, sets curriculum progress = min(1, t / horizon).

    When resuming training, `timestep_offset` shifts the progress calculation
    so the curriculum picks up where the previous session left off.
    """

    def __init__(self, horizon: int, update_freq: int = 10_000,
                 timestep_offset: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.horizon = horizon
        self.update_freq = update_freq
        self.timestep_offset = timestep_offset

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            effective_timesteps = self.timestep_offset + self.num_timesteps
            alpha = min(1.0, effective_timesteps / self.horizon)
            self.training_env.env_method("set_curriculum_progress", alpha)
            if self.verbose:
                print(f"[curriculum] step={self.num_timesteps} "
                      f"(cumulative={effective_timesteps}) alpha={alpha:.2f}")
        return True


class ExplorationDecayCallback(BaseCallback):
    """Linearly decays the TD3 action-noise sigma over `decay_steps`.

    When resuming training, `timestep_offset` shifts the decay schedule
    so exploration picks up where the previous session left off.
    """

    def __init__(self, action_dim: int, sigma_init: float, sigma_final: float,
                 decay_steps: int, update_freq: int = 10_000,
                 timestep_offset: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.action_dim = action_dim
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.decay_steps = decay_steps
        self.update_freq = update_freq
        self.timestep_offset = timestep_offset

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            effective_timesteps = self.timestep_offset + self.num_timesteps
            progress = min(1.0, effective_timesteps / self.decay_steps)
            sigma = self.sigma_init + progress * (self.sigma_final - self.sigma_init)
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(self.action_dim), sigma=sigma * np.ones(self.action_dim)
            )
            if self.verbose:
                print(f"[exploration] step={self.num_timesteps} "
                      f"(cumulative={effective_timesteps}) sigma={sigma:.3f}")
        return True


def make_env(disturbances: bool, log_dir: str, gui: bool = False):
    def _init():
        env = PaperHoverEnv(
            motor_time_constant=0.15,
            max_disturbance_force=0.05 if disturbances else 0.0,   # Newtons
            max_disturbance_torque=0.002 if disturbances else 0.0, # N*m
            obs_noise_std=0.01,
            gui=gui,
        )
        return Monitor(env, log_dir)
    return _init


def main(total_steps: int, disturbances: bool, log_dir: str, seed: int,
         gui: bool = False, fresh: bool = False):
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(make_env(disturbances, log_dir, gui=gui), n_envs=1, seed=seed)
    action_dim = env.action_space.shape[-1]

    # ------------------------------------------------------------------ #
    # Load or create model
    # ------------------------------------------------------------------ #
    metadata = _load_metadata(log_dir)
    cumulative_timesteps = metadata["cumulative_timesteps"]
    resumed = False

    latest_model = find_latest_model(log_dir) if not fresh else None

    if latest_model is not None:
        print(f"[train] ✓ RESUMING from {latest_model}")
        print(f"[train]   cumulative timesteps so far: {cumulative_timesteps}")
        model = TD3.load(
            latest_model,
            env=env,
            learning_rate=3e-4,
            seed=seed,
            verbose=1,
            tensorboard_log=log_dir,
        )
        # Restore the replay buffer if it exists
        buffer_path = os.path.join(log_dir, "td3_replay_buffer.pkl")
        if os.path.exists(buffer_path):
            model.load_replay_buffer(buffer_path)
            print(f"[train]   loaded replay buffer from {buffer_path} "
                  f"({model.replay_buffer.size()} transitions)")
        else:
            print("[train]   no replay buffer found — starting with empty buffer")
        resumed = True
    else:
        if fresh:
            print("[train] ✗ FRESH start requested — ignoring any existing weights")
        else:
            print("[train] ✗ No existing weights found — starting FRESH training")
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=NormalActionNoise(
                mean=np.zeros(action_dim), sigma=0.3 * np.ones(action_dim)
            ),
            policy_delay=2,
            seed=seed,
            verbose=1,
            tensorboard_log=log_dir,
        )
        cumulative_timesteps = 0

    # ------------------------------------------------------------------ #
    # Callbacks — offset by cumulative timesteps so curriculum/exploration
    # schedules continue from where the previous session left off
    # ------------------------------------------------------------------ #
    # Use a horizon that accounts for all training done so far + this session
    total_lifetime_steps = cumulative_timesteps + total_steps
    curriculum_horizon = max(1, total_lifetime_steps // 3)

    callbacks = CallbackList([
        CurriculumCallback(
            horizon=curriculum_horizon, update_freq=10_000,
            timestep_offset=cumulative_timesteps, verbose=1,
        ),
        ExplorationDecayCallback(
            action_dim=action_dim, sigma_init=0.3, sigma_final=0.05,
            decay_steps=curriculum_horizon, update_freq=10_000,
            timestep_offset=cumulative_timesteps, verbose=1,
        ),
    ])

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print(f"[train] training for {total_steps} steps "
          f"(cumulative after this session: {cumulative_timesteps + total_steps})")
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not resumed,
    )

    # ------------------------------------------------------------------ #
    # Save: versioned + canonical + replay buffer + metadata
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Versioned save
    versioned_path = os.path.join(log_dir, f"td3_paper_hover_{timestamp}")
    model.save(versioned_path)
    print(f"[train] saved versioned model to {versioned_path}.zip")

    # Canonical save (always the "latest")
    canonical_path = os.path.join(log_dir, "td3_paper_hover")
    model.save(canonical_path)
    print(f"[train] saved canonical model to {canonical_path}.zip")

    # Replay buffer
    buffer_path = os.path.join(log_dir, "td3_replay_buffer.pkl")
    model.save_replay_buffer(buffer_path)
    print(f"[train] saved replay buffer to {buffer_path}")

    # Update metadata
    new_cumulative = cumulative_timesteps + total_steps
    metadata["cumulative_timesteps"] = new_cumulative
    metadata["sessions"].append({
        "timestamp": timestamp,
        "steps_this_session": total_steps,
        "cumulative_timesteps": new_cumulative,
        "resumed": resumed,
        "disturbances": disturbances,
        "seed": seed,
    })
    _save_metadata(log_dir, metadata)
    print(f"[train] updated metadata — cumulative timesteps: {new_cumulative}")
    print(f"[train] tensorboard logs at {log_dir} — run: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--disturbances", action="store_true",
                         help="enable the random per-episode force/torque disturbance")
    parser.add_argument("--log-dir", type=str, default="results/td3_paper_baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true",
                         help="open PyBullet GUI to visualize training in real time (slower)")
    parser.add_argument("--fresh", action="store_true",
                         help="force a fresh start, ignoring any existing saved weights")
    args = parser.parse_args()
    main(args.total_steps, args.disturbances, args.log_dir, args.seed,
         gui=args.gui, fresh=args.fresh)