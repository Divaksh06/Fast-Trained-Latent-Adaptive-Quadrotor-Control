"""
Train PaperHoverEnv with TD3 (the off-policy algorithm the paper uses,
Section III/IV), with:
  - a curriculum callback that anneals the reward weights from lenient to
    strict over `curriculum_horizon` steps (paper: every 100_000 steps)
  - an exploration-noise-decay callback (paper decays exploration noise on
    the same schedule as the curriculum)

Run:
    python train/train_td3_paper.py --total-steps 300000
    python train/train_td3_paper.py --total-steps 300000 --disturbances

The paper's headline result is ~300_000 steps (~18s wall-clock on their
massively-parallel GPU simulator) for a reliable policy; on a single
CPU-stepped gym-pybullet-drones env this will take much longer wall-clock,
but is the right first target for "does this learn to hover at all".
"""
import argparse
import os

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from envs.paper_hover_env import PaperHoverEnv


class CurriculumCallback(BaseCallback):
    """Every `update_freq` steps, sets curriculum progress = min(1, t / horizon)."""

    def __init__(self, horizon: int, update_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.horizon = horizon
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            alpha = min(1.0, self.num_timesteps / self.horizon)
            self.training_env.env_method("set_curriculum_progress", alpha)
            if self.verbose:
                print(f"[curriculum] step={self.num_timesteps} alpha={alpha:.2f}")
        return True


class ExplorationDecayCallback(BaseCallback):
    """Linearly decays the TD3 action-noise sigma over `decay_steps`."""

    def __init__(self, action_dim: int, sigma_init: float, sigma_final: float,
                 decay_steps: int, update_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.action_dim = action_dim
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.decay_steps = decay_steps
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            progress = min(1.0, self.num_timesteps / self.decay_steps)
            sigma = self.sigma_init + progress * (self.sigma_final - self.sigma_init)
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(self.action_dim), sigma=sigma * np.ones(self.action_dim)
            )
            if self.verbose:
                print(f"[exploration] step={self.num_timesteps} sigma={sigma:.3f}")
        return True


def make_env(disturbances: bool, log_dir: str):
    def _init():
        env = PaperHoverEnv(
            motor_time_constant=0.15,
            max_disturbance_force=0.05 if disturbances else 0.0,   # Newtons
            max_disturbance_torque=0.002 if disturbances else 0.0, # N*m
            obs_noise_std=0.01,
            gui=False,
        )
        return Monitor(env, log_dir)
    return _init


def main(total_steps: int, disturbances: bool, log_dir: str, seed: int):
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(make_env(disturbances, log_dir), n_envs=1, seed=seed)
    action_dim = env.action_space.shape[-1]

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=NormalActionNoise(mean=np.zeros(action_dim), sigma=0.3 * np.ones(action_dim)),
        policy_delay=2,
        seed=seed,
        verbose=1,
        tensorboard_log=log_dir,
    )

    curriculum_horizon = max(1, total_steps // 3)  # anneal over the first third of training
    callbacks = CallbackList([
        CurriculumCallback(horizon=curriculum_horizon, update_freq=10_000, verbose=1),
        ExplorationDecayCallback(
            action_dim=action_dim, sigma_init=0.3, sigma_final=0.05,
            decay_steps=curriculum_horizon, update_freq=10_000, verbose=1,
        ),
    ])

    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)

    save_path = os.path.join(log_dir, "td3_paper_hover")
    model.save(save_path)
    print(f"[train] saved model to {save_path}.zip")
    print(f"[train] tensorboard logs at {log_dir} -- run: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--disturbances", action="store_true",
                         help="enable the random per-episode force/torque disturbance")
    parser.add_argument("--log-dir", type=str, default="results/td3_paper_baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.total_steps, args.disturbances, args.log_dir, args.seed)