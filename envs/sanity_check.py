"""
Sanity check #1: confirms gym-pybullet-drones, PyBullet, and your conda env
are wired up correctly before any RL work starts. No RL involved -- just
random actions, to prove the simulator itself steps correctly.

Run:
    python envs/sanity_check.py                 # headless, random actions, fast
    python envs/sanity_check.py --gui           # GUI, random actions, real-time pace
    python envs/sanity_check.py --gui --hover   # GUI, near-hover actions -- calmer visual

Note on the GUI view: with pure random actions the drone tips over and the
episode resets within a handful of steps almost every time -- that's
expected (the env correctly detects the crash and truncates). Without
real-time pacing, many crash-resets per second can look like the window is
"blinking" as the drone snaps back to the start pose repeatedly. This
version paces steps to real time and offers a --hover flag (small random
perturbations around the hover RPM instead of fully random RPM) so you can
actually watch stable-ish flight if you want a calmer sanity check.
"""
import argparse
import time
import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def main(gui: bool, hover: bool):
    env = HoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,   # level 5.1 in the paper's taxonomy -- direct RPM setpoints
        gui=gui,
    )

    obs, info = env.reset(seed=0)
    print(f"[sanity check] observation shape: {obs.shape}")
    print(f"[sanity check] action space:       {env.action_space}")
    print(f"[sanity check] HOVER_RPM:          {env.HOVER_RPM:.1f}")

    total_reward = 0.0
    n_steps = 500
    n_resets = 0
    dt = 1.0 / env.CTRL_FREQ  # real-time pacing target when gui=True

    for step in range(n_steps):
        start = time.time()

        if hover:
            # small perturbations around hover (action=0 -> HOVER_RPM exactly)
            # instead of fully random RPM -- stays airborne much longer
            action = np.clip(np.random.normal(0, 0.05, size=env.action_space.shape), -1, 1)
        else:
            action = env.action_space.sample()  # fully random RPM commands in [-1, 1]^4

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            n_resets += 1
            obs, info = env.reset()

        if gui:
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    env.close()
    print(f"[sanity check] ran {n_steps} steps, {n_resets} episode reset(s) "
          f"under a RANDOM policy (episodes ending early is expected here).")
    print(f"[sanity check] cumulative reward: {total_reward:.2f}")
    print("[sanity check] PASSED -- environment, physics, and spaces all working.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="open a PyBullet GUI window")
    parser.add_argument("--hover", action="store_true",
                         help="use small perturbations around hover instead of fully random RPM")
    args = parser.parse_args()
    main(gui=args.gui, hover=args.hover)