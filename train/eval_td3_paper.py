"""
Evaluate / visualize a trained TD3 policy in the PyBullet GUI.

Loads a saved TD3 checkpoint and runs it with rendering so you can
watch the learned hover policy fly.

Run:
    python train/eval_td3_paper.py                              # defaults: last saved model, GUI on
    python train/eval_td3_paper.py --model results/td3_paper_baseline/td3_paper_hover.zip
    python train/eval_td3_paper.py --episodes 5 --disturbances  # 5 episodes with disturbances
    python train/eval_td3_paper.py --no-gui                     # headless, print metrics only
"""
import argparse
import time

import numpy as np
from stable_baselines3 import TD3

from envs.paper_hover_env import PaperHoverEnv


def main(model_path: str, episodes: int, disturbances: bool, gui: bool):
    env = PaperHoverEnv(
        motor_time_constant=0.15,
        max_disturbance_force=0.05 if disturbances else 0.0,
        max_disturbance_torque=0.002 if disturbances else 0.0,
        obs_noise_std=0.01,
        gui=gui,
    )
    # Set curriculum to fully converged (strict) weights for evaluation
    env.set_curriculum_progress(1.0)

    model = TD3.load(model_path)
    print(f"[eval] loaded model from {model_path}")
    print(f"[eval] running {episodes} episode(s), gui={gui}, disturbances={disturbances}")

    dt = 1.0 / env.CTRL_FREQ  # real-time pacing target

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            if gui:
                elapsed = time.time() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        pos = env._getDroneStateVector(0)[0:3]
        print(f"[eval] episode {ep+1}/{episodes}: "
              f"steps={ep_steps}, reward={ep_reward:.2f}, "
              f"final_pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    env.close()
    print("[eval] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="results/td3_paper_baseline/td3_paper_hover.zip",
                        help="path to the saved TD3 model .zip")
    parser.add_argument("--episodes", type=int, default=3,
                        help="number of episodes to run")
    parser.add_argument("--disturbances", action="store_true",
                        help="enable random force/torque disturbances")
    parser.add_argument("--no-gui", action="store_true",
                        help="run headless (no PyBullet window)")
    args = parser.parse_args()
    main(args.model, args.episodes, args.disturbances, gui=not args.no_gui)
