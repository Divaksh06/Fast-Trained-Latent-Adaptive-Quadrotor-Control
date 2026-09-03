"""
Simulate a trained TD3 hover policy in PyBullet and generate diagnostic reports.

This script does NOT train — it loads the latest saved weights and runs
hover episodes to evaluate how well the model has learned.  Detailed
reports (human-readable .txt + machine-readable .json) are saved to the
``simulation_reports/`` directory with timestamps.

The report includes:
  - Per-episode hover time, reward, position/velocity errors
  - Aggregate statistics across episodes
  - Learning assessment (comparison against a random-action baseline)
  - Overfitting check (clean vs. disturbance performance comparison)
  - An overall verdict

Run:
    python train/simulate_hover.py                           # headless, 10 episodes
    python train/simulate_hover.py --gui --episodes 5        # PyBullet GUI, 5 eps
    python train/simulate_hover.py --model path/to/model.zip # specific checkpoint
"""
import argparse
import glob
import json
import os
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import TD3

from envs.paper_hover_env import PaperHoverEnv


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
REPORTS_DIR = "simulation_reports"
OVERFITTING_THRESHOLD = 0.30  # 30% degradation → flag overfitting


# ------------------------------------------------------------------ #
# Model discovery
# ------------------------------------------------------------------ #
def find_latest_model(log_dir: str) -> str | None:
    """Find the most recently saved model .zip in `log_dir`."""
    canonical = os.path.join(log_dir, "td3_paper_hover.zip")
    if os.path.exists(canonical):
        return canonical
    pattern = os.path.join(log_dir, "td3_paper_hover_*.zip")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


# ------------------------------------------------------------------ #
# Episode runner
# ------------------------------------------------------------------ #
def run_episodes(env: PaperHoverEnv, model, episodes: int,
                 deterministic: bool = True, gui: bool = False) -> list[dict]:
    """Run `episodes` and collect per-episode metrics."""
    dt = 1.0 / env.CTRL_FREQ
    results = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        pos_errors = []
        velocities = []
        ang_rates = []

        while not done:
            start = time.time()

            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # Random policy baseline
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            # Collect per-step metrics
            state = env._getDroneStateVector(0)
            pos = state[0:3]
            vel = state[10:13]
            ang_v = state[13:16]

            pos_err = float(np.linalg.norm(pos - env.TARGET_POS))
            vel_mag = float(np.linalg.norm(vel))
            ang_rate_mag = float(np.linalg.norm(ang_v))

            pos_errors.append(pos_err)
            velocities.append(vel_mag)
            ang_rates.append(ang_rate_mag)

            if gui:
                elapsed = time.time() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        hover_time = ep_steps * dt
        final_pos = env._getDroneStateVector(0)[0:3].tolist()

        results.append({
            "episode": ep + 1,
            "steps": ep_steps,
            "hover_time_s": round(hover_time, 3),
            "max_hover_time_s": round(env.EPISODE_LEN_SEC, 3),
            "reward": round(ep_reward, 4),
            "crashed": bool(terminated),
            "truncated": bool(truncated),
            "final_pos": [round(x, 4) for x in final_pos],
            "mean_pos_error": round(float(np.mean(pos_errors)), 4),
            "max_pos_error": round(float(np.max(pos_errors)), 4),
            "mean_velocity": round(float(np.mean(velocities)), 4),
            "max_velocity": round(float(np.max(velocities)), 4),
            "mean_ang_rate": round(float(np.mean(ang_rates)), 4),
            "max_ang_rate": round(float(np.max(ang_rates)), 4),
        })

    return results


# ------------------------------------------------------------------ #
# Aggregate statistics
# ------------------------------------------------------------------ #
def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate statistics from per-episode results."""
    rewards = [r["reward"] for r in results]
    hover_times = [r["hover_time_s"] for r in results]
    pos_errors = [r["mean_pos_error"] for r in results]
    velocities = [r["mean_velocity"] for r in results]
    ang_rates = [r["mean_ang_rate"] for r in results]
    n_crashed = sum(1 for r in results if r["crashed"])
    max_hover = results[0]["max_hover_time_s"] if results else 0

    return {
        "num_episodes": len(results),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "std_reward": round(float(np.std(rewards)), 4),
        "mean_hover_time_s": round(float(np.mean(hover_times)), 3),
        "std_hover_time_s": round(float(np.std(hover_times)), 3),
        "max_possible_hover_s": max_hover,
        "hover_coverage_pct": round(float(np.mean(hover_times)) / max_hover * 100, 1) if max_hover > 0 else 0,
        "mean_pos_error": round(float(np.mean(pos_errors)), 4),
        "std_pos_error": round(float(np.std(pos_errors)), 4),
        "mean_velocity": round(float(np.mean(velocities)), 4),
        "std_velocity": round(float(np.std(velocities)), 4),
        "mean_ang_rate": round(float(np.mean(ang_rates)), 4),
        "std_ang_rate": round(float(np.std(ang_rates)), 4),
        "crash_rate_pct": round(n_crashed / len(results) * 100, 1) if results else 0,
        "stability_score": round(1.0 / (1.0 + float(np.std(rewards))), 4),
    }


# ------------------------------------------------------------------ #
# Report generation
# ------------------------------------------------------------------ #
def generate_txt_report(
    model_path: str,
    timestamp: str,
    clean_results: list[dict],
    clean_agg: dict,
    random_agg: dict,
    disturb_results: list[dict] | None,
    disturb_agg: dict | None,
) -> str:
    """Generate a human-readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("    QUADROTOR HOVER SIMULATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Timestamp       : {timestamp}")
    lines.append(f"Model           : {model_path}")
    lines.append(f"Episodes (clean): {clean_agg['num_episodes']}")
    lines.append("")

    # --- Per-Episode Results ---
    lines.append("-" * 60)
    lines.append("PER-EPISODE RESULTS (no disturbances)")
    lines.append("-" * 60)
    for r in clean_results:
        status = "CRASHED" if r["crashed"] else ("FULL" if r["truncated"] else "OK")
        lines.append(
            f"  Ep {r['episode']:>2d}: hover={r['hover_time_s']:.2f}s  "
            f"reward={r['reward']:>8.2f}  "
            f"pos_err={r['mean_pos_error']:.4f}  "
            f"[{status}]"
        )
    lines.append("")

    # --- Aggregate Metrics ---
    lines.append("-" * 60)
    lines.append("AGGREGATE METRICS")
    lines.append("-" * 60)
    lines.append(f"  Mean hover time   : {clean_agg['mean_hover_time_s']:.2f}s "
                 f"± {clean_agg['std_hover_time_s']:.2f}s "
                 f"(max possible: {clean_agg['max_possible_hover_s']:.2f}s)")
    lines.append(f"  Hover coverage    : {clean_agg['hover_coverage_pct']:.1f}%")
    lines.append(f"  Mean reward       : {clean_agg['mean_reward']:.2f} "
                 f"± {clean_agg['std_reward']:.2f}")
    lines.append(f"  Mean position err : {clean_agg['mean_pos_error']:.4f} "
                 f"± {clean_agg['std_pos_error']:.4f}")
    lines.append(f"  Mean velocity     : {clean_agg['mean_velocity']:.4f} "
                 f"± {clean_agg['std_velocity']:.4f}")
    lines.append(f"  Mean angular rate : {clean_agg['mean_ang_rate']:.4f} "
                 f"± {clean_agg['std_ang_rate']:.4f}")
    lines.append(f"  Crash rate        : {clean_agg['crash_rate_pct']:.1f}%")
    lines.append(f"  Stability score   : {clean_agg['stability_score']:.4f} "
                 f"(1.0 = perfectly consistent)")
    lines.append("")

    # --- Learning Assessment ---
    lines.append("-" * 60)
    lines.append("LEARNING ASSESSMENT")
    lines.append("-" * 60)
    lines.append(f"  Random policy reward : {random_agg['mean_reward']:.2f} "
                 f"± {random_agg['std_reward']:.2f}")
    lines.append(f"  Trained policy reward: {clean_agg['mean_reward']:.2f} "
                 f"± {clean_agg['std_reward']:.2f}")

    reward_improvement = clean_agg['mean_reward'] - random_agg['mean_reward']
    # "Has learned" = trained reward meaningfully exceeds random baseline.
    # We require trained reward to be higher AND the gap to be substantial
    # (at least 50% of the absolute random baseline reward, or > 1.0 absolute).
    min_gap = max(1.0, abs(random_agg['mean_reward']) * 0.5)
    has_learned = reward_improvement > min_gap
    if has_learned:
        lines.append(f"  Status: ✓ MODEL HAS LEARNED "
                     f"(+{reward_improvement:.2f} improvement over random)")
    else:
        lines.append(f"  Status: ✗ MODEL HAS NOT LEARNED SIGNIFICANTLY "
                     f"({reward_improvement:+.2f} vs random baseline)")
    lines.append("")

    # --- Overfitting Check ---
    lines.append("-" * 60)
    lines.append("OVERFITTING CHECK")
    lines.append("-" * 60)
    if disturb_agg is not None:
        lines.append(f"  Performance (clean)       : {clean_agg['mean_reward']:.2f} "
                     f"± {clean_agg['std_reward']:.2f}")
        lines.append(f"  Performance (disturbances): {disturb_agg['mean_reward']:.2f} "
                     f"± {disturb_agg['std_reward']:.2f}")

        if clean_agg['mean_reward'] != 0:
            degradation = (clean_agg['mean_reward'] - disturb_agg['mean_reward']) / abs(clean_agg['mean_reward'])
        else:
            degradation = 0.0

        degradation_pct = max(0, degradation) * 100
        is_overfit = degradation > OVERFITTING_THRESHOLD

        if is_overfit:
            lines.append(f"  Status: ⚠ POSSIBLE OVERFITTING DETECTED "
                         f"({degradation_pct:.1f}% degradation > "
                         f"{OVERFITTING_THRESHOLD*100:.0f}% threshold)")
        else:
            lines.append(f"  Status: ✓ NO OVERFITTING DETECTED "
                         f"({degradation_pct:.1f}% degradation < "
                         f"{OVERFITTING_THRESHOLD*100:.0f}% threshold)")
    else:
        lines.append("  Skipped (no disturbance test run)")
    lines.append("")

    # --- Verdict ---
    lines.append("-" * 60)
    lines.append("VERDICT")
    lines.append("-" * 60)

    verdicts = []
    if has_learned:
        verdicts.append("The model shows learning — it outperforms a random policy.")
    else:
        verdicts.append("The model has NOT yet learned meaningful hover behavior.")

    coverage = clean_agg['hover_coverage_pct']
    if coverage > 90:
        verdicts.append(f"Excellent hover duration ({coverage:.0f}% of max episode length).")
    elif coverage > 50:
        verdicts.append(f"Moderate hover duration ({coverage:.0f}% of max). More training recommended.")
    else:
        verdicts.append(f"Poor hover duration ({coverage:.0f}% of max). Significantly more training needed.")

    if clean_agg['crash_rate_pct'] > 50:
        verdicts.append(f"High crash rate ({clean_agg['crash_rate_pct']:.0f}%). "
                        "Policy is unstable — continue training.")
    elif clean_agg['crash_rate_pct'] > 0:
        verdicts.append(f"Some crashes ({clean_agg['crash_rate_pct']:.0f}%). "
                        "Policy needs refinement.")
    else:
        verdicts.append("No crashes detected — policy is stable within tested conditions.")

    if disturb_agg is not None and degradation > OVERFITTING_THRESHOLD:
        verdicts.append("Consider training WITH disturbances enabled to improve robustness.")

    for v in verdicts:
        lines.append(f"  • {v}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_json_report(
    model_path: str,
    timestamp: str,
    clean_results: list[dict],
    clean_agg: dict,
    random_results: list[dict],
    random_agg: dict,
    disturb_results: list[dict] | None,
    disturb_agg: dict | None,
) -> dict:
    """Generate a machine-readable JSON report."""
    report = {
        "timestamp": timestamp,
        "model_path": model_path,
        "overfitting_threshold": OVERFITTING_THRESHOLD,
        "clean": {
            "episodes": clean_results,
            "aggregate": clean_agg,
        },
        "random_baseline": {
            "episodes": random_results,
            "aggregate": random_agg,
        },
    }
    if disturb_results is not None:
        report["disturbances"] = {
            "episodes": disturb_results,
            "aggregate": disturb_agg,
        }
        if clean_agg['mean_reward'] != 0:
            deg = (clean_agg['mean_reward'] - disturb_agg['mean_reward']) / abs(clean_agg['mean_reward'])
        else:
            deg = 0.0
        report["overfitting_degradation_pct"] = round(max(0, deg) * 100, 2)
        report["overfitting_flagged"] = deg > OVERFITTING_THRESHOLD

    reward_improvement = clean_agg['mean_reward'] - random_agg['mean_reward']
    min_gap = max(1.0, abs(random_agg['mean_reward']) * 0.5)
    report["has_learned"] = reward_improvement > min_gap
    report["reward_improvement_over_random"] = round(reward_improvement, 4)

    return report


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main(model_path: str | None, episodes: int, gui: bool, log_dir: str):
    # --- Find model ---
    if model_path is None:
        model_path = find_latest_model(log_dir)
        if model_path is None:
            print("[simulate] ✗ No saved model found in", log_dir)
            print("[simulate]   Train first: python train/train_td3_paper.py --total-steps 300000")
            return
    elif not os.path.exists(model_path):
        print(f"[simulate] ✗ Model not found: {model_path}")
        return

    print(f"[simulate] Loading model from {model_path}")
    model = TD3.load(model_path)

    # --- 1. Clean episodes (no disturbances) ---
    print(f"[simulate] Running {episodes} clean episodes (no disturbances)...")
    clean_env = PaperHoverEnv(
        motor_time_constant=0.15,
        max_disturbance_force=0.0,
        max_disturbance_torque=0.0,
        obs_noise_std=0.01,
        gui=gui,
    )
    clean_env.set_curriculum_progress(1.0)
    clean_results = run_episodes(clean_env, model, episodes, gui=gui)
    clean_agg = compute_aggregate(clean_results)
    clean_env.close()

    # --- 2. Random baseline (3 episodes, no GUI) ---
    print("[simulate] Running 3 random-baseline episodes...")
    rand_env = PaperHoverEnv(
        motor_time_constant=0.15,
        max_disturbance_force=0.0,
        max_disturbance_torque=0.0,
        obs_noise_std=0.01,
        gui=False,
    )
    rand_env.set_curriculum_progress(1.0)
    random_results = run_episodes(rand_env, None, 3, gui=False)
    random_agg = compute_aggregate(random_results)
    rand_env.close()

    # --- 3. Disturbance episodes (overfitting check) ---
    print(f"[simulate] Running {episodes} disturbance episodes (overfitting check)...")
    dist_env = PaperHoverEnv(
        motor_time_constant=0.15,
        max_disturbance_force=0.05,
        max_disturbance_torque=0.002,
        obs_noise_std=0.01,
        gui=False,
    )
    dist_env.set_curriculum_progress(1.0)
    disturb_results = run_episodes(dist_env, model, episodes, gui=False)
    disturb_agg = compute_aggregate(disturb_results)
    dist_env.close()

    # --- Generate reports ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Text report
    txt_report = generate_txt_report(
        model_path, timestamp, clean_results, clean_agg,
        random_agg, disturb_results, disturb_agg,
    )
    txt_path = os.path.join(REPORTS_DIR, f"sim_report_{file_timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(txt_report)

    # JSON report
    json_report = generate_json_report(
        model_path, timestamp, clean_results, clean_agg,
        random_results, random_agg, disturb_results, disturb_agg,
    )
    json_path = os.path.join(REPORTS_DIR, f"sim_report_{file_timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    # Print to console too
    print()
    print(txt_report)
    print()
    print(f"[simulate] Reports saved:")
    print(f"  TXT : {txt_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a trained hover policy and generate diagnostic reports."
    )
    parser.add_argument("--model", type=str, default=None,
                        help="path to a saved TD3 model .zip (auto-detects latest if omitted)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of evaluation episodes to run (default: 10)")
    parser.add_argument("--gui", action="store_true",
                        help="open PyBullet GUI for the clean episodes (slower)")
    parser.add_argument("--log-dir", type=str, default="results/td3_paper_baseline",
                        help="directory to search for saved models")
    args = parser.parse_args()
    main(args.model, args.episodes, args.gui, args.log_dir)
