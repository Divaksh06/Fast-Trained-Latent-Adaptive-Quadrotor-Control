"""
PaperHoverEnv: a subclass of gym-pybullet-drones' HoverAviary that adds the
pieces of Eschmann, Albani, Loianno, "Learning to Fly in Seconds" (RA-L 2024)
that HoverAviary doesn't provide out of the box.

Already provided for free by HoverAviary / BaseRLAviary (verified against
the library source -- nothing to add here):
  - RPM-level (taxonomy level 5.1) action space, normalized to [-1, 1]^4,
    mapped to RPM = HOVER_RPM * (1 + 0.05 * action).
  - An action-history buffer (last 0.5s of *commanded* actions) automatically
    appended to the observation -- this is exactly the "action history"
    input the paper uses to handle partial observability from motor delay.

Added in this subclass:
  - A first-order low-pass filter on RPM (motor delay), applied once per
    control step -- the paper's single biggest ablation finding (RA-L '24,
    Table II: training without it produces "no usable policy" at all).
  - A random force/torque disturbance sampled once per episode and applied
    every physics substep (paper's asymmetric-actor-critic disturbance
    inputs -- here just used to make the policy robust, since the critic
    isn't privileged in this first version, see note below).
  - Observation noise on the 12-dim kinematic state.
  - The paper's exact reward formula (Section IV), with curriculum-
    adjustable weights, using the true quaternion (self.quat) for the
    attitude term even though the base KIN observation only exposes Euler
    angles.

Deliberately NOT implemented yet (first working version -- extend later):
  - Asymmetric actor-critic (critic seeing privileged RPM + disturbance
    state). SB3's default TD3 policy doesn't support different actor/critic
    observations without custom policy code; this version trains a
    standard (symmetric) TD3 policy on the noisy observation.
  - Reward recalculation of the whole replay buffer after each curriculum
    update (the paper does this; skipping it here is a minor simplification
    that only matters once you've reproduced the paper's headline number).
"""
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, DroneModel, Physics


class PaperHoverEnv(HoverAviary):

    def __init__(
        self,
        motor_time_constant: float = 0.15,   # seconds -- paper's empirically chosen value (Sec. IV)
        max_disturbance_force: float = 0.0,  # Newtons; 0 disables disturbances
        max_disturbance_torque: float = 0.0, # N*m; 0 disables disturbances
        obs_noise_std: float = 0.01,
        reward_init: dict | None = None,
        reward_target: dict | None = None,
        gui: bool = False,
        **kwargs,
    ):
        super().__init__(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            gui=gui,
            **kwargs,
        )

        self._motor_tau = motor_time_constant
        self._motor_alpha = self.CTRL_TIMESTEP / (self._motor_tau + self.CTRL_TIMESTEP)
        self._true_rpm = None

        self._dist_force_max = max_disturbance_force
        self._dist_torque_max = max_disturbance_torque
        self._dist_force = np.zeros(3)
        self._dist_torque = np.zeros(3)

        self._obs_noise_std = obs_noise_std

        # Reward weights -- defaults loosely follow the paper's structure
        # (position, attitude, velocity, angular-rate, action, survival).
        # "init" is the lenient early-training config, "target" is the
        # stricter converged config the curriculum anneals towards.
        self._reward_init = reward_init or dict(Crp=1.0, Crq=0.1, Crv=0.05, Crw=0.01, Cra=0.01, Crs=1.0)
        self._reward_target = reward_target or dict(Crp=5.0, Crq=1.0, Crv=0.2, Crw=0.05, Cra=0.1, Crs=1.0)
        self._curriculum_alpha = 0.0
        self._apply_curriculum(0.0)

    # ------------------------------------------------------------------ #
    # Curriculum
    # ------------------------------------------------------------------ #
    def _apply_curriculum(self, alpha: float):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        for name, init_val in self._reward_init.items():
            target_val = self._reward_target[name]
            setattr(self, f"_{name}", init_val + alpha * (target_val - init_val))
        self._curriculum_alpha = alpha

    def set_curriculum_progress(self, alpha: float):
        """Call this from a training callback, e.g. every 100_000 steps,
        with alpha = min(1.0, num_timesteps / curriculum_horizon)."""
        self._apply_curriculum(alpha)

    # ------------------------------------------------------------------ #
    # Episode reset -- sample this episode's disturbance
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._true_rpm = None
        self._dist_force = self.np_random.uniform(-1, 1, size=3) * self._dist_force_max
        self._dist_torque = self.np_random.uniform(-1, 1, size=3) * self._dist_torque_max
        return obs, info

    # ------------------------------------------------------------------ #
    # Motor delay: filter the commanded RPM before it reaches the physics
    # ------------------------------------------------------------------ #
    def _preprocessAction(self, action):
        # super()._preprocessAction also appends the *raw commanded* action
        # to self.action_buffer -- that's what feeds the observation's
        # action-history block, exactly as the paper intends.
        target_rpm = super()._preprocessAction(action)
        if self._true_rpm is None:
            self._true_rpm = target_rpm.copy()
        self._true_rpm = self._true_rpm + self._motor_alpha * (target_rpm - self._true_rpm)
        return self._true_rpm

    # ------------------------------------------------------------------ #
    # Disturbance force/torque, applied every physics substep
    # ------------------------------------------------------------------ #
    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        if self._dist_force_max > 0 or self._dist_torque_max > 0:
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone], -1,
                forceObj=self._dist_force.tolist(),
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )
            p.applyExternalTorque(
                self.DRONE_IDS[nth_drone], -1,
                torqueObj=self._dist_torque.tolist(),
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )

    # ------------------------------------------------------------------ #
    # Observation noise on the 12-dim kinematic block only (not the
    # action-history block, which should stay an exact proprioceptive log)
    # ------------------------------------------------------------------ #
    def _computeObs(self):
        obs = super()._computeObs()
        if self._obs_noise_std > 0:
            obs = obs.copy()
            obs[:, 0:12] += self.np_random.normal(0, self._obs_noise_std, size=(self.NUM_DRONES, 12))
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # The paper's reward (Section IV), using the true quaternion for the
    # attitude term -- self.quat is tracked by the base class even though
    # the KIN observation itself only exposes Euler angles.
    # ------------------------------------------------------------------ #
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]        # PyBullet order: [x, y, z, w]
        vel = state[10:13]
        ang_v = state[13:16]
        last_action = state[16:20]  # normalized commanded action in [-1, 1]

        qw = quat[3]
        err_pos = float(np.sum((pos - self.TARGET_POS) ** 2))
        err_att = float(1.0 - qw ** 2)
        err_vel = float(np.sum(vel ** 2))
        err_rate = float(np.sum(ang_v ** 2))
        err_act = float(np.sum(last_action ** 2))  # bias term omitted (0 == hover in this action encoding)

        reward = (
            - self._Crp * err_pos
            - self._Crq * err_att
            - self._Crv * err_vel
            - self._Crw * err_rate
            - self._Cra * err_act
            + self._Crs
        )
        return reward