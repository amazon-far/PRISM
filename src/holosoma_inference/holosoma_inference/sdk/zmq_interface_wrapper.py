from __future__ import annotations

import numpy as np

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.utils.math.quat import quat_rotate_inverse, xyzw_to_wxyz
from holosoma_inference.utils.sim_control import SimControlPush
from holosoma_inference.utils.sim_state import SimStateSub


class ZmqSimInterfaceWrapper:
    """Split sim2sim interface using ZMQ state/control channels instead of Unitree DDS."""

    def __init__(
        self,
        robot_config: RobotConfig,
        *,
        sim_state_port: int = 5557,
        sim_control_port: int = 5559,
        use_joystick: bool = False,
    ) -> None:
        self.robot_config = robot_config
        self.backend = "zmq"
        self.use_joystick = use_joystick
        self.no_action = 0
        self.kp_level = 1.0
        self.kd_level = 1.0
        self._wc_key_map: dict[int, str] = {}
        self._last_robot_state_data: np.ndarray | None = None
        self._last_sim_time_ms: float | None = None
        self._lowcmd_seq = 0

        self._sim_state_sub = SimStateSub(port=sim_state_port)
        self._sim_state_sub.start()
        self._sim_control_pub = SimControlPush(port=sim_control_port)
        self._sim_control_pub.start()

    def update_config(self, robot_config: RobotConfig) -> None:
        self.robot_config = robot_config

    def reset_runtime_state(self) -> None:
        self._last_robot_state_data = None
        self._last_sim_time_ms = None
        self._lowcmd_seq = 0
        if hasattr(self._sim_state_sub, "last_state"):
            self._sim_state_sub.last_state = None

    def _joint_gains_from_robot_config(self) -> tuple[np.ndarray, np.ndarray]:
        joint_kp = np.zeros(self.robot_config.num_joints, dtype=np.float32)
        joint_kd = np.zeros(self.robot_config.num_joints, dtype=np.float32)
        motor_kp = getattr(self.robot_config, "motor_kp", None)
        motor_kd = getattr(self.robot_config, "motor_kd", None)
        if motor_kp is None or motor_kd is None:
            return joint_kp, joint_kd
        for motor_id, joint_id in enumerate(self.robot_config.motor2joint):
            if 0 <= joint_id < self.robot_config.num_joints:
                joint_kp[joint_id] = float(motor_kp[motor_id])
                joint_kd[joint_id] = float(motor_kd[motor_id])
        return joint_kp, joint_kd

    def get_low_state(self) -> np.ndarray | None:
        state = self._sim_state_sub.get_state()
        if state is None:
            return self._last_robot_state_data
        try:
            self._last_sim_time_ms = float(state.get("sim_time_ms"))
        except (TypeError, ValueError):
            pass

        robot_root_state = state.get("robot_root_state")
        robot_dof_pos = state.get("robot_dof_pos")
        robot_dof_vel = state.get("robot_dof_vel")
        if robot_root_state is None or robot_dof_pos is None or robot_dof_vel is None:
            return self._last_robot_state_data

        root_state = np.asarray(robot_root_state, dtype=np.float64)
        dof_pos = np.asarray(robot_dof_pos, dtype=np.float64)
        dof_vel = np.asarray(robot_dof_vel, dtype=np.float64)
        if (
            root_state.shape[0] < 13
            or dof_pos.shape[0] < self.robot_config.num_joints
            or dof_vel.shape[0] < self.robot_config.num_joints
        ):
            return self._last_robot_state_data

        quat_xyzw = root_state[3:7].reshape(1, 4)
        quat_wxyz = xyzw_to_wxyz(quat_xyzw).reshape(-1).astype(np.float64, copy=False)
        base_lin_vel_b = quat_rotate_inverse(quat_wxyz.reshape(1, 4), root_state[7:10].reshape(1, 3)).reshape(-1)
        base_ang_vel_b = quat_rotate_inverse(quat_wxyz.reshape(1, 4), root_state[10:13].reshape(1, 3)).reshape(-1)
        q = np.concatenate([root_state[:3], quat_wxyz, dof_pos[: self.robot_config.num_joints]], axis=0)
        dq = np.concatenate([base_lin_vel_b, base_ang_vel_b, dof_vel[: self.robot_config.num_joints]], axis=0)
        tau_est = np.zeros_like(dq)
        ddq = np.zeros_like(dq)
        robot_state_data = np.concatenate([q, dq, tau_est, ddq], axis=0).reshape(1, -1)
        self._last_robot_state_data = robot_state_data
        return robot_state_data

    def get_sim_time_ms(self) -> float | None:
        return self._last_sim_time_ms

    def send_low_command(
        self,
        cmd_q,
        cmd_dq,
        cmd_tau,
        dof_pos_latest=None,
        kp_override=None,
        kd_override=None,
    ) -> None:
        del dof_pos_latest
        q_target = np.asarray(cmd_q, dtype=np.float32).reshape(-1)
        dq_target = np.asarray(cmd_dq, dtype=np.float32).reshape(-1)
        tau_ff = np.asarray(cmd_tau, dtype=np.float32).reshape(-1)

        if self.no_action:
            kp = np.zeros_like(q_target, dtype=np.float32)
            kd = np.zeros_like(q_target, dtype=np.float32)
            tau_ff = np.zeros_like(q_target, dtype=np.float32)
        else:
            default_joint_kp, default_joint_kd = self._joint_gains_from_robot_config()
            kp = np.asarray(kp_override, dtype=np.float32).reshape(-1) if kp_override is not None else default_joint_kp
            kd = np.asarray(kd_override, dtype=np.float32).reshape(-1) if kd_override is not None else default_joint_kd
            kp *= float(self.kp_level)
            kd *= float(self.kd_level)

        self._sim_control_pub.publish(
            {
                "action": "lowcmd",
                "seq": int(self._lowcmd_seq),
                "policy_sim_time_ms": None if self._last_sim_time_ms is None else float(self._last_sim_time_ms),
                "q_target": q_target.tolist(),
                "dq_target": dq_target.tolist(),
                "tau_ff": tau_ff.tolist(),
                "kp": kp.tolist(),
                "kd": kd.tolist(),
            }
        )
        self._lowcmd_seq += 1

    def process_joystick_input(self, *args):
        if len(args) >= 3:
            return args[0], args[1], {}
        return np.zeros((1, 3), dtype=np.float32)

    def get_joystick_key(self):
        return 0

    def get_joystick_msg(self):
        return None

    def close(self) -> None:
        self._sim_state_sub.close()
        self._sim_control_pub.close()
