from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jump_1d_model import Jump1DModel
from jump_params import JumpParams
from jump_phase_observer import PHASE_NAMES, JumpPhase
from jump_pitch_controller import JumpPitchController


@dataclass
class Jump2DPitchInitialState:
    theta: float
    theta_dot: float


class Jump2DPitchModel:
    """Pure Python 2D z/pitch smoke model.

    Vertical dynamics and hybrid jump phases are reused from the 1D jump model.
    Pitch dynamics are intentionally simplified:

        Iyy * theta_ddot = T_pitch + disturbance_torque - damping_theta * theta_dot

    T_pitch is a virtual pitch torque for this smoke test; it is not yet mapped
    to wheel motor torque or hip/leg actuator torque.
    """

    def __init__(self, params: JumpParams, initial: Jump2DPitchInitialState):
        self.p = params
        self.vertical = Jump1DModel(params)
        self.pitch_controller = JumpPitchController()
        self.theta = float(initial.theta)
        self.theta_dot = float(initial.theta_dot)

    def step(self) -> dict[str, float | int | str | bool]:
        p = self.p
        torque = self.pitch_controller.compute(
            self.theta,
            self.theta_dot,
            p.theta_ref,
            p.theta_dot_ref,
            p,
        )
        theta_ddot = (
            torque.clipped
            + p.disturbance_torque
            - p.damping_theta * self.theta_dot
        ) / p.Iyy
        self.theta_dot += theta_ddot * p.dt
        self.theta += self.theta_dot * p.dt

        vertical_sample = self.vertical.step()
        phase = JumpPhase(int(vertical_sample["phase"]))
        contact = phase in {JumpPhase.SQUAT, JumpPhase.THRUST, JumpPhase.LANDING, JumpPhase.RECOVER}

        return {
            "t": float(vertical_sample["t"]),
            "z": float(vertical_sample["z_com"]),
            "zdot": float(vertical_sample["z_com_dot"]),
            "z_body": float(vertical_sample["z_b"]),
            "theta": self.theta,
            "theta_dot": self.theta_dot,
            "theta_ddot": theta_ddot,
            "leg_length": float(vertical_sample["L"]),
            "leg_length_dot": float(vertical_sample["L_dot"]),
            "leg_length_ref": float(vertical_sample["L_ref"]),
            "phase": int(phase),
            "phase_name": PHASE_NAMES[phase],
            "contact": bool(contact),
            "Fz": float(vertical_sample["F_leg"]),
            "Fz_raw": float(vertical_sample["F_raw"]),
            "force_saturated": bool(vertical_sample["force_saturated"]),
            "T_pitch": torque.clipped,
            "T_pitch_raw": torque.raw,
            "pitch_torque_saturated": torque.saturated,
            "disturbance_torque": p.disturbance_torque,
        }

    def run(self) -> dict[str, np.ndarray | bool]:
        samples: list[dict] = []
        n_steps = int(np.ceil(self.p.t_final / self.p.dt))
        for _ in range(n_steps):
            samples.append(self.step())
            if self.vertical.finished:
                break

        numeric_keys = [
            "t",
            "z",
            "zdot",
            "z_body",
            "theta",
            "theta_dot",
            "theta_ddot",
            "leg_length",
            "leg_length_dot",
            "leg_length_ref",
            "phase",
            "contact",
            "Fz",
            "Fz_raw",
            "force_saturated",
            "T_pitch",
            "T_pitch_raw",
            "pitch_torque_saturated",
            "disturbance_torque",
        ]
        out: dict[str, np.ndarray | bool] = {}
        for key in numeric_keys:
            out[key] = np.asarray([s[key] for s in samples], dtype=float)

        out["phase_names_per_sample"] = np.asarray([s["phase_name"] for s in samples])
        out["phase_name_table"] = np.asarray([PHASE_NAMES[JumpPhase(i)] for i in range(len(PHASE_NAMES))])
        out["takeoff_time"] = np.array([self.vertical.takeoff_time])
        out["landing_time"] = np.array([self.vertical.landing_time])
        out["recover_time"] = np.array([self.vertical.recover_time])
        takeoff_time = float(out["takeoff_time"][0])
        if np.isfinite(takeoff_time):
            takeoff_idx = int(np.searchsorted(out["t"], takeoff_time))
            takeoff_idx = int(np.clip(takeoff_idx, 0, len(out["t"]) - 1))
            flight_mask = np.logical_or(
                out["phase"] == int(JumpPhase.FLIGHT_UP),
                out["phase"] == int(JumpPhase.FLIGHT_DOWN),
            )
            takeoff_z = float(out["z"][takeoff_idx])
            max_height = float(np.max(out["z"][flight_mask]) - takeoff_z) if np.any(flight_mask) else 0.0
        else:
            max_height = 0.0
        vertical_success = bool(self.vertical.finished)

        out["max_height"] = np.array([max_height])
        out["peak_force"] = np.array([float(np.max(out["Fz"]))])
        out["peak_pitch_torque"] = np.array([float(np.max(np.abs(out["T_pitch"])))])
        out["pitch_torque_saturation_ratio"] = np.array([float(np.mean(out["pitch_torque_saturated"]))])
        out["h_target"] = np.array([self.p.h_target])
        out["dt"] = np.array([self.p.dt])
        out["vertical_success"] = np.array([vertical_success])
        return out
