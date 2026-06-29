from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jump_params import JumpParams
from jump_phase_observer import PHASE_NAMES, JumpPhase, JumpPhaseObserver
from jump_pitch_controller import JumpPitchController
from jump_planner_vertical import VerticalJumpPlanner
from jump_vmc_controller import JumpVMCController
from jump_x_controller import JumpXController


@dataclass
class Jump2DXZPitchInitialState:
    x: float
    xdot: float
    theta: float
    theta_dot: float
    x_force_offset: float


@dataclass
class _VerticalStateProxy:
    L: float
    L_dot: float


class Jump2DXZPitchModel:
    """Pure Python x/z/pitch coupled jump smoke.

    This v0 model keeps the vertical 1D VMC/phase-observer structure but adds
    horizontal contact force and a pitch torque balance:

        m * xddot = Fx
        m * zddot = Fz - m*g
        Iyy * theta_ddot = x_force_offset*Fz - z_com_height*Fx
                           + T_pitch - damping_theta*theta_dot

    Fx is only available in contact phases. T_pitch is still virtual and is not
    mapped to wheel/hip actuators in this smoke layer.
    """

    def __init__(self, params: JumpParams, initial: Jump2DXZPitchInitialState):
        self.p = params
        self.initial = initial
        self.planner = VerticalJumpPlanner()
        self.vmc = JumpVMCController()
        self.observer = JumpPhaseObserver()
        self.pitch_controller = JumpPitchController()
        self.x_controller = JumpXController()
        self.reset()

    def reset(self) -> None:
        p = self.p
        self.t = 0.0
        self.t_phase = 0.0
        self.phase = JumpPhase.SQUAT
        self.x = float(self.initial.x)
        self.xdot = float(self.initial.xdot)
        self.z = p.wheel_radius + p.L_stand
        self.zdot = 0.0
        self.z_wheel = p.wheel_radius
        self.z_wheel_dot = 0.0
        self.leg_length = p.L_stand
        self.leg_length_dot = 0.0
        self.theta = float(self.initial.theta)
        self.theta_dot = float(self.initial.theta_dot)
        self.x_force_offset = float(self.initial.x_force_offset)
        self.flight_up_start_L = p.L_takeoff
        self.flight_down_start_L = p.L_air_min
        self.takeoff_time = np.nan
        self.landing_time = np.nan
        self.recover_time = np.nan
        self.finished = False

    def _contact_phase(self) -> bool:
        return self.phase in {JumpPhase.SQUAT, JumpPhase.THRUST, JumpPhase.LANDING, JumpPhase.RECOVER}

    def _set_phase(self, phase: JumpPhase) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self.t_phase = 0.0
        if phase == JumpPhase.FLIGHT_UP:
            self.takeoff_time = self.t
            self.flight_up_start_L = self.leg_length
        elif phase == JumpPhase.FLIGHT_DOWN:
            self.flight_down_start_L = self.leg_length
        elif phase == JumpPhase.LANDING:
            self.landing_time = self.t
            self.z_wheel = self.p.wheel_radius
            self.z_wheel_dot = 0.0
            self.leg_length = float(np.clip(self.z - self.p.wheel_radius, self.p.L_min, self.p.L_takeoff))
            self.leg_length_dot = self.zdot
        elif phase == JumpPhase.RECOVER:
            self.recover_time = self.t

    def _leg_ref(self):
        state = _VerticalStateProxy(self.leg_length, self.leg_length_dot)
        if self.phase == JumpPhase.FLIGHT_UP:
            return self.planner.flight_ref_from_lengths(
                self.phase, self.t_phase, self.flight_up_start_L, self.p.L_air_min, self.p
            )
        if self.phase == JumpPhase.FLIGHT_DOWN:
            return self.planner.flight_ref_from_lengths(
                self.phase, self.t_phase, self.flight_down_start_L, self.p.L_landing, self.p
            )
        return self.planner.get_ref(self.phase, self.t_phase, state, self.p)

    def step(self) -> dict[str, float | int | str | bool]:
        p = self.p
        contact = self._contact_phase()
        ref = self._leg_ref()
        state_proxy = _VerticalStateProxy(self.leg_length, self.leg_length_dot)
        fz_debug = self.vmc.compute_leg_force(state_proxy, ref, self.phase, self.t_phase, p)
        fx_debug = self.x_controller.compute(self.x, self.xdot, contact, p)
        pitch_debug = self.pitch_controller.compute(
            self.theta, self.theta_dot, p.theta_ref, p.theta_dot_ref, p
        )

        Fz = fz_debug.clipped if contact else 0.0
        Fx = fx_debug.clipped if contact else 0.0
        z_com_height = max(self.z - p.wheel_radius, 0.0)
        tau_coupling = self.x_force_offset * Fz - z_com_height * Fx
        theta_ddot = (
            tau_coupling
            + pitch_debug.clipped
            + p.disturbance_torque
            - p.damping_theta * self.theta_dot
        ) / p.Iyy

        self.theta_dot += theta_ddot * p.dt
        self.theta += self.theta_dot * p.dt

        xddot = Fx / p.m_body
        self.xdot += xddot * p.dt
        self.x += self.xdot * p.dt

        if contact:
            zddot = Fz / p.m_body - p.g
            self.zdot += zddot * p.dt
            self.z += self.zdot * p.dt
            self.z_wheel = p.wheel_radius
            self.z_wheel_dot = 0.0
            self.leg_length = float(np.clip(self.z - self.z_wheel, p.L_min, p.L_takeoff + 0.03))
            self.leg_length_dot = self.zdot
            if self.z < p.wheel_radius + p.L_min:
                self.z = p.wheel_radius + p.L_min
                self.zdot = max(0.0, self.zdot)
                self.leg_length = p.L_min
                self.leg_length_dot = self.zdot
        else:
            old_L = self.leg_length
            old_z_wheel = self.z_wheel
            self.zdot += -p.g * p.dt
            self.z += self.zdot * p.dt
            self.leg_length = float(np.clip(ref.L, p.L_min, p.L_takeoff))
            self.leg_length_dot = (self.leg_length - old_L) / p.dt
            self.z_wheel = self.z - self.leg_length
            self.z_wheel_dot = (self.z_wheel - old_z_wheel) / p.dt

        if self.phase == JumpPhase.SQUAT and self.observer.squat_done(
            self.t_phase, self.leg_length, self.leg_length_dot, p
        ):
            self._set_phase(JumpPhase.THRUST)
        elif self.phase == JumpPhase.THRUST and self.observer.takeoff_detected(
            self.t_phase, self.zdot, self.leg_length, p
        ):
            self._set_phase(JumpPhase.FLIGHT_UP)
        elif self.phase == JumpPhase.FLIGHT_UP and self.observer.apex_detected(self.zdot):
            self._set_phase(JumpPhase.FLIGHT_DOWN)
        elif self.phase == JumpPhase.FLIGHT_DOWN and self.observer.landing_detected(
            self.z_wheel, self.z_wheel_dot, p
        ):
            self._set_phase(JumpPhase.LANDING)
        elif self.phase == JumpPhase.LANDING and self.observer.landing_done(
            self.t_phase, self.zdot, self.leg_length, self.leg_length_dot, p
        ):
            self._set_phase(JumpPhase.RECOVER)
        elif self.phase == JumpPhase.RECOVER and self.observer.recover_done(
            self.t_phase, self.zdot, self.leg_length, self.leg_length_dot, p
        ):
            self.finished = True

        sample = {
            "t": self.t,
            "x": self.x,
            "xdot": self.xdot,
            "z": self.z,
            "zdot": self.zdot,
            "z_wheel": self.z_wheel,
            "z_wheel_dot": self.z_wheel_dot,
            "theta": self.theta,
            "theta_dot": self.theta_dot,
            "theta_ddot": theta_ddot,
            "leg_length": self.leg_length,
            "leg_length_dot": self.leg_length_dot,
            "leg_length_ref": ref.L,
            "phase": int(self.phase),
            "phase_name": PHASE_NAMES[self.phase],
            "contact": bool(contact),
            "Fz": Fz,
            "Fz_raw": fz_debug.raw,
            "force_z_saturated": bool(fz_debug.saturated),
            "Fx": Fx,
            "Fx_raw": fx_debug.raw,
            "force_x_saturated": bool(fx_debug.saturated),
            "T_pitch": pitch_debug.clipped,
            "T_pitch_raw": pitch_debug.raw,
            "pitch_torque_saturated": bool(pitch_debug.saturated),
            "tau_coupling": tau_coupling,
            "z_com_height": z_com_height,
            "x_force_offset": self.x_force_offset,
        }
        self.t += p.dt
        self.t_phase += p.dt
        return sample

    def run(self) -> dict[str, np.ndarray]:
        samples: list[dict] = []
        n_steps = int(np.ceil(self.p.t_final / self.p.dt))
        for _ in range(n_steps):
            samples.append(self.step())
            if self.finished:
                break

        numeric_keys = [
            "t",
            "x",
            "xdot",
            "z",
            "zdot",
            "z_wheel",
            "z_wheel_dot",
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
            "force_z_saturated",
            "Fx",
            "Fx_raw",
            "force_x_saturated",
            "T_pitch",
            "T_pitch_raw",
            "pitch_torque_saturated",
            "tau_coupling",
            "z_com_height",
            "x_force_offset",
        ]
        out: dict[str, np.ndarray] = {}
        for key in numeric_keys:
            out[key] = np.asarray([s[key] for s in samples], dtype=float)

        out["phase_names_per_sample"] = np.asarray([s["phase_name"] for s in samples])
        out["phase_name_table"] = np.asarray([PHASE_NAMES[JumpPhase(i)] for i in range(len(PHASE_NAMES))])
        out["takeoff_time"] = np.array([self.takeoff_time])
        out["landing_time"] = np.array([self.landing_time])
        out["recover_time"] = np.array([self.recover_time])
        if np.isfinite(self.takeoff_time):
            takeoff_idx = int(np.searchsorted(out["t"], self.takeoff_time))
            takeoff_idx = int(np.clip(takeoff_idx, 0, len(out["t"]) - 1))
            flight_mask = np.logical_or(
                out["phase"] == int(JumpPhase.FLIGHT_UP),
                out["phase"] == int(JumpPhase.FLIGHT_DOWN),
            )
            takeoff_z = float(out["z"][takeoff_idx])
            max_height = float(np.max(out["z"][flight_mask]) - takeoff_z) if np.any(flight_mask) else 0.0
        else:
            max_height = 0.0
        out["max_height"] = np.array([max_height])
        out["peak_force_z"] = np.array([float(np.max(np.abs(out["Fz"])))])
        out["peak_force_x"] = np.array([float(np.max(np.abs(out["Fx"])))])
        out["peak_pitch_torque"] = np.array([float(np.max(np.abs(out["T_pitch"])))])
        out["peak_tau_coupling"] = np.array([float(np.max(np.abs(out["tau_coupling"])))])
        out["pitch_torque_saturation_ratio"] = np.array([float(np.mean(out["pitch_torque_saturated"]))])
        out["wheel_force_saturation_ratio"] = np.array([float(np.mean(out["force_x_saturated"]))])
        out["h_target"] = np.array([self.p.h_target])
        out["dt"] = np.array([self.p.dt])
        out["vertical_success"] = np.array([bool(self.finished)])
        return out
