from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jump_params import JumpParams
from jump_phase_observer import PHASE_NAMES, JumpPhase, JumpPhaseObserver
from jump_planner_vertical import VerticalJumpPlanner
from jump_vmc_controller import JumpVMCController


@dataclass
class JumpState:
    z_b: float
    z_b_dot: float
    z_w: float
    z_w_dot: float
    L: float
    L_dot: float
    phase: JumpPhase


class Jump1DModel:
    """Pure Python 1D vertical jump hybrid model.

    Important modeling boundary from the revised PDF:
    the old support-phase leg-height model is valid only when the wheel is in
    contact with the ground. It is not used during FLIGHT_UP/FLIGHT_DOWN. In
    flight, the COM follows ballistic z_com_ddot = -g and the leg length follows
    an internal cubic trajectory.
    """

    def __init__(self, params: JumpParams):
        self.p = params
        self.planner = VerticalJumpPlanner()
        self.controller = JumpVMCController()
        self.observer = JumpPhaseObserver()
        self.state = self.reset()
        self.t = 0.0
        self.t_phase = 0.0
        self.z_com = 0.0
        self.z_com_dot = 0.0
        self.takeoff_time = np.nan
        self.landing_time = np.nan
        self.recover_time = np.nan
        self.finished = False
        self.flight_up_start_L = params.L_takeoff
        self.flight_down_start_L = params.L_air_min

    def reset(self) -> JumpState:
        p = self.p
        self.t = 0.0
        self.t_phase = 0.0
        self.takeoff_time = np.nan
        self.landing_time = np.nan
        self.recover_time = np.nan
        self.finished = False
        self.flight_up_start_L = p.L_takeoff
        self.flight_down_start_L = p.L_air_min
        state = JumpState(
            z_b=p.wheel_radius + p.L_stand,
            z_b_dot=0.0,
            z_w=p.wheel_radius,
            z_w_dot=0.0,
            L=p.L_stand,
            L_dot=0.0,
            phase=JumpPhase.SQUAT,
        )
        self.state = state
        self._update_com_from_state()
        return state

    def _update_com_from_state(self) -> None:
        p = self.p
        s = self.state
        self.z_com = (p.m_body * s.z_b + p.m_wheel * s.z_w) / p.M_total
        self.z_com_dot = (p.m_body * s.z_b_dot + p.m_wheel * s.z_w_dot) / p.M_total

    def _set_phase(self, phase: JumpPhase) -> None:
        if phase == self.state.phase:
            return
        self.state.phase = phase
        self.t_phase = 0.0
        if phase == JumpPhase.FLIGHT_UP:
            # Use the body takeoff speed as the ballistic COM speed for the
            # first 1D smoke. This intentionally favors a tiny visible jump
            # before a later two-mass calibration.
            self.z_com = self.state.z_w + (self.p.m_body / self.p.M_total) * self.state.L
            self.z_com_dot = self.state.z_b_dot
            self.flight_up_start_L = self.state.L
            self.takeoff_time = self.t
        elif phase == JumpPhase.FLIGHT_DOWN:
            self.flight_down_start_L = self.state.L
        elif phase == JumpPhase.LANDING:
            self.landing_time = self.t
            self.state.z_w = self.p.wheel_radius
            self.state.z_w_dot = 0.0
            self.state.L = float(np.clip(self.state.z_b - self.state.z_w, self.p.L_min, self.p.L_takeoff))
            self.state.L_dot = self.state.z_b_dot
        elif phase == JumpPhase.RECOVER:
            self.recover_time = self.t

    def _integrate_support(self, force: float) -> None:
        p = self.p
        s = self.state
        z_b_ddot = force / p.m_body - p.g
        s.z_b_dot += z_b_ddot * p.dt
        s.z_b += s.z_b_dot * p.dt
        s.z_w = p.wheel_radius
        s.z_w_dot = 0.0
        s.L = float(np.clip(s.z_b - s.z_w, p.L_min, p.L_takeoff + 0.03))
        s.L_dot = s.z_b_dot
        if s.z_b < p.wheel_radius + p.L_min:
            s.z_b = p.wheel_radius + p.L_min
            s.z_b_dot = max(0.0, s.z_b_dot)
            s.L = p.L_min
            s.L_dot = s.z_b_dot
        self._update_com_from_state()

    def _integrate_flight(self, ref) -> None:
        p = self.p
        s = self.state
        old_L = s.L
        old_z_b = s.z_b
        old_z_w = s.z_w

        self.z_com_dot += -p.g * p.dt
        self.z_com += self.z_com_dot * p.dt
        L = float(np.clip(ref.L, p.L_min, p.L_takeoff))
        L_dot = (L - old_L) / p.dt

        s.L = L
        s.L_dot = L_dot
        s.z_b = self.z_com + (p.m_wheel / p.M_total) * L
        s.z_w = self.z_com - (p.m_body / p.M_total) * L
        s.z_b_dot = (s.z_b - old_z_b) / p.dt
        s.z_w_dot = (s.z_w - old_z_w) / p.dt

    def step(self) -> dict[str, float | int | str | bool]:
        p = self.p
        s = self.state
        if s.phase == JumpPhase.FLIGHT_UP:
            ref = self.planner.flight_ref_from_lengths(
                s.phase, self.t_phase, self.flight_up_start_L, p.L_air_min, p
            )
        elif s.phase == JumpPhase.FLIGHT_DOWN:
            ref = self.planner.flight_ref_from_lengths(
                s.phase, self.t_phase, self.flight_down_start_L, p.L_landing, p
            )
        else:
            ref = self.planner.get_ref(s.phase, self.t_phase, s, p)
        force_debug = self.controller.compute_leg_force(s, ref, s.phase, self.t_phase, p)

        if s.phase in {JumpPhase.SQUAT, JumpPhase.THRUST, JumpPhase.LANDING, JumpPhase.RECOVER}:
            self._integrate_support(force_debug.clipped)
        else:
            self._integrate_flight(ref)

        if s.phase == JumpPhase.SQUAT and self.observer.squat_done(self.t_phase, s.L, s.L_dot, p):
            self._set_phase(JumpPhase.THRUST)
        elif s.phase == JumpPhase.THRUST and self.observer.takeoff_detected(self.t_phase, s.z_b_dot, s.L, p):
            self._set_phase(JumpPhase.FLIGHT_UP)
        elif s.phase == JumpPhase.FLIGHT_UP and self.observer.apex_detected(self.z_com_dot):
            self._set_phase(JumpPhase.FLIGHT_DOWN)
        elif s.phase == JumpPhase.FLIGHT_DOWN and self.observer.landing_detected(s.z_w, s.z_w_dot, p):
            self._set_phase(JumpPhase.LANDING)
        elif s.phase == JumpPhase.LANDING and self.observer.landing_done(self.t_phase, s.z_b_dot, s.L, s.L_dot, p):
            self._set_phase(JumpPhase.RECOVER)
        elif s.phase == JumpPhase.RECOVER and self.observer.recover_done(self.t_phase, s.z_b_dot, s.L, s.L_dot, p):
            self.finished = True

        sample = {
            "t": self.t,
            "z_b": s.z_b,
            "z_b_dot": s.z_b_dot,
            "z_w": s.z_w,
            "z_w_dot": s.z_w_dot,
            "L": s.L,
            "L_dot": s.L_dot,
            "phase": int(s.phase),
            "phase_name": PHASE_NAMES[s.phase],
            "L_ref": ref.L,
            "L_ref_dot": ref.L_dot,
            "L_ref_ddot": ref.L_ddot,
            "F_leg": force_debug.clipped,
            "F_raw": force_debug.raw,
            "force_saturated": bool(force_debug.saturated),
            "Kp": force_debug.Kp,
            "Kd": force_debug.Kd,
            "z_com": self.z_com,
            "z_com_dot": self.z_com_dot,
        }

        self.t += p.dt
        self.t_phase += p.dt
        return sample

    def run(self) -> dict[str, np.ndarray | float | bool]:
        samples: list[dict] = []
        n_steps = int(np.ceil(self.p.t_final / self.p.dt))
        for _ in range(n_steps):
            samples.append(self.step())
            if self.finished:
                break

        out: dict[str, np.ndarray | float | bool] = {}
        numeric_keys = [
            "t",
            "z_b",
            "z_b_dot",
            "z_w",
            "z_w_dot",
            "L",
            "L_dot",
            "phase",
            "L_ref",
            "L_ref_dot",
            "L_ref_ddot",
            "F_leg",
            "F_raw",
            "force_saturated",
            "Kp",
            "Kd",
            "z_com",
            "z_com_dot",
        ]
        for key in numeric_keys:
            out[key] = np.asarray([s[key] for s in samples], dtype=float)

        phase_names = np.asarray([s["phase_name"] for s in samples])
        out["phase_names_per_sample"] = phase_names
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
            takeoff_com = float(out["z_com"][takeoff_idx])
            max_height = float(np.max(out["z_com"][flight_mask]) - takeoff_com) if np.any(flight_mask) else 0.0
        else:
            max_height = 0.0
        out["max_height"] = np.array([max_height])
        out["peak_force"] = np.array([float(np.max(out["F_leg"]))])
        phases_seen = set(int(x) for x in out["phase"])
        success = (
            self.finished
            and JumpPhase.FLIGHT_UP in phases_seen
            and JumpPhase.FLIGHT_DOWN in phases_seen
            and JumpPhase.LANDING in phases_seen
            and float(out["max_height"][0]) >= 0.5 * self.p.h_target
            and float(np.max(out["L"])) <= self.p.L_takeoff + 0.035
            and float(np.min(out["L"])) >= self.p.L_min - 1e-6
        )
        out["success"] = np.array([bool(success)])
        out["h_target"] = np.array([self.p.h_target])
        out["v_takeoff_target"] = np.array([self.p.v_takeoff_target])
        out["dt"] = np.array([self.p.dt])
        return out
