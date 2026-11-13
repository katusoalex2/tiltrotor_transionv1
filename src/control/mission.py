"""Timed transition references and state-driven mission sequencing."""
from dataclasses import dataclass

import numpy as np


class MissionProfile:
    def __init__(self):
        # We tilt from 0 to 90 degrees over 20 seconds
        self.hover_time = 1.0
        self.transition_duration = 20.0
        self.total_time = self.hover_time + self.transition_duration
    
    def at(self, t: float) -> dict:
        if t < self.hover_time:
            # Pure Hover
            return {"altitude": 100.0, "speed": 0.0, "tilt": 0.0, "phase": "HOVER"}
            
        elif t >= self.total_time:
            # Pure Cruise
            return {"altitude": 100.0, "speed": 12.0, "tilt": np.pi/2, "phase": "CRUISE"}
            
        else:
            # Transition Phase
            # Calculate progress from 0.0 to 1.0
            progress = (t - self.hover_time) / self.transition_duration
            
            # S-Curve (Cosine Blending)
            # smooth_progress goes from 0 -> 1 with zero slope at both ends
            smooth_progress = (1.0 - np.cos(np.pi * progress)) / 2.0
            
            # Calculate speed and tilt based on smooth progress
            speed = smooth_progress * 12.0
            tilt_rad = smooth_progress * (np.pi / 2)
            
            return {
                "altitude": 100.0, 
                "speed": speed, 
                "tilt": tilt_rad, 
                "phase": "TRANSITION"
            }


@dataclass(frozen=True)
class Waypoint:
    """A target in the local NED frame, with altitude expressed positive-up."""

    north_m: float
    east_m: float
    altitude_m: float
    speed_mps: float = 0.0
    tilt_rad: float = 0.0
    name: str = ""
    acceptance_radius_m: float = 5.0
    altitude_tolerance_m: float = 2.0

    def __post_init__(self):
        if self.acceptance_radius_m <= 0.0:
            raise ValueError("acceptance_radius_m must be positive")
        if self.altitude_tolerance_m < 0.0:
            raise ValueError("altitude_tolerance_m must be non-negative")
        if self.speed_mps < 0.0:
            raise ValueError("speed_mps must be non-negative")

    def as_reference(self) -> dict:
        return {
            "north": float(self.north_m),
            "east": float(self.east_m),
            "altitude": float(self.altitude_m),
            "speed": float(self.speed_mps),
            "tilt": float(self.tilt_rad),
            "waypoint": self.name,
        }


class WaypointSequencer:
    """Advance through waypoints when the aircraft enters each acceptance volume."""

    def __init__(self, waypoints):
        self.waypoints = tuple(waypoints)
        if not all(isinstance(waypoint, Waypoint) for waypoint in self.waypoints):
            raise TypeError("waypoints must contain only Waypoint instances")
        self.index = 0

    @property
    def current(self):
        if self.complete:
            return None
        return self.waypoints[self.index]

    @property
    def complete(self) -> bool:
        return self.index >= len(self.waypoints)

    def update(self, state):
        """Return the active target; advance at most one waypoint per update."""
        waypoint = self.current
        if waypoint is None:
            return None

        horizontal_error = np.hypot(
            float(state.pos[0]) - waypoint.north_m,
            float(state.pos[1]) - waypoint.east_m,
        )
        altitude_error = abs(-float(state.pos[2]) - waypoint.altitude_m)
        if (
            horizontal_error <= waypoint.acceptance_radius_m
            and altitude_error <= waypoint.altitude_tolerance_m
        ):
            self.index += 1
        return self.current


class MissionManager:
    """Manage mission progression and AUTO, HOLD, and MANUAL mode changes."""

    MODES = frozenset({"AUTO", "HOLD", "MANUAL"})

    def __init__(self, waypoints):
        self.sequencer = WaypointSequencer(waypoints)
        self.mode = "AUTO"
        self._hold_reference = None

    def set_mode(self, mode: str, state=None):
        """Switch mode; HOLD captures the current position and altitude."""
        normalized_mode = str(mode).upper()
        if normalized_mode not in self.MODES:
            raise ValueError(f"Unsupported mission mode: {mode}")
        if normalized_mode == "HOLD" and state is None:
            raise ValueError("state is required to enter HOLD mode")

        self.mode = normalized_mode
        if normalized_mode == "HOLD":
            self._hold_reference = {
                "north": float(state.pos[0]),
                "east": float(state.pos[1]),
                "altitude": float(-state.pos[2]),
                "speed": 0.0,
                "tilt": float(state.tilt),
                "waypoint": "HOLD",
            }
        return self.mode

    def update(self, state) -> dict:
        """Return the active mission target and mode status for this state."""
        if self.mode == "MANUAL":
            return {
                "mode": self.mode,
                "active": False,
                "complete": self.sequencer.complete,
                "reference": None,
            }

        if self.mode == "HOLD":
            reference = dict(self._hold_reference)
        else:
            waypoint = self.sequencer.update(state)
            reference = waypoint.as_reference() if waypoint is not None else None

        return {
            "mode": self.mode,
            "active": reference is not None,
            "complete": self.sequencer.complete,
            "reference": reference,
        }
