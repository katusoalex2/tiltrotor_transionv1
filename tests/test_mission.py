import numpy as np
import pytest

from src.control.mission import MissionManager, Waypoint, WaypointSequencer
from src.dynamics.state import AircraftState


def test_waypoint_sequencer_advances_only_when_position_and_altitude_are_reached():
    sequencer = WaypointSequencer([
        Waypoint(10.0, 0.0, 100.0, name="outbound"),
        Waypoint(20.0, 0.0, 100.0, name="return"),
    ])
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))

    assert sequencer.update(state).name == "outbound"
    state.pos = np.array([10.0, 0.0, -100.0])
    assert sequencer.update(state).name == "return"
    state.pos = np.array([20.0, 0.0, -100.0])
    assert sequencer.update(state) is None
    assert sequencer.complete


def test_mission_manager_holds_position_until_auto_resumes():
    manager = MissionManager([Waypoint(50.0, 0.0, 100.0, name="target")])
    state = AircraftState(pos=np.array([5.0, 2.0, -90.0]), tilt=0.4)

    manager.set_mode("hold", state)
    held = manager.update(state)["reference"]
    state.pos[:] = [25.0, 10.0, -80.0]
    assert manager.update(state)["reference"] == held
    assert held == {
        "north": 5.0,
        "east": 2.0,
        "altitude": 90.0,
        "speed": 0.0,
        "tilt": 0.4,
        "waypoint": "HOLD",
    }

    manager.set_mode("AUTO")
    assert manager.update(state)["reference"]["waypoint"] == "target"


def test_mission_manager_manual_mode_disables_automatic_reference():
    manager = MissionManager([Waypoint(10.0, 0.0, 100.0)])
    manager.set_mode("MANUAL")

    status = manager.update(AircraftState())
    assert status["active"] is False
    assert status["reference"] is None
    assert status["mode"] == "MANUAL"


def test_mission_manager_requires_state_for_hold_and_rejects_unknown_mode():
    manager = MissionManager([])
    with pytest.raises(ValueError, match="state is required"):
        manager.set_mode("HOLD")
    with pytest.raises(ValueError, match="Unsupported mission mode"):
        manager.set_mode("RTL")
