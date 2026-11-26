import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest


APP = """
from types import SimpleNamespace
import numpy as np
from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from scripts.dashboard_transition_analyses import render_transition_analyses

positions = ((.3, -.4, 0., True), (.3, .4, 0., True),
             (-.3, -.4, 0., False), (-.3, .4, 0., False))
props = SimpleNamespace(motor_positions=positions, tilt_pivot_xz=(.1, 0.))
tail = VtailModel(.1, .5)
dynamics = TiltrotorDynamics(5., RotorModel(1e-5, 3000.),
                             mass_props=props, aero=AeroModel(.6, tail))
render_transition_analyses(dynamics, 10.0)
"""


def test_tilt_schedule_controls_update_curves_and_rate_warning(monkeypatch):
    observed = {}
    original = st.line_chart

    def capture(data, **kwargs):
        observed[kwargs["alt"]] = data.copy()
        return original(data, **kwargs)

    monkeypatch.setattr(st, "line_chart", capture)
    app = AppTest.from_string(APP, default_timeout=60).run()
    app.selectbox(key="additional_transition_analysis").select("Tilt schedules").run()
    frame = observed["Cosine and linear tilt references versus time"]
    assert frame["Cosine"].iloc[0] == 90.0
    assert np.isclose(frame["Cosine"].iloc[-1], 0.0)
    assert any("exceeds" in item.value for item in app.warning)

    app.slider(key="extra_tilt_duration").set_value(30.0).run()
    assert not app.exception
    assert not any("exceeds" in item.value for item in app.warning)
    assert observed["Cosine and linear tilt references versus time"]["Time (s)"].iloc[-1] == 34.0


def test_force_and_envelope_follow_controls_and_angle_convention(monkeypatch):
    observed = []
    original = st.plotly_chart

    def capture(figure, **kwargs):
        observed.append((kwargs["alt"], figure))
        return original(figure, **kwargs)

    monkeypatch.setattr(st, "plotly_chart", capture)
    app = AppTest.from_string(APP, default_timeout=60).run()
    app.selectbox(key="additional_transition_analysis").select("Force sweeps").run()
    assert not app.exception
    force = next(fig for alt, fig in observed if alt.startswith("Fixed-rotor-speed"))
    assert np.isclose(force.data[1].x[-1], 90.0)
    assert np.isclose(force.data[1].y[-1], 5.0 * 9.81)
    assert len(force.data) == 20
    app.slider(key="extra_force_speed").set_value(30.0).run()
    assert any(trace.name == "30 m/s" for trace in observed[-4][1].data)

    observed.clear()
    app.selectbox(key="additional_transition_analysis").select("Analytical tilt-speed envelope").run()
    first = observed[-1][1]
    assert np.isclose(first.data[0].y[0], 90.0)
    assert np.isclose(first.data[0].y[-1], 0.0)
    np.testing.assert_allclose(first.data[-1].y, [90.0, 0.0])
    assert not app.exception


def test_pid_diagnostic_hides_stale_results():
    app = AppTest.from_string(APP, default_timeout=60).run()
    app.selectbox(key="additional_transition_analysis").select("Exploratory PID diagnostic").run()
    app.slider(key="extra_pid_duration").set_value(1.0).run()
    app.button(key="extra_run_pid").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["extra_pid_result"]["log"]["time"][-1] == 1.0
    assert any("Maximum absolute altitude error" in item.label for item in app.metric)

    app.slider(key="extra_pid_sigma").set_value(2.0).run()
    assert not app.exception
    assert any("Inputs changed" in item.value for item in app.info)
    assert not any("Maximum absolute altitude error" in item.label for item in app.metric)


def test_pid_diagnostic_detects_changed_profile(monkeypatch, tmp_path):
    from scripts import plot_master_pid_diagnostics as diagnostics

    profile = tmp_path / "profile.json"
    profile.write_bytes(diagnostics.PROFILE_PATH.read_bytes())
    monkeypatch.setattr(diagnostics, "PROFILE_PATH", profile)
    app = AppTest.from_string(APP, default_timeout=60).run()
    app.selectbox(key="additional_transition_analysis").select("Exploratory PID diagnostic").run()
    app.slider(key="extra_pid_duration").set_value(1.0).run()
    app.button(key="extra_run_pid").click().run(timeout=60)
    assert any("Maximum absolute altitude error" in item.label for item in app.metric)

    profile.write_bytes(profile.read_bytes() + b"\n")
    app.run()
    assert not app.exception
    assert any("Inputs changed" in item.value for item in app.info)
    assert not any("Maximum absolute altitude error" in item.label for item in app.metric)


def test_optimizer_surfaces_pitch_infeasibility_and_discards_stale_results(monkeypatch):
    from scripts import optimize_longitudinal_transition as optimizer

    app = AppTest.from_string(APP, default_timeout=120).run()
    app.selectbox(key="additional_transition_analysis").select("Reduced-order optimization").run()
    app.button(key="extra_run_optimizer").click().run(timeout=120)
    assert not app.exception
    assert any("pitch-moment balance" in item.value for item in app.error)
    assert "extra_optimizer_result" not in app.session_state

    def fail():
        raise RuntimeError("solver did not converge")

    monkeypatch.setattr(optimizer, "solve_transition", fail)
    app.button(key="extra_run_optimizer").click().run(timeout=120)
    assert not app.exception
    assert any("solver did not converge" in item.value for item in app.error)
    assert "extra_optimizer_result" not in app.session_state
