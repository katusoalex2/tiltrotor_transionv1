import ast
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.control.mission import MissionProfile
from src.simulation.report import SimulationReport


def test_streamlit_line_and_scatter_charts_declare_both_axis_labels():
    dashboard_path = Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"
    dashboard = ast.parse(dashboard_path.read_text(encoding="utf-8-sig"))
    chart_calls = [
        node
        for node in ast.walk(dashboard)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"line_chart", "scatter_chart"}
    ]

    assert chart_calls
    missing_labels = [
        (
            node.lineno,
            node.func.attr,
            {"x_label", "y_label"} - {keyword.arg for keyword in node.keywords},
        )
        for node in chart_calls
        if {"x_label", "y_label"} - {keyword.arg for keyword in node.keywords}
    ]
    assert not missing_labels, f"Charts missing axis titles: {missing_labels}"


def test_transition_time_plots_only_include_recorded_transition_samples(monkeypatch):
    observed = {}
    original = SimulationReport.line_chart

    def capture(self, target, data, **kwargs):
        alt = kwargs.get("alt", "")
        if alt.startswith("Transition ") and alt.endswith("versus mission time"):
            observed[alt] = data.copy()
        return original(self, target, data, **kwargs)

    monkeypatch.setattr(SimulationReport, "line_chart", capture)
    dashboard = Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=180).run()

    assert not app.exception
    assert len(observed) == 3
    mission = MissionProfile()
    times = [frame["time_s"].tolist() for frame in observed.values()]
    assert times[0]
    assert times[0] == times[1] == times[2]
    assert min(times[0]) >= mission.hover_time
    assert max(times[0]) < mission.total_time

    observed.clear()
    app.selectbox(key="mission_mode").select("HOLD").run()
    assert not app.exception
    assert not observed
    assert any("No transition-phase samples" in item.value for item in app.info)


def test_reference_animation_is_opt_in_and_keeps_simulation_warnings():
    dashboard = Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=180).run()
    assert not app.exception
    warning_before = [item.value for item in app.warning if "Simulation stopped" in item.value]
    assert warning_before
    chart_count = len(app.get("plotly_chart"))

    app.checkbox(key="show_transition_reference").check().run()

    assert not app.exception
    assert len(app.get("plotly_chart")) == chart_count + 1
    assert any("REFERENCE ONLY" in item.value for item in app.warning)
    assert [item.value for item in app.warning if "Simulation stopped" in item.value] == warning_before


def test_tilt_airspeed_curve_uses_recorded_transition_in_time_order(monkeypatch):
    observed = {}
    original = SimulationReport.plotly_chart

    def capture(self, target, figure, **kwargs):
        alt = kwargs.get("alt", "")
        if alt in {
            "Recorded transition nacelle tilt versus true airspeed in time order",
            "Transition corridor with airspeed on the horizontal axis and nacelle tilt on the vertical axis",
        }:
            observed[alt] = figure
        return original(self, target, figure, **kwargs)

    monkeypatch.setattr(SimulationReport, "plotly_chart", capture)
    dashboard = Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=180).run()

    assert not app.exception
    curve = observed["Recorded transition nacelle tilt versus true airspeed in time order"]
    corridor = observed["Transition corridor with airspeed on the horizontal axis and nacelle tilt on the vertical axis"]
    simulated = next(trace for trace in corridor.data if trace.name == "Recorded transition")
    times = np.asarray(curve.data[0].customdata)
    assert times[0] >= MissionProfile().hover_time
    assert times[-1] < MissionProfile().total_time
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(curve.data[0].x, simulated.x)
    np.testing.assert_allclose(curve.data[0].y, simulated.y)
    assert curve.layout.xaxis.title.text == "True airspeed (m/s)"
    assert curve.layout.yaxis.title.text == "Nacelle tilt (deg)"
    assert corridor.layout.xaxis.title.text == "True airspeed (m/s)"
    assert corridor.layout.yaxis.title.text == "Nacelle tilt (deg)"
    assert any("Simulation stopped" in item.value for item in app.warning)

    observed.clear()
    app.selectbox(key="mission_mode").select("HOLD").run()
    assert not app.exception
    assert "Recorded transition nacelle tilt versus true airspeed in time order" not in observed


def test_computed_corridor_displays_unqualified_trim_points(monkeypatch):
    observed = {}
    original = SimulationReport.plotly_chart
    original_pyplot = SimulationReport.pyplot
    corridor_alt = (
        "Transition corridor with airspeed on the horizontal axis and nacelle tilt on the vertical axis"
    )
    histories_alt = (
        "Recorded transition histories for airspeed, pitch, tilt, angle of attack, "
        "motor commands and V-tail commands"
    )

    def capture(self, target, figure, **kwargs):
        if kwargs.get("alt") == corridor_alt:
            observed["figure"] = figure
        elif kwargs.get("alt") == histories_alt:
            observed["histories"] = figure
        elif kwargs.get("alt") == "Recorded transition nacelle tilt versus true airspeed in time order":
            observed["transition"] = figure
        return original(self, target, figure, **kwargs)

    def capture_alpha(self, target, figure, **kwargs):
        if kwargs.get("alt") == "Angle of attack through the flight with stall-angle references":
            observed["raw_alpha"] = np.asarray(figure.axes[0].lines[0].get_ydata()).copy()
        return original_pyplot(self, target, figure, **kwargs)

    monkeypatch.setattr(SimulationReport, "plotly_chart", capture)
    monkeypatch.setattr(SimulationReport, "pyplot", capture_alpha)
    dashboard = Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=240).run()
    assert not app.exception
    assert all(trace.name != "Trim only; not qualified" for trace in observed["figure"].data)
    assert any("Select" in item.value and "Compute constrained corridor" in item.value for item in app.info)

    app.button(key="compute_constrained_transition_corridor").click().run()
    assert not app.exception
    result = app.session_state["constrained_transition_corridor_result"]["result"]
    trim_only = (
        result["analytic_mask"] & result["trim_mask"]
        & result["rotor_thrust_mask"] & ~result["known_constraints_mask"]
    )
    trace = next(trace for trace in observed["figure"].data if trace.name == "Trim only; not qualified")
    speed_indices, tilt_indices = np.where(trim_only)
    assert speed_indices.size > 0
    np.testing.assert_array_equal(trace.x, result["airspeeds_mps"][speed_indices])
    np.testing.assert_array_equal(trace.y, result["tilt_angles_deg"][tilt_indices])
    np.testing.assert_array_equal(
        trace.customdata[:, 0], result["stability_map"]["trim_front_motor_rpm"][trim_only]
    )
    np.testing.assert_array_equal(
        trace.customdata[:, 1], result["stability_map"]["trim_rear_motor_rpm"][trim_only]
    )
    summary = next(
        item.value for item in app.dataframe
        if "Constraint" in item.value.columns
    )
    stability_row = summary.loc[
        summary["Constraint"] == "Full static derivative signs (including Cl_beta)"
    ].iloc[0]
    assert stability_row["Assessed cells"] == np.count_nonzero(
        result["analytic_mask"] & result["static_stability_assessed_mask"]
    )
    assessed_trim_points = result["stability_map"]["assessment_mask"]
    assert np.all(
        result["static_stability_status"][assessed_trim_points] == "unassessed"
    )
    assert not result["qualified_mask"].any()
    assert any("Cl_beta) is unassessed" in item.value for item in app.warning)
    assert any("Simulation stopped" in item.value for item in app.warning)

    histories = observed["histories"]
    aoa_assessed = next(trace for trace in histories.data if trace.name == "AoA: target assessed")
    aoa_unassessed = next(trace for trace in histories.data if trace.name == "AoA: target unassessed")
    recorded_times = np.asarray(observed["transition"].data[0].customdata)
    indices = np.rint(recorded_times / 0.01).astype(int)
    actual_alpha = np.where(np.isnan(aoa_assessed.y), aoa_unassessed.y, aoa_assessed.y)
    np.testing.assert_allclose(actual_alpha, observed["raw_alpha"][indices])
    np.testing.assert_allclose(aoa_assessed.x, recorded_times - recorded_times[0])
    assert histories.layout.yaxis4.range is None
    assert any(shape.y0 == 5.0 and shape.y1 == 12.0 for shape in histories.layout.shapes)

    observed.pop("histories")
    app.slider(key="gust_magnitude_mps").set_value(5.5).run()
    assert not app.exception
    assert all(trace.name != "Trim only; not qualified" for trace in observed["figure"].data)
    assert "histories" not in observed
    assert any("Recompute the constrained corridor" in item.value for item in app.info)
