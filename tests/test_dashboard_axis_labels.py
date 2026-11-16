import ast
from pathlib import Path


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
