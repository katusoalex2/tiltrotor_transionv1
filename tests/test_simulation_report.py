from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest
from docx import Document

from src.simulation.report import ReportItem, SimulationReport


class ChartTarget:
    def line_chart(self, data, **kwargs):
        return "displayed"


def test_report_includes_graph_data_tables_warnings_and_profile():
    report = SimulationReport(True)
    frame = pd.DataFrame({"time": [1.0, 2.0], "altitude": [4.0, 5.0]})
    result = report.line_chart(
        ChartTarget(),
        frame,
        x="time",
        y=["altitude"],
        x_label="Time (s)",
        y_label="Altitude (m)",
        alt="Transition altitude",
    )
    report.items.append(ReportItem("text", "", "WARNING: Simulation ended early."))
    report.items.append(ReportItem("table", "Trim results", pd.DataFrame({"Feasible": [False]})))
    word, pdf = report.render("Example aircraft", '{"mass_kg": 5}')

    assert result == "displayed"
    assert pdf.startswith(b"%PDF-")
    document = Document(BytesIO(word))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Transition altitude" in text
    assert "WARNING: Simulation ended early." in text
    assert '{"mass_kg": 5}' in text
    assert document.tables[0].cell(1, 1).text == "False"
    with ZipFile(BytesIO(word)) as archive:
        assert any(name.startswith("word/media/") for name in archive.namelist())


def test_report_fails_explicitly_when_a_graph_cannot_be_rendered():
    class BrokenFigure:
        def to_image(self, **kwargs):
            raise RuntimeError("Renderer unavailable")

    report = SimulationReport(True)
    report.items.append(ReportItem("image", "Critical graph", BrokenFigure()))

    with pytest.raises(RuntimeError, match="Critical graph.*Renderer unavailable"):
        report.render("Example aircraft", "{}")


def test_disabled_report_does_not_capture_graphs():
    report = SimulationReport(False)
    result = report.line_chart(ChartTarget(), pd.DataFrame({"x": [1], "y": [2]}), x="x", y="y")
    assert result == "displayed"
    assert not report.items
