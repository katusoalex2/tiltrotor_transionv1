"""Build downloadable simulation reports from explicitly captured dashboard output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.shared import Inches
from PIL import Image
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as PdfImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


@dataclass
class ReportItem:
    kind: str
    title: str
    value: Any


class SimulationReport:
    """Capture dashboard output while leaving the underlying Streamlit calls unchanged."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.items: list[ReportItem] = []

    def line_chart(self, target: Any, data: Any, **kwargs: Any) -> Any:
        result = target.line_chart(data, **kwargs)
        if self.enabled:
            frame = pd.DataFrame(data)
            fig, ax = plt.subplots(figsize=(9, 4))
            x_name = kwargs.get("x")
            x = frame[x_name] if x_name else frame.index
            ys = kwargs.get("y")
            if isinstance(ys, str):
                ys = [ys]
            if ys is None:
                ys = [name for name in frame.columns if name != x_name]
            for name in ys:
                ax.plot(x, frame[name], label=str(name))
            ax.set_xlabel(kwargs.get("x_label") or str(x_name or "Index"))
            ax.set_ylabel(kwargs.get("y_label") or "Value")
            ax.grid(True, alpha=0.3)
            if ys:
                ax.legend(fontsize="small")
            fig.tight_layout()
            self._matplotlib_figure(fig, kwargs.get("alt", "Line chart"))
            plt.close(fig)
        return result

    def scatter_chart(self, target: Any, data: Any, **kwargs: Any) -> Any:
        result = target.scatter_chart(data, **kwargs)
        if self.enabled:
            frame = pd.DataFrame(data)
            x_name, y_name = kwargs["x"], kwargs["y"]
            fig, ax = plt.subplots(figsize=(9, 4))
            color_name = kwargs.get("color")
            if isinstance(color_name, str) and color_name in frame:
                points = ax.scatter(frame[x_name], frame[y_name], c=frame[color_name])
                fig.colorbar(points, ax=ax, label=color_name)
            else:
                ax.scatter(frame[x_name], frame[y_name])
            ax.set_xlabel(kwargs.get("x_label") or str(x_name))
            ax.set_ylabel(kwargs.get("y_label") or str(y_name))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            self._matplotlib_figure(fig, kwargs.get("alt", "Scatter chart"))
            plt.close(fig)
        return result

    def pyplot(self, target: Any, fig: Any, **kwargs: Any) -> Any:
        if self.enabled:
            self._matplotlib_figure(fig, kwargs.get("alt", "Figure"))
        return target.pyplot(fig, **kwargs)

    def plotly_chart(self, target: Any, fig: Any, **kwargs: Any) -> Any:
        result = target.plotly_chart(fig, **kwargs)
        if self.enabled:
            self.items.append(
                ReportItem("image", str(kwargs.get("alt", "Plotly chart")), fig)
            )
        return result

    def dataframe(self, target: Any, data: Any, **kwargs: Any) -> Any:
        result = target.dataframe(data, **kwargs)
        if self.enabled:
            self.items.append(
                ReportItem("table", str(kwargs.get("alt", "Results table")), pd.DataFrame(data).copy())
            )
        return result

    def text(self, kind: str, target: Any, content: Any, **kwargs: Any) -> Any:
        result = getattr(target, kind)(content, **kwargs)
        if self.enabled:
            label = f"{kind.upper()}: " if kind in {"warning", "error", "info"} else ""
            self.items.append(
                ReportItem("heading" if kind == "subheader" else "text", "", label + str(content))
            )
        return result

    def metric(self, target: Any, label: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        result = target.metric(label, value, *args, **kwargs)
        if self.enabled:
            self.items.append(ReportItem("text", "", f"{label}: {value}"))
        return result

    def _matplotlib_figure(self, fig: Any, title: str) -> None:
        image = BytesIO()
        fig.savefig(image, format="png", dpi=130, bbox_inches="tight")
        self.items.append(ReportItem("image", str(title), image.getvalue()))

    def render(self, aircraft_name: str, profile_json: str) -> tuple[bytes, bytes]:
        """Render both formats; fail explicitly if any chart cannot be exported."""
        for item in self.items:
            if item.kind == "image" and not isinstance(item.value, bytes):
                try:
                    item.value = item.value.to_image(format="png", width=1100, height=650)
                except (RuntimeError, ValueError, OSError) as exc:
                    raise RuntimeError(f"Could not export graph '{item.title}': {exc}") from exc

        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        disclaimer = (
            "Simulation results only. Not flight-ready, not a certified safety envelope, "
            "and not a PX4 configuration. Unassessed constraints are not passes."
        )
        document = Document()
        document.add_heading("Aircraft transition simulation report", 0)
        document.add_paragraph(f"Aircraft: {aircraft_name}\nGenerated: {generated}")
        document.add_paragraph(disclaimer)

        styles = getSampleStyleSheet()
        font_path = Path(plt.matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        pdfmetrics.registerFont(TTFont("ReportDejaVu", str(font_path)))
        for style in styles.byName.values():
            style.fontName = "ReportDejaVu"
        styles["Title"].alignment = TA_CENTER
        story: list[Any] = [
            Paragraph("Aircraft transition simulation report", styles["Title"]),
            Paragraph(escape(f"Aircraft: {aircraft_name} | Generated: {generated}"), styles["Normal"]),
            Paragraph(escape(disclaimer), styles["Normal"]),
            Spacer(1, 10),
        ]

        for item in self.items:
            if item.kind == "image":
                document.add_heading(item.title, level=2)
                stream = BytesIO(item.value)
                with Image.open(stream) as image:
                    width, height = image.size
                width_inches = min(6.0, 7.5 * width / height)
                document.add_picture(BytesIO(item.value), width=Inches(width_inches))
                story.append(Paragraph(escape(item.title), styles["Heading2"]))
                story.append(PdfImage(BytesIO(item.value), width=width_inches * inch,
                                      height=width_inches * inch * height / width))
                story.append(Spacer(1, 8))
            elif item.kind == "table":
                document.add_heading(item.title, level=2)
                story.append(Paragraph(escape(item.title), styles["Heading2"]))
                frame = item.value
                table = document.add_table(rows=1, cols=len(frame.columns) + 1)
                table.style = "Table Grid"
                for cell, value in zip(table.rows[0].cells, ["Index", *frame.columns]):
                    cell.text = str(value)
                for index, row in frame.iterrows():
                    cells = table.add_row().cells
                    for cell, value in zip(cells, [index, *row.tolist()]):
                        cell.text = str(value)
                    line = "; ".join(
                        f"{name}: {value}" for name, value in zip(frame.columns, row.tolist())
                    )
                    story.append(Paragraph(escape(f"{index}: {line}"), styles["BodyText"]))
            elif item.kind == "heading":
                document.add_heading(item.value, level=1)
                story.append(Paragraph(escape(item.value), styles["Heading1"]))
            else:
                document.add_paragraph(item.value)
                story.append(Paragraph(escape(item.value).replace("\n", "<br/>"), styles["BodyText"]))
        document.add_heading("Aircraft profile and dashboard settings", level=1)
        document.add_paragraph(profile_json)
        story.append(Paragraph("Aircraft profile and dashboard settings", styles["Heading1"]))
        for line in profile_json.splitlines():
            story.append(Paragraph(escape(line) or "&nbsp;", styles["Code"]))

        word = BytesIO()
        document.save(word)
        pdf = BytesIO()
        SimpleDocTemplate(pdf, pagesize=(8.27 * inch, 11.69 * inch),
                          leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                          topMargin=0.6 * inch, bottomMargin=0.6 * inch).build(story)
        return word.getvalue(), pdf.getvalue()
