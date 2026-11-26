"""Safe Excel import/export for versioned aircraft profiles."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator, Mapping
from typing import Any
from xml.etree.ElementTree import ParseError

from defusedxml.common import DefusedXmlException
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils.exceptions import InvalidFileException

from src.configuration.aircraft_profile import (
    MAX_PROFILE_BYTES,
    validate_aircraft_profile,
)


_DATA_HEADERS = (
    "json_pointer",
    "value_json",
    "unit",
    "source",
    "status",
)
_MAX_ARCHIVE_MEMBERS = 512
_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_WORKSHEET_ROWS = 100_001
_MAX_CELL_CHARACTERS = 32_767

_UNIT_SUFFIXES = (
    ("_kg_m2", "kg·m²"),
    ("_nm_s2_rad2", "N·m·s²/rad²"),
    ("_n_s2_rad2", "N·s²/rad²"),
    ("_rpm_per_volt", "RPM/V"),
    ("_mps2", "m/s²"),
    ("_deg_s", "deg/s"),
    ("_rpm_s", "RPM/s"),
    ("_m2", "m²"),
    ("_m3", "m³"),
    ("_mps", "m/s"),
    ("_mah", "mAh"),
    ("_rpm", "RPM"),
    ("_deg", "deg"),
    ("_rad", "rad"),
    ("_hz", "Hz"),
    ("_kg", "kg"),
    ("_in", "in"),
    ("_kv", "RPM/V"),
    ("_ohm", "Ω"),
    ("_nm", "N·m"),
    ("_h", "H"),
    ("_v", "V"),
    ("_m", "m"),
)

_SHEET_ORDER = (
    "Profile",
    "Mass-Geometry",
    "Propulsion",
    "Aerodynamics",
    "Actuators-Sensors",
    "Control-Transition",
    "Environment-Faults",
    "Hardware",
    "Simulation",
    "Traceability",
    "Extensions",
)


def serialize_aircraft_profile_excel(profile: Mapping[str, Any]) -> bytes:
    """Create an editable .xlsx workbook that preserves the complete profile."""
    validate_aircraft_profile(profile)
    workbook = Workbook()
    readme = workbook.active
    readme.title = "Read me"
    readme.append(["Aircraft profile workbook"])
    readme.append(["Purpose", "Editable Excel representation of a versioned aircraft profile."])
    readme.append(["Editing", "Edit Value (JSON); keep JSON Pointer values unchanged."])
    readme.append([
        "Value format",
        'Use JSON values such as 3.5, true, null, "PX4", [1, 2, 3], or {"key": 1}.',
    ])
    readme.append([
        "Import",
        "Upload this .xlsx file in the dashboard's aircraft profile uploader.",
    ])
    readme.append([
        "Safety",
        "Profile validation still applies; hardware actuator outputs must remain disabled.",
    ])
    readme.append([
        "Traceability",
        "Source/status notes are shown beside values where available and in the Traceability sheet.",
    ])
    readme.freeze_panes = "A2"
    readme.column_dimensions["A"].width = 22
    readme.column_dimensions["B"].width = 110
    readme.sheet_view.showGridLines = False
    readme["A1"].font = Font(bold=True, size=16, color="FFFFFF")
    readme["A1"].fill = PatternFill("solid", fgColor="17365D")
    readme.merge_cells("A1:B1")
    for row in readme.iter_rows(min_row=2, max_col=2):
        row[0].font = Font(bold=True)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    records_by_sheet: dict[str, list[tuple[str, Any, str, str]]] = {
        sheet_name: [] for sheet_name in _SHEET_ORDER
    }
    trace_records = _traceability_records(profile)
    for segments, value in _flatten_profile(profile):
        sheet_name = _sheet_for_path(segments)
        pointer = _json_pointer(segments)
        source, status = _trace_for_path(segments, trace_records)
        records_by_sheet[sheet_name].append((
            pointer,
            value,
            _infer_unit(segments[-1]),
            source,
            status,
        ))
    if sum(len(rows) for rows in records_by_sheet.values()) > _MAX_WORKSHEET_ROWS - 1:
        raise ValueError("aircraft profile workbook exceeds the data-row limit")

    for sheet_name in _SHEET_ORDER:
        rows = records_by_sheet[sheet_name]
        if not rows:
            continue
        if len(rows) > _MAX_WORKSHEET_ROWS - 1:
            raise ValueError(
                f"worksheet {sheet_name!r} exceeds the {_MAX_WORKSHEET_ROWS - 1:,} data-row limit"
            )
        worksheet = workbook.create_sheet(sheet_name)
        worksheet.append(list(_DATA_HEADERS))
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = f"A1:E{len(rows) + 1}"
        worksheet.sheet_view.showGridLines = False
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17365D")
        for pointer, value, unit, source, status in rows:
            row = worksheet.max_row + 1
            encoded_value = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if any(
                len(text) > _MAX_CELL_CHARACTERS
                for text in (pointer, encoded_value, unit, source, status)
            ):
                raise ValueError(
                    f"worksheet {sheet_name!r} contains a value exceeding Excel's {_MAX_CELL_CHARACTERS:,}-character cell limit"
                )
            _write_text(worksheet.cell(row=row, column=1), pointer)
            _write_text(worksheet.cell(row=row, column=2), encoded_value)
            _write_text(worksheet.cell(row=row, column=3), unit)
            _write_text(worksheet.cell(row=row, column=4), source)
            _write_text(worksheet.cell(row=row, column=5), status)
        for column, width in zip("ABCDE", (54, 46, 16, 54, 36)):
            worksheet.column_dimensions[column].width = width

    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    contents = stream.getvalue()
    if len(contents) > MAX_PROFILE_BYTES:
        raise ValueError("generated aircraft profile workbook exceeds the 32 MiB limit")
    return contents


def load_aircraft_profile_excel(contents: bytes) -> dict[str, Any]:
    """Load and validate an .xlsx aircraft profile workbook without evaluating formulas."""
    if not isinstance(contents, bytes):
        raise TypeError("aircraft profile workbook contents must be bytes")
    if not contents:
        raise ValueError("aircraft profile workbook is empty")
    if len(contents) > MAX_PROFILE_BYTES:
        raise ValueError("aircraft profile workbook exceeds the 32 MiB limit")

    _validate_xlsx_archive(contents)
    try:
        workbook = load_workbook(
            io.BytesIO(contents),
            read_only=True,
            data_only=False,
            keep_links=False,
        )
    except (
        DefusedXmlException,
        InvalidFileException,
        KeyError,
        OSError,
        ParseError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise ValueError(f"could not read aircraft profile workbook: {exc}") from exc

    try:
        profile: dict[str, Any] = {}
        row_count = 0
        found_data_sheet = False
        for worksheet in workbook.worksheets:
            if worksheet.title == "Read me":
                continue
            if worksheet.max_row is not None and worksheet.max_row > _MAX_WORKSHEET_ROWS:
                raise ValueError(
                    f"worksheet {worksheet.title!r} exceeds the {_MAX_WORKSHEET_ROWS - 1:,} data-row limit"
                )

            rows = worksheet.iter_rows(values_only=False)
            header = next(rows, None)
            if header is None or tuple(
                cell.value for cell in header[:len(_DATA_HEADERS)]
            ) != _DATA_HEADERS or any(
                cell.value is not None for cell in header[len(_DATA_HEADERS):]
            ):
                if _worksheet_has_values(worksheet):
                    raise ValueError(
                        f"worksheet {worksheet.title!r} must use the generated aircraft-profile workbook format"
                    )
                continue

            found_data_sheet = True
            for row_number, cells in enumerate(rows, start=2):
                row_count += 1
                if row_count > _MAX_WORKSHEET_ROWS - 1:
                    raise ValueError("aircraft profile workbook exceeds the data-row limit")
                if any(cell.data_type == "f" for cell in cells):
                    raise ValueError(
                        f"worksheet {worksheet.title!r} row {row_number} contains a formula; use literal values"
                    )
                if any(cell.value is not None for cell in cells[len(_DATA_HEADERS):]):
                    raise ValueError(
                        f"worksheet {worksheet.title!r} row {row_number} has data outside the profile workbook columns"
                    )
                pointer = cells[0].value if len(cells) > 0 else None
                encoded_value = cells[1].value if len(cells) > 1 else None
                if pointer is None and encoded_value is None:
                    continue
                if not isinstance(pointer, str) or not isinstance(
                    encoded_value,
                    (str, bool, int, float),
                ):
                    raise ValueError(
                        f"worksheet {worksheet.title!r} row {row_number} needs a JSON Pointer and a JSON value"
                    )
                segments = _parse_json_pointer(pointer)
                if isinstance(encoded_value, str):
                    try:
                        value = json.loads(
                            encoded_value,
                            parse_constant=_reject_json_constant,
                            object_pairs_hook=_reject_duplicate_keys,
                        )
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(
                            f"worksheet {worksheet.title!r} row {row_number} has invalid JSON in value_json: {exc}"
                        ) from exc
                else:
                    value = encoded_value
                _insert_profile_value(profile, segments, value, worksheet.title, row_number)

        if not found_data_sheet:
            raise ValueError("workbook contains no aircraft profile data sheets")
        validate_aircraft_profile(profile)
        return profile
    except (DefusedXmlException, KeyError, OSError, ParseError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not read aircraft profile workbook: {exc}") from exc
    finally:
        workbook.close()


def _validate_xlsx_archive(contents: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("aircraft profile workbook has too many archive entries")
            if "[Content_Types].xml" not in archive.namelist():
                raise ValueError("file is not a valid .xlsx workbook")
            if any(info.filename.lower().endswith("vbaproject.bin") for info in members):
                raise ValueError("macro-enabled Excel workbooks are not supported")
            if sum(info.file_size for info in members) > _MAX_UNCOMPRESSED_BYTES:
                raise ValueError("aircraft profile workbook expands beyond the 64 MiB limit")
            if any(
                info.file_size > 1024 * 1024
                and info.file_size / max(info.compress_size, 1) > 1000
                for info in members
            ):
                raise ValueError("aircraft profile workbook has an unsafe compression ratio")
    except zipfile.BadZipFile as exc:
        raise ValueError("file is not a valid .xlsx workbook") from exc


def _flatten_profile(
    value: Any,
    segments: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping) and value:
        for key, child in value.items():
            yield from _flatten_profile(child, (*segments, key))
    elif segments:
        yield segments, value


def _json_pointer(segments: tuple[str, ...]) -> str:
    return "/" + "/".join(segment.replace("~", "~0").replace("/", "~1") for segment in segments)


def _parse_json_pointer(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/") or pointer == "/":
        raise ValueError(f"invalid JSON Pointer {pointer!r}")
    parts = pointer[1:].split("/")
    decoded = []
    for part in parts:
        index = 0
        while index < len(part):
            if part[index] == "~" and (
                index + 1 == len(part) or part[index + 1] not in "01"
            ):
                raise ValueError(f"invalid escape in JSON Pointer {pointer!r}")
            index += 2 if part[index] == "~" else 1
        decoded.append(part.replace("~1", "/").replace("~0", "~"))
    return tuple(decoded)


def _sheet_for_path(segments: tuple[str, ...]) -> str:
    if segments[0] in ("schema_version", "metadata"):
        return "Profile"
    if segments[0] == "aircraft" and len(segments) > 1:
        group = segments[1]
        if group in ("mass_properties", "geometry"):
            return "Mass-Geometry"
        if group == "propulsion":
            return "Propulsion"
        if group == "aerodynamics":
            return "Aerodynamics"
        if group in ("actuators", "sensors"):
            return "Actuators-Sensors"
        if group in ("control", "transition"):
            return "Control-Transition"
        if group in ("disturbance", "environment", "failures"):
            return "Environment-Faults"
    if segments[0] == "hardware":
        return "Hardware"
    if segments[0] == "simulation":
        return "Simulation"
    if segments[0] == "traceability":
        return "Traceability"
    return "Extensions"


def _traceability_records(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    traceability = profile.get("traceability")
    if not isinstance(traceability, Mapping):
        return []
    records = traceability.get("parameter_group_records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _trace_for_path(
    segments: tuple[str, ...],
    records: list[Mapping[str, Any]],
) -> tuple[str, str]:
    path = ".".join(segments)
    matches = []
    for record in records:
        source_path = record.get("json_path")
        if not isinstance(source_path, str) or not source_path:
            continue
        candidates = [source_path]
        if not source_path.startswith("aircraft."):
            candidates.append(f"aircraft.{source_path}")
        if any(path == candidate or path.startswith(f"{candidate}.") for candidate in candidates):
            matches.append(record)
    if not matches:
        return "", ""
    record = max(matches, key=lambda item: len(str(item.get("json_path", ""))))
    source = record.get("source", "")
    status = record.get("status", "")
    return (
        source if isinstance(source, str) else "",
        status if isinstance(status, str) else "",
    )


def _infer_unit(key: str) -> str:
    for suffix, unit in _UNIT_SUFFIXES:
        if key.endswith(suffix):
            return unit
    return ""


def _write_text(cell: Any, value: str) -> None:
    cell.value = value
    cell.data_type = "s"


def _worksheet_has_values(worksheet: Any) -> bool:
    for row in worksheet.iter_rows(values_only=True):
        if any(value is not None for value in row):
            return True
    return False


def _insert_profile_value(
    profile: dict[str, Any],
    segments: tuple[str, ...],
    value: Any,
    sheet_name: str,
    row_number: int,
) -> None:
    current: dict[str, Any] = profile
    for segment in segments[:-1]:
        if segment not in current:
            existing: Any = {}
            current[segment] = existing
        else:
            existing = current[segment]
        if not isinstance(existing, dict):
            raise ValueError(
                f"worksheet {sheet_name!r} row {row_number} conflicts with a parent value"
            )
        current = existing
    leaf = segments[-1]
    if leaf in current:
        raise ValueError(
            f"worksheet {sheet_name!r} row {row_number} duplicates JSON Pointer {_json_pointer(segments)!r}"
        )
    current[leaf] = value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result
