"""JSON output for verification reports."""

from __future__ import annotations

import json
from pathlib import Path

from halref.models import BatchReport


def write_json_report(batch: BatchReport, output_path: Path | None = None) -> str:
    """Write batch report as JSON.

    Returns the JSON string. If output_path provided, also writes to file.
    """
    data = batch.model_dump(mode="json")
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)

    return json_str
