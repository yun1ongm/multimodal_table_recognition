#!/usr/bin/env python3
"""Stitch page-level TableIR files and attach financial metadata."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from build_table_ir import normalize_ir
from dataset_utils import utc_now, write_json
from financial_utils import annotate_financial_ir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="inputs", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-file", default="stitched.pdf")
    parser.add_argument("--no-financial", action="store_true")
    return parser.parse_args()


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def row_signature(document: dict[str, Any], row: int) -> tuple[tuple[int, str, int, int], ...]:
    values = []
    for cell in document.get("cells", []):
        if cell.get("row") == row and normalized_text(cell.get("text")):
            values.append(
                (
                    int(cell.get("col", 0)),
                    normalized_text(cell.get("text")),
                    int(cell.get("rowspan", 1)),
                    int(cell.get("colspan", 1)),
                )
            )
    return tuple(sorted(values))


def header_rows(document: dict[str, Any], limit: int = 5) -> set[int]:
    rows: set[int] = set()
    for cell in document.get("cells", []):
        row = int(cell.get("row", 0))
        if row >= limit:
            continue
        if cell.get("role") == "header":
            rows.add(row)
    return rows


def repeated_header_skip(first: dict[str, Any], current: dict[str, Any]) -> int:
    """Return the number of leading rows to remove from a continuation page."""

    first_header_signatures = {
        row_signature(first, row)
        for row in header_rows(first)
        if row_signature(first, row)
    }
    current_headers = header_rows(current)
    repeated: set[int] = set()
    for row in current_headers:
        signature = row_signature(current, row)
        if signature in first_header_signatures:
            repeated.add(row)
            continue
        current_text = {item[1] for item in signature}
        for first_signature in first_header_signatures:
            first_text = {item[1] for item in first_signature}
            if current_text and len(current_text & first_text) >= min(2, len(current_text)):
                repeated.add(row)
                break
    if not repeated:
        return 0
    # A table header is contiguous at the top of a continuation page.  Keep
    # the rule conservative: only skip if the repeated rows start at row 0.
    if min(repeated) != 0:
        return 0
    return max(repeated) + 1


def stitch_pages(documents: list[dict[str, Any]], source_file: str) -> dict[str, Any]:
    if not documents:
        raise ValueError("No TableIR documents to stitch")
    documents = sorted(documents, key=lambda item: int(item.get("page_index", 0)))
    first = documents[0]
    output: dict[str, Any] = {
        "schema_version": "table_ir.v1",
        "source_file": source_file,
        "title": next((str(item.get("title")) for item in documents if item.get("title")), ""),
        "unit": next((item.get("unit") for item in documents if item.get("unit")), None),
        "row_count": 0,
        "col_count": max(int(item.get("col_count", 0)) for item in documents),
        "cells": [],
        "notes": [],
    }
    row_offset = 0
    for page_index, document in enumerate(documents):
        skip = repeated_header_skip(first, document) if page_index else 0
        if skip:
            output["notes"].append(f"page {page_index + 1}: skipped {skip} repeated header rows")
        if document.get("unit") and output["unit"] and document["unit"] != output["unit"]:
            output["notes"].append(
                f"page {page_index + 1}: unit differs from first page ({document['unit']!r})"
            )
        for note in document.get("notes", []):
            output["notes"].append(f"page {page_index + 1}: {note}")
        local_rows = max(0, int(document.get("row_count", 0)) - skip)
        for original in document.get("cells", []):
            cell = copy.deepcopy(original)
            local_row = int(cell["row"])
            if local_row < skip:
                continue
            cell["row"] = row_offset + local_row - skip
            cell["source_page"] = page_index
            output["cells"].append(cell)
        row_offset += local_rows
    output["row_count"] = row_offset
    stitched = normalize_ir(output, source_file)
    if not stitched["notes"]:
        stitched["notes"] = ["stitched from page-level TableIR"]
    if output["unit"] is None:
        stitched["unit"] = None
    return stitched


def main() -> int:
    args = parse_args()
    documents = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.is_file():
            raise SystemExit(f"TableIR input not found: {path}")
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    result = stitch_pages(documents, args.source_file)
    if not args.no_financial:
        result = annotate_financial_ir(result)
    result["generated_at"] = utc_now()
    result["page_inputs"] = [str(path) for path in args.inputs]
    write_json(args.output, result)
    print(
        f"stitched pages={len(documents)} rows={result['row_count']} "
        f"cols={result['col_count']} cells={len(result['cells'])} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
