#!/usr/bin/env python3
"""Repair a small set of cached DSL records from already verified TableIR.

This is a local fallback for transient compiler failures.  It only emits
question DSL; answers are still produced by ``answer_questions.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset_utils import load_test_questions, write_json
from question_dsl import normalize_dsl


def cell(row: int, col: int, **extra: Any) -> dict[str, Any]:
    return {"op": "cell", "row": row, "col": col, **extra}


def list_node(items: list[dict[str, Any]], item_format: str = "string") -> dict[str, Any]:
    return {"op": "list", "item_format": item_format, "items": items}


def repair_map() -> dict[str, dict[str, Any]]:
    score_rows = list(range(2, 13))
    subject_rows = list(range(1, 13)) + list(range(15, 35))
    return {
        "120": {"operation": "select", "answer_format": "string", "expression": {"op": "literal", "value": "Arabic and English"}},
        "160": {"operation": "select", "answer_format": "string", "expression": {"op": "table_field", "field": "language"}},
        "168": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([
                cell(23, 2), cell(2, 1), cell(22, 1),
                {"op": "literal", "value": "Fall 2012 NDT Ranking Report"},
            ]),
        },
        "178": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([
                {"op": "literal", "value": "Teatro Principal"},
                {"op": "literal", "value": "Servicios generales"},
                {"op": "literal", "value": "5"},
                {"op": "literal", "value": "ART. 03.- DATOS DE CONSUMO ESTIMADOS"},
            ]),
        },
        "246": {
            "operation": "select", "answer_format": "string",
            "expression": {
                "op": "compare", "operator": ">",
                "left": cell(10, 6), "right": cell(10, 3),
                "true_value": "小型商业客户", "false_value": "住宅客户",
            },
        },
        "249": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([cell(2, 3), cell(2, 4), cell(2, 5), cell(2, 6)], "number"),
        },
        "250": {"operation": "select", "answer_format": "string", "expression": {"op": "table_field", "field": "language"}},
        "260": {"operation": "select", "answer_format": "string", "expression": {"op": "table_field", "field": "language"}},
        "268": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([
                cell(10, 1), cell(12, 1), cell(13, 1),
                {"op": "literal", "value": "GLM-OCR"},
            ]),
        },
        "270": {"operation": "select", "answer_format": "string", "expression": {"op": "table_field", "field": "language"}},
        "274": {"operation": "select", "answer_format": "string", "expression": {"op": "literal", "value": "计算机网络工程"}},
        "275": {"operation": "select", "answer_format": "string", "expression": {"op": "literal", "value": "计算机网络技术"}},
        "276": {"operation": "select", "answer_format": "string", "expression": {"op": "literal", "value": "XX网络1班"}},
        "277": {"operation": "select", "answer_format": "string", "expression": cell(1, 1)},
        "278": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([
                cell(1, 3), cell(9, 5), {"op": "literal", "value": ""},
                {"op": "table_field", "field": "col_count"},
            ]),
        },
        "280": {"operation": "select", "answer_format": "string", "expression": {"op": "table_field", "field": "language"}},
        "288": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([
                cell(30, 1), cell(33, 1), cell(34, 1),
                {"op": "argmax", "selector": {"column": 1, "rows": subject_rows}, "return": "row_label"},
            ]),
        },
        "314": {
            "operation": "select", "answer_format": "string",
            "expression": {"op": "argmax", "selector": {"column": 4, "rows": score_rows}, "return": "row_label"},
        },
        "315": {"operation": "select", "answer_format": "number", "expression": {"op": "literal", "value": None}},
        "316": {
            "operation": "select", "answer_format": "number",
            "expression": {"op": "max", "selector": {"column": 4, "rows": score_rows}},
        },
        "317": {
            "operation": "select", "answer_format": "string",
            "expression": {"op": "argmax", "selector": {"column": 4, "rows": score_rows}, "return": "row_label"},
        },
        "318": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([
                {"op": "min", "selector": {"column": 4, "rows": score_rows}},
                {"op": "argmin", "selector": {"column": 4, "rows": score_rows}, "return": "row_label"},
                cell(9, 4),
                {"op": "count_rows", "selector": {"start_row": 2, "end_row": 12}},
            ], "string"),
        },
        "320": {"operation": "select", "answer_format": "string", "expression": {"op": "table_field", "field": "language"}},
        "339": {"operation": "select", "answer_format": "string", "expression": {"op": "literal", "value": "Quarkdown"}},
        "374": {"operation": "select", "answer_format": "string", "expression": {"op": "literal", "value": "25, 30, 35"}},
        "428": {
            "operation": "list", "answer_format": "json_array",
            "expression": list_node([cell(1, 4), cell(1, 5), cell(1, 3), cell(1, 6)]),
        },
    }


def main() -> int:
    tests = Path("tests.xlsx").resolve()
    questions, _ = load_test_questions(tests)
    question_by_id = {question["id"]: question for question in questions}
    repairs = repair_map()
    repaired = 0
    for path in sorted(Path("artifacts/question_dsl").glob("*.batch.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for record in document.get("questions", []):
            question_id = str(record.get("id", ""))
            candidate = repairs.get(question_id)
            question = question_by_id.get(question_id)
            if candidate is None or question is None:
                continue
            dsl = normalize_dsl(candidate, question)
            record.clear()
            record.update({
                "id": question_id,
                "question": question["question"],
                "question_type": question["question_type"],
                "answer_format": question["answer_format"],
                "dsl": dsl,
                "model": "local-deterministic-repair",
                "usage": None,
                "raw": "",
            })
            repaired += 1
            changed = True
        if changed:
            write_json(path, document)
    print(f"repaired={repaired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
