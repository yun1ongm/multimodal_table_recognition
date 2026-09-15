#!/usr/bin/env python3
"""Merge internal answer JSON files into the competition submission workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from dataset_utils import load_test_questions, utc_now, write_json
from validate_submission import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, action="append", required=True, help="answers.v1 JSON; repeatable")
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("submission.xlsx"))
    parser.add_argument("--report", type=Path, default=Path("reports/submission_export.json"))
    parser.add_argument("--allow-blank", action="store_true", help="Allow missing answer records and write blank answers")
    return parser.parse_args()


def load_answer_records(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for record in document.get("answers", []):
            question_id = str(record.get("id", ""))
            if not question_id:
                conflicts.append({"kind": "blank_answer_id", "file": str(path)})
                continue
            if question_id in by_id and by_id[question_id].get("answer") != record.get("answer"):
                conflicts.append(
                    {
                        "kind": "conflicting_answer",
                        "id": question_id,
                        "first": by_id[question_id].get("answer"),
                        "second": record.get("answer"),
                        "file": str(path),
                    }
                )
                continue
            by_id[question_id] = record
    return by_id, conflicts


def main() -> int:
    args = parse_args()
    for path in args.answers:
        if not path.is_file():
            raise SystemExit(f"answer JSON not found: {path}")
    questions, _ = load_test_questions(args.tests.resolve())
    expected_ids = {question["id"] for question in questions}
    records, conflicts = load_answer_records(args.answers)
    missing = [question["id"] for question in questions if question["id"] not in records]
    extra = sorted(set(records) - expected_ids)
    if conflicts:
        raise SystemExit("conflicting answer records: " + json.dumps(conflicts[:5], ensure_ascii=False))
    if missing and not args.allow_blank:
        raise SystemExit(
            f"{len(missing)} question ids have no answer record; use --allow-blank only for a draft workbook"
        )

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "submission"
    worksheet.append(["id", "answer"])
    for question in questions:
        record = records.get(question["id"], {})
        answer = record.get("answer", "")
        worksheet.append([question["id"], str(answer) if answer is not None else ""])
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)

    validation = validate(args.tests.resolve(), output_path)
    report = {
        "schema_version": "submission_export.v1",
        "generated_at": utc_now(),
        "answers": [str(path) for path in args.answers],
        "output": str(output_path),
        "missing_ids": missing,
        "extra_ids": extra,
        "validation": validation,
    }
    write_json(args.report.resolve(), report)
    print(
        f"submission={output_path} rows={len(questions)} missing={len(missing)} "
        f"extra={len(extra)} validation_errors={len(validation['errors'])}"
    )
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
