#!/usr/bin/env python3
"""Execute compiled question DSL against a TableIR cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from answer_engine import AnswerEngine, AnswerEngineError
from dataset_utils import load_test_questions, utc_now, write_json
from question_dsl import DSLValidationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--dsl", type=Path, required=True)
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--id", dest="question_ids", action="append")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.ir.is_file() or not args.dsl.is_file():
        raise SystemExit("both --ir and --dsl must point to files")
    ir = json.loads(args.ir.read_text(encoding="utf-8"))
    dsl_document = json.loads(args.dsl.read_text(encoding="utf-8"))
    question_by_id = {question["id"]: question for question in load_test_questions(args.tests.resolve())[0]}
    requested_ids = set(args.question_ids or [])
    records: list[dict[str, Any]] = []
    engine = AnswerEngine(ir)
    for item in dsl_document.get("questions", []):
        question_id = str(item.get("id", ""))
        if requested_ids and question_id not in requested_ids:
            continue
        question = question_by_id.get(question_id)
        if question is None:
            records.append({"id": question_id, "error": "question id not found in tests.xlsx"})
            continue
        record: dict[str, Any] = {"id": question_id, "question": question["question"]}
        try:
            if "dsl" not in item:
                raise DSLValidationError(item.get("error", "compiled DSL is missing"))
            answer = engine.answer(question, item["dsl"])
            record.update(
                {
                    "answer": answer["answer"],
                    "sources": answer["sources"],
                    "dsl": answer["dsl"],
                }
            )
            print(f"question {question_id}: {answer['answer']}")
        except (AnswerEngineError, DSLValidationError, ValueError, TypeError) as exc:
            record["error"] = str(exc)
            print(f"question {question_id}: ERROR {exc}")
        records.append(record)

    output_path = (args.output or Path("artifacts/answers") / f"{Path(str(ir.get('source_file', 'table'))).stem}.json").resolve()
    result = {
        "schema_version": "answers.v1",
        "source_ir": str(args.ir),
        "source_dsl": str(args.dsl),
        "generated_at": utc_now(),
        "answers": records,
    }
    write_json(output_path, result)
    errors = sum(1 for record in records if record.get("error"))
    print(f"answered={len(records) - errors} errors={errors} -> {output_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
