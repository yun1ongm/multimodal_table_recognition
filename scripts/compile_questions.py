#!/usr/bin/env python3
"""Compile natural-language questions into executable question_dsl.v1 JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_utils import load_test_questions, utc_now, write_json
from qwen_client import QwenClient, QwenConfig
from question_dsl import DSLValidationError, build_compile_prompt, normalize_dsl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True, help="TableIR JSON")
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--id", dest="question_ids", action="append", help="Question id; repeatable")
    parser.add_argument("--hint", help="Only compile questions with this table hint")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_questions(args: argparse.Namespace, ir: dict[str, Any], tests_path: Path) -> list[dict[str, Any]]:
    questions, _ = load_test_questions(tests_path)
    source_file = str(ir.get("source_file", ""))
    requested_ids = set(args.question_ids or [])
    selected: list[dict[str, Any]] = []
    for question in questions:
        if requested_ids and question["id"] not in requested_ids:
            continue
        if not requested_ids:
            if question["file_name"] != source_file and Path(question["file_name"]).stem != Path(source_file).stem:
                continue
        if args.hint is not None and question.get("table_hint", "") != args.hint:
            continue
        selected.append(question)
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def default_output(ir: dict[str, Any]) -> Path:
    source = Path(str(ir.get("source_file", "table")))
    return Path("artifacts/question_dsl") / f"{source.stem}.json"


def main() -> int:
    args = parse_args()
    if args.max_tokens <= 0 or args.schema_retries < 0:
        raise SystemExit("max-tokens must be positive and schema-retries cannot be negative")
    if not args.ir.is_file():
        raise SystemExit(f"TableIR not found: {args.ir}")
    ir = json.loads(args.ir.read_text(encoding="utf-8"))
    questions = select_questions(args, ir, args.tests.resolve())
    if not questions:
        raise SystemExit("No questions selected for this TableIR")

    output_path = (args.output or default_output(ir)).resolve()
    if args.dry_run:
        for question in questions:
            print(f"### question {question['id']}")
            print(build_compile_prompt(question, ir))
        return 0

    client = QwenClient(QwenConfig.from_env(args.env_file.resolve()))
    records: list[dict[str, Any]] = []
    for question in questions:
        prompt = build_compile_prompt(question, ir)
        current_prompt = prompt
        record: dict[str, Any] = {
            "id": question["id"],
            "question": question["question"],
            "question_type": question["question_type"],
            "answer_format": question["answer_format"],
        }
        last_error: str | None = None
        for attempt in range(args.schema_retries + 1):
            try:
                response = client.chat_text_json(
                    required_keys={"operation", "expression"},
                    prompt=current_prompt,
                    system_prompt="你是严格的 JSON 题目意图编译器。只返回可执行 DSL JSON。",
                    model=args.model,
                    max_tokens=args.max_tokens,
                )
                dsl = normalize_dsl(response["json"], question)
                record.update(
                    {
                        "dsl": dsl,
                        "model": response["model"],
                        "usage": response.get("usage"),
                        "raw": response["text"],
                    }
                )
                print(f"question {question['id']}: compiled")
                break
            except (DSLValidationError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
                if attempt == args.schema_retries:
                    record["error"] = last_error
                    print(f"question {question['id']}: ERROR {last_error}")
                    break
                current_prompt = (
                    prompt
                    + "\n\n上一次输出无法执行，错误如下："
                    + last_error
                    + "\n请重新输出完整 JSON；expression 必须是允许的节点，禁止直接填写答案。"
                )
        records.append(record)

    result = {
        "schema_version": "question_dsl.v1",
        "source_ir": str(args.ir),
        "source_file": ir.get("source_file"),
        "generated_at": utc_now(),
        "questions": records,
    }
    write_json(output_path, result)
    errors = sum(1 for record in records if record.get("error"))
    print(f"compiled={len(records) - errors} errors={errors} -> {output_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
