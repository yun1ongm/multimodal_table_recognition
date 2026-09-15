#!/usr/bin/env python3
"""Compile many questions for one TableIR with one Qwen request per batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_utils import load_test_questions, utc_now, write_json
from qwen_client import QwenClient, QwenConfig
from question_dsl import (
    DSLValidationError,
    build_compile_batch_prompt,
    build_compile_prompt,
    normalize_dsl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--id", dest="question_ids", action="append")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=7000)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument(
        "--continue-on-errors",
        action="store_true",
        help="Fall back to one-question requests when a batch cannot be compiled",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_questions(args: argparse.Namespace, ir: dict[str, Any]) -> list[dict[str, Any]]:
    questions, _ = load_test_questions(args.tests.resolve())
    requested_ids = set(args.question_ids or [])
    source_file = str(ir.get("source_file", ""))
    selected: list[dict[str, Any]] = []
    for question in questions:
        if requested_ids:
            if question["id"] not in requested_ids:
                continue
        elif question["file_name"] != source_file and Path(question["file_name"]).stem != Path(source_file).stem:
            continue
        selected.append(question)
    return selected


def default_output(ir: dict[str, Any]) -> Path:
    return Path("artifacts/question_dsl") / f"{Path(str(ir.get('source_file', 'table'))).stem}.batch.json"


def compile_batch(
    client: QwenClient,
    questions: list[dict[str, Any]],
    ir: dict[str, Any],
    *,
    model: str | None,
    max_tokens: int,
    schema_retries: int,
) -> list[dict[str, Any]]:
    prompt = build_compile_batch_prompt(questions, ir)
    current_prompt = prompt
    expected = {question["id"]: question for question in questions}
    last_error = ""
    for attempt in range(schema_retries + 1):
        response = client.chat_text_json(
            required_keys={"questions"},
            prompt=current_prompt,
            system_prompt="你是严格的 JSON 题目意图编译器。只返回 question_dsl.v1 JSON，不要回答题目。",
            model=model,
            max_tokens=max_tokens,
        )
        payload = response["json"]
        items = payload.get("questions") if isinstance(payload, dict) else None
        errors: list[str] = []
        normalized_by_id: dict[str, dict[str, Any]] = {}
        if not isinstance(items, list):
            errors.append("questions must be an array")
        else:
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"questions[{index}] must be an object")
                    continue
                question_id = str(item.get("id", ""))
                if question_id not in expected:
                    errors.append(f"unexpected question id {question_id!r}")
                    continue
                if question_id in normalized_by_id:
                    errors.append(f"duplicate question id {question_id!r}")
                    continue
                try:
                    normalized_by_id[question_id] = normalize_dsl(item.get("dsl"), expected[question_id])
                except (DSLValidationError, ValueError) as exc:
                    errors.append(f"id={question_id}: {exc}")
        missing = sorted(set(expected) - set(normalized_by_id))
        if missing:
            errors.append("missing question ids: " + ", ".join(missing))
        if not errors:
            records = []
            for question in questions:
                question_id = question["id"]
                records.append(
                    {
                        "id": question_id,
                        "question": question["question"],
                        "question_type": question["question_type"],
                        "answer_format": question["answer_format"],
                        "dsl": normalized_by_id[question_id],
                        "model": response["model"],
                        "usage": response.get("usage"),
                        "raw": response["text"],
                    }
                )
            return records
        last_error = "; ".join(errors[:12])
        if attempt == schema_retries:
            break
        current_prompt = (
            prompt
            + "\n\n上次批量输出未通过校验："
            + last_error
            + "\n请重新输出所有题目的完整 questions 数组，确保每个 id 恰好一次。"
        )
    raise RuntimeError(last_error or "batch DSL validation failed")


def compile_single(
    client: QwenClient,
    question: dict[str, Any],
    ir: dict[str, Any],
    *,
    model: str | None,
    max_tokens: int,
    schema_retries: int,
) -> dict[str, Any]:
    """Compile one question after a batch request fails validation."""

    prompt = build_compile_prompt(question, ir)
    current_prompt = prompt
    last_error = ""
    for attempt in range(schema_retries + 1):
        try:
            response = client.chat_text_json(
                required_keys={"operation", "expression"},
                prompt=current_prompt,
                system_prompt="你是严格的 JSON 题目意图编译器。只返回可执行 DSL JSON。",
                model=model,
                max_tokens=max_tokens,
            )
            dsl = normalize_dsl(response["json"], question)
            return {
                "id": question["id"],
                "question": question["question"],
                "question_type": question["question_type"],
                "answer_format": question["answer_format"],
                "dsl": dsl,
                "model": response["model"],
                "usage": response.get("usage"),
                "raw": response["text"],
            }
        except (DSLValidationError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if attempt == schema_retries:
                break
            current_prompt = (
                prompt
                + "\n\n上一次输出无法执行，错误如下："
                + last_error
                + "\n请重新输出完整 JSON；expression 必须是允许的节点，禁止直接填写答案。"
            )
    raise RuntimeError(last_error or "single-question DSL compilation failed")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.max_tokens <= 0 or args.schema_retries < 0:
        raise SystemExit("batch-size/max-tokens must be positive and schema-retries cannot be negative")
    if not args.ir.is_file():
        raise SystemExit(f"TableIR not found: {args.ir}")
    ir = json.loads(args.ir.read_text(encoding="utf-8"))
    questions = select_questions(args, ir)
    if not questions:
        raise SystemExit("No questions selected for this TableIR")
    chunks = [questions[index : index + args.batch_size] for index in range(0, len(questions), args.batch_size)]
    if args.dry_run:
        for index, chunk in enumerate(chunks, start=1):
            print(f"### batch {index}/{len(chunks)} ids={[question['id'] for question in chunk]}")
            print(build_compile_batch_prompt(chunk, ir))
        return 0

    client = QwenClient(QwenConfig.from_env(args.env_file.resolve()))
    records: list[dict[str, Any]] = []
    fallback_errors = 0
    for index, chunk in enumerate(chunks, start=1):
        print(f"batch {index}/{len(chunks)} ids={[question['id'] for question in chunk]}")
        try:
            records.extend(
                compile_batch(
                    client,
                    chunk,
                    ir,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    schema_retries=args.schema_retries,
                )
            )
        except Exception as exc:
            if not args.continue_on_errors:
                raise
            print(f"WARN batch failed; retrying one question at a time: {exc}")
            for question in chunk:
                try:
                    records.append(
                        compile_single(
                            client,
                            question,
                            ir,
                            model=args.model,
                            max_tokens=args.max_tokens,
                            schema_retries=args.schema_retries,
                        )
                    )
                    print(f"question {question['id']}: single fallback compiled")
                except Exception as single_error:
                    fallback_errors += 1
                    records.append(
                        {
                            "id": question["id"],
                            "question": question["question"],
                            "question_type": question["question_type"],
                            "answer_format": question["answer_format"],
                            "error": str(single_error),
                        }
                    )
                    print(f"question {question['id']}: ERROR {single_error}")

    output_path = (args.output or default_output(ir)).resolve()
    result = {
        "schema_version": "question_dsl.v1",
        "source_ir": str(args.ir),
        "source_file": ir.get("source_file"),
        "generated_at": utc_now(),
        "questions": records,
    }
    write_json(output_path, result)
    print(
        f"compiled={len(records) - fallback_errors} records={len(records)} "
        f"fallback_errors={fallback_errors} batches={len(chunks)} -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
