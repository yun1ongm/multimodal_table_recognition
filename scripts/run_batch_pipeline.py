#!/usr/bin/env python3
"""Run cached TableIR -> batch DSL -> answers -> submission generation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_utils import index_files, load_test_questions, resolve_file_name


def safe_slug(value: str) -> str:
    import re

    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", value).strip("_")
    return slug[:100] or "table"


@dataclass(frozen=True)
class TableGroup:
    actual_file: str
    hint: str
    question_ids: tuple[str, ...]

    @property
    def stem(self) -> str:
        return Path(self.actual_file).stem

    @property
    def label(self) -> str:
        return f"{self.actual_file}::{self.hint or '<all>'}"

    @property
    def slug(self) -> str:
        return safe_slug(f"{self.stem}__{self.hint or 'all'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ir", "dsl", "answer", "export", "all"), default="all")
    parser.add_argument("--files-dir", type=Path, default=Path("files"))
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--ir-dir", type=Path, default=Path("artifacts/table_ir"))
    parser.add_argument("--financial-pages-dir", type=Path, default=Path("artifacts/financial_pages"))
    parser.add_argument("--render-dir", type=Path, default=Path("artifacts/prepared/rendered"))
    parser.add_argument("--dsl-dir", type=Path, default=Path("artifacts/question_dsl"))
    parser.add_argument("--answers-dir", type=Path, default=Path("artifacts/answers"))
    parser.add_argument("--output", type=Path, default=Path("submission.xlsx"))
    parser.add_argument("--report", type=Path, default=Path("reports/submission_export.json"))
    parser.add_argument("--file", dest="requested_files", action="append")
    parser.add_argument("--id", dest="question_ids", action="append")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge selected repaired DSL records into an existing group cache",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Compile selected DSL repairs one question per request",
    )
    parser.add_argument("--allow-blank", action="store_true")
    parser.add_argument(
        "--continue-on-errors",
        action="store_true",
        help="Keep other groups running when a group answer stage returns errors",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_groups(args: argparse.Namespace) -> list[TableGroup]:
    questions, _ = load_test_questions(args.tests.resolve())
    exact, aliases = index_files(args.files_dir.resolve())
    requested_ids = set(args.question_ids or [])
    requested_files = set(args.requested_files or [])
    groups: dict[tuple[str, str], list[str]] = {}
    for question in questions:
        if requested_ids and question["id"] not in requested_ids:
            continue
        resolution = resolve_file_name(question["file_name"], exact, aliases)
        actual_file = resolution.get("resolved")
        if not actual_file:
            continue
        if requested_files:
            requested_resolution = resolve_file_name(question["file_name"], exact, aliases)
            if question["file_name"] not in requested_files and actual_file not in requested_files:
                # Also accept a requested alias such as 58.pdf for 058.pdf.
                matched_alias = any(
                    resolve_file_name(requested, exact, aliases).get("resolved") == actual_file
                    for requested in requested_files
                )
                if not matched_alias:
                    continue
        key = (actual_file, question.get("table_hint", ""))
        groups.setdefault(key, []).append(question["id"])
    ordered = [
        TableGroup(actual_file=file_name, hint=hint, question_ids=tuple(ids))
        for (file_name, hint), ids in sorted(groups.items())
    ]
    if args.limit_groups is not None:
        ordered = ordered[: args.limit_groups]
    return ordered


def paths_for_group(args: argparse.Namespace, group: TableGroup) -> tuple[Path, Path, Path]:
    suffix = group.actual_file.lower().rsplit(".", 1)[-1]
    if suffix == "pdf":
        ir_path = args.ir_dir.resolve() / f"{group.slug}.financial.json"
    else:
        ir_path = args.ir_dir.resolve() / f"{group.stem}.json"
    dsl_path = args.dsl_dir.resolve() / f"{group.slug}.batch.json"
    answer_path = args.answers_dir.resolve() / f"{group.slug}.answers.json"
    return ir_path, dsl_path, answer_path


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def run_command(command: list[str], *, dry_run: bool, allow_failure: bool = False) -> bool:
    print("$ " + command_text(command))
    if dry_run:
        return True
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        if allow_failure:
            print(f"WARN command failed with exit code {completed.returncode}; continuing")
            return False
        raise SystemExit(f"command failed with exit code {completed.returncode}: {command_text(command)}")
    return True


def build_ir_command(args: argparse.Namespace, group: TableGroup, ir_path: Path) -> list[str]:
    script_root = Path(__file__).resolve().parent
    source_path = args.files_dir.resolve() / group.actual_file
    if source_path.suffix.lower() == ".pdf":
        command = [
            sys.executable,
            str(script_root / "build_financial_pages.py"),
            "--pdf",
            group.actual_file,
            "--hint",
            group.hint,
            "--files-dir",
            str(args.files_dir.resolve()),
            "--tests",
            str(args.tests.resolve()),
            "--render-dir",
            str(args.render_dir.resolve()),
            "--output-dir",
            str(args.financial_pages_dir.resolve()),
            "--stitched-output",
            str(ir_path),
            "--env-file",
            str(args.env_file.resolve()),
            "--dpi",
            str(args.dpi),
            "--max-tokens",
            str(args.max_tokens),
            "--schema-retries",
            str(args.schema_retries),
        ]
    else:
        command = [
            sys.executable,
            str(script_root / "build_table_ir.py"),
            "--file",
            group.actual_file,
            "--files-dir",
            str(args.files_dir.resolve()),
            "--tests",
            str(args.tests.resolve()),
            "--output-dir",
            str(args.ir_dir.resolve()),
            "--env-file",
            str(args.env_file.resolve()),
            "--max-tokens",
            str(args.max_tokens),
            "--schema-retries",
            str(args.schema_retries),
        ]
    if args.force:
        command.append("--force")
    return command


def compile_command(args: argparse.Namespace, group: TableGroup, ir_path: Path, dsl_path: Path) -> list[str]:
    script_root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(script_root / "compile_questions_batch.py"),
        "--ir",
        str(ir_path),
        "--tests",
        str(args.tests.resolve()),
        "--batch-size",
        str(args.batch_size),
        "--output",
        str(dsl_path),
        "--env-file",
        str(args.env_file.resolve()),
        "--max-tokens",
        str(args.max_tokens),
        "--schema-retries",
        str(args.schema_retries),
    ]
    for question_id in group.question_ids:
        command.extend(["--id", question_id])
    if args.model:
        command.extend(["--model", args.model])
    if args.continue_on_errors:
        command.append("--continue-on-errors")
    if args.merge_existing:
        command.append("--merge-existing")
    if args.single_only:
        command.append("--single-only")
    return command


def answer_command(args: argparse.Namespace, ir_path: Path, dsl_path: Path, answer_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "answer_questions.py"),
        "--ir",
        str(ir_path),
        "--dsl",
        str(dsl_path),
        "--tests",
        str(args.tests.resolve()),
        "--output",
        str(answer_path),
    ]


def export_command(args: argparse.Namespace, answer_paths: list[Path]) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "export_submission.py"),
        "--tests",
        str(args.tests.resolve()),
        "--output",
        str(args.output.resolve()),
        "--report",
        str(args.report.resolve()),
    ]
    for path in answer_paths:
        command.extend(["--answers", str(path)])
    if (
        args.allow_blank
        or args.continue_on_errors
        or args.limit_groups is not None
        or args.requested_files
        or args.question_ids
    ):
        command.append("--allow-blank")
    return command


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.max_tokens <= 0 or args.schema_retries < 0:
        raise SystemExit("batch-size/max-tokens must be positive and schema-retries cannot be negative")
    groups = collect_groups(args)
    if not groups:
        raise SystemExit("No question groups selected")
    print(f"selected groups={len(groups)} questions={sum(len(group.question_ids) for group in groups)}")
    if args.dry_run:
        for group in groups:
            print(f"GROUP {group.label} ids={list(group.question_ids)}")

    answer_paths: list[Path] = []
    for group in groups:
        ir_path, dsl_path, answer_path = paths_for_group(args, group)
        ir_path.parent.mkdir(parents=True, exist_ok=True)
        dsl_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        if args.stage in {"ir", "all"}:
            if ir_path.exists() and not args.force:
                print(f"IR cache: {group.label} -> {ir_path}")
            else:
                ok = run_command(
                    build_ir_command(args, group, ir_path),
                    dry_run=args.dry_run,
                    allow_failure=args.continue_on_errors,
                )
                if not ok and not args.dry_run:
                    print(f"WARN skip group after IR failure: {group.label}")
                    continue
        if args.stage in {"dsl", "all"}:
            if not ir_path.is_file() and not args.dry_run:
                if args.continue_on_errors:
                    print(f"WARN skip group after missing IR: {group.label}")
                    continue
                raise SystemExit(f"TableIR missing before DSL stage: {ir_path}")
            if dsl_path.exists() and not args.force:
                print(f"DSL cache: {group.label} -> {dsl_path}")
            else:
                ok = run_command(
                    compile_command(args, group, ir_path, dsl_path),
                    dry_run=args.dry_run,
                    allow_failure=args.continue_on_errors,
                )
                if not ok and not args.dry_run:
                    print(f"WARN skip group after DSL failure: {group.label}")
                    continue
        if args.stage in {"answer", "all"}:
            if (not ir_path.is_file() or not dsl_path.is_file()) and not args.dry_run:
                if args.continue_on_errors:
                    print(f"WARN skip group after missing IR/DSL: {group.label}")
                    continue
                raise SystemExit(f"IR/DSL missing before answer stage: {group.label}")
            if answer_path.exists() and not args.force:
                print(f"answer cache: {group.label} -> {answer_path}")
            else:
                run_command(
                    answer_command(args, ir_path, dsl_path, answer_path),
                    dry_run=args.dry_run,
                    allow_failure=args.continue_on_errors,
                )
        if answer_path.is_file() or args.dry_run:
            answer_paths.append(answer_path)

    if args.stage in {"export", "all"}:
        if not args.dry_run and any(not path.is_file() for path in answer_paths):
            missing = [str(path) for path in answer_paths if not path.is_file()]
            raise SystemExit("answer files missing before export: " + ", ".join(missing[:5]))
        run_command(export_command(args, answer_paths), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
