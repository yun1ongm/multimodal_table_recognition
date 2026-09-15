#!/usr/bin/env python3
"""Audit the competition inputs and optionally render referenced PDFs.

Examples:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --render-pdfs --dpi 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_utils import (
    image_dimensions,
    index_files,
    load_test_questions,
    pdf_metadata,
    resolve_file_name,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--files", dest="files_dir", type=Path, default=Path("files"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/prepared"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--render-pdfs", action="store_true", help="Render referenced PDFs to PNG pages")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--force-render", action="store_true", help="Re-render existing PDF page files")
    return parser.parse_args()


def render_pdf(pdf_path: Path, output_root: Path, dpi: int, force: bool = False) -> dict[str, Any]:
    """Render a PDF into zero-padded page PNGs and return a render record."""

    metadata = pdf_metadata(pdf_path)
    pages = metadata.get("pages")
    record: dict[str, Any] = {
        "source": pdf_path.name,
        "pages": pages,
        "dpi": dpi,
        "status": "pending",
        "output_dir": str(output_root),
        "outputs": [],
    }
    if not pages:
        record["status"] = "failed"
        record["error"] = "Could not determine PDF page count"
        return record
    if shutil.which("pdftoppm") is None:
        record["status"] = "failed"
        record["error"] = "pdftoppm is not installed"
        return record

    output_root.mkdir(parents=True, exist_ok=True)
    expected = [output_root / f"page-{index:03d}.png" for index in range(1, pages + 1)]
    if not force and all(path.exists() and path.stat().st_size > 0 for path in expected):
        record["status"] = "skipped_existing"
        record["outputs"] = [str(path) for path in expected]
        return record

    with tempfile.TemporaryDirectory(prefix="mmtr_pdf_") as temporary_dir:
        prefix = Path(temporary_dir) / "page"
        command = [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            str(pdf_path),
            str(prefix),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            record["status"] = "failed"
            record["error"] = (completed.stderr or completed.stdout).strip()[-1000:]
            return record

        generated = sorted(
            Path(temporary_dir).glob("page-*.png"),
            key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
        )
        if len(generated) != pages:
            record["status"] = "failed"
            record["error"] = f"Expected {pages} pages, rendered {len(generated)}"
            return record
        for index, source in enumerate(generated, start=1):
            target = output_root / f"page-{index:03d}.png"
            source.replace(target)
            record["outputs"].append(str(target))

    record["status"] = "rendered"
    return record


def build_manifest(
    tests_path: Path,
    files_dir: Path,
    output_dir: Path,
    render_pdfs: bool,
    dpi: int,
    force_render: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    questions, workbook_info = load_test_questions(tests_path)
    exact, aliases = index_files(files_dir)
    actual_names = set(exact)

    resolutions: dict[str, dict[str, Any]] = {}
    normalized_questions: list[dict[str, Any]] = []
    for question in questions:
        requested = question["file_name"]
        if requested not in resolutions:
            resolutions[requested] = resolve_file_name(requested, exact, aliases)
        resolution = resolutions[requested]
        normalized = dict(question)
        normalized["file_resolution"] = resolution
        normalized["resolved_file"] = resolution.get("resolved")
        normalized_questions.append(normalized)

    referenced_actual = {
        resolution["resolved"]
        for resolution in resolutions.values()
        if resolution.get("resolved")
    }
    file_records: list[dict[str, Any]] = []
    render_records: list[dict[str, Any]] = []
    for name in sorted(actual_names):
        path = exact[name]
        suffix = path.suffix.lower()
        dimensions = image_dimensions(path) if suffix in {".jpg", ".jpeg", ".png", ".webp"} else None
        record: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "extension": suffix.lstrip("."),
            "bytes": path.stat().st_size,
            "referenced": name in referenced_actual,
            "dimensions": {"width": dimensions[0], "height": dimensions[1]} if dimensions else None,
        }
        if suffix == ".pdf":
            record["pdf"] = pdf_metadata(path)
            if render_pdfs and name in referenced_actual:
                render_record = render_pdf(
                    path,
                    output_dir / "rendered" / path.stem,
                    dpi=dpi,
                    force=force_render,
                )
                render_records.append(render_record)
        file_records.append(record)

    question_type_counts_raw = Counter(question["raw_question_type"] for question in normalized_questions)
    question_type_counts = Counter(question["question_type"] for question in normalized_questions)
    answer_format_counts = Counter(question["answer_format"] for question in normalized_questions)
    hint_values = [question["table_hint"] for question in normalized_questions if question["table_hint"]]
    id_counts = Counter(question["id"] for question in normalized_questions)
    duplicate_ids = sorted(question_id for question_id, count in id_counts.items() if count > 1)
    blank_id_rows = [question["excel_row"] for question in normalized_questions if not question["id"]]

    manifest = {
        "generated_at": utc_now(),
        "tests": str(tests_path),
        "files_dir": str(files_dir),
        "question_count": len(normalized_questions),
        "blank_id_rows": blank_id_rows,
        "duplicate_question_ids": duplicate_ids,
        "question_type_counts_raw": dict(question_type_counts_raw),
        "question_type_counts": dict(question_type_counts),
        "answer_format_counts": dict(answer_format_counts),
        "table_hint_nonempty": len(hint_values),
        "table_hint_unique": len(set(hint_values)),
        "file_count_on_disk": len(actual_names),
        "referenced_file_count": len(referenced_actual),
        "file_resolutions": resolutions,
        "missing_file_references": [
            resolution for resolution in resolutions.values() if resolution["status"] == "missing"
        ],
        "invalid_file_references": [
            resolution
            for resolution in resolutions.values()
            if resolution["status"] == "invalid_path"
        ],
        "unused_files": sorted(actual_names - referenced_actual),
        "workbook_warnings": workbook_info["warnings"],
        "files": file_records,
        "renders": render_records,
    }
    return manifest, normalized_questions


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# 输入审计报告",
        "",
        f"生成时间（UTC）：`{manifest['generated_at']}`",
        "",
        "## 概览",
        "",
        f"- 题目数：{manifest['question_count']}",
        f"- 被引用文件数：{manifest['referenced_file_count']}",
        f"- 目录文件数：{manifest['file_count_on_disk']}",
        f"- 非空 `table_hint`：{manifest['table_hint_nonempty']}（唯一值 {manifest['table_hint_unique']}）",
        "",
        "## 题型统计",
        "",
        "| 题型 | 数量 |",
        "| --- | ---: |",
    ]
    for key, value in manifest["question_type_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## 答案格式统计", "", "| 格式 | 数量 |", "| --- | ---: |"])
    for key, value in manifest["answer_format_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    if manifest["blank_id_rows"] or manifest["duplicate_question_ids"]:
        lines.extend(["", "## ID 完整性告警", ""])
        if manifest["blank_id_rows"]:
            lines.append(f"- 空 ID 所在 Excel 行：{manifest['blank_id_rows']}")
        if manifest["duplicate_question_ids"]:
            lines.append(f"- 重复 ID：{manifest['duplicate_question_ids']}")

    lines.extend(["", "## 文件名解析", "", "| 清单文件名 | 解析结果 | 状态 |", "| --- | --- | --- |"])
    for requested, resolution in sorted(manifest["file_resolutions"].items()):
        lines.append(
            f"| `{requested}` | `{resolution.get('resolved') or ''}` | `{resolution['status']}` |"
        )

    if manifest["workbook_warnings"]:
        lines.extend(["", "## 清单告警", ""])
        for warning in manifest["workbook_warnings"]:
            lines.append(f"- `{json.dumps(warning, ensure_ascii=False)}`")
    if manifest["missing_file_references"]:
        lines.extend(["", "## 缺失文件", ""])
        for item in manifest["missing_file_references"]:
            lines.append(f"- `{item['requested']}`")
    if manifest["unused_files"]:
        lines.extend(["", "## 未被引用的文件", ""])
        for item in manifest["unused_files"]:
            lines.append(f"- `{item}`")

    if manifest["renders"]:
        lines.extend(["", "## PDF 渲染", "", "| 文件 | 页数 | 状态 |", "| --- | ---: | --- |"])
        for item in manifest["renders"]:
            lines.append(f"| `{item['source']}` | {item.get('pages') or ''} | `{item['status']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    tests_path = args.tests.resolve()
    files_dir = args.files_dir.resolve()
    output_dir = args.output_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    if not tests_path.is_file():
        raise SystemExit(f"tests workbook not found: {tests_path}")
    if not files_dir.is_dir():
        raise SystemExit(f"files directory not found: {files_dir}")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    manifest, questions = build_manifest(
        tests_path,
        files_dir,
        output_dir,
        render_pdfs=args.render_pdfs,
        dpi=args.dpi,
        force_render=args.force_render,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "questions.normalized.json", questions)
    write_json(output_dir / "file_manifest.json", manifest)
    write_json(reports_dir / "input_audit.json", manifest)
    (reports_dir / "input_audit.md").write_text(render_report(manifest), encoding="utf-8")

    print(f"questions: {manifest['question_count']}")
    print(f"question types: {manifest['question_type_counts']}")
    print(f"answer formats: {manifest['answer_format_counts']}")
    print(f"referenced files: {manifest['referenced_file_count']}/{manifest['file_count_on_disk']}")
    print(f"missing references: {len(manifest['missing_file_references'])}")
    print(f"warnings: {len(manifest['workbook_warnings'])}")
    if args.render_pdfs:
        rendered = Counter(item["status"] for item in manifest["renders"])
        print(f"pdf renders: {dict(rendered)}")
    print(f"manifest: {output_dir / 'file_manifest.json'}")
    print(f"report: {reports_dir / 'input_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
