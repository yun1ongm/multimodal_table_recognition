#!/usr/bin/env python3
"""Build a visual TableIR cache with the supplied Qwen multimodal API.

Start with a few representative files to control API cost:

    python scripts/build_table_ir.py --file 078.png --file 090.png --file 093.jpg

The output is an internal cache, not a submission workbook.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset_utils import index_files, load_test_questions, resolve_file_name, utc_now, write_json
from qwen_client import QwenClient, QwenConfig


SYSTEM_PROMPT = """你是复杂表格视觉解析器。你只能根据输入图像和题目上下文工作，不要猜测图像之外的信息。
请输出一个合法 JSON 对象，禁止 Markdown、解释文字和代码围栏。要保留原文语言、大小写、数字、括号、单位和符号。
把视觉上的合并单元格表示为左上角真实单元格，并填写 rowspan/colspan；被覆盖的位置不要重复输出。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-dir", type=Path, default=Path("files"))
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/table_ir"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--file", dest="requested_files", action="append", help="File name; repeatable")
    parser.add_argument("--limit", type=int, help="Limit automatically selected files")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_ir(candidate: Any, source_name: str) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("TableIR must be a JSON object")
    required = ("title", "unit", "row_count", "col_count", "cells", "notes")
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ValueError(f"TableIR is missing fields: {', '.join(missing)}")
    if not isinstance(candidate["title"], str):
        raise ValueError("TableIR.title must be a string")
    if candidate["unit"] is not None and not isinstance(candidate["unit"], str):
        raise ValueError("TableIR.unit must be a string or null")
    if not isinstance(candidate["row_count"], int) or candidate["row_count"] <= 0:
        raise ValueError("TableIR.row_count must be a positive integer")
    if not isinstance(candidate["col_count"], int) or candidate["col_count"] <= 0:
        raise ValueError("TableIR.col_count must be a positive integer")
    if not isinstance(candidate["cells"], list):
        raise ValueError("TableIR.cells must be an array")
    if not isinstance(candidate["notes"], list) or not all(isinstance(note, str) for note in candidate["notes"]):
        raise ValueError("TableIR.notes must be an array of strings")

    cells: list[dict[str, Any]] = []
    occupied: dict[tuple[int, int], int] = {}
    for index, original in enumerate(candidate["cells"]):
        if not isinstance(original, dict):
            raise ValueError(f"cells[{index}] must be an object")
        for field in ("text", "row", "col", "rowspan", "colspan"):
            if field not in original:
                raise ValueError(f"cells[{index}] missing {field}")
        if not isinstance(original["text"], str):
            raise ValueError(f"cells[{index}].text must be a string")
        if any(
            not isinstance(original[field], int) or isinstance(original[field], bool)
            for field in ("row", "col", "rowspan", "colspan")
        ):
            raise ValueError(f"cells[{index}] row/col/span fields must be integers")
        row, col = original["row"], original["col"]
        rowspan, colspan = original["rowspan"], original["colspan"]
        if row < 0 or col < 0 or rowspan <= 0 or colspan <= 0:
            raise ValueError(f"cells[{index}] has invalid position or span")
        if row + rowspan > candidate["row_count"] or col + colspan > candidate["col_count"]:
            raise ValueError(f"cells[{index}] exceeds logical table bounds")
        for occupied_row in range(row, row + rowspan):
            for occupied_col in range(col, col + colspan):
                key = (occupied_row, occupied_col)
                if key in occupied:
                    raise ValueError(f"cells[{index}] overlaps cells[{occupied[key]}] at {key}")
                occupied[key] = index
        cell = {
            "text": original["text"],
            "row": row,
            "col": col,
            "rowspan": rowspan,
            "colspan": colspan,
        }
        for optional in ("role", "header_path", "bbox", "fill_color", "confidence", "source_page"):
            if optional in original:
                cell[optional] = original[optional]
        cells.append(cell)

    normalized = {
        "schema_version": "table_ir.v1",
        "source_file": source_name,
        "title": candidate["title"],
        "unit": candidate["unit"],
        "row_count": candidate["row_count"],
        "col_count": candidate["col_count"],
        "cells": sorted(cells, key=lambda cell: (cell["row"], cell["col"])),
        "notes": candidate["notes"],
    }
    for optional in ("table_bbox", "page_index", "confidence"):
        if optional in candidate:
            normalized[optional] = candidate[optional]
    return normalized


def build_prompt(file_name: str, table_hints: list[str], questions: list[str]) -> str:
    hints = json.dumps(table_hints, ensure_ascii=False)
    context = json.dumps(questions[:8], ensure_ascii=False)
    return f"""请解析图片 `{file_name}` 中的目标表格，生成完整的 TableIR。

表格提示（可能为空或不完全可靠）：{hints}
相关问题（只用于定位字段，不要直接回答）：{context}

严格返回以下 JSON 结构，不要增加顶层字段：
{{
  "title": "表格标题；找不到时为空字符串",
  "unit": "表格或表头声明的单位；没有则为 null",
  "row_count": 逻辑行数,
  "col_count": 逻辑列数,
  "cells": [
    {{
      "text": "单元格可见文本",
      "row": 0,
      "col": 0,
      "rowspan": 1,
      "colspan": 1,
      "role": "header|row_label|value|note|other",
      "header_path": ["继承后的多级表头，从上到下"]
    }}
  ],
  "notes": ["页眉、脚注、续表、旋转或识别不确定性等说明"]
}}

规则：
1. row/col 从 0 开始；row_count/col_count 是完整逻辑表尺寸。
2. 每个真实单元格只输出一次；合并单元格只输出左上角，并填写真实 rowspan/colspan。
3. 普通空白占位格不要伪造文本；但如果空白格承担合并关系，仍需按真实合并区域表达。
4. 保留金额、百分比、日期、负数括号、上下标可见文本和中英文大小写，不要自行换算。
5. 如果图片中有多个表格，优先选择提示和相关问题共同指向的表；如果无法确定，在 notes 中说明。
6. 输出前在脑中展开每个单元格覆盖的坐标，任何两个单元格不得覆盖同一坐标。只有同一个真实单元格跨越多行时才使用 rowspan；相邻行有不同可见文本时必须拆成不同单元格。
"""


def selected_files(args: argparse.Namespace, files_dir: Path, tests_path: Path) -> list[tuple[str, Path, list[str], list[str]]]:
    exact, aliases = index_files(files_dir)
    questions, _ = load_test_questions(tests_path)
    grouped_hints: dict[str, set[str]] = defaultdict(set)
    grouped_questions: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        resolution = resolve_file_name(question["file_name"], exact, aliases)
        resolved = resolution.get("resolved")
        if not resolved:
            continue
        grouped_hints[resolved].add(question["table_hint"])
        grouped_questions[resolved].append(question["question"])

    if args.requested_files:
        requested = args.requested_files
    else:
        # The second implementation stage focuses on single-page images in
        # the 011-057 and 078-093 ranges.  Keep only files actually referenced
        # by the question list.
        requested = sorted(
            name
            for name in grouped_questions
            if name.split(".", 1)[0].isdigit()
            and (11 <= int(name.split(".", 1)[0]) <= 57 or 78 <= int(name.split(".", 1)[0]) <= 93)
        )
    selected: list[tuple[str, Path, list[str], list[str]]] = []
    seen: set[str] = set()
    for requested_name in requested:
        resolution = resolve_file_name(requested_name, exact, aliases)
        resolved = resolution.get("resolved")
        if not resolved or resolved in seen:
            continue
        path = exact[resolved]
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        seen.add(resolved)
        hints = sorted(hint for hint in grouped_hints.get(resolved, set()) if hint)
        selected.append((resolved, path, hints, grouped_questions.get(resolved, [])))
        if args.limit and len(selected) >= args.limit:
            break
    return selected


def main() -> int:
    args = parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.schema_retries < 0:
        raise SystemExit("--schema-retries cannot be negative")
    files_dir = args.files_dir.resolve()
    output_dir = args.output_dir.resolve()
    selected = selected_files(args, files_dir, args.tests.resolve())
    if not selected:
        raise SystemExit("No matching image files selected")

    config = None if args.dry_run else QwenConfig.from_env(args.env_file.resolve())
    client = QwenClient(config) if config else None
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"selected files: {len(selected)}")
    for file_name, path, hints, questions in selected:
        output_path = output_dir / f"{path.stem}.json"
        raw_path = output_dir / f"{path.stem}.raw.txt"
        error_path = output_dir / f"{path.stem}.error.json"
        prompt = build_prompt(file_name, hints, questions)
        print(f"[{file_name}] hints={len(hints)} questions={len(questions)}")
        if args.dry_run:
            print(prompt)
            continue
        if output_path.exists() and not args.force:
            print(f"[{file_name}] skip existing {output_path}")
            continue
        assert client is not None
        response: dict[str, Any] | None = None
        try:
            required_keys = {"title", "unit", "row_count", "col_count", "cells", "notes"}
            current_prompt = prompt
            normalized: dict[str, Any] | None = None
            schema_error: Exception | None = None
            for attempt in range(args.schema_retries + 1):
                response = client.chat_json(
                    required_keys=required_keys,
                    image_path=path,
                    prompt=current_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    model=args.model,
                    max_tokens=args.max_tokens,
                )
                attempt_raw_path = raw_path if attempt == args.schema_retries else output_dir / f"{path.stem}.attempt{attempt + 1}.raw.txt"
                attempt_raw_path.write_text(response["text"] + "\n", encoding="utf-8")
                try:
                    normalized = normalize_ir(response["json"], file_name)
                    break
                except ValueError as exc:
                    schema_error = exc
                    if attempt == args.schema_retries:
                        raise
                    current_prompt = (
                        prompt
                        + "\n\n上一次输出未通过结构校验，必须重新完整输出。具体错误："
                        + str(exc)
                        + "\n请重新检查每个单元格的逻辑坐标覆盖，禁止重叠；不要解释，只返回 JSON。"
                    )
            if normalized is None:
                raise schema_error or ValueError("TableIR normalization returned no result")
            normalized["generated_at"] = utc_now()
            normalized["model"] = response["model"]
            normalized["usage"] = response.get("usage")
            write_json(output_path, normalized)
            if error_path.exists():
                error_path.unlink()
            print(f"[{file_name}] ok -> {output_path}")
        except Exception as exc:
            if response is not None and not raw_path.exists():
                raw_path.write_text(response.get("text", "") + "\n", encoding="utf-8")
            error = {"file": file_name, "error": str(exc), "generated_at": utc_now()}
            write_json(error_path, error)
            print(f"[{file_name}] ERROR: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
