#!/usr/bin/env python3
"""Build and stitch page-level financial TableIR with Qwen."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from build_table_ir import normalize_ir
from dataset_utils import index_files, load_test_questions, resolve_file_name, utc_now, write_json
from prepare_dataset import render_pdf
from qwen_client import QwenClient, QwenConfig
from financial_utils import annotate_financial_ir
from stitch_table_ir import stitch_pages


SYSTEM_PROMPT = """你是财务报表分页解析器。只根据输入页面和目标表名识别表格，不回答题目。
必须返回合法 JSON，不要 Markdown 或解释。保留中文/英文原文、金额逗号、负数括号、百分号和单位。
只输出目标表在当前页面可见的部分；如果当前页完全没有目标表，visible=false 且 cells=[]。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="PDF file name in files/")
    parser.add_argument("--hint", required=True, help="Target table hint/title")
    parser.add_argument("--files-dir", type=Path, default=Path("files"))
    parser.add_argument("--tests", type=Path, default=Path("tests.xlsx"))
    parser.add_argument("--render-dir", type=Path, default=Path("artifacts/prepared/rendered"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/financial_pages"))
    parser.add_argument("--stitched-output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", value).strip("_")
    return slug[:80] or "table"


def build_prompt(file_name: str, page_index: int, hint: str, questions: list[str]) -> str:
    return f"""解析 `{file_name}` 的第 {page_index + 1} 页，只关注目标表：`{hint}`。

相关题目（只用于确定行列和字段，不要直接回答）：{json.dumps(questions[:12], ensure_ascii=False)}

严格返回：
{{
  "visible": true,
  "title": "当前页可见的表题，没有则为空字符串",
  "unit": "当前页声明的单位，没有则为 null",
  "row_count": 逻辑行数,
  "col_count": 逻辑列数,
  "cells": [
    {{"text": "可见文本", "row": 0, "col": 0, "rowspan": 1, "colspan": 1,
     "role": "header|row_label|value|note|other", "header_path": ["多级表头路径"]}}
  ],
  "notes": ["续表、页眉页脚、表头重复、单位或识别疑点"]
}}

规则：
1. 当前页没有目标表时返回 visible=false、cells=[]；不要把其他表格当成目标表。
2. 当前页是续表时，仍从本页表格左上角重新编号；后续程序会去掉重复表头并拼接。
3. 每个真实单元格只输出一次，合并单元格只输出左上角；展开 rowspan/colspan 后不得有任何坐标重叠。
4. 空白值可以省略，但不能省略有意义的行标题、列标题、合计和单位。
"""


def main() -> int:
    args = parse_args()
    if args.dpi <= 0 or args.max_tokens <= 0 or args.schema_retries < 0:
        raise SystemExit("dpi/max-tokens must be positive and schema-retries cannot be negative")
    files_dir = args.files_dir.resolve()
    exact, aliases = index_files(files_dir)
    resolution = resolve_file_name(args.pdf, exact, aliases)
    actual_name = resolution.get("resolved")
    if not actual_name:
        raise SystemExit(f"PDF not found: {args.pdf}")
    pdf_path = exact[actual_name]
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Not a PDF: {pdf_path}")

    questions, _ = load_test_questions(args.tests.resolve())
    target_questions = []
    for question in questions:
        question_resolution = resolve_file_name(question["file_name"], exact, aliases)
        if question_resolution.get("resolved") != actual_name:
            continue
        if question.get("table_hint") in {"", args.hint}:
            target_questions.append(question["question"])
    render_root = args.render_dir.resolve() / pdf_path.stem
    render_record = render_pdf(pdf_path, render_root, dpi=args.dpi, force=False)
    if render_record["status"] == "failed":
        raise SystemExit(render_record["error"])
    page_paths = sorted(render_root.glob("page-*.png"), key=lambda path: int(path.stem.rsplit("-", 1)[-1]))
    if not page_paths:
        raise SystemExit(f"No rendered pages found under {render_root}")

    output_dir = args.output_dir.resolve() / f"{pdf_path.stem}__{safe_slug(args.hint)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"pdf={actual_name} pages={len(page_paths)} hint={args.hint!r}")
        for index, page_path in enumerate(page_paths):
            print(build_prompt(actual_name, index, args.hint, target_questions))
        return 0

    client = QwenClient(QwenConfig.from_env(args.env_file.resolve()))
    page_documents: list[dict[str, Any]] = []
    required_keys = {"visible", "title", "unit", "row_count", "col_count", "cells", "notes"}
    for page_index, page_path in enumerate(page_paths):
        output_path = output_dir / f"page-{page_index + 1:03d}.json"
        raw_path = output_dir / f"page-{page_index + 1:03d}.raw.txt"
        if output_path.exists() and not args.force:
            document = json.loads(output_path.read_text(encoding="utf-8"))
            page_documents.append(document)
            print(f"page {page_index + 1}: skip existing")
            continue
        prompt = build_prompt(actual_name, page_index, args.hint, target_questions)
        current_prompt = prompt
        response: dict[str, Any] | None = None
        normalized: dict[str, Any] | None = None
        for attempt in range(args.schema_retries + 1):
            response = client.chat_json(
                required_keys=required_keys,
                image_path=page_path,
                prompt=current_prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=args.max_tokens,
            )
            raw_path.write_text(response["text"] + "\n", encoding="utf-8")
            candidate = response["json"]
            if not isinstance(candidate, dict):
                error = "page response is not an object"
            else:
                error = None
                try:
                    normalized = normalize_ir(candidate, actual_name)
                except ValueError as exc:
                    error = str(exc)
            if error is None and normalized is not None:
                normalized["visible"] = bool(candidate.get("visible", True))
                normalized["page_index"] = page_index
                normalized["model"] = response["model"]
                normalized["usage"] = response.get("usage")
                normalized["generated_at"] = utc_now()
                write_json(output_path, normalized)
                break
            if attempt == args.schema_retries:
                raise SystemExit(f"page {page_index + 1} TableIR invalid: {error}")
            current_prompt = prompt + f"\n上次输出未通过校验：{error}。请重新输出完整 JSON，禁止坐标重叠。"
        assert normalized is not None
        page_documents.append(normalized)
        print(f"page {page_index + 1}: visible={normalized.get('visible', True)} -> {output_path}")

    visible_documents = [document for document in page_documents if document.get("visible", True)]
    if not visible_documents:
        raise SystemExit("Qwen found no visible target table pages")
    stitched = stitch_pages(visible_documents, actual_name)
    stitched = annotate_financial_ir(stitched)
    stitched["target_hint"] = args.hint
    stitched["page_inputs"] = [str(path) for path in page_paths]
    stitched["generated_at"] = utc_now()
    stitched_output = args.stitched_output.resolve() if args.stitched_output else Path(
        "artifacts/table_ir"
    ).resolve() / f"{pdf_path.stem}__{safe_slug(args.hint)}.financial.json"
    write_json(stitched_output, stitched)
    print(
        f"stitched visible_pages={len(visible_documents)} rows={stitched['row_count']} "
        f"cols={stitched['col_count']} cells={len(stitched['cells'])} -> {stitched_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
