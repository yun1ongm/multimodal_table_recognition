"""Question intent DSL and prompts for the deterministic answer engine."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "question_dsl.v1"
ANSWER_FORMATS = {"string", "number", "json_array", "json"}
OPERATIONS = {"select", "calculate", "count", "list", "boolean", "structure"}
NODE_OPERATIONS = {
    "literal",
    "cell",
    "text",
    "table_field",
    "list",
    "add",
    "sum",
    "subtract",
    "difference",
    "multiply",
    "divide",
    "ratio",
    "average",
    "max",
    "min",
    "count",
    "count_rows",
    "count_cells",
    "span_sum",
    "merge_cells",
    "row_labels",
    "cells",
    "argmax",
    "argmin",
    "compare",
    "contains",
    "structure",
}

_OPERATION_ALIASES = {
    "extract": "select",
    "query": "select",
    "compute": "calculate",
    "calculation": "calculate",
    "filter": "list",
    "判断": "boolean",
    "计算": "calculate",
    "提取": "select",
    "结构": "structure",
}

_NODE_ALIASES = {
    "value": "cell",
    "cell_value": "cell",
    "select_cell": "cell",
    "find_text": "text",
    "field": "table_field",
    "rows": "row_labels",
    "sum_rows": "sum",
    "subtract": "subtract",
    "diff": "difference",
    "eq": "compare",
    "equal": "compare",
    "is_equal": "compare",
    "table_structure": "structure",
}


class DSLValidationError(ValueError):
    """Raised when a model-produced question DSL cannot be executed safely."""


def _canonical_operation(value: Any) -> str:
    operation = str(value or "select").strip().lower()
    return _OPERATION_ALIASES.get(operation, operation)


def _canonical_node_operation(value: Any) -> str:
    operation = str(value or "").strip().lower()
    return _NODE_ALIASES.get(operation, operation)


def _copy_and_normalize_node(node: Any, path: str = "expression") -> dict[str, Any]:
    if not isinstance(node, dict):
        raise DSLValidationError(f"{path} must be an object")
    normalized = deepcopy(node)
    operation = _canonical_node_operation(normalized.get("op"))
    if operation not in NODE_OPERATIONS:
        raise DSLValidationError(f"{path}.op has unsupported value {operation!r}")
    normalized["op"] = operation

    child_keys = {
        "left",
        "right",
        "value",
        "operand",
        "condition",
        "then",
        "else",
    }
    for key in child_keys:
        if key in normalized and isinstance(normalized[key], dict):
            normalized[key] = _copy_and_normalize_node(normalized[key], f"{path}.{key}")
    for key in ("operands", "items", "values"):
        if key in normalized:
            if not isinstance(normalized[key], list):
                raise DSLValidationError(f"{path}.{key} must be an array")
            normalized[key] = [
                _copy_and_normalize_node(item, f"{path}.{key}[{index}]")
                if isinstance(item, dict) and "op" in item
                else item
                for index, item in enumerate(normalized[key])
            ]

    if operation == "cell":
        selectors = ("row_label", "column", "row", "col", "text", "contains", "match")
        if not any(key in normalized for key in selectors):
            raise DSLValidationError(f"{path} cell node has no selector")
    if operation == "table_field" and not normalized.get("field"):
        raise DSLValidationError(f"{path}.field is required")
    if operation in {"list", "sum", "average", "max", "min", "count", "argmax", "argmin", "span_sum", "merge_cells"}:
        if not any(key in normalized for key in ("items", "operands", "values", "selector", "where")):
            raise DSLValidationError(f"{path} requires items, operands, values, selector or where")
    if operation in {"add", "subtract", "difference", "multiply", "divide", "ratio", "compare"}:
        if not (
            ("left" in normalized and "right" in normalized)
            or isinstance(normalized.get("operands"), list)
        ):
            raise DSLValidationError(f"{path} requires left/right or operands")
    if operation == "structure":
        if not normalized.get("full") and not any(
            key in normalized for key in ("rows", "row_indices", "cols", "col_indices", "row_range", "col_range")
        ):
            raise DSLValidationError(f"{path} structure node requires full or a row/column scope")
    return normalized


def normalize_dsl(candidate: Any, question: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and canonicalize one model-produced DSL object.

    The validator is deliberately strict about executable node types and
    deliberately permissive about selector fields.  New table-specific
    selectors can therefore be carried through without allowing arbitrary
    executable code.
    """

    if not isinstance(candidate, dict):
        raise DSLValidationError("question DSL must be an object")
    if isinstance(candidate.get("dsl"), dict):
        candidate = candidate["dsl"]
    operation = _canonical_operation(candidate.get("operation"))
    if operation not in OPERATIONS:
        raise DSLValidationError(f"operation has unsupported value {operation!r}")
    expression = candidate.get("expression")
    if expression is None:
        raise DSLValidationError("expression is required")
    normalized = deepcopy(candidate)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["operation"] = operation
    normalized["expression"] = _copy_and_normalize_node(expression)

    expected_format = question.get("answer_format") if question else None
    candidate_format = normalized.get("answer_format")
    if expected_format and candidate_format and candidate_format != expected_format:
        raise DSLValidationError(
            f"answer_format {candidate_format!r} does not match question format {expected_format!r}"
        )
    answer_format = str(candidate_format or expected_format or "string")
    if answer_format not in ANSWER_FORMATS:
        raise DSLValidationError(f"unsupported answer_format {answer_format!r}")
    normalized["answer_format"] = answer_format
    if question:
        normalized.setdefault("question_id", question.get("id", ""))
        normalized.setdefault("question_type", question.get("question_type", "extract"))
    if not isinstance(normalized.get("notes", []), list):
        raise DSLValidationError("notes must be an array")
    return normalized


def table_context(ir: dict[str, Any], max_chars: int = 24000) -> str:
    """Render compact, coordinate-rich TableIR context for the text model."""

    lines = [
        f"source_file={ir.get('source_file', '')!r}",
        f"title={ir.get('title', '')!r}",
        f"unit={ir.get('unit')!r}",
        f"row_count={ir.get('row_count', 0)} col_count={ir.get('col_count', 0)}",
        "cells:",
    ]
    cells_by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in ir.get("cells", []):
        cells_by_row.setdefault(int(cell.get("row", 0)), []).append(cell)
    for row in sorted(cells_by_row):
        cells = sorted(cells_by_row[row], key=lambda item: int(item.get("col", 0)))
        parts = []
        for cell in cells:
            role = cell.get("role", "")
            span = f"r{cell.get('rowspan', 1)}c{cell.get('colspan', 1)}"
            text = str(cell.get("text", "")).replace("\n", "\\n")
            parts.append(
                f"({cell.get('row', 0)},{cell.get('col', 0)},{span},{role})={text!r}"
            )
        lines.append(f"row {row}: " + " | ".join(parts))
    result = "\n".join(lines)
    if len(result) <= max_chars:
        return result
    return result[: max_chars - 80] + "\n...[TableIR context truncated; use exact coordinates from visible rows]"


def build_compile_prompt(question: dict[str, Any], ir: dict[str, Any]) -> str:
    """Build the Qwen prompt for compiling one natural-language question."""

    examples = r'''
示例 1（单元格提取）
问题：2017 Net product sales 是多少？
输出：{"operation":"select","answer_format":"number","expression":{"op":"cell","row_label":"Net product sales","column":"2017","return":"raw"}}

示例 2（确定性计算）
问题：计算 2017 Total net sales 减 2017 Total operating expenses。
输出：{"operation":"calculate","answer_format":"number","expression":{"op":"subtract","left":{"op":"cell","row_label":"Total net sales","column":"2017","return":"raw"},"right":{"op":"cell","row_label":"Total operating expenses","column":"2017","return":"raw"}}}

示例 3（结构题）
问题：恢复整个表格。
输出：{"operation":"structure","answer_format":"json","expression":{"op":"structure","full":true}}

示例 4（表格元数据）
问题：请给出该表可见主表的列数。
输出：{"operation":"count","answer_format":"number","expression":{"op":"table_field","field":"col_count"}}
'''
    return f"""你是表格问答意图编译器。你只能使用给定的 TableIR，不要直接回答问题，不要猜测图像之外的信息。
请把一条自然语言题目编译成可由本地确定性引擎执行的 question_dsl.v1 JSON。只返回 JSON，不要 Markdown、解释或代码围栏。

题目元数据：
{json.dumps({key: question.get(key) for key in ('id', 'question_type', 'question', 'table_hint', 'answer_format')}, ensure_ascii=False)}

可用 TableIR：
{table_context(ir)}

可用顶层 operation：select、calculate、count、list、boolean、structure。
可用 expression op：literal、cell、text、table_field、list、add、sum、subtract、difference、multiply、divide、ratio、average、max、min、count、count_rows、count_cells、merge_cells、span_sum、row_labels、cells、argmax、argmin、compare、contains、structure。

规则：
1. cell 优先用 TableIR 中可见的精确 row_label、column 和坐标；row_label/column 必须尽量复制原文，不要翻译。
2. 数字计算必须拆成 cell 节点和算术节点，不能把计算结果写死；本地程序使用 Decimal 执行。
3. 多个答案用 list 节点，数字列表设置 item_format="number"，名称/文本列表设置 item_format="string"。
4. 表格行数、列数、标题使用 table_field；判断合并关系或恢复结构使用 structure。
5. 返回原始显示值时 cell 的 return 使用 raw；返回单元格文本时使用 text。
6. 无法从 TableIR 安全定位时仍输出最小 DSL，并在 notes 中说明疑点；不要编造值。
7. cell 的 row/col 是 0 起始的精确坐标；按行标题和列标题定位时使用 row_label/column。
8. 顶层 operation=count 时，expression 应返回 list、cells 或 count_rows；顶层 operation=boolean 时，expression 必须使用 compare/contains 返回布尔值。
9. 需要统计合并跨度时，使用 {{"op":"span_sum","span":"rowspan","selector":{{"rowspan":3}}}}；需要判断是否有同时跨行跨列合并时，使用顶层 operation=boolean 加 expression={{"op":"structure","full":true}}。

{examples}

现在只编译这一题：
{json.dumps(question, ensure_ascii=False)}
"""
