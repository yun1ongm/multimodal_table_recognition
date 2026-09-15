"""Deterministic execution engine for question_dsl.v1."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Iterable

from financial_utils import decimal_to_string, parse_financial_value
from question_dsl import normalize_dsl


class AnswerEngineError(ValueError):
    """Raised when a DSL cannot be resolved unambiguously from a TableIR."""


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[`'\"“”‘’.,，:：;；/／\\|()（）\[\]【】{}<>《》]+", "", text)


def _score_match(target: Any, candidate: Any) -> int:
    left = normalize_text(target)
    right = normalize_text(candidate)
    if not left or not right:
        return 0
    if left == right:
        return 100
    if left in right:
        return 82 + min(12, int(10 * len(left) / len(right)))
    if right in left and len(right) >= 2:
        return 72 + min(10, int(10 * len(right) / len(left)))
    ratio = SequenceMatcher(None, left, right).ratio()
    return int(ratio * 70) if ratio >= 0.72 else 0


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass(frozen=True)
class CellRef:
    cell: dict[str, Any]
    row_label: str
    column_label: str

    @property
    def text(self) -> str:
        return str(self.cell.get("text", ""))

    @property
    def row(self) -> int:
        return int(self.cell.get("row", 0))

    @property
    def col(self) -> int:
        return int(self.cell.get("col", 0))

    def as_source(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "row": self.row,
            "col": self.col,
            "row_label": self.row_label,
            "column_label": self.column_label,
            "source_page": self.cell.get("source_page"),
        }


class TableView:
    """Indexes a TableIR while retaining the original cell text."""

    def __init__(self, ir: dict[str, Any]):
        self.ir = ir
        self.cells: list[dict[str, Any]] = sorted(
            [cell for cell in ir.get("cells", []) if isinstance(cell, dict)],
            key=lambda cell: (int(cell.get("row", 0)), int(cell.get("col", 0))),
        )
        self.rows: dict[int, list[dict[str, Any]]] = {}
        self._cell_at: dict[tuple[int, int], dict[str, Any]] = {}
        for cell in self.cells:
            row = int(cell.get("row", 0))
            col = int(cell.get("col", 0))
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            self.rows.setdefault(row, []).append(cell)
            for occupied_row in range(row, row + rowspan):
                for occupied_col in range(col, col + colspan):
                    self._cell_at.setdefault((occupied_row, occupied_col), cell)

        self.header_by_col: dict[int, list[str]] = {}
        for cell in self.cells:
            if cell.get("role") != "header":
                continue
            labels = [str(cell.get("text", ""))]
            path = cell.get("header_path")
            if isinstance(path, list):
                labels.extend(str(item) for item in path)
            for col in range(int(cell.get("col", 0)), int(cell.get("col", 0)) + int(cell.get("colspan", 1))):
                self.header_by_col[col] = _unique_preserving_order(
                    [*self.header_by_col.get(col, []), *labels]
                )

    def row_texts(self, row: int) -> list[str]:
        return [str(cell.get("text", "")) for cell in self.rows.get(row, []) if str(cell.get("text", ""))]

    def row_label(self, row: int) -> str:
        cells = self.rows.get(row, [])
        preferred = [
            str(cell.get("text", ""))
            for cell in cells
            if cell.get("role") == "row_label" and str(cell.get("text", ""))
        ]
        if preferred:
            return preferred[0]
        non_values = [
            str(cell.get("text", ""))
            for cell in cells
            if cell.get("role") not in {"value", "note"} and str(cell.get("text", ""))
        ]
        if non_values:
            return non_values[0]
        texts = self.row_texts(row)
        return texts[0] if texts else ""

    def column_labels(self, col: int) -> list[str]:
        return list(self.header_by_col.get(col, []))

    def column_label(self, col: int) -> str:
        labels = self.column_labels(col)
        return labels[-1] if labels else str(col)

    def cell_column_labels(self, cell: dict[str, Any]) -> list[str]:
        labels: list[str] = []
        for col in range(int(cell.get("col", 0)), int(cell.get("col", 0)) + int(cell.get("colspan", 1))):
            labels.extend(self.column_labels(col))
        path = cell.get("header_path")
        if isinstance(path, list):
            labels.extend(str(item) for item in path)
        row = int(cell.get("row", 0))
        cell_col = int(cell.get("col", 0))
        preceding_headers = [
            candidate
            for candidate in self.rows.get(row, [])
            if candidate.get("role") == "header"
            and int(candidate.get("col", 0)) + int(candidate.get("colspan", 1)) <= cell_col
        ]
        if preceding_headers:
            nearest = max(preceding_headers, key=lambda candidate: int(candidate.get("col", 0)))
            labels.append(str(nearest.get("text", "")))
        return _unique_preserving_order([label for label in labels if label])

    def make_ref(self, cell: dict[str, Any]) -> CellRef:
        labels = self.cell_column_labels(cell)
        return CellRef(
            cell=cell,
            row_label=self.row_label(int(cell.get("row", 0))),
            column_label=labels[-1] if labels else self.column_label(int(cell.get("col", 0))),
        )

    def _row_score(self, target: Any, row: int) -> int:
        return max((_score_match(target, text) for text in self.row_texts(row)), default=0)

    def _column_score(self, target: Any, cell: dict[str, Any]) -> int:
        if isinstance(target, int) or (isinstance(target, str) and target.strip().isdigit()):
            target_col = int(target)
            if 0 <= target_col < int(self.ir.get("col_count", 0)):
                start = int(cell.get("col", 0))
                end = start + int(cell.get("colspan", 1))
                return 100 if start <= target_col < end else 0
        labels = self.cell_column_labels(cell)
        return max((_score_match(target, label) for label in labels), default=0)

    def _form_pair_bonus(self, target: Any, cell: dict[str, Any]) -> int:
        """Prefer the value paired with a matching label in form-like rows."""

        if cell.get("role") != "value":
            return 0
        row = int(cell.get("row", 0))
        cell_col = int(cell.get("col", 0))
        labels = [
            candidate
            for candidate in self.rows.get(row, [])
            if candidate.get("role") in {"header", "row_label", "other"}
            and int(candidate.get("col", 0)) + int(candidate.get("colspan", 1)) <= cell_col
        ]
        if not labels:
            return 0
        best = 0
        for label in labels:
            label_score = _score_match(target, label.get("text", ""))
            if label_score <= 0:
                continue
            distance = cell_col - (int(label.get("col", 0)) + int(label.get("colspan", 1)))
            best = max(best, 110 + label_score - min(distance, 20))
        return best

    def _role_bonus(self, cell: dict[str, Any], selector: dict[str, Any]) -> int:
        requested = selector.get("role")
        if requested:
            return 10 if cell.get("role") == requested else -1000
        return 8 if cell.get("role") == "value" else 0

    def resolve_cells(self, selector: dict[str, Any]) -> list[CellRef]:
        if not isinstance(selector, dict):
            raise AnswerEngineError("cell selector must be an object")
        if isinstance(selector.get("row"), int) and isinstance(selector.get("col"), int):
            cell = self._cell_at.get((selector["row"], selector["col"]))
            if cell is None:
                row = int(selector["row"])
                col = int(selector["col"])
                if 0 <= row < int(self.ir.get("row_count", 0)) and 0 <= col < int(self.ir.get("col_count", 0)):
                    return [
                        self.make_ref(
                            {
                                "text": "",
                                "row": row,
                                "col": col,
                                "rowspan": 1,
                                "colspan": 1,
                                "role": "value",
                                "virtual": True,
                            }
                        )
                    ]
                # Some sparse OCR layouts contain one logical cell in a row,
                # while the model's coordinate estimate follows the visual
                # column position and lands outside the compact IR grid.  A
                # unique row cell is a safe recovery; ambiguous rows must
                # still fail instead of guessing.
                if 0 <= row < int(self.ir.get("row_count", 0)):
                    row_cells = self.rows.get(row, [])
                    if len(row_cells) == 1:
                        return [self.make_ref(row_cells[0])]
                return []
            return [self.make_ref(cell)]

        target_row_index = selector.get("row") if isinstance(selector.get("row"), int) else None
        target_col_index = selector.get("col") if isinstance(selector.get("col"), int) else None
        target_row = selector.get("row_label", selector.get("row_text"))
        target_column = selector.get("column", selector.get("column_label"))
        if target_col_index is None and isinstance(target_column, int):
            target_col_index = target_column
            target_column = None
        target_text = selector.get("text")
        target_contains = selector.get("contains")
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for index, cell in enumerate(self.cells):
            if selector.get("role") and cell.get("role") != selector.get("role"):
                continue
            text = str(cell.get("text", ""))
            row = int(cell.get("row", 0))
            score = self._role_bonus(cell, selector)
            for span_key in ("rowspan", "colspan"):
                if span_key in selector and int(cell.get(span_key, 1)) != int(selector[span_key]):
                    score = -1000
            if score <= -1000:
                continue
            if target_row_index is not None and not (
                row <= target_row_index < row + int(cell.get("rowspan", 1))
            ):
                continue
            if target_col_index is not None and not (
                int(cell.get("col", 0))
                <= target_col_index
                < int(cell.get("col", 0)) + int(cell.get("colspan", 1))
            ):
                continue
            if target_row is not None:
                row_score = self._row_score(target_row, row)
                if row_score <= 0:
                    continue
                score += row_score
                score += self._form_pair_bonus(target_row, cell)
            if target_column is not None:
                column_score = self._column_score(target_column, cell)
                if column_score <= 0:
                    continue
                score += column_score
            if target_text is not None:
                text_score = _score_match(target_text, text)
                if text_score <= 0:
                    continue
                score += text_score
            if target_contains is not None:
                if normalize_text(target_contains) not in normalize_text(text):
                    continue
                score += 80
            if target_row is None and target_column is None and target_text is None and target_contains is None:
                score += 1
            candidates.append((score, -index, cell))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        refs: list[CellRef] = []
        seen: set[int] = set()
        for score, _, cell in candidates:
            if score <= 0:
                continue
            marker = id(cell)
            if marker in seen:
                continue
            seen.add(marker)
            refs.append(self.make_ref(cell))
        return refs

    def resolve_one(self, selector: dict[str, Any]) -> CellRef:
        refs = self.resolve_cells(selector)
        if not refs:
            raise AnswerEngineError(f"no cell matched selector: {selector}")
        return refs[0]

    def matching_rows(self, where: Any = None) -> list[int]:
        if where is None or where == {}:
            return sorted(self.rows)
        if isinstance(where, str):
            where = {"label_contains": where}
        if not isinstance(where, dict):
            raise AnswerEngineError("row filter must be an object or string")
        result: list[int] = []
        for row in sorted(self.rows):
            if where.get("start_row") is not None and row < int(where["start_row"]):
                continue
            if where.get("end_row") is not None and row > int(where["end_row"]):
                continue
            texts = self.row_texts(row)
            joined = " | ".join(texts)
            include = True
            if where.get("label") is not None:
                include = self._row_score(where["label"], row) >= 72
            if where.get("label_contains") is not None:
                include = include and normalize_text(where["label_contains"]) in normalize_text(joined)
            if where.get("text_contains") is not None:
                include = include and normalize_text(where["text_contains"]) in normalize_text(joined)
            if where.get("not_contains") is not None:
                include = include and normalize_text(where["not_contains"]) not in normalize_text(joined)
            if where.get("row") is not None:
                include = include and row == int(where["row"])
            if include:
                result.append(row)
        return result

    def structure(self, node: dict[str, Any]) -> dict[str, Any]:
        full = bool(node.get("full"))
        row_count = int(self.ir.get("row_count", 0))
        col_count = int(self.ir.get("col_count", 0))
        row_scope = node.get("rows", node.get("row_indices"))
        col_scope = node.get("cols", node.get("col_indices"))
        if row_scope is None and node.get("row_range") is not None:
            row_scope = self._expand_range(node["row_range"], row_count)
        if col_scope is None and node.get("col_range") is not None:
            col_scope = self._expand_range(node["col_range"], col_count)
        selected_rows = set(range(row_count)) if full or row_scope is None else self._as_index_set(row_scope)
        selected_cols = set(range(col_count)) if full or col_scope is None else self._as_index_set(col_scope)

        output_cells: list[dict[str, Any]] = []
        for cell in self.cells:
            row = int(cell.get("row", 0))
            col = int(cell.get("col", 0))
            rows = set(range(row, row + int(cell.get("rowspan", 1))))
            cols = set(range(col, col + int(cell.get("colspan", 1))))
            if not (rows & selected_rows and cols & selected_cols):
                continue
            output_cells.append(
                {
                    "text": str(cell.get("text", "")),
                    "row": row,
                    "col": col,
                    "rowspan": int(cell.get("rowspan", 1)),
                    "colspan": int(cell.get("colspan", 1)),
                }
            )
        return {
            "row_count": row_count,
            "col_count": col_count,
            "cells": sorted(output_cells, key=lambda item: (item["row"], item["col"])),
        }

    @staticmethod
    def _as_index_set(value: Any) -> set[int]:
        if not isinstance(value, list):
            raise AnswerEngineError("structure indices must be an array")
        return {int(item) for item in value}

    @staticmethod
    def _expand_range(value: Any, limit: int) -> list[int]:
        if not isinstance(value, list) or len(value) != 2:
            raise AnswerEngineError("structure range must contain [start, end]")
        start, end = int(value[0]), int(value[1])
        if start < 0 or end < start:
            raise AnswerEngineError("invalid structure range")
        return list(range(start, min(end, limit)))


class AnswerEngine:
    def __init__(self, ir: dict[str, Any]):
        self.view = TableView(ir)
        self.sources: list[dict[str, Any]] = []

    def answer(self, question: dict[str, Any], dsl: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_dsl(dsl, question)
        self.sources = []
        expression = normalized["expression"]
        if normalized["operation"] == "structure" and expression.get("op") == "cells":
            # Some model outputs use `cells` as the structure selector.  It
            # carries the same row/column scope, so normalize it here rather
            # than serializing a list of CellRef objects as JSON.
            value = self.view.structure(expression)
        else:
            value = self.evaluate(expression)
        value = self._apply_top_operation(normalized["operation"], value)
        answer = self.format_answer(value, normalized["answer_format"], normalized)
        return {
            "id": question.get("id", normalized.get("question_id", "")),
            "answer": answer,
            "sources": self.sources,
            "dsl": normalized,
        }

    @staticmethod
    def _apply_top_operation(operation: str, value: Any) -> Any:
        if operation == "count":
            if isinstance(value, (list, dict, str, tuple, set)):
                return len(value)
            return value
        if operation == "boolean":
            if isinstance(value, dict) and isinstance(value.get("cells"), list):
                value = any(
                    int(cell.get("rowspan", 1)) > 1 and int(cell.get("colspan", 1)) > 1
                    for cell in value["cells"]
                    if isinstance(cell, dict)
                )
            if isinstance(value, bool):
                return "是" if value else "否"
            return value
        return value

    def evaluate(self, node: Any) -> Any:
        if not isinstance(node, dict) or "op" not in node:
            return node
        operation = node["op"]
        if operation == "literal":
            return node.get("value")
        if operation == "cell":
            ref = self.view.resolve_one(node)
            self.sources.append(ref.as_source())
            return self._cell_return(ref, node.get("return", "raw"))
        if operation == "text":
            selector = dict(node)
            selector.pop("op", None)
            if "text" not in selector and "contains" not in selector and "value" in selector:
                selector["text"] = selector["value"]
            ref = self.view.resolve_one(selector)
            self.sources.append(ref.as_source())
            return self._cell_return(ref, node.get("return", "text"))
        if operation == "table_field":
            return self._table_field(str(node.get("field", "")))
        if operation in {"list", "cells"}:
            if operation == "cells" and node.get("selector") is not None:
                refs = self.view.resolve_cells(node["selector"])
                self.sources.extend(ref.as_source() for ref in refs)
                return refs
            if operation == "cells" and any(
                key in node for key in ("rows", "row_indices", "cols", "col_indices")
            ):
                row_scope = node.get("rows", node.get("row_indices"))
                col_scope = node.get("cols", node.get("col_indices"))
                if row_scope is None and isinstance(node.get("row"), int):
                    row_scope = [node["row"]]
                if col_scope is None and isinstance(node.get("col"), int):
                    col_scope = [node["col"]]
                rows = (
                    self.view._as_index_set(row_scope)
                    if row_scope is not None
                    else set(range(int(self.view.ir.get("row_count", 0))))
                )
                cols = (
                    self.view._as_index_set(col_scope)
                    if col_scope is not None
                    else set(range(int(self.view.ir.get("col_count", 0))))
                )
                refs: list[CellRef] = []
                seen: set[int] = set()
                for row in sorted(rows):
                    for col in sorted(cols):
                        for ref in self.view.resolve_cells({"row": row, "col": col}):
                            marker = id(ref.cell)
                            if marker in seen:
                                continue
                            seen.add(marker)
                            refs.append(ref)
                self.sources.extend(ref.as_source() for ref in refs)
                return refs
            items = node.get("items", node.get("values", []))
            if not isinstance(items, list):
                raise AnswerEngineError(f"{operation} requires an array")
            return [self.evaluate(item) for item in items]
        if operation in {"add", "sum", "subtract", "difference", "multiply", "divide", "ratio", "average", "max", "min"}:
            return self._arithmetic(operation, node)
        if operation in {"count", "count_rows", "count_cells"}:
            return self._count(operation, node)
        if operation == "merge_cells":
            selector = dict(node.get("selector", {}))
            for key in ("rowspan", "colspan", "row_label", "column", "text", "contains"):
                if key in node:
                    selector[key] = node[key]
            refs = self.view.resolve_cells(selector)
            self.sources.extend(ref.as_source() for ref in refs)
            return refs
        if operation == "span_sum":
            selector = dict(node.get("selector", {}))
            for key in ("rowspan", "colspan", "row_label", "column", "text", "contains"):
                if key in node:
                    selector[key] = node[key]
            refs = self.view.resolve_cells(selector)
            self.sources.extend(ref.as_source() for ref in refs)
            span = str(node.get("span", "rowspan"))
            if span not in {"rowspan", "colspan"}:
                raise AnswerEngineError("span_sum span must be rowspan or colspan")
            return sum(int(ref.cell.get(span, 1)) for ref in refs)
        if operation == "row_labels":
            rows = self.view.matching_rows(node.get("where", node.get("selector")))
            return [self.view.row_label(row) for row in rows if self.view.row_label(row)]
        if operation in {"argmax", "argmin"}:
            return self._arg_extreme(operation, node)
        if operation == "compare":
            return self._compare(node)
        if operation == "contains":
            value = self.evaluate(node.get("value"))
            needle = str(self.evaluate(node.get("needle", node.get("contains", ""))))
            return normalize_text(needle) in normalize_text(self._as_text(value))
        if operation == "structure":
            return self.view.structure(node)
        raise AnswerEngineError(f"unsupported expression operation: {operation}")

    def _cell_return(self, ref: CellRef, mode: str) -> Any:
        if mode in {"raw", "text", "value"}:
            if ref.cell.get("virtual") and not ref.text:
                return ref
            return ref.text
        if mode in {"number", "numeric"}:
            return self._to_decimal(ref)
        if mode in {"row_label", "row"}:
            return ref.row_label
        if mode in {"column_label", "column"}:
            return ref.column_label
        if mode == "cell":
            return ref
        raise AnswerEngineError(f"unsupported cell return mode: {mode}")

    def _table_field(self, field: str) -> Any:
        ir = self.view.ir
        if field in {"title", "unit", "row_count", "col_count"}:
            return ir.get(field)
        header_cells = [cell for cell in self.view.cells if cell.get("role") == "header" and cell.get("text", "")]
        if field in {"leftmost_header", "first_header"}:
            if not header_cells:
                return ""
            cell = min(header_cells, key=lambda item: (int(item.get("col", 0)), int(item.get("row", 0))))
            return str(cell.get("text", ""))
        if field in {"rightmost_header", "last_header"}:
            if not header_cells:
                return ""
            cell = max(header_cells, key=lambda item: (int(item.get("col", 0)), -int(item.get("row", 0))))
            return str(cell.get("text", ""))
        if field == "language":
            text = " ".join(str(cell.get("text", "")) for cell in self.view.cells)
            has_cjk = bool(re.search(r"[\u3400-\u9fff]", text))
            has_latin = bool(re.search(r"[A-Za-z]", text))
            if has_cjk and has_latin:
                return "中文和英文"
            if has_cjk:
                return "中文"
            if has_latin:
                return "英文"
            return ""
        if field in {"visible_rows", "rows"}:
            return len(self.view.rows)
        raise AnswerEngineError(f"unsupported table field: {field}")

    def _arithmetic(self, operation: str, node: dict[str, Any]) -> Decimal:
        if operation in {"subtract", "difference", "divide", "ratio"}:
            operands = [node.get("left"), node.get("right")]
        else:
            operands = node.get("operands", node.get("items", node.get("values")))
            if operands is None and "left" in node and "right" in node:
                operands = [node["left"], node["right"]]
            if operands is None and "value" in node:
                operands = [node["value"]]
        if not isinstance(operands, list) or not operands:
            raise AnswerEngineError(f"{operation} requires operands")
        numbers = [self._to_decimal(self.evaluate(operand)) for operand in operands]
        if operation in {"add", "sum"}:
            return sum(numbers, Decimal("0"))
        if operation in {"subtract", "difference"}:
            if len(numbers) != 2:
                raise AnswerEngineError("subtract/difference requires exactly two operands")
            return numbers[0] - numbers[1]
        if operation == "multiply":
            result = Decimal("1")
            for number in numbers:
                result *= number
            return result
        if operation in {"divide", "ratio"}:
            if len(numbers) != 2:
                raise AnswerEngineError("divide/ratio requires exactly two operands")
            if numbers[1] == 0:
                raise AnswerEngineError("division by zero")
            result = numbers[0] / numbers[1]
            if node.get("as_percent"):
                result *= Decimal("100")
            return result
        if operation == "average":
            return sum(numbers, Decimal("0")) / Decimal(len(numbers))
        if operation == "max":
            return max(numbers)
        if operation == "min":
            return min(numbers)
        raise AnswerEngineError(f"unsupported arithmetic operation: {operation}")

    def _count(self, operation: str, node: dict[str, Any]) -> int:
        if operation == "count_rows":
            return len(self.view.matching_rows(node.get("where", node.get("selector"))))
        if operation == "count_cells":
            selector = node.get("selector", node.get("where", {}))
            refs = self.view.resolve_cells(selector)
            self.sources.extend(ref.as_source() for ref in refs)
            return len(refs)
        items = node.get("items", node.get("values"))
        if isinstance(items, list):
            return len(items)
        if node.get("where") is not None:
            return len(self.view.matching_rows(node["where"]))
        raise AnswerEngineError("count requires items, values or where")

    def _arg_extreme(self, operation: str, node: dict[str, Any]) -> Any:
        items = node.get("items", node.get("operands", node.get("values")))
        if not isinstance(items, list) or not items:
            raise AnswerEngineError(f"{operation} requires a non-empty items array")
        evaluated = [self.evaluate(item) for item in items]
        pairs = [(self._to_decimal(value), value) for value in evaluated]
        chosen_index, (_, chosen) = (max if operation == "argmax" else min)(
            enumerate(pairs), key=lambda pair: pair[1][0]
        )
        labels = node.get("labels")
        if isinstance(labels, list) and len(labels) == len(items):
            return self.evaluate(labels[chosen_index])
        if node.get("return") in {"row", "row_label"}:
            if isinstance(chosen, CellRef):
                return chosen.row_label
            raise AnswerEngineError("argmax row return requires cell operands")
        return chosen

    def _compare(self, node: dict[str, Any]) -> Any:
        left = self.evaluate(node.get("left"))
        right = self.evaluate(node.get("right"))
        operator = str(node.get("operator", "equals")).lower()
        try:
            left_number = self._to_decimal(left)
            right_number = self._to_decimal(right)
            comparable_left: Any = left_number
            comparable_right: Any = right_number
        except AnswerEngineError:
            comparable_left = self._as_text(left)
            comparable_right = self._as_text(right)
        if operator in {"equals", "eq", "=="}:
            result = comparable_left == comparable_right
        elif operator in {"not_equals", "ne", "!="}:
            result = comparable_left != comparable_right
        elif operator in {"greater", "gt", ">"}:
            result = comparable_left > comparable_right
        elif operator in {"greater_equal", "ge", ">="}:
            result = comparable_left >= comparable_right
        elif operator in {"less", "lt", "<"}:
            result = comparable_left < comparable_right
        elif operator in {"less_equal", "le", "<="}:
            result = comparable_left <= comparable_right
        else:
            raise AnswerEngineError(f"unsupported comparison operator: {operator}")
        if result:
            return node.get("true_value", True)
        return node.get("false_value", False)

    def _to_decimal(self, value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, bool):
            return Decimal(1 if value else 0)
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            return Decimal(str(value))
        if isinstance(value, CellRef):
            if value.cell.get("virtual") and not value.text:
                return Decimal("0")
            financial = value.cell.get("financial_value")
            if isinstance(financial, dict) and financial.get("raw_value") is not None:
                try:
                    return Decimal(str(financial["raw_value"]))
                except InvalidOperation:
                    pass
            parsed = parse_financial_value(
                value.text,
                unit_context=self.view.ir.get("unit"),
                header_context=value.row_label,
            )
            if parsed.raw_value is not None:
                return Decimal(parsed.raw_value)
            raise AnswerEngineError(f"cell is not numeric: {value.text!r}")
        if isinstance(value, str):
            parsed = parse_financial_value(value, unit_context=None)
            if parsed.raw_value is not None:
                return Decimal(parsed.raw_value)
        raise AnswerEngineError(f"value is not numeric: {value!r}")

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, CellRef):
            return value.text
        if isinstance(value, Decimal):
            return decimal_to_string(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def format_answer(self, value: Any, answer_format: str, dsl: dict[str, Any]) -> str:
        if answer_format == "number":
            return decimal_to_string(self._to_decimal(value))
        if answer_format == "string":
            return self._as_text(value)
        if answer_format == "json_array":
            values = value if isinstance(value, list) else [value]
            item_format = dsl.get("item_format")
            if item_format is None and isinstance(dsl.get("expression"), dict):
                item_format = dsl["expression"].get("item_format")
            serialized = [self._format_json_item(item, item_format) for item in values]
            return json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
        if answer_format == "json":
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            return json.dumps(self._format_json_item(value, dsl.get("item_format")), ensure_ascii=False)
        raise AnswerEngineError(f"unsupported answer format: {answer_format}")

    def _format_json_item(self, value: Any, item_format: str | None) -> Any:
        if isinstance(value, CellRef):
            if not value.text:
                return ""
            if item_format == "number":
                try:
                    return self._decimal_json(self._to_decimal(value))
                except AnswerEngineError:
                    return ""
            if item_format == "row_label":
                return value.row_label
            if item_format == "column_label":
                return value.column_label
            return value.text
        if isinstance(value, Decimal):
            if item_format == "string":
                return decimal_to_string(value)
            return self._decimal_json(value)
        if isinstance(value, list):
            return [self._format_json_item(item, item_format) for item in value]
        return value

    @staticmethod
    def _decimal_json(value: Decimal) -> int | float | str:
        if value == value.to_integral_value():
            return int(value)
        return float(value)
