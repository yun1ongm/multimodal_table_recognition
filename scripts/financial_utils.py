"""Deterministic normalization helpers for financial TableIR documents."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


@dataclass(frozen=True)
class UnitScale:
    factor: Decimal
    label: str
    source: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "factor": decimal_to_string(self.factor),
            "label": self.label,
            "source": self.source,
        }


@dataclass(frozen=True)
class FinancialValue:
    raw_text: str
    kind: str
    raw_value: str | None = None
    scaled_value: str | None = None
    factor: str = "1"
    unit: str | None = None
    negative: bool = False
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "kind": self.kind,
            "raw_value": self.raw_value,
            "scaled_value": self.scaled_value,
            "factor": self.factor,
            "unit": self.unit,
            "negative": self.negative,
            "reason": self.reason,
        }


_UNIT_RULES: tuple[tuple[re.Pattern[str], Decimal, str], ...] = (
    (re.compile(r"(?:in\s+)?billions?|十亿元|亿元", re.IGNORECASE), Decimal("1000000000"), "billion"),
    (re.compile(r"(?:in\s+)?millions?|百万元", re.IGNORECASE), Decimal("1000000"), "million"),
    (re.compile(r"(?:in\s+)?thousands?|千元", re.IGNORECASE), Decimal("1000"), "thousand"),
    (re.compile(r"(?:in\s+)?hundreds?|百元", re.IGNORECASE), Decimal("100"), "hundred"),
    (re.compile(r"(?:人民币|美元|港币|日元)?元\b|\byuan\b|\bRMB\b", re.IGNORECASE), Decimal("1"), "unit"),
)

_MISSING_VALUES = {
    "",
    "-",
    "—",
    "–",
    "−",
    "－",
    "na",
    "n/a",
    "n.a.",
    "nm",
    "n.m.",
    "不适用",
    "无",
}

_NUMBER_TOKEN = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+")


def decimal_to_string(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def detect_unit(text: str | None) -> UnitScale:
    context = (text or "").replace("，", ",").strip()
    for pattern, factor, label in _UNIT_RULES:
        match = pattern.search(context)
        if match:
            return UnitScale(factor=factor, label=label, source=match.group(0))
    return UnitScale(factor=Decimal("1"), label="unit", source=None)


def _is_missing(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text).lower()
    return normalized in {re.sub(r"\s+", "", item).lower() for item in _MISSING_VALUES}


def _is_percent(text: str, context: str) -> bool:
    combined = f"{text} {context}".lower()
    return "%" in combined or "百分比" in combined or "percentage" in combined or "percent" in combined


def _is_per_share(text: str, context: str) -> bool:
    combined = f"{text} {context}".lower()
    # Statement-wide unit notes often say "except per share data".  That
    # clause describes an exception, not the current cell; only the cell or
    # row/header label should trigger the no-scale rule.
    combined = re.sub(r"except\s+per\s+share\s+data", "", combined)
    return "per share" in combined or "每股" in combined or "每股收益" in combined


def parse_financial_value(
    text: str,
    *,
    unit_context: str | None = None,
    header_context: str | None = None,
) -> FinancialValue:
    """Parse one visible financial cell without losing its raw representation.

    ``raw_value`` remains in the table's displayed unit. ``scaled_value`` is
    converted to the unit declared by ``unit_context``. Percentages and per
    share values are intentionally not multiplied by a statement-wide scale.
    """

    raw = str(text or "").strip()
    context = f"{unit_context or ''} {header_context or ''}"
    if _is_missing(raw):
        return FinancialValue(raw_text=raw, kind="missing", reason="blank_or_dash")

    percent = _is_percent(raw, context)
    per_share = _is_per_share(raw, context)
    kind = "percent" if percent else "per_share" if per_share else "number"
    negative_by_parentheses = bool(re.match(r"^\s*[\(（].*[\)）]\s*$", raw))
    cleaned = raw.replace("\u00a0", " ").replace("−", "-").replace("－", "-")
    if negative_by_parentheses:
        cleaned = cleaned.strip()[1:-1]
    token_text = cleaned
    if re.fullmatch(r"\s*[+-]?\d+,\d{1,2}\s*%?\s*", token_text):
        token_text = token_text.replace(",", ".")
    else:
        token_text = token_text.replace(",", "")
    tokens = _NUMBER_TOKEN.findall(token_text)
    if len(tokens) != 1:
        return FinancialValue(raw_text=raw, kind="unparsed", reason=f"expected_one_number_got_{len(tokens)}")

    token = tokens[0]
    try:
        number = Decimal(token)
    except InvalidOperation:
        return FinancialValue(raw_text=raw, kind="unparsed", reason="invalid_decimal")
    if negative_by_parentheses:
        number = -abs(number)
    factor = Decimal("1") if percent or per_share else detect_unit(unit_context).factor
    unit = "percent" if percent else "per_share" if per_share else detect_unit(unit_context).label
    scaled = number * factor
    return FinancialValue(
        raw_text=raw,
        kind=kind,
        raw_value=decimal_to_string(number),
        scaled_value=decimal_to_string(scaled),
        factor=decimal_to_string(factor),
        unit=unit,
        negative=number < 0,
    )


def canonical_period(text: str | None) -> str | None:
    """Map common Chinese/English financial period labels to stable keys."""

    value = re.sub(r"\s+", " ", (text or "")).strip().lower()
    year = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    if year:
        return f"year_{year.group(1)}"
    if any(token in value for token in ("期初", "上年末", "beginning balance", "opening balance")):
        return "prior_end"
    if any(token in value for token in ("上期期末", "上期末", "prior ending", "prior end", "previous ending")):
        return "prior_end"
    if any(token in value for token in ("期末", "本年末", "ending balance", "closing balance", "as of")):
        return "current_end"
    if any(token in value for token in ("本期", "本年度", "current period", "current year")):
        return "current_period"
    if any(token in value for token in ("上期", "上年度", "prior period", "prior year", "previous period")):
        return "prior_period"
    return None


def _unique_texts(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def infer_column_headers(ir: dict[str, Any]) -> dict[str, list[str]]:
    """Collect visible and inherited headers for each logical column."""

    col_count = int(ir.get("col_count", 0))
    result: dict[str, list[str]] = {str(col): [] for col in range(col_count)}
    cells = ir.get("cells", [])
    for cell in cells:
        role = cell.get("role")
        path = cell.get("header_path")
        labels: list[str] = []
        if role not in {"header", "value"}:
            continue
        if isinstance(path, list):
            labels.extend(str(item) for item in path)
        if role == "header" and cell.get("text"):
            labels.append(str(cell["text"]))
        if not labels:
            continue
        for col in range(cell["col"], cell["col"] + cell["colspan"]):
            result[str(col)] = _unique_texts([*result[str(col)], *labels])
    return result


def _row_label_for_cell(cells: list[dict[str, Any]], cell: dict[str, Any]) -> str:
    same_row = [candidate for candidate in cells if candidate.get("row") == cell.get("row")]
    labels = [
        candidate.get("text", "")
        for candidate in same_row
        if candidate.get("col", 0) < cell.get("col", 0)
        and candidate.get("role") in {"row_label", "header", "other"}
    ]
    return " | ".join(_unique_texts(labels))


def annotate_financial_ir(ir: dict[str, Any]) -> dict[str, Any]:
    """Add non-destructive numeric and period metadata to a TableIR."""

    result = copy.deepcopy(ir)
    cells = result.get("cells", [])
    unit_context = str(result.get("unit") or "")
    if not unit_context:
        unit_context = " ".join(
            [str(result.get("title") or ""), *[str(note) for note in result.get("notes", [])]]
        )
    unit_scale = detect_unit(unit_context)
    header_by_col = infer_column_headers(result)
    period_columns: dict[str, dict[str, Any]] = {}
    for col, labels in header_by_col.items():
        period = next((canonical_period(label) for label in labels if canonical_period(label)), None)
        period_columns[col] = {"labels": labels, "period": period}

    for cell in cells:
        role = cell.get("role")
        if role != "value":
            continue
        column_info = period_columns.get(str(cell.get("col", 0)), {})
        header_labels = cell.get("header_path")
        if not isinstance(header_labels, list):
            header_labels = column_info.get("labels", [])
        row_label = _row_label_for_cell(cells, cell)
        context = " | ".join(_unique_texts([*map(str, header_labels), row_label]))
        parsed = parse_financial_value(
            str(cell.get("text", "")),
            unit_context=unit_context,
            header_context=context,
        )
        cell["financial_value"] = parsed.as_dict()
        cell["row_label"] = row_label
        cell["period"] = canonical_period(row_label) or column_info.get("period")

    result["financial"] = {
        "unit_context": unit_context,
        "unit_scale": unit_scale.as_dict(),
        "period_columns": period_columns,
    }
    return result
