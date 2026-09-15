#!/usr/bin/env python3
"""Small OpenAI-compatible client for the supplied Alibaba Cloud Qwen API."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


@dataclass(frozen=True)
class QwenConfig:
    api_key: str
    base_url: str
    model: str
    fallback_model: str | None = None
    timeout_seconds: int = 180
    verify_ssl: bool = True
    ca_bundle: str | None = None

    @classmethod
    def from_env(cls, env_path: Path = Path(".env")) -> "QwenConfig":
        values = load_env_file(env_path)
        api_key = os.getenv("QWEN_API_KEY") or values.get("QWEN_API_KEY", "")
        base_url = os.getenv("QWEN_BASE_URL") or values.get("QWEN_BASE_URL", "")
        model = os.getenv("QWEN_MODEL") or values.get("QWEN_MODEL", "")
        fallback = os.getenv("QWEN_MODEL_FALLBACK") or values.get("QWEN_MODEL_FALLBACK") or None
        verify_value = os.getenv("QWEN_VERIFY_SSL") or values.get("QWEN_VERIFY_SSL", "true")
        verify_ssl = verify_value.strip().lower() not in {"0", "false", "no", "off"}
        ca_bundle = os.getenv("QWEN_CA_BUNDLE") or values.get("QWEN_CA_BUNDLE") or None
        if not api_key:
            raise ValueError("QWEN_API_KEY is missing from the environment or .env")
        if not base_url:
            raise ValueError("QWEN_BASE_URL is missing from the environment or .env")
        if not model:
            raise ValueError("QWEN_MODEL is missing from the environment or .env")
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            fallback_model=fallback,
            verify_ssl=verify_ssl,
            ca_bundle=ca_bundle,
        )


def image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"Unsupported Qwen image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                chunks.append(block["text"])
        return "".join(chunks)
    return str(content)


def extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Qwen response has no choices")
    message = choices[0].get("message", {})
    if not isinstance(message, dict) or "content" not in message:
        raise ValueError("Qwen response has no message content")
    return _content_to_text(message["content"]).strip()


def extract_json(text: str, required_keys: set[str] | None = None) -> Any:
    """Parse JSON even if the model wrapped it in a Markdown code fence."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if required_keys and (not isinstance(parsed, dict) or not required_keys.issubset(parsed)):
            raise ValueError(
                "Qwen JSON is missing required keys: " + ", ".join(sorted(required_keys))
            )
        return parsed
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates: list[Any] = []
        for index, character in enumerate(cleaned):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                if required_keys and isinstance(value, dict):
                    if required_keys.issubset(value):
                        return value
                elif not required_keys:
                    return value
                candidates.append(value)
            except json.JSONDecodeError:
                continue
        if required_keys:
            raise ValueError(
                "Qwen JSON is incomplete or missing required keys: "
                + ", ".join(sorted(required_keys))
            )
    raise ValueError(f"Qwen did not return valid JSON: {text[:500]}")


class QwenClient:
    def __init__(self, config: QwenConfig):
        self.config = config

    def _ssl_context(self) -> ssl.SSLContext:
        if not self.config.verify_ssl:
            return ssl._create_unverified_context()  # noqa: S323 - explicit opt-in only
        if self.config.ca_bundle:
            return ssl.create_default_context(cafile=self.config.ca_bundle)
        # The system Python on macOS may not be connected to the Keychain CA
        # store.  certifi is optional, so retain the standard context as a
        # fallback for installations without it.
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
                context=self._ssl_context(),
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen API HTTP {exc.code}: {body_text[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen API connection failed: {exc.reason}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Qwen API returned non-JSON response: {raw[:500]}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Qwen API response is not a JSON object")
        return payload

    def chat(
        self,
        *,
        image_path: Path,
        prompt: str,
        system_prompt: str,
        model: str | None = None,
        max_tokens: int = 5000,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        selected_model = model or self.config.model
        body: dict[str, Any] = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload = self._request(body)
        return {
            "model": selected_model,
            "text": extract_message_text(payload),
            "usage": payload.get("usage"),
            "raw": payload,
        }

    def chat_text(
        self,
        *,
        prompt: str,
        system_prompt: str,
        model: str | None = None,
        max_tokens: int = 5000,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send a text-only request through the same Qwen endpoint.

        TableIR already contains the visual evidence needed by the question
        compiler.  Keeping this request text-only avoids uploading the same
        page image once for every question in a table.
        """

        selected_model = model or self.config.model
        body: dict[str, Any] = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload = self._request(body)
        return {
            "model": selected_model,
            "text": extract_message_text(payload),
            "usage": payload.get("usage"),
            "raw": payload,
        }

    def chat_json(self, required_keys: set[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Call the primary model and optionally retry once with its fallback."""

        try:
            response = self.chat(**kwargs)
            response["json"] = extract_json(response["text"], required_keys=required_keys)
            return response
        except Exception as primary_error:
            fallback = self.config.fallback_model
            if not fallback or fallback == kwargs.get("model") or fallback == self.config.model:
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model"] = fallback
            try:
                response = self.chat(**fallback_kwargs)
                response["json"] = extract_json(response["text"], required_keys=required_keys)
                response["fallback_reason"] = str(primary_error)
                return response
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Primary Qwen attempt failed: {primary_error}; "
                    f"fallback attempt failed: {fallback_error}"
                ) from fallback_error

    def chat_text_json(self, required_keys: set[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Text-only counterpart of :meth:`chat_json`."""

        try:
            response = self.chat_text(**kwargs)
            response["json"] = extract_json(response["text"], required_keys=required_keys)
            return response
        except Exception as primary_error:
            fallback = self.config.fallback_model
            if not fallback or fallback == kwargs.get("model") or fallback == self.config.model:
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model"] = fallback
            try:
                response = self.chat_text(**fallback_kwargs)
                response["json"] = extract_json(response["text"], required_keys=required_keys)
                response["fallback_reason"] = str(primary_error)
                return response
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Primary Qwen text attempt failed: {primary_error}; "
                    f"fallback text attempt failed: {fallback_error}"
                ) from fallback_error
