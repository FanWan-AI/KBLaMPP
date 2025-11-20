"""Lightweight OpenAI-compatible client used for LLM-based data generation.

The implementation mirrors the ``LocalGPT`` helper used in previous
KBLaM tooling and relies on an OpenAI-compatible endpoint (e.g.
DashScope/Qwen).  API credentials and defaults are read from the local
``.env`` file or process environment variables:

- ``DASHSCOPE_API_KEY`` (preferred) or ``OPENAI_API_KEY``
- ``DASHSCOPE_BASE_URL`` to override the default DashScope URL
- ``MODEL`` / ``QWEN_MODEL`` to override the default model name.

This module purposely avoids any project-level dependencies so it can be
imported from both the offline data generation scripts and future
interactive tools.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, cast

try:  # pragma: no cover - thin wrapper around openai client
    from openai import OpenAI
    from openai._exceptions import OpenAIError
    from openai.types.chat import ChatCompletionMessageParam
except ImportError as exc:  # pragma: no cover
    raise ImportError("The 'openai' package is required. Install it via 'pip install openai'.") from exc

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_SYSTEM_MSG = (
    "You are a world-class knowledge engineer who writes rigorously formatted JSON facts."
)
ENV_FILENAME = ".env"

_ENV_CACHE: Optional[Dict[str, str]] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_env_paths() -> List[Path]:
    paths: List[Path] = []
    unique = set()
    for base in (
        _project_root(),
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
    ):
        if base is None:
            continue
        env_path = base / ENV_FILENAME
        if env_path.exists() and env_path not in unique:
            paths.append(env_path)
            unique.add(env_path)
    custom = os.getenv("KBLaMPP_ENV_FILE")
    if custom:
        custom_path = Path(custom).expanduser().resolve()
        if custom_path.exists() and custom_path not in unique:
            paths.append(custom_path)
    return paths


def _load_env_pairs() -> Dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    pairs: Dict[str, str] = {}
    for env_path in _candidate_env_paths():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalized = line.replace("：", ":", 1)
            if "=" in normalized:
                key, _, value = normalized.partition("=")
            elif ":" in normalized:
                key, _, value = normalized.partition(":")
            else:
                continue
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"').strip("'")
            pairs[key] = value
            upper = key.upper()
            if upper not in pairs:
                pairs[upper] = value
        if pairs:
            break
    _ENV_CACHE = pairs
    return pairs


def _lookup(env: Dict[str, str], keys: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        if key in env and env[key]:
            return env[key]
    return default


def _pick(*values: Optional[str], default: Optional[str] = None) -> Optional[str]:
    for value in values:
        if value:
            return value
    return default


class LocalGPT:
    """Thin helper around an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_msg: str = DEFAULT_SYSTEM_MSG,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_retries: int = 3,
        request_timeout: float = 120.0,
        seed: Optional[int] = None,
    ) -> None:
        env = _load_env_pairs()
        self.api_key = _pick(
            api_key,
            os.getenv("DASHSCOPE_API_KEY"),
            env.get("DASHSCOPE_API_KEY"),
            env.get("dashscope_api_key"),
            os.getenv("OPENAI_API_KEY"),
            env.get("OPENAI_API_KEY"),
        )
        if not self.api_key:
            raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or OPENAI_API_KEY in the environment/.env.")

        self.base_url = _pick(
            endpoint_url,
            os.getenv("DASHSCOPE_BASE_URL"),
            env.get("DASHSCOPE_BASE_URL"),
            env.get("dashscope_base_url"),
            DEFAULT_BASE_URL,
        )
        self.base_url = (self.base_url or DEFAULT_BASE_URL).rstrip("/")

        picked_model = _pick(
            model_name,
            os.getenv("MODEL"),
            env.get("MODEL"),
            os.getenv("QWEN_MODEL"),
            env.get("QWEN_MODEL"),
            DEFAULT_MODEL,
        )
        self.model_name: str = picked_model or DEFAULT_MODEL
        self.system_msg = system_msg
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.seed = seed
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def api_call_chat(self, messages: List[Dict[str, str]]) -> str:
        backoff = 1.5
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=cast(Sequence[ChatCompletionMessageParam], messages),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    timeout=self.request_timeout,
                    seed=self.seed,
                )
                if response and response.choices:
                    content = response.choices[0].message.content
                    if content is None:
                        raise RuntimeError("LLM response missing content")
                    return content
                raise RuntimeError("LLM returned empty choices")
            except OpenAIError as err:
                last_error = err
            except Exception as err:  # pragma: no cover
                last_error = err
            if attempt < self.max_retries:
                time.sleep(min(backoff ** attempt, 8.0))
        if last_error:
            raise RuntimeError(f"LLM chat request failed: {last_error}")
        raise RuntimeError("LLM chat request failed without further details")

    def generate_response(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": prompt},
        ]
        return self.api_call_chat(messages)