"""Shared DeepSeek V4 chat helper for robustness data builders.

Reads DEEPSEEK_API_KEY from env or .env. Default base_url = api.deepseek.com.
Override model via DEEPSEEK_MODEL env (default: deepseek-chat = V4 non-thinking).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock

from openai import OpenAI


_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_MODEL = "deepseek-chat"


def _load_env_key(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


def get_client(base_url: str | None = None) -> OpenAI:
    key = _load_env_key("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set in env or .env")
    return OpenAI(api_key=key, base_url=base_url or _DEFAULT_BASE_URL, timeout=180.0)


def get_model_name() -> str:
    return os.environ.get("DEEPSEEK_MODEL", _DEFAULT_MODEL)


class RateLimiter:
    """Simple min-interval limiter shared across worker threads."""

    def __init__(self, min_interval_s: float = 0.05) -> None:
        self._lock = Lock()
        self._next_ok = 0.0
        self._dt = float(min_interval_s)

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_s = self._next_ok - now
            if wait_s > 0:
                time.sleep(wait_s)
            self._next_ok = time.monotonic() + self._dt


def chat(
    client: OpenAI,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    limiter: RateLimiter | None = None,
    retries: int = 3,
) -> str:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})

    last_err: Exception | None = None
    for attempt in range(retries):
        if limiter is not None:
            limiter.wait()
        try:
            r = client.chat.completions.create(
                model=model or get_model_name(),
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content or ""
        except Exception as exc:  # network / 429 / 5xx
            last_err = exc
            time.sleep(min(8.0, 1.5 ** attempt))
    raise RuntimeError(f"DeepSeek chat failed after {retries} retries: {last_err}")
