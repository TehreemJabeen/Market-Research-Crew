"""Model wrapper: one interface over Gemini (free tier) and Grok (xAI).

Adds what a free tier needs: a rate limiter, retry with backoff on 429/5xx,
a hard call budget, usage counters, and JSON-validated structured output.
"""
import json
import re
import threading
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


def extract_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return raw[start : end + 1]


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart; safe across threads."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


_RETRYABLE = ("429", "rate", "quota", "503", "500", "overloaded", "timeout", "unavailable", "resource_exhausted")


def _retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in _RETRYABLE)


class BaseLLM:
    def __init__(self, rpm: int = 8, max_calls: int = 80):
        self._limiter = RateLimiter(rpm)
        self.max_calls = max_calls
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- provider hook -----------------------------------------------------
    def _generate(self, system: str, prompt: str, json_mode: bool) -> tuple[str, int, int]:
        raise NotImplementedError

    # -- bookkeeping -------------------------------------------------------
    def _reserve(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"LLM call budget of {self.max_calls} reached")
            self.calls += 1

    def _record(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    def usage(self) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    # -- public API --------------------------------------------------------
    def complete(self, system: str, prompt: str, json_mode: bool = False) -> str:
        self._reserve()
        attempts = 5
        for attempt in range(attempts):
            self._limiter.wait()
            try:
                text, p, c = self._generate(system, prompt, json_mode)
                self._record(p, c)
                return text
            except Exception as exc:  # provider SDKs raise many different types
                if attempt == attempts - 1 or not _retryable(exc):
                    raise LLMError(str(exc)) from exc
                time.sleep(min(60, 5 * 2**attempt))
        raise LLMError("unreachable")

    def structured(self, system: str, prompt: str, schema: type[T], retries: int = 2) -> T:
        schema_json = json.dumps(schema.model_json_schema())
        full_system = f"{system}\n\nRespond with ONLY a JSON object matching this JSON schema:\n{schema_json}"
        current = prompt
        last_error: Exception | None = None
        for _ in range(retries + 1):
            raw = self.complete(full_system, current, json_mode=True)
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                current = f"{prompt}\n\nYour previous reply was invalid ({str(exc)[:300]}). Return only valid JSON."
        raise LLMError(f"could not get valid {schema.__name__} JSON: {last_error}")


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(**kwargs)
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set (free key: https://aistudio.google.com/apikey)")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model

    def _generate(self, system, prompt, json_mode):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        usage = getattr(resp, "usage_metadata", None)
        return (
            resp.text or "",
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
        )


class GrokLLM(BaseLLM):
    """xAI exposes an OpenAI-compatible endpoint, so the openai SDK works."""

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(**kwargs)
        if not api_key:
            raise LLMError("XAI_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.model = model

    def _generate(self, system, prompt, json_mode):
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **extra,
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


def make_llm(settings) -> BaseLLM:
    kwargs = {"rpm": settings.rpm, "max_calls": settings.max_llm_calls}
    if settings.provider == "gemini":
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model, **kwargs)
    if settings.provider in ("grok", "xai"):
        return GrokLLM(settings.xai_api_key, settings.grok_model, **kwargs)
    raise LLMError(f"unknown LLM_PROVIDER '{settings.provider}' (use 'gemini' or 'grok')")
