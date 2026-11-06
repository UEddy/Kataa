"""Minimal Mistral chat client for this session. Runs in: cloud session.

The API key never appears here: the session's network proxy injects it for api.mistral.ai, so
requests carry a placeholder bearer token. Paces itself from the rate-limit headers (the free
plan gives mistral-large-4 50 requests and 20,000 tokens per minute), retries 429s with backoff,
and returns only the text parts of a reply (Large 4 also returns "thinking" blocks; their tokens
are counted in usage.completion_tokens and billed as output).
"""

from __future__ import annotations

import time

import httpx

API = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-4"

# USD per 1M tokens. Main figure: Oct 22 list price. Sensitivity: earlier launch-sale price.
PRICES = {"list_2026_10_22": {"input": 1.36, "cached_input": 0.14, "output": 4.18},
          "launch_sale": {"input": 0.68, "cached_input": 0.07, "output": 2.09}}


def cost_usd(usage: dict, price: dict) -> float:
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    fresh = usage.get("prompt_tokens", 0) - cached
    return (fresh * price["input"] + cached * price["cached_input"]
            + usage.get("completion_tokens", 0) * price["output"]) / 1e6


def text_of(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    parts = []
    for block in c or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class MistralClient:
    def __init__(self, model: str = MODEL, max_tokens: int = 2048, timeout: float = 300,
                 reasoning_effort: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort  # Large 4 accepts low / medium / high
        self.http = httpx.Client(timeout=timeout,
                                 headers={"Authorization": "Bearer proxy-injected"})
        self.tokens_left: int | None = None
        self.reqs_left: int | None = None
        self.window_start = time.time()
        self.usage_log: list[dict] = []

    def _pace(self, est_tokens: int) -> None:
        if self.tokens_left is None:
            return
        if self.tokens_left < est_tokens or (self.reqs_left is not None and self.reqs_left < 2):
            wait = max(0.0, 61 - (time.time() - self.window_start))
            time.sleep(wait if wait > 0 else 5)
            self.tokens_left = None

    def chat(self, messages: list[dict], temperature: float = 0.0) -> tuple[str, dict]:
        est = sum(len(m.get("content") or "") for m in messages) // 3 + self.max_tokens // 2
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "top_p": 1, "max_tokens": self.max_tokens}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        delay = 20.0
        for attempt in range(8):
            self._pace(est)
            t0 = time.time()
            try:
                r = self.http.post(API, json=body)
            except httpx.HTTPError:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            if "x-ratelimit-remaining-tokens-minute" in r.headers:
                left = int(r.headers["x-ratelimit-remaining-tokens-minute"])
                if self.tokens_left is None or left > (self.tokens_left or 0):
                    self.window_start = time.time()
                self.tokens_left = left
                self.reqs_left = int(r.headers.get("x-ratelimit-remaining-req-minute", 50))
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                self.tokens_left = None
                continue
            r.raise_for_status()
            j = r.json()
            usage = dict(j.get("usage") or {})
            usage["latency_s"] = round(time.time() - t0, 2)
            self.usage_log.append(usage)
            return text_of(j["choices"][0]["message"]), usage
        raise RuntimeError("Mistral request failed after retries")
