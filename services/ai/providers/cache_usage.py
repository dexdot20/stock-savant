"""Helpers for provider prompt-cache accounting and logging."""

from __future__ import annotations

from typing import Any, Dict


def extract_prompt_cache_usage(usage: Any) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        return {}

    prompt_token_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_token_details, dict):
        prompt_token_details = {}

    cache_usage: Dict[str, Any] = {}

    for key in (
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "cache_discount",
    ):
        value = usage.get(key)
        if value is not None:
            cache_usage[key] = value

    cached_tokens = prompt_token_details.get("cached_tokens")
    if cached_tokens is not None:
        cache_usage["cached_tokens"] = cached_tokens

    cache_write_tokens = prompt_token_details.get("cache_write_tokens")
    if cache_write_tokens is not None:
        cache_usage["cache_write_tokens"] = cache_write_tokens

    return cache_usage


def log_prompt_cache_usage(
    logger: Any,
    provider_label: str,
    model: str,
    cache_usage: Dict[str, Any],
) -> None:
    if not cache_usage or logger is None:
        return

    prompt_hit_tokens = int(cache_usage.get("prompt_cache_hit_tokens") or 0)
    prompt_miss_tokens = int(cache_usage.get("prompt_cache_miss_tokens") or 0)
    cached_tokens = int(cache_usage.get("cached_tokens") or 0)
    cache_write_tokens = int(cache_usage.get("cache_write_tokens") or 0)
    cache_discount = cache_usage.get("cache_discount")

    if (
        prompt_hit_tokens <= 0
        and cached_tokens <= 0
        and cache_write_tokens <= 0
        and not cache_discount
    ):
        return

    logger.info(
        "%s prompt cache usage: model=%s hit_tokens=%s miss_tokens=%s cached_tokens=%s cache_write_tokens=%s cache_discount=%s",
        provider_label,
        model,
        prompt_hit_tokens,
        prompt_miss_tokens,
        cached_tokens,
        cache_write_tokens,
        cache_discount if cache_discount is not None else 0,
    )