"""Shared provider metadata helpers."""

from __future__ import annotations

from typing import Optional


PROVIDER_DISPLAY_NAMES = {
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
}


def get_provider_display_name(provider_name: Optional[str]) -> str:
    if not provider_name:
        return "Unknown Provider"
    return PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.title())