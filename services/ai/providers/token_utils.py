from __future__ import annotations

from typing import Any, Dict, Iterable


def _stringify_message(message: Dict[str, Any]) -> str:
    content = message.get("content", "")

    parts = []
    if content:
        parts.append(str(content))
    return "\n".join(parts)


def estimate_tokens_for_messages(
    messages: Iterable[Dict[str, Any]], encoding_name: str
) -> int:
    text = "\n".join(_stringify_message(message) for message in messages if message)
    if not text:
        return 0

    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback heuristic: ~4 chars per token
        return max(1, len(text) // 4)
