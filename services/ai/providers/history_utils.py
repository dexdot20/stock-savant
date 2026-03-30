from __future__ import annotations

from typing import Dict, List, Any

from config import get_config
from services.ai.providers.token_utils import estimate_tokens_for_messages
from services.ai.providers.tool_call_parser import parse_tool_calls_from_content


def _is_working_memory_call(msg: Dict[str, Any]) -> bool:
    """Check if an assistant message contains an update_working_memory call."""
    if msg.get("role") != "assistant":
        return False
    tool_calls, _ = parse_tool_calls_from_content(msg.get("content", ""))
    return any(tc.get("name") == "update_working_memory" for tc in tool_calls)


def select_immediate_context(
    history: List[Dict[str, Any]],
    token_limit: int,
    encoding: str,
    min_keep: int = 4,
    max_tail_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    if len(history) <= 1:
        return []

    ai_cfg = get_config().get("ai", {})
    token_cfg = ai_cfg.get("token_budget", {}) if isinstance(ai_cfg, dict) else {}
    immediate_context_ratio = float(token_cfg.get("immediate_context_ratio", 0.5) or 0.5)
    working_memory_rescue_ratio = float(token_cfg.get("working_memory_rescue_ratio", 0.5) or 0.5)
    immediate_context_ratio = max(0.2, min(0.85, immediate_context_ratio))
    working_memory_rescue_ratio = max(0.2, min(1.0, working_memory_rescue_ratio))

    tail_budget = max_tail_tokens
    if tail_budget is None:
        tail_budget = max(800, int(max(1, token_limit) * immediate_context_ratio))
    else:
        tail_budget = max(200, int(tail_budget))

    # Step 1: Normal tail — from latest messages backward until token budget is exhausted.
    tail: List[Dict[str, Any]] = []
    for msg in reversed(history[1:]):
        tail.insert(0, msg)
        if len(tail) < min_keep:
            continue
        token_count = estimate_tokens_for_messages(tail, encoding)
        if token_count >= tail_budget:
            break

    # Step 2: Find update_working_memory messages that couldn't fit in the tail.
    # These messages represent the research trail; if large search results exhaust the budget
    # they may remain outside the tail. We save them with a small additional budget.
    tail_ids = {id(m) for m in tail}
    wm_msgs_outside_tail = [
        msg
        for msg in history[1:]
        if id(msg) not in tail_ids and _is_working_memory_call(msg)
    ]

    if not wm_msgs_outside_tail:
        return tail

    wm_extra_budget = max(200, int(tail_budget * working_memory_rescue_ratio))
    rescued: List[Dict[str, Any]] = []
    for msg in reversed(wm_msgs_outside_tail):  # Start from newest
        cost = estimate_tokens_for_messages([msg], encoding)
        if cost <= wm_extra_budget:
            rescued.append(msg)
            wm_extra_budget -= cost
        if wm_extra_budget <= 0:
            break

    if not rescued:
        return tail

    # Step 3: Merge rescued messages with tail in original order.
    rescued_ids = {id(m) for m in rescued}
    merged = [
        msg for msg in history[1:] if id(msg) in tail_ids or id(msg) in rescued_ids
    ]
    return merged
