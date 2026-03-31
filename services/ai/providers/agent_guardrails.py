from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set

VERIFIED_MEMORY_TOOLS: frozenset[str] = frozenset(
    {
        "fetch_url_content",
        "summarize_url_content",
        "kap_get_disclosure_detail",
        "kap_batch_disclosure_details",
        "yfinance_company_data",
        "yfinance_overview",
        "yfinance_price_history",
        "yfinance_dividends",
        "yfinance_analyst",
        "yfinance_earnings",
        "yfinance_ownership",
        "yfinance_sustainability",
        "yfinance_index_data",
        "yfinance_sector_analysis",
    }
)

BIST_INDEX_SYMBOL_ALIASES = {
    "BIST": "XU100.IS",
    "BIST 100": "XU100.IS",
    "BORSA ISTANBUL": "XU100.IS",
    "BORSA İSTANBUL": "XU100.IS",
    "XU100": "XU100.IS",
    "^XU100": "XU100.IS",
    "XU100 INDEX": "XU100.IS",
    "BIST100": "XU100.IS",
}

SPECULATIVE_FACT_PATTERNS: tuple[str, ...] = (
    r"\bappears?\b",
    r"\bseems?\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bpotentially\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bshould\b",
    r"\bqualif(?:y|ies|ied)\b",
    r"\brecommend(?:ed|ation)?\b",
    r"\blooks? like\b",
    r"\bupper half\b",
    r"\blower half\b",
    r"\blower quartile\b",
    r"\bgenuine relative low\b",
)

TOOL_ARG_ALIASES: Dict[str, Dict[str, str]] = {
    "search_memory": {"max_hits": "limit"},
    "yfinance_sector_analysis": {"index": "symbol"},
    "yfinance_search": {
        "count": "max_results",
        "limit": "max_results",
        "asset_type": "type_filter",
    },
}

STRICT_TOOL_ARGS: Dict[str, Set[str]] = {
    "search_memory": {
        "query",
        "collection",
        "symbol",
        "limit",
        "recent_days",
        "from_timestamp",
        "to_timestamp",
        "detail_level",
        "hit_ids",
        "context_window",
    },
    "yfinance_sector_analysis": {
        "sector_key",
        "industry_key",
        "sector",
        "symbol",
        "ticker",
        "country",
        "region",
        "exchange",
        "max_tickers",
    },
    "yfinance_search": {
        "query",
        "max_results",
        "news_results",
        "type_filter",
    },
}

FINAL_REPORT_MARKERS: tuple[str, ...] = (
    "FINAL ANALYSIS COMPLETE",
    "PRIMARY RECOMMENDATION",
    "RANKING:",
    "RECOMMENDATION FOR 2-MONTH",
    "BIST Stock Recommendation Report",
)

_BIST_CONTEXT_VALUES = {
    "BIST",
    "BORSA ISTANBUL",
    "BORSA İSTANBUL",
    "IST",
    "XU100",
    "XU100.IS",
    "^XU100",
}

_BIST_SPECIFIC_QUERY_PATTERN = re.compile(r"^[A-ZÇĞİÖŞÜ]{2,10}\.IS$")

_BIST_BROAD_DISCOVERY_GUIDANCE = (
    "BIST broad discovery is not reliable in yfinance. Use search_web in Turkish "
    "or kap_search_disclosures for discovery; reserve yfinance_search for a specific "
    "ticker such as FROTO.IS or an index symbol such as XU100.IS."
)

_BIST_INDEX_QUERY_TOKENS = {"XU100", "XU100.IS", "^XU100"}


def canonicalize_bist_market_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    normalized_call = dict(tool_call or {})
    args = normalized_call.get("args")
    if not isinstance(args, dict):
        return normalized_call

    normalized_args = normalize_tool_args(
        str(normalized_call.get("name") or ""),
        args,
    )
    tool_name = str(normalized_call.get("name") or "")

    if tool_name == "yfinance_index_data":
        for key in ("symbol", "ticker", "index_symbol"):
            candidate = _canonicalize_bist_index_symbol(normalized_args.get(key))
            if candidate:
                normalized_args[key] = candidate
                break
    elif tool_name == "yfinance_sector_analysis":
        for key in ("symbol", "ticker"):
            candidate = _canonicalize_bist_index_symbol(normalized_args.get(key))
            if candidate:
                normalized_args[key] = candidate
                break
        if not any(
            normalized_args.get(key)
            for key in ("sector_key", "industry_key", "symbol", "ticker")
        ):
            exchange = str(normalized_args.get("exchange") or "").strip().upper()
            if exchange in {"BIST", "BORSA ISTANBUL", "BORSA İSTANBUL", "XU100", "^XU100"}:
                normalized_args["symbol"] = "XU100.IS"

    normalized_call["args"] = normalized_args
    return normalized_call


def is_bist_market_context(value: Any) -> bool:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().upper())
    return normalized in _BIST_CONTEXT_VALUES


def _is_specific_bist_query(query: Any) -> bool:
    normalized = re.sub(r"\s+", " ", str(query or "").strip().upper())
    if not normalized:
        return False

    if normalized in _BIST_CONTEXT_VALUES or normalized in BIST_INDEX_SYMBOL_ALIASES:
        return True

    for token in normalized.split():
        cleaned = token.strip(",;:!?()[]{}")
        if not cleaned:
            continue
        if cleaned in _BIST_INDEX_QUERY_TOKENS:
            return True
        if cleaned in _BIST_CONTEXT_VALUES:
            continue
        if _BIST_SPECIFIC_QUERY_PATTERN.fullmatch(cleaned):
            return True

    return False


def should_block_bist_yfinance_search(
    *,
    exchange: Any,
    query: Any,
    type_filter: Any = None,
) -> Optional[str]:
    if not is_bist_market_context(exchange):
        return None

    normalized_type = str(type_filter or "").strip().lower()
    if normalized_type == "index" and _is_specific_bist_query(query):
        return None

    if _is_specific_bist_query(query):
        return None

    return _BIST_BROAD_DISCOVERY_GUIDANCE


def normalize_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    normalized_args = dict(args or {})
    alias_map = TOOL_ARG_ALIASES.get(str(tool_name or ""), {})
    for alias, canonical in alias_map.items():
        if alias in normalized_args and canonical not in normalized_args:
            normalized_args[canonical] = normalized_args.pop(alias)
        elif alias in normalized_args:
            normalized_args.pop(alias)

    if str(tool_name or "") == "yfinance_search":
        # `region` is a common carry-over from market-summary style prompts,
        # but the search tool itself does not use it. Drop it so a harmless
        # locale hint does not block an otherwise valid search call.
        normalized_args.pop("region", None)
    return normalized_args


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    if not isinstance(args, dict):
        return None

    normalized_tool_name = str(tool_name or "")
    allowed = STRICT_TOOL_ARGS.get(normalized_tool_name)
    if not allowed:
        return None

    unknown_keys = sorted(key for key in args.keys() if key not in allowed)
    if unknown_keys:
        allowed_str = ", ".join(sorted(allowed))
        unknown_str = ", ".join(unknown_keys)
        return (
            f"{normalized_tool_name}: unsupported argument(s): {unknown_str}. "
            f"Allowed arguments: {allowed_str}."
        )

    return None


def build_tool_plan_preview(
    tool_calls: Iterable[Dict[str, Any]],
    *,
    max_parallel_tools: Optional[int],
    non_dedup_tools: Set[str],
    executed_signatures: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    preview_signatures = set(executed_signatures or set())
    preview_rows: List[Dict[str, Any]] = []
    preview_executable_count = 0

    for tool_call in tool_calls or []:
        fn_name = tool_call.get("name")
        fn_args = tool_call.get("args", {})
        if isinstance(fn_args, dict):
            fn_args = normalize_tool_args(str(fn_name or ""), fn_args)
        plan = {
            "name": fn_name or "unknown",
            "args": fn_args if isinstance(fn_args, dict) else {},
            "status": "execute_now",
            "note": "",
        }

        if fn_name == "update_working_memory":
            plan["status"] = "memory_update"
        elif fn_name == "finish":
            plan["status"] = "finish"
        elif not isinstance(fn_args, dict):
            plan["status"] = "blocked"
            plan["note"] = "invalid JSON arguments"
        else:
            validation_error = validate_tool_args(str(fn_name or ""), fn_args)
            if validation_error:
                plan["status"] = "blocked"
                plan["note"] = validation_error
            elif fn_name == "search_memory":
                has_query = bool(str(fn_args.get("query") or "").strip())
                has_hit_ids = bool(fn_args.get("hit_ids"))
                if not has_query and not has_hit_ids:
                    plan["status"] = "blocked"
                    plan["note"] = "missing query or hit_ids"

        if plan["status"] == "execute_now":
            if max_parallel_tools is not None and preview_executable_count >= max_parallel_tools:
                plan["status"] = "deferred"
                plan["note"] = "deferred by concurrency limit"
            elif fn_name not in non_dedup_tools:
                sig = make_tool_signature(fn_name, fn_args)
                if sig and sig in preview_signatures:
                    plan["status"] = "skipped_duplicate"
                    plan["note"] = "already executed in a previous step"
                else:
                    if sig:
                        preview_signatures.add(sig)
                    preview_executable_count += 1
            else:
                preview_executable_count += 1

        preview_rows.append(plan)

    return preview_rows


def sanitize_memory_args(args: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(args or {})
    source_summary = str(sanitized.get("source_summary") or "").strip()
    default_provenance = _normalize_fact_provenance(sanitized.get("fact_provenance"))
    if default_provenance:
        sanitized["fact_provenance"] = default_provenance
    elif "fact_provenance" in sanitized:
        sanitized["fact_provenance"] = {}
    evidence_type = str(default_provenance.get("evidence_type") or "").strip().lower()
    memory_only_context = "memory" in source_summary.lower() or evidence_type == "rag"
    default_anchor = bool(source_summary or default_provenance) and not memory_only_context

    raw_facts = _normalize_fact_payload(sanitized.get("new_facts"))
    sanitized["new_facts"] = raw_facts
    if raw_facts is None:
        return sanitized

    fact_items = raw_facts if isinstance(raw_facts, list) else [raw_facts]
    verified_facts: List[Any] = []
    inferred_facts: List[str] = []
    unanchored_facts = 0
    bulk_list_facts = 0

    for item in fact_items:
        if isinstance(item, dict):
            fact_text = str(item.get("text") or item.get("fact") or "").strip()
            if not fact_text:
                continue
            item_provenance = (
                item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
            )
            item_anchor = bool(
                item.get("source")
                or item.get("date")
                or item_provenance
                or default_anchor
            )
            if looks_like_speculative_fact(fact_text):
                inferred_facts.append(fact_text)
                continue
            if looks_like_bulk_list_fact(fact_text):
                bulk_list_facts += 1
                continue
            if not item_anchor:
                unanchored_facts += 1
                continue
            verified_facts.append(item)
            continue

        fact_text = str(item).strip()
        if not fact_text:
            continue
        if looks_like_speculative_fact(fact_text):
            inferred_facts.append(fact_text)
            continue
        if looks_like_bulk_list_fact(fact_text):
            bulk_list_facts += 1
            continue
        if not default_anchor:
            unanchored_facts += 1
            continue
        verified_facts.append(fact_text)

    sanitized["new_facts"] = verified_facts

    milestones = [
        str(item).strip()
        for item in (sanitized.get("research_milestones") or [])
        if str(item).strip()
    ]
    questions = [
        str(item).strip()
        for item in (sanitized.get("new_questions") or [])
        if str(item).strip()
    ]

    if inferred_facts:
        milestones.append(
            f"Held back {len(inferred_facts)} inferred statement(s) from verified memory facts."
        )
    if unanchored_facts:
        milestones.append(
            f"Held back {unanchored_facts} unanchored statement(s); verified facts require source-backed context."
        )
    if bulk_list_facts:
        milestones.append(
            f"Held back {bulk_list_facts} bulk list-like statement(s); summarize the screening result instead of storing long ticker lists as facts."
        )
    for inferred_text in inferred_facts[:3]:
        question = f"Verify inference before storing as fact: {inferred_text}"
        if question not in questions:
            questions.append(question)
    if bulk_list_facts:
        question = (
            "Condense any large ticker/constituent list into a short screening summary before storing it in memory."
        )
        if question not in questions:
            questions.append(question)

    sanitized["research_milestones"] = milestones
    sanitized["new_questions"] = questions
    return sanitized


def _normalize_fact_provenance(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {
            str(key).strip(): raw
            for key, raw in value.items()
            if str(key).strip() and raw is not None and str(raw).strip()
        }
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return {"source_id": ", ".join(parts)} if parts else {}
    text = str(value or "").strip()
    return {"source_id": text} if text else {}


def _normalize_fact_payload(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if any(key in value for key in ("text", "fact", "source", "date", "provenance")):
        return value

    normalized_items: List[Dict[str, str]] = []
    for key, raw in value.items():
        key_text = str(key).strip()
        value_text = str(raw).strip() if not isinstance(raw, (dict, list)) else ""
        if not value_text and isinstance(raw, (dict, list)):
            try:
                value_text = json.dumps(raw, ensure_ascii=False)
            except (TypeError, ValueError):
                value_text = str(raw).strip()
        if not key_text or not value_text:
            continue
        normalized_items.append({"text": f"{key_text}: {value_text}"})

    return normalized_items


def looks_like_final_report_payload(args: Dict[str, Any]) -> bool:
    new_facts = args.get("new_facts") if isinstance(args, dict) else None
    flattened = (
        "\n".join(str(item) for item in (new_facts or []))
        if isinstance(new_facts, list)
        else str(new_facts or "")
    )
    return any(marker in flattened for marker in FINAL_REPORT_MARKERS)


def has_actionable_memory_payload(args: Dict[str, Any]) -> bool:
    return bool(
        args.get("new_facts")
        or args.get("new_questions")
        or args.get("contradictions")
        or args.get("research_milestones")
    )


def make_tool_signature(fn_name: Any, fn_args: Any) -> Optional[str]:
    try:
        return f"{fn_name}::{json.dumps(fn_args, sort_keys=True, ensure_ascii=False)}"
    except (TypeError, ValueError):
        return None


def looks_like_speculative_fact(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in SPECULATIVE_FACT_PATTERNS)


def looks_like_bulk_list_fact(text: str) -> bool:
    normalized = str(text or "").strip()
    ticker_like = re.findall(r"\b[A-Z0-9]{2,8}(?:\.IS)?\b", normalized)
    comma_count = normalized.count(",")
    if len(ticker_like) >= 12 or comma_count >= 14:
        return True
    return len(normalized) >= 160 and (len(ticker_like) >= 8 or comma_count >= 10)


def get_pre_research_pivot_notice(
    memory_state: Dict[str, Any],
    executed_tools: Iterable[str],
) -> Optional[str]:
    rejected = [
        str(item).strip()
        for item in (memory_state.get("rejected_hypotheses") or [])
        if str(item).strip()
    ]
    rejected_relative_low = [
        item
        for item in rejected
        if "relative low" in item.lower() or "not near" in item.lower()
    ]
    price_history_calls = sum(
        1 for tool_name in executed_tools if tool_name == "yfinance_price_history"
    )

    if price_history_calls < 3 or len(rejected_relative_low) < 5:
        return None

    return (
        "⚠️ Screening drift detected: multiple candidates failed the user's core `relative low` criterion. "
        "Stop brute-force ticker scanning. Reassess whether no stable BIST candidate currently fits, "
        "or pivot to a different screening method such as verified 52-week-low candidates or a narrower defensive-sector filter."
    )


def _canonicalize_bist_index_symbol(value: Any) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().upper())
    return BIST_INDEX_SYMBOL_ALIASES.get(normalized)
