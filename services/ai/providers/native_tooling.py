from __future__ import annotations

import inspect
import json
from typing import Any, Dict, Iterable, List, Optional, Union, get_args, get_origin

from services.tools import get_tool


_AGENT_REQUEST_TOOL_NAMES: Dict[str, tuple[str, ...]] = {
    "pre_research": (
        "search_web",
        "search_news",
        "search_google_news",
        "search_memory",
        "summarize_url_content",
        "fetch_url_content",
        "yfinance_search",
        "yfinance_index_data",
        "yfinance_overview",
        "yfinance_price_history",
        "yfinance_dividends",
        "yfinance_analyst",
        "yfinance_earnings",
        "yfinance_ownership",
        "yfinance_sustainability",
        "yfinance_sector_analysis",
        "yfinance_company_data",
        "kap_search_disclosures",
        "kap_get_disclosure_detail",
        "kap_batch_disclosure_details",
        "get_tool_quality_metrics",
        "python_exec",
        "report",
        "update_working_memory",
        "finish",
    ),
    "comparison": (
        "search_news",
        "search_google_news",
        "summarize_url_content",
        "fetch_url_content",
        "yfinance_search",
        "yfinance_index_data",
        "yfinance_overview",
        "yfinance_price_history",
        "yfinance_dividends",
        "yfinance_analyst",
        "yfinance_earnings",
        "yfinance_ownership",
        "yfinance_sustainability",
        "yfinance_company_data",
        "kap_search_disclosures",
        "kap_get_disclosure_detail",
        "kap_batch_disclosure_details",
        "get_tool_quality_metrics",
        "python_exec",
        "report",
        "update_working_memory",
        "finish",
    ),
    "news": (
        "search_memory",
        "search_news",
        "search_google_news",
        "summarize_url_content",
        "fetch_url_content",
        "yfinance_search",
        "yfinance_index_data",
        "yfinance_overview",
        "yfinance_price_history",
        "yfinance_dividends",
        "yfinance_analyst",
        "yfinance_earnings",
        "yfinance_ownership",
        "yfinance_sustainability",
        "yfinance_company_data",
        "python_exec",
        "report",
        "update_working_memory",
        "finish",
    ),
}

_PARAMETER_DESCRIPTIONS: Dict[str, str] = {
    "category": "Short classification label for the report or query.",
    "code": "Python source code to execute inside the sandbox.",
    "collection": "Optional memory collection filter.",
    "context": "Optional structured context that helps explain the issue.",
    "context_window": "How many surrounding memory chunks to include.",
    "count": "Maximum number of results to return.",
    "days": "How many recent days to include.",
    "detail_level": "How much detail to return for memory hits.",
    "details": "Extra details for the report or result.",
    "disclosure_index": "Single KAP disclosure index to fetch.",
    "disclosure_indexes": "List of KAP disclosure indexes to fetch.",
    "exchange": "Exchange or market identifier.",
    "fact_importance": "Importance score for saved facts.",
    "fact_provenance": "Optional provenance metadata for stored facts.",
    "fact_tags": "Optional tags for saved facts.",
    "final_analysis": "Full final report returned to the user.",
    "fingerprint": "Stable deduplication fingerprint for the report.",
    "focus_query": "Optional focus query used to extract the most relevant passages.",
    "from_date": "Start date filter.",
    "from_timestamp": "Start timestamp filter.",
    "hit_ids": "Specific memory hit identifiers to hydrate.",
    "input_data": "Optional structured input passed into sandboxed Python.",
    "interval": "Requested price-history interval.",
    "language": "Language code for search or output.",
    "limit": "Maximum number of items to return.",
    "max_results": "Maximum number of results to return.",
    "max_workers": "Maximum worker count for batched operations.",
    "new_facts": "Verified facts to store in working memory.",
    "new_questions": "Open questions that still need verification.",
    "news_results": "How many related news hits to request from search.",
    "period": "Requested price-history period.",
    "query": "Natural language query or short search phrase.",
    "recent_days": "Recent-day lookback filter.",
    "rejected_hypotheses": "Hypotheses that were checked and rejected.",
    "research_milestones": "Short notes about important research decisions or progress.",
    "research_summary": "Short summary of the completed research path.",
    "reset": "Whether to reset the metrics after reading them.",
    "resolve_questions": "Questions that can now be marked as resolved.",
    "severity": "Issue severity level.",
    "source_summary": "Short source summary for the memory update.",
    "stock_codes": "One or more stock codes for KAP disclosure search.",
    "suggested_fix": "Suggested remediation or improvement.",
    "summary": "Short summary text.",
    "ticker": "Ticker symbol, including exchange suffix when needed.",
    "timeout_seconds": "Maximum seconds allowed for the action.",
    "timelimit": "Relative time filter for news search.",
    "title": "Short report title.",
    "to_date": "End date filter.",
    "to_timestamp": "End timestamp filter.",
    "tool_name": "Specific tool name to inspect.",
    "type_filter": "Optional asset type filter.",
    "url": "Source URL to fetch or summarize.",
}

_REQUEST_TOOL_SCHEMA_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def build_native_tool_request_kwargs(
    request_type: str,
    *,
    max_parallel_tools: Optional[int] = None,
) -> Dict[str, Any]:
    tools = build_native_tools_for_request(request_type)
    if not tools:
        return {}

    kwargs: Dict[str, Any] = {
        "tools": tools,
        "tool_choice": "auto",
    }
    if request_type in {"pre_research", "comparison", "news"}:
        kwargs["parallel_tool_calls"] = (
            max_parallel_tools is None or int(max_parallel_tools) > 1
        )
    return kwargs


def build_native_tools_for_request(request_type: str) -> List[Dict[str, Any]]:
    normalized_request_type = str(request_type or "").strip().lower()
    cached = _REQUEST_TOOL_SCHEMA_CACHE.get(normalized_request_type)
    if cached is not None:
        return [dict(item) for item in cached]

    tool_names = _AGENT_REQUEST_TOOL_NAMES.get(normalized_request_type, ())
    definitions = [
        tool_definition
        for tool_name in tool_names
        if (tool_definition := _build_tool_definition(tool_name)) is not None
    ]
    _REQUEST_TOOL_SCHEMA_CACHE[normalized_request_type] = definitions
    return [dict(item) for item in definitions]


def build_tool_result_history_message(
    *,
    tool_name: str,
    result: Any,
    tool_call: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tool_call_id = ""
    if isinstance(tool_call, dict):
        tool_call_id = str(tool_call.get("id") or "").strip()

    content = stringify_tool_result_content(result)
    if tool_call_id:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    return {
        "role": "user",
        "content": f"<tool_result name=\"{tool_name}\">\n{content}\n</tool_result>",
    }


def stringify_tool_result_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def merge_stream_tool_calls(
    tool_calls_by_index: Dict[int, Dict[str, Any]],
    tool_calls_delta: Any,
) -> None:
    if not isinstance(tool_calls_delta, list):
        return

    for fallback_index, item in enumerate(tool_calls_delta):
        if not isinstance(item, dict):
            continue
        raw_index = item.get("index", fallback_index)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = fallback_index

        merged_call = tool_calls_by_index.setdefault(
            index,
            {
                "id": f"tool_call_{index}",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )

        tool_call_id = str(item.get("id") or "").strip()
        if tool_call_id:
            merged_call["id"] = tool_call_id

        tool_type = str(item.get("type") or "").strip()
        if tool_type:
            merged_call["type"] = tool_type

        function_payload = item.get("function")
        if not isinstance(function_payload, dict):
            continue

        merged_function = merged_call.setdefault(
            "function",
            {"name": "", "arguments": ""},
        )

        function_name = str(function_payload.get("name") or "")
        if function_name:
            if not merged_function.get("name"):
                merged_function["name"] = function_name
            elif function_name not in str(merged_function.get("name") or ""):
                merged_function["name"] = (
                    str(merged_function.get("name") or "") + function_name
                )

        function_arguments = function_payload.get("arguments")
        if isinstance(function_arguments, str) and function_arguments:
            merged_function["arguments"] = (
                str(merged_function.get("arguments") or "") + function_arguments
            )
        elif isinstance(function_arguments, dict):
            merged_function["arguments"] = json.dumps(
                function_arguments,
                ensure_ascii=False,
            )


def finalize_stream_tool_calls(
    tool_calls_by_index: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    finalized: List[Dict[str, Any]] = []
    for index in sorted(tool_calls_by_index.keys()):
        tool_call = tool_calls_by_index[index]
        function_payload = tool_call.get("function") or {}
        function_name = str(function_payload.get("name") or "").strip()
        if not function_name:
            continue
        finalized.append(tool_call)
    return finalized


def ensure_tool_call_ids(tool_calls: Any) -> Any:
    if not isinstance(tool_calls, list):
        return tool_calls

    normalized: List[Dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        normalized_call = dict(tool_call)
        if not str(normalized_call.get("id") or "").strip():
            normalized_call["id"] = f"tool_call_{index}"
        normalized.append(normalized_call)
    return normalized


def _build_tool_definition(tool_name: str) -> Optional[Dict[str, Any]]:
    if tool_name == "update_working_memory":
        return _build_update_working_memory_definition()
    if tool_name == "finish":
        return _build_finish_definition()

    tool = get_tool(tool_name)
    if tool is None:
        return None

    signature = inspect.signature(tool.execute)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        schema = _annotation_to_schema(parameter.annotation)
        schema["description"] = _describe_parameter(parameter.name)
        if parameter.default is not inspect._empty:
            default_value = parameter.default
            if default_value is not None and isinstance(
                default_value, (str, int, float, bool, list, dict)
            ):
                schema["default"] = default_value
        else:
            required.append(parameter.name)

        properties[parameter.name] = schema

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": str(getattr(tool, "description", tool_name)).strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _build_update_working_memory_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "update_working_memory",
            "description": (
                "Persist verified research facts, open questions, contradictions, and short milestones "
                "into working memory. Use it only for source-backed findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _describe_parameter("new_facts"),
                    },
                    "new_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _describe_parameter("new_questions"),
                    },
                    "contradictions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conflicting source-backed statements that need reconciliation.",
                    },
                    "research_milestones": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _describe_parameter("research_milestones"),
                    },
                    "rejected_hypotheses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _describe_parameter("rejected_hypotheses"),
                    },
                    "source_summary": {
                        "type": "string",
                        "description": _describe_parameter("source_summary"),
                    },
                    "resolve_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _describe_parameter("resolve_questions"),
                    },
                    "fact_importance": {
                        "type": "integer",
                        "description": _describe_parameter("fact_importance"),
                    },
                    "question_importance": {
                        "type": "integer",
                        "description": "Importance score for new questions.",
                    },
                    "contradiction_importance": {
                        "type": "integer",
                        "description": "Importance score for contradictions.",
                    },
                    "fact_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _describe_parameter("fact_tags"),
                    },
                    "fact_pinned": {
                        "type": "boolean",
                        "description": "Whether the saved facts should be pinned in memory.",
                    },
                    "fact_provenance": {
                        "type": "object",
                        "description": _describe_parameter("fact_provenance"),
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": False,
            },
        },
    }


def _build_finish_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "End the agent loop and return the complete final report to the user. "
                "Use finish only when the research is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "final_analysis": {
                        "type": "string",
                        "description": _describe_parameter("final_analysis"),
                    },
                    "research_summary": {
                        "type": "string",
                        "description": _describe_parameter("research_summary"),
                    },
                },
                "required": ["final_analysis"],
                "additionalProperties": False,
            },
        },
    }


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    if annotation in (inspect._empty, Any):
        return {"type": "string"}

    origin = get_origin(annotation)
    if origin is Union:
        union_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(union_args) == 1:
            return _annotation_to_schema(union_args[0])
        return {"type": "string"}

    if origin in (list, List, Iterable):
        item_args = get_args(annotation)
        item_schema = (
            _annotation_to_schema(item_args[0]) if item_args else {"type": "string"}
        )
        return {"type": "array", "items": item_schema}

    if origin in (dict, Dict):
        return {"type": "object", "additionalProperties": True}

    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is str:
        return {"type": "string"}

    return {"type": "string"}


def _describe_parameter(name: str) -> str:
    normalized_name = str(name or "").strip()
    if not normalized_name:
        return "Tool parameter."
    return _PARAMETER_DESCRIPTIONS.get(
        normalized_name,
        f"Parameter `{normalized_name}` for the tool call.",
    )
