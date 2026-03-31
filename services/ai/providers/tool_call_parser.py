from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Primary format: ```json or ```tool_calls fenced block
_JSON_FENCE_RE = re.compile(
    r"```(?:json|tool_calls)\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)
# Backward-compat: legacy <tool_calls>...</tool_calls> XML wrapper
_XML_BLOCK_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_FUNCTION_RE = re.compile(
    r"<function(?:\s*=\s*|\s+name\s*=\s*[\"']?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"']?\s*>(?P<body>.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_PARAMETER_RE = re.compile(
    r"<parameter(?:\s*=\s*|\s+name\s*=\s*[\"']?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"']?\s*>(?P<value>.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)
# Low-priority fallback: any generic ``` block
_ANY_FENCE_RE = re.compile(r"```\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
# Function-call style: tool_name({"key": "value"})
_FUNCTION_CALL_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")


# ── JSON lenient fixup ────────────────────────────────────────────────────────

def _try_fix_json(candidate: str) -> str | None:
    """Repair common LLM JSON mistakes. Returns a valid JSON string or None."""
    text = candidate.strip()

    def _fix_literals(s: str) -> str:
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        return s

    def _remove_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([\}\]])", r"\1", s)

    def _single_to_double(s: str) -> str:
        result: List[str] = []
        i = 0
        while i < len(s):
            if s[i] == "'" and (i == 0 or s[i - 1] != "\\"):
                j = i + 1
                inner: List[str] = []
                while j < len(s):
                    if s[j] == "\\" and j + 1 < len(s):
                        inner.append(s[j : j + 2])
                        j += 2
                        continue
                    if s[j] == "'":
                        break
                    if s[j] == '"':
                        inner.append('\\"')
                    else:
                        inner.append(s[j])
                    j += 1
                result.append('"' + "".join(inner) + '"')
                i = j + 1
            else:
                result.append(s[i])
                i += 1
        return "".join(result)

    def _strip_trailing_junk(s: str) -> str:
        """Remove stray characters after the outermost array/object closes."""
        # Find the position where the outermost structure closes; strip anything after.
        if not s:
            return s
        opener, closer = ("[", "]") if s[0] == "[" else ("{", "}")
        depth = 0
        in_string = False
        escape = False
        close_idx = -1
        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
        if close_idx >= 0 and close_idx < len(s) - 1:
            return s[: close_idx + 1]
        return s

    def _close_unclosed_braces(s: str) -> str:
        """Append missing closing `}` / `]` to balance an unclosed JSON structure."""
        stack: List[str] = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()
        if stack:
            return s + "".join(reversed(stack))
        return s

    def _insert_missing_braces_before_closer(s: str) -> str:
        """Fix: last JSON object in array missing its closing }.
        Pattern: [..., {"key": "val"}] where inner { is not closed before ].
        Inserts the required } chars before the last ] or }.
        """
        last_bracket = s.rfind("]")
        if last_bracket < 0:
            return s
        before = s[:last_bracket]
        after = s[last_bracket:]
        open_count = 0
        in_string = False
        escape = False
        for ch in before:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                open_count += 1
            elif ch == "}":
                open_count -= 1
        if open_count > 0:
            return before + ("}" * open_count) + after
        return s

    targets = [
        text,
        _remove_trailing_commas(text),
    ]
    for target in targets:
        for transform in (
            lambda t: _fix_literals(t),
            lambda t: _fix_literals(_remove_trailing_commas(t)),
            lambda t: _fix_literals(_single_to_double(t)),
            lambda t: _fix_literals(_remove_trailing_commas(_single_to_double(t))),
        ):
            try:
                fixed = transform(target)
                json.loads(fixed)
                return fixed
            except (json.JSONDecodeError, Exception):
                pass

    # Extra repair passes for structural issues
    for repair in (
        _strip_trailing_junk,
        _insert_missing_braces_before_closer,
        _close_unclosed_braces,
        lambda t: _insert_missing_braces_before_closer(_strip_trailing_junk(t)),
        lambda t: _close_unclosed_braces(_strip_trailing_junk(t)),
        lambda t: _remove_trailing_commas(_insert_missing_braces_before_closer(_strip_trailing_junk(t))),
        lambda t: _fix_literals(_remove_trailing_commas(_insert_missing_braces_before_closer(_strip_trailing_junk(t)))),
    ):
        try:
            repaired = repair(text)
            if repaired != text:
                json.loads(repaired)
                return repaired
        except (json.JSONDecodeError, Exception):
            pass

    return None


# ── Core normalizers ──────────────────────────────────────────────────────────

def _coerce_args(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_tool_call(item: Any) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if "function" in item and isinstance(item.get("function"), dict):
        fn = item["function"]
        name = fn.get("name")
        args = _coerce_args(fn.get("arguments", {}))
    else:
        name = item.get("name")
        args = _coerce_args(item.get("args", item.get("arguments", {})))
    if not isinstance(name, str) or not name.strip():
        return None
    return {"name": name.strip(), "args": args}


def _normalize_tool_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return normalize_tool_calls(payload)
    if isinstance(payload, dict):
        row = _normalize_tool_call(payload)
        return [row] if row else []
    return []


def normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in tool_calls:
        row = _normalize_tool_call(item)
        if row:
            normalized.append(row)
    return normalized


# ── Array parser (with lenient fixup) ────────────────────────────────────────

def _parse_tool_call_array(raw_block: str) -> List[Dict[str, Any]]:
    candidate = raw_block.strip()
    if not candidate:
        return []

    # Fast path: strict JSON
    try:
        return _normalize_tool_payload(json.loads(candidate))
    except json.JSONDecodeError:
        pass

    # Try to isolate an array or object sub-string
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pattern, candidate)
        if m:
            try:
                return _normalize_tool_payload(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass

    # Lenient fixup pass
    for target in (candidate, *[
        m.group(0)
        for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}")
        for m in [re.search(pat, candidate)]
        if m
    ]):
        fixed = _try_fix_json(target)
        if fixed is not None:
            try:
                return _normalize_tool_payload(json.loads(fixed))
            except json.JSONDecodeError:
                pass

    return []


def _parse_legacy_parameter_value(raw_value: str) -> Any:
    value = str(raw_value or "").strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value

    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)", value):
        try:
            return float(value)
        except ValueError:
            return value

    if value[0] in {'[', '{', '"'}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def _parse_legacy_tool_call_markup(content: str) -> Tuple[List[Dict[str, Any]], str]:
    tool_calls: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for match in _LEGACY_TOOL_CALL_RE.finditer(content):
        function_match = _LEGACY_FUNCTION_RE.search(match.group(1))
        if not function_match:
            continue

        args: Dict[str, Any] = {}
        for param_match in _LEGACY_PARAMETER_RE.finditer(function_match.group("body")):
            args[param_match.group("name").strip()] = _parse_legacy_parameter_value(
                param_match.group("value")
            )

        tool_calls.append(
            {
                "name": function_match.group("name").strip(),
                "args": args,
            }
        )
        spans.append((match.start(), match.end()))

    if not tool_calls:
        return [], content.strip()

    cleaned_parts: List[str] = []
    last_idx = 0
    for start, end in spans:
        cleaned_parts.append(content[last_idx:start])
        last_idx = end
    cleaned_parts.append(content[last_idx:])
    cleaned = " ".join(part.strip() for part in cleaned_parts if part.strip()).strip()
    cleaned = re.sub(r"(?:\s*,\s*)+", " ", cleaned).strip(" ,\n\t")
    return tool_calls, cleaned


# ── Function-call style parser ────────────────────────────────────────────────

def _find_balanced_segment(text: str, start_idx: int, opener: str, closer: str) -> int:
    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _parse_function_style_tool_calls(content: str) -> Tuple[List[Dict[str, Any]], str]:
    tool_calls: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for match in _FUNCTION_CALL_RE.finditer(content):
        name = match.group("name")
        open_paren_idx = content.find("(", match.start())
        if open_paren_idx < 0:
            continue
        close_paren_idx = _find_balanced_segment(content, open_paren_idx, "(", ")")
        if close_paren_idx < 0:
            continue
        args_text = content[open_paren_idx + 1 : close_paren_idx].strip()
        if args_text:
            try:
                parsed_args = json.loads(args_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_args, dict):
                continue
        else:
            parsed_args = {}
        tool_calls.append({"name": name.strip(), "args": parsed_args})
        spans.append((match.start(), close_paren_idx + 1))
    if not tool_calls:
        return [], content.strip()
    cleaned_parts: List[str] = []
    last_idx = 0
    for start, end in spans:
        cleaned_parts.append(content[last_idx:start])
        last_idx = end
    cleaned_parts.append(content[last_idx:])
    cleaned = " ".join(p.strip() for p in cleaned_parts if p.strip()).strip()
    cleaned = re.sub(r"(?:\s*,\s*)+", " ", cleaned).strip(" ,\n\t")
    return tool_calls, cleaned


def content_has_tool_call_markup(content: Any) -> bool:
    if not isinstance(content, str) or not content.strip():
        return False
    return bool(
        _JSON_FENCE_RE.search(content)
        or _XML_BLOCK_RE.search(content)
        or _LEGACY_TOOL_CALL_RE.search(content)
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def parse_tool_calls_from_content(content: Any) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse tool calls from an LLM response string.

    Priority order:
      1. ```json or ```tool_calls fenced block  (primary — new format)
      2. <tool_calls>...</tool_calls> XML block  (backward compat)
            3. <tool_call><function=...>...</tool_call> legacy markup
            4. Generic ``` fenced block
            5. Bare JSON array at top level
            6. Function-call style: name({...})
    """
    if not isinstance(content, str):
        return [], ""

    # ── 1. Primary: tagged JSON code block ────────────────────────────────────
    for match in _JSON_FENCE_RE.finditer(content):
        calls = _parse_tool_call_array(match.group(1))
        if calls:
            cleaned = _JSON_FENCE_RE.sub("", content, count=1).strip()
            return calls, cleaned
        logger.warning(
            "Found ```json/tool_calls block but JSON parsing failed. "
            "Block (first 200 chars): %r",
            match.group(1).strip()[:200],
        )

    # ── 2. Backward compat: XML wrapper ───────────────────────────────────────
    xml_matches = list(_XML_BLOCK_RE.finditer(content))
    if xml_matches:
        if len(xml_matches) > 1:
            logger.warning(
                "Multiple <tool_calls> blocks detected (%d). Using last valid block.",
                len(xml_matches),
            )
        for match in reversed(xml_matches):
            calls = _parse_tool_call_array(match.group(1))
            if calls:
                cleaned = _XML_BLOCK_RE.sub("", content).strip()
                return calls, cleaned
        bad = xml_matches[-1].group(1).strip()
        logger.warning(
            "Found <tool_calls> block but JSON parsing failed. "
            "Block (first 200 chars): %r",
            bad[:200],
        )
        cleaned = _XML_BLOCK_RE.sub("", content).strip()
        return [], cleaned

    # ── 3. Legacy singular tool-call markup ──────────────────────────────────
    calls, cleaned = _parse_legacy_tool_call_markup(content)
    if calls:
        return calls, cleaned

    # ── 4. Generic ``` block ──────────────────────────────────────────────────
    for match in _ANY_FENCE_RE.finditer(content):
        calls = _parse_tool_call_array(match.group(1))
        if calls:
            cleaned = _ANY_FENCE_RE.sub("", content, count=1).strip()
            return calls, cleaned

    # ── 5. Bare JSON array ────────────────────────────────────────────────────
    stripped = content.strip()
    if stripped.startswith("["):
        calls = _parse_tool_call_array(stripped)
        if calls:
            return calls, ""

    # ── 6. Function-call style ────────────────────────────────────────────────
    return _parse_function_style_tool_calls(content)
