from __future__ import annotations

from typing import Any, Dict, Optional

from config import get_config

_LANGUAGE_LOCK_MARKER = "🔒 LANGUAGE LOCK:"
_TOOLING_DIRECTIVE_MARKER = "## STRUCT AI TOOL DIRECTIVE"


def _build_tooling_directive(config: Dict[str, Any]) -> str:
    ai_cfg = config.get("ai", {}) if isinstance(config, dict) else {}
    python_cfg = ai_cfg.get("python_exec", {}) if isinstance(ai_cfg, dict) else {}
    report_cfg = ai_cfg.get("report_tool", {}) if isinstance(ai_cfg, dict) else {}

    lines = [_TOOLING_DIRECTIVE_MARKER]

    if bool(python_cfg.get("enabled", True)):
        lines.append(
            "- `python_exec(code, input_data?, timeout_seconds?)`: use this for exact calculations, multi-step quantitative reasoning, and bounded local Python execution. "
            "Execution is isolated per call with sandboxed temporary files only. No outbound network, subprocesses, package installation, or filesystem access outside the sandbox are allowed."
        )

    if bool(report_cfg.get("enabled", True)):
        lines.append(
            "- `report(title, category, severity, summary, details?, suggested_fix?, context?, fingerprint?)`: use this to record developer-facing technical issues, risks, or optimization ideas in a redacted JSONL report log. "
            "This tool is always available, but use it only for actionable engineering feedback and avoid duplicate or noisy reports. Never include secrets or raw sensitive data."
        )

    lines.append(
        "- If the current interaction supports tool calls, invoke these tools with the standard tool-call format. If tool calls are not available in the current internal helper request, do not invent tool syntax."
    )
    return "\n".join(lines)


def augment_system_prompt(
    prompt: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    output_language: Optional[str] = None,
) -> str:
    if not prompt:
        return ""

    resolved_config = config or get_config()
    ai_cfg = resolved_config.get("ai", {}) if isinstance(resolved_config, dict) else {}
    resolved_output_language = (
        output_language
        or (
            ai_cfg.get("output_language")
            if isinstance(ai_cfg, dict)
            else None
        )
        or "English"
    )

    augmented = str(prompt).strip()

    if _TOOLING_DIRECTIVE_MARKER not in augmented:
        augmented = f"{augmented}\n\n{_build_tooling_directive(resolved_config)}"

    if resolved_output_language and _LANGUAGE_LOCK_MARKER not in augmented:
        augmented += (
            f"\n\n{_LANGUAGE_LOCK_MARKER} You MUST respond exclusively in {resolved_output_language}. "
            "This applies to all generated text including headers, bullet points, tables, and explanatory content. "
            "Responding in another language is an error."
        )

    return augmented
