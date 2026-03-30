from __future__ import annotations

from typing import Any, Dict, List


def _normalize_facts(working_memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_facts = working_memory.get("facts_learned", [])
    facts: List[Dict[str, Any]] = []
    if not isinstance(raw_facts, list):
        return facts

    for item in raw_facts:
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            facts.append(
                {
                    "text": text,
                    "importance": int(item.get("importance", 5) or 5),
                }
            )
            continue

        text = str(item or "").strip()
        if text:
            facts.append({"text": text, "importance": 5})
    return facts


def should_inject_reflection(
    working_memory: Dict[str, Any],
    *,
    step: int,
    interval_steps: int,
) -> bool:
    if step <= 0:
        return False
    if interval_steps <= 0 or step % interval_steps != 0:
        return False
    if not isinstance(working_memory, dict):
        return False

    return any(
        bool(working_memory.get(key))
        for key in (
            "facts_learned",
            "unanswered_questions",
            "contradictions_found",
            "research_milestones",
        )
    )


def build_reflection_prompt(
    working_memory: Dict[str, Any],
    *,
    step: int,
    output_language: str,
    max_facts: int = 4,
    max_questions: int = 3,
    max_contradictions: int = 2,
) -> str:
    facts = sorted(
        _normalize_facts(working_memory),
        key=lambda item: (
            int(item.get("importance", 0) or 0),
            len(str(item.get("text") or "")),
        ),
        reverse=True,
    )[: max(1, int(max_facts))]

    questions = [
        str(item).strip()
        for item in working_memory.get("unanswered_questions", [])
        if str(item).strip()
    ][: max(1, int(max_questions))]

    contradictions = [
        str(item).strip()
        for item in working_memory.get("contradictions_found", [])
        if str(item).strip()
    ][: max(1, int(max_contradictions))]

    milestones = [
        str(item).strip()
        for item in working_memory.get("research_milestones", [])
        if str(item).strip()
    ][:2]

    if not (facts or questions or contradictions or milestones):
        return ""

    lines: List[str] = [
        f"SELF-REFLECTION CHECKPOINT (step {step})",
        f"Respond in {output_language}.",
        "Use the working memory below to decide the next highest-value action.",
        "Do not repeat this checklist verbatim unless it improves the final answer.",
        "",
        "Verified facts to anchor on:",
    ]

    if facts:
        lines.extend(
            f"- (importance={int(item.get('importance', 5) or 5)}) {str(item.get('text') or '').strip()}"
            for item in facts
        )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Open questions to resolve next:")
    lines.extend(f"- {item}" for item in questions) if questions else lines.append("- None")

    lines.append("")
    lines.append("Contradictions to verify:")
    lines.extend(f"- {item}" for item in contradictions) if contradictions else lines.append("- None")

    lines.append("")
    lines.append("Recent milestones:")
    lines.extend(f"- {item}" for item in milestones) if milestones else lines.append("- None")

    lines.extend(
        [
            "",
            "Before choosing tools, silently answer:",
            "1. What are the two most decision-critical verified facts?",
            "2. Which unresolved question or contradiction most threatens the current conclusion?",
            "3. What single next tool call would most reduce uncertainty?",
            "4. If evidence is already sufficient, finish instead of re-searching.",
        ]
    )
    return "\n".join(lines)


__all__ = ["build_reflection_prompt", "should_inject_reflection"]