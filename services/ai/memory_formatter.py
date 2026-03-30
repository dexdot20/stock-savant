from __future__ import annotations

from typing import Any, Dict, List

from domain.utils import safe_int_strict as _safe_int


def _normalize_provenance(provenance: Any) -> Dict[str, Any]:
    if not isinstance(provenance, dict):
        return {}

    normalized: Dict[str, Any] = {}
    for key, raw in provenance.items():
        key_text = str(key).strip()
        if not key_text or raw is None:
            continue
        if isinstance(raw, (str, int, float, bool)):
            if isinstance(raw, str) and not raw.strip():
                continue
            normalized[key_text] = raw
    return normalized


def _format_provenance_inline(provenance: Dict[str, Any]) -> str:
    if not provenance:
        return ""

    ordered_keys = [
        "evidence_type",
        "source_id",
        "doc_group_id",
        "chunk_index",
        "retrieval_score",
        "timestamp",
    ]
    parts: List[str] = []
    for key in ordered_keys:
        if key in provenance:
            parts.append(f"{key}={provenance[key]}")
    for key, value in provenance.items():
        if key not in ordered_keys:
            parts.append(f"{key}={value}")
    return f" [provenance: {'; '.join(parts)}]" if parts else ""


def _normalize_working_memory(working_memory: Dict[str, Any]) -> Dict[str, Any]:
    raw_facts = working_memory.get("facts_learned", [])
    facts: List[Dict[str, Any]] = []
    if isinstance(raw_facts, list):
        for item in raw_facts:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    importance = _safe_int(item.get("importance", 5) or 5, 5)
                    tags_raw = item.get("tags", [])
                    tags = (
                        [str(t).strip() for t in tags_raw if str(t).strip()]
                        if isinstance(tags_raw, list)
                        else []
                    )
                    facts.append(
                        {
                            "text": text,
                            "importance": importance,
                            "tags": tags,
                            "provenance": _normalize_provenance(
                                item.get("provenance", {})
                            ),
                        }
                    )
            else:
                text = str(item).strip()
                if text:
                    facts.append(
                        {
                            "text": text,
                            "importance": 5,
                            "tags": [],
                            "provenance": {},
                        }
                    )

    sources = [
        str(s).strip()
        for s in working_memory.get("sources_consulted", [])
        if str(s).strip()
    ]
    contradictions = [
        str(c).strip()
        for c in working_memory.get("contradictions_found", [])
        if str(c).strip()
    ]
    questions = [
        str(q).strip()
        for q in working_memory.get("unanswered_questions", [])
        if str(q).strip()
    ]
    rejected = [
        str(r).strip()
        for r in working_memory.get("rejected_hypotheses", [])
        if str(r).strip()
    ]
    milestones = [
        str(m).strip()
        for m in working_memory.get("research_milestones", [])
        if str(m).strip()
    ]
    raw_evidence: List[Dict[str, Any]] = []
    for item in working_memory.get("evidence_trail", []):
        if not isinstance(item, dict):
            continue
        preview = str(item.get("preview", "")).strip()
        if not preview:
            continue
        raw_evidence.append(
            {
                "tool_name": str(item.get("tool_name") or "tool").strip() or "tool",
                "source_id": str(item.get("source_id") or item.get("url") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                "captured_at": str(item.get("captured_at") or "").strip(),
                "preview": preview,
                "payload_kind": str(item.get("payload_kind") or "").strip(),
            }
        )
    depth_score = _safe_int(working_memory.get("research_depth_score", 0) or 0, 0)

    return {
        "facts": facts,
        "sources": sources,
        "contradictions": contradictions,
        "questions": questions,
        "rejected_hypotheses": rejected,
        "milestones": milestones,
        "evidence": raw_evidence,
        "depth_score": depth_score,
    }


def _format_evidence_line(item: Dict[str, Any]) -> str:
    tool_name = str(item.get("tool_name") or "tool").strip() or "tool"
    source_id = str(item.get("source_id") or "").strip()
    title = str(item.get("title") or "").strip()
    captured_at = str(item.get("captured_at") or "").strip()
    payload_kind = str(item.get("payload_kind") or "").strip()
    preview = str(item.get("preview") or "").strip()

    meta_parts: List[str] = [f"tool={tool_name}"]
    if payload_kind:
        meta_parts.append(f"kind={payload_kind}")
    if source_id:
        meta_parts.append(f"source={source_id}")
    if captured_at:
        meta_parts.append(f"captured_at={captured_at}")

    lead = f"- ({'; '.join(meta_parts)})"
    if title:
        return f"{lead} {title} | {preview}"
    return f"{lead} {preview}"


def format_working_memory_for_llm(
    working_memory: Dict[str, Any], style: str = "snapshot"
) -> str:
    if not isinstance(working_memory, dict):
        return ""

    if style == "evidence_pack":
        return format_working_memory_evidence_pack(working_memory)

    normalized = _normalize_working_memory(working_memory)
    facts = sorted(
        normalized["facts"],
        key=lambda x: (int(x.get("importance", 0)), str(x.get("text", "")).lower()),
        reverse=True,
    )
    sources = normalized["sources"]
    contradictions = normalized["contradictions"]
    questions = normalized["questions"]
    rejected = normalized["rejected_hypotheses"]
    milestones = normalized["milestones"]
    evidence = normalized["evidence"]
    depth_score = normalized["depth_score"]

    if not (
        facts
        or sources
        or contradictions
        or questions
        or rejected
        or milestones
        or evidence
        or depth_score
    ):
        return ""

    if style == "reasoner":
        lines: List[str] = [
            "\n\nSTRUCTURED RESEARCH FINDINGS (WORKING MEMORY):",
            f"- Research depth score: {depth_score}",
            f"- Facts captured: {len(facts)}",
            f"- Sources consulted: {len(sources)}",
            f"- Contradictions found: {len(contradictions)}",
            f"- Unresolved questions: {len(questions)}",
            f"- Rejected hypotheses: {len(rejected)}",
            f"- Research milestones: {len(milestones)}",
            f"- Raw evidence records: {len(evidence)}",
        ]

        if facts:
            lines.append("\nVerified Facts:")
            for fact in sorted(
                facts, key=lambda x: int(x.get("importance", 0)), reverse=True
            ):
                fact_text = str(fact.get("text", "")).strip()
                fact_importance = int(fact.get("importance", 5) or 5)
                fact_tags = list(fact.get("tags", []))
                provenance_suffix = _format_provenance_inline(
                    fact.get("provenance", {})
                )
                if fact_tags:
                    lines.append(
                        f"- (importance={fact_importance}, tags={', '.join(fact_tags)}) {fact_text}{provenance_suffix}"
                    )
                else:
                    lines.append(
                        f"- (importance={fact_importance}) {fact_text}{provenance_suffix}"
                    )

        if contradictions:
            lines.append("\nRecorded Contradictions:")
            lines.extend(f"- {item}" for item in contradictions)

        if questions:
            lines.append("\nOpen Questions:")
            lines.extend(f"- {item}" for item in questions)

        if rejected:
            lines.append("\nRejected Hypotheses:")
            lines.extend(f"- {item}" for item in rejected)

        if milestones:
            lines.append("\nResearch Milestones:")
            lines.extend(f"- {item}" for item in milestones)

        if sources:
            lines.append("\nSource Summaries:")
            lines.extend(f"- {item}" for item in sources)

        if evidence:
            lines.append("\nRecent Raw Evidence Trail:")
            for item in list(reversed(evidence[-4:])):
                lines.append(_format_evidence_line(item))

        return "\n".join(lines)

    fact_lines: List[str] = []
    for fact in facts:
        fact_text = str(fact.get("text", "")).strip()
        fact_importance = int(fact.get("importance", 5) or 5)
        fact_tags = list(fact.get("tags", []))
        provenance_suffix = _format_provenance_inline(fact.get("provenance", {}))
        if fact_tags:
            tags_text = ", ".join(fact_tags)
            fact_lines.append(
                f"- (importance={fact_importance} tags={tags_text}) {fact_text}{provenance_suffix}"
            )
        else:
            fact_lines.append(
                f"- (importance={fact_importance}) {fact_text}{provenance_suffix}"
            )

    lines = [
        "=== WORKING MEMORY SNAPSHOT ===",
        f"Depth score: {depth_score}",
        f"Facts: {len(facts)} | Sources: {len(sources)} | Contradictions: {len(contradictions)} | Open questions: {len(questions)}",
        "",
        "[FACTS]",
        *(fact_lines or ["- None"]),
        "",
        "[CONTRADICTIONS]",
        *([f"- {c}" for c in contradictions] or ["- None"]),
        "",
        "[OPEN QUESTIONS]",
        *([f"- {q}" for q in questions] or ["- None"]),
        "",
        "[REJECTED HYPOTHESES]",
        *([f"- {r}" for r in rejected] or ["- None"]),
        "",
        "[RESEARCH MILESTONES]",
        *([f"- {m}" for m in milestones] or ["- None"]),
        "",
        "[EVIDENCE TRAIL]",
        *([_format_evidence_line(item) for item in list(reversed(evidence))] or ["- None"]),
        "",
        "[SOURCES]",
        *([f"- {s}" for s in sources] or ["- None"]),
    ]
    return "\n".join(lines)


def format_working_memory_evidence_pack(
    working_memory: Dict[str, Any],
    *,
    max_facts: int | None = 8,
    max_contradictions: int | None = 4,
    max_questions: int | None = 4,
    max_sources: int | None = 4,
    max_evidence: int | None = 4,
) -> str:
    if not isinstance(working_memory, dict):
        return ""

    normalized = _normalize_working_memory(working_memory)
    facts = sorted(
        normalized["facts"],
        key=lambda x: (int(x.get("importance", 0)), str(x.get("text", "")).lower()),
        reverse=True,
    )

    def _apply_limit(items: List[Any], limit: int | None) -> List[Any]:
        if limit is None:
            return list(items)
        return list(items[: max(1, int(limit))])

    facts = _apply_limit(facts, max_facts)
    contradictions = _apply_limit(normalized["contradictions"], max_contradictions)
    questions = _apply_limit(normalized["questions"], max_questions)
    sources = _apply_limit(normalized["sources"], max_sources)
    evidence = _apply_limit(list(reversed(normalized["evidence"])), max_evidence)

    if not (facts or contradictions or questions or sources or evidence):
        return ""

    lines: List[str] = ["=== EVIDENCE PACK ==="]

    if facts:
        lines.append("[TOP FACTS]")
        for fact in facts:
            text = str(fact.get("text", "")).strip()
            importance = int(fact.get("importance", 5) or 5)
            tags = list(fact.get("tags", []))
            tag_text = f" tags={', '.join(tags)}" if tags else ""
            provenance_suffix = _format_provenance_inline(fact.get("provenance", {}))
            lines.append(
                f"- (importance={importance}{tag_text}) {text}{provenance_suffix}"
            )

    if contradictions:
        lines.append("[CRITICAL CONTRADICTIONS]")
        lines.extend(f"- {item}" for item in contradictions)

    if questions:
        lines.append("[OPEN QUESTIONS]")
        lines.extend(f"- {item}" for item in questions)

    if sources:
        lines.append("[SOURCE TRAIL]")
        lines.extend(f"- {item}" for item in sources)

    if evidence:
        lines.append("[RECENT RAW EVIDENCE]")
        lines.extend(_format_evidence_line(item) for item in evidence)

    return "\n".join(lines)
