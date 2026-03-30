from __future__ import annotations

from services.ai.providers.tool_journal_utils import (
    format_tool_journal_for_prompt,
    normalize_tool_journal,
    normalize_tool_journal_step,
)


def test_format_tool_journal_for_prompt_groups_by_step() -> None:
    journal = [
        normalize_tool_journal_step(
            1,
            [
                {"name": "search_web", "args": {"query": "AKBNK outlook"}},
                {"name": "fetch_url_content", "args": {"url": "https://example.com"}},
            ],
            assistant_summary="First search for broad coverage, then inspect the best source.",
            tool_results=[
                {"name": "search_web", "status": "ok", "summary": "items=5, query=AKBNK outlook"},
                {"name": "fetch_url_content", "status": "ok", "summary": "url=https://example.com"},
            ],
        ),
        normalize_tool_journal_step(
            2,
            [
                {"name": "update_working_memory", "args": {"new_facts": ["Revenue up"]}},
            ],
            deferred_tools=["search_news"],
            memory_updates=[{"summary": "facts=1"}],
            notes=["prepared the next news lookup"],
        ),
    ]

    prompt = format_tool_journal_for_prompt(journal)

    assert "Chronological step trace in this run:" in prompt
    assert "Step 1" in prompt
    assert "assistant: First search for broad coverage" in prompt
    assert "search_web" in prompt
    assert "ok: items=5" in prompt
    assert "fetch_url_content" in prompt
    assert "Step 2" in prompt
    assert "memory_update(facts=1)" in prompt
    assert "deferred: search_news" in prompt
    assert "notes: prepared the next news lookup" in prompt


def test_normalize_tool_journal_discards_invalid_entries() -> None:
    journal = normalize_tool_journal(
        [
            {"step": "3", "tools": [{"name": "search_memory", "args": {"query": "ABC"}}]},
            {"step": "bad", "tools": [{"name": "finish", "args": {}}]},
            {"step": 4, "tools": "invalid"},
        ]
    )

    assert len(journal) == 2
    assert journal[0]["step"] == 3
    assert journal[0]["tools"][0]["name"] == "search_memory"
    assert journal[1]["step"] == 2
    assert journal[1]["tools"][0]["name"] == "finish"
