"""Pre-research command for exchange discovery."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
import re
from datetime import datetime, timezone

from config import get_config
from core import get_standard_logger
from services.ai.providers import AIService
from services.factories import get_rag_service

logger = get_standard_logger(__name__)


def _prompt_exchange(console: Console) -> Optional[str]:
    exchange = Prompt.ask(
        "Which exchange should we research? (e.g., BIST, NYSE, NASDAQ, LSE)",
        default="",
    ).strip()
    if not exchange:
        console.print("[red]❌ Exchange name cannot be empty.[/red]")
        return None
    return exchange


def _prompt_criteria() -> Optional[str]:
    if Confirm.ask("Would you like to add additional criteria?", default=False):
        criteria = Prompt.ask(
            "Enter criteria (e.g., dividend, growth, sector filter, market cap)",
            default="",
        ).strip()
        return criteria or None
    return None


def handle_pre_research(console: Console) -> None:
    exchange = _prompt_exchange(console)
    if not exchange:
        return

    criteria = _prompt_criteria()

    console.print(
        "\n[dim]🔎 Starting pre-research, this process may take some time...[/dim]"
    )

    cfg = get_config()
    ai_steps_cfg = cfg.get("ai", {}).get("agent_steps", {})
    depth_mode = str(ai_steps_cfg.get("pre_research_depth_mode", "standard"))

    ai_service = AIService(cfg)
    try:
        result = ai_service.pre_research_exchange(
            exchange=exchange,
            criteria=criteria,
            console=console,
            depth_mode=depth_mode,
        )
    except KeyboardInterrupt:
        logger.info("Pre-research interrupted by user before completion")
        console.print("[yellow]⚠️ Pre-research cancelled.[/yellow]")
        return

    if isinstance(result, dict) and result.get("cancelled"):
        console.print("[yellow]⚠️ Pre-research cancelled.[/yellow]")
        return

    if not result or result.get("error"):
        error_msg = (
            result.get("error", "Unknown error")
            if isinstance(result, dict)
            else "Unknown error"
        )
        logger.error("Pre-research failed: %s", error_msg)
        console.print(f"[red]❌ Pre-research failed: {error_msg}[/red]")
        return

    analysis_text = result.get("analysis") if isinstance(result, dict) else None
    if analysis_text in ("</tool_call>", "<tool_call>"):
        analysis_text = ""
    if analysis_text:
        console.print(
            Panel(
                Markdown(analysis_text),
                title="🔎 Pre-Research Report",
                border_style="cyan",
            )
        )

        session_id = str(result.get("session_id") or "").strip()
        session_path = str(result.get("session_path") or "").strip()
        if session_id:
            console.print(f"[dim]Session: {session_id}[/dim]")
        if session_path:
            console.print(f"[dim]Session file: {session_path}[/dim]")

        # Raporu kaydet
        try:
            from core.paths import get_runtime_dir

            reports_dir = get_runtime_dir() / "instance" / "reports" / "pre_research"
            reports_dir.mkdir(parents=True, exist_ok=True)

            safe_exchange = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", exchange)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{safe_exchange}_research.md"
            filepath = reports_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Pre-Research Report: {exchange}\n")
                f.write(f"**Date:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
                if criteria:
                    f.write(f"**Criteria:** {criteria}\n")
                f.write("\n---\n\n")
                f.write(analysis_text)

            try:
                rag = get_rag_service(cfg)
                rag.index_pre_research(
                    exchange=exchange,
                    markdown_content=analysis_text,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as rag_exc:
                logger.debug("Pre-research RAG indexing skipped: %s", rag_exc)

            console.print(
                f"\n[green]✅ Report saved:[/green] [link={filepath}]{filepath}[/link]"
            )
        except Exception as e:
            logger.error("Report could not be saved: %s", e)
            console.print(f"[red]❌ Report could not be saved: {e}[/red]")

    else:
        console.print("[yellow]⚠️ Pre-research result came back empty.[/yellow]")
