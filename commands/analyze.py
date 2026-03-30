"""Stock analysis commands"""

from __future__ import annotations

import json
import hashlib
import time
import threading
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from rich.console import Console

from config import get_config
from core import (
    get_standard_logger,
    log_operation_start,
    log_operation_end,
    log_exception,
    BorsaException,
    console as core_console,
)
from core.cache_manager import get_unified_cache
from domain.data_processors import (
    process_and_enrich_company_data,
    sanitize_raw_company_data,
)
from services.factories import get_financial_service
from services.factories import get_rag_service
from services.ai.providers import AIService
from core.paths import get_analysis_cache_dir, get_history_dir

logger = get_standard_logger(__name__)

_cache_cfg = get_config().get("cache", {}).get("analysis_daily", {})
_analysis_mem_cache = get_unified_cache().get_ttl_cache(
    namespace="analysis_daily",
    maxsize=int(_cache_cfg.get("max_entries", 512)),
    ttl_seconds=int(_cache_cfg.get("ttl_seconds", 86400)),
)


class AnalysisCancelled(Exception):
    """Raised when an analysis is explicitly cancelled by the user."""

    pass


from domain.utils import safe_int_strict as _safe_int


def _resolve_batch_worker_limit(
    symbols: List[str],
    requested_workers: Optional[int],
    config: Optional[Dict[str, Any]],
) -> int:
    resolved_config = config or get_config()
    configured_limit = max(1, _safe_int(resolved_config.get("max_workers", 4), 4))
    requested = configured_limit if requested_workers is None else max(1, requested_workers)
    worker_count = min(len(symbols), requested, configured_limit)

    if requested_workers is not None and requested_workers > worker_count:
        logger.info(
            "Batch worker count clamped from %s to %s (configured max_workers=%s).",
            requested_workers,
            worker_count,
            configured_limit,
        )

    return max(1, worker_count)


def _safe_symbol_for_cache(symbol: str) -> str:
    return "".join(
        ch if (ch.isalnum() or ch in {"-", ".", "_"}) else "_" for ch in symbol.upper()
    )


def _stable_cache_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _analysis_cache_variant(
    investment_horizon: Optional[str], user_context: Optional[Any]
) -> str:
    payload = {
        "investment_horizon": _stable_cache_text(investment_horizon),
        "user_context": _stable_cache_text(user_context),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _daily_analysis_cache_path(
    symbol: str,
    investment_horizon: Optional[str],
    user_context: Optional[Any],
    *,
    legacy: bool = False,
) -> Path:
    day_key = datetime.now().strftime("%Y%m%d")
    variant = _analysis_cache_variant(investment_horizon, user_context)
    filename = f"cache_{_safe_symbol_for_cache(symbol)}_{day_key}_{variant}.json"
    cache_dir = get_history_dir() if legacy else get_analysis_cache_dir()
    return cache_dir / filename


def _read_daily_analysis_cache(
    symbol: str,
    investment_horizon: Optional[str],
    user_context: Optional[Any],
) -> Optional[Dict[str, Any]]:
    memory_key = f"{symbol.upper()}::{_analysis_cache_variant(investment_horizon, user_context)}"
    memory_cached = _analysis_mem_cache.get(memory_key)
    if isinstance(memory_cached, dict) and memory_cached.get("status") == "completed":
        cached_result = memory_cached.get("result")
        if isinstance(cached_result, dict):
            return cached_result

    cache_candidates = [
        _daily_analysis_cache_path(symbol, investment_horizon, user_context),
        _daily_analysis_cache_path(
            symbol,
            investment_horizon,
            user_context,
            legacy=True,
        ),
    ]
    seen_paths: set[Path] = set()
    for cache_path in cache_candidates:
        if cache_path in seen_paths:
            continue
        seen_paths.add(cache_path)
        if not cache_path.exists():
            continue

        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        if payload.get("status") != "completed":
            continue

        cached_result = payload.get("result")
        if isinstance(cached_result, dict):
            _analysis_mem_cache[memory_key] = payload
            return cached_result
    return None


def _write_daily_analysis_cache(
    symbol: str,
    result: Dict[str, Any],
    investment_horizon: Optional[str],
    user_context: Optional[Any],
) -> None:
    cache_path = _daily_analysis_cache_path(symbol, investment_horizon, user_context)
    payload = {
        "symbol": symbol,
        "status": "completed",
        "cached_at": datetime.now().isoformat(),
        "cache_variant": _analysis_cache_variant(investment_horizon, user_context),
        "investment_horizon": investment_horizon,
        "user_context": user_context,
        "result": result,
    }

    memory_key = f"{symbol.upper()}::{payload['cache_variant']}"
    _analysis_mem_cache[memory_key] = payload

    fd, temp_path = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=str(cache_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _extract_working_memory(news_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(news_analysis, dict):
        return {}
    working_memory = news_analysis.get("working_memory")
    return working_memory if isinstance(working_memory, dict) else {}


def _count_news_sources(news_analysis: Optional[Dict[str, Any]]) -> int:
    working_memory = _extract_working_memory(news_analysis)
    sources = working_memory.get("sources_consulted")
    if isinstance(sources, list):
        return len([source for source in sources if source])
    return 1 if isinstance(news_analysis, dict) and news_analysis.get("has_news_data") else 0


def _normalize_decision_summary(workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    decision_summary = workflow_result.get("decision_summary")
    if isinstance(decision_summary, dict):
        return decision_summary

    investment_decision = workflow_result.get("investment_decision")
    decision_text = ""
    if isinstance(investment_decision, dict):
        decision_text = str(investment_decision.get("content") or "")
    elif isinstance(investment_decision, str):
        decision_text = investment_decision

    return {
        "analysis": decision_text,
        "decision": "N/A",
        "risk_score": 50,
        "reasoning": decision_text,
    }


def _normalize_news_analysis(
    workflow_result: Dict[str, Any],
    fallback: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    news_analysis = workflow_result.get("news_analysis")
    if isinstance(news_analysis, dict):
        return news_analysis
    return fallback if isinstance(fallback, dict) else None


def extract_analysis_highlights(result: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(result, dict):
        return {"news_summary": "", "final_decision": ""}

    workflow_result = result.get("workflow")
    if not isinstance(workflow_result, dict):
        workflow_result = {}

    news_analysis = result.get("news_analysis")
    if not isinstance(news_analysis, dict):
        news_analysis = _normalize_news_analysis(workflow_result)

    ai_analysis = result.get("ai_analysis")
    if not isinstance(ai_analysis, dict):
        ai_analysis = _normalize_decision_summary(workflow_result)

    final_decision = (
        ai_analysis.get("analysis")
        or ai_analysis.get("reasoning")
        or ""
    )

    return {
        "news_summary": str((news_analysis or {}).get("analysis") or ""),
        "final_decision": str(final_decision or ""),
    }


def _run_agentic_news_discovery(
    *,
    ai_service: AIService,
    company_data: Dict[str, Any],
    symbol: str,
    is_batch: bool,
    console: Optional[Console],
    check_cancel: Callable[[], None],
    max_continuations: int = 5,
) -> Dict[str, Any]:
    def ask_user_continue(state: Dict[str, Any]) -> bool:
        """Ask user if they want to continue when max steps reached."""
        if is_batch or not console:
            return False

        try:
            from rich.prompt import Confirm

            return Confirm.ask(
                "\\n[bold yellow]Maximum operation limit reached. Do you want to continue?[/bold yellow]",
                default=True,
            )
        except Exception as exc:
            logger.warning("[%s] User interaction failed: %s", symbol, exc)
            return False

    resume_state = None
    continuation_count = 0

    while True:
        result = ai_service.analyze_company_with_ai(
            company_data=company_data,
            news_context=None,
            console=console,
            resume_state=resume_state,
            on_max_steps_callback=ask_user_continue,
        )

        check_cancel()

        if not isinstance(result, dict):
            raise RuntimeError("Agentic AI returned invalid response type")

        if not result.get("needs_continuation"):
            return result

        resume_state = result.get("state")
        if not isinstance(resume_state, dict):
            return result

        continuation_count += 1
        if continuation_count > max_continuations:
            logger.warning(
                "[%s] Agentic AI continuation budget exhausted after %d retries.",
                symbol,
                max_continuations,
            )
            return result

        if console:
            console.print(f"[dim]Continuation {continuation_count}/{max_continuations}[/dim]")


def analyze_single_stock(
    symbol: str,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    investment_horizon: Optional[str] = None,
    user_context: Optional[str] = None,
    is_batch: bool = False,
    console: Optional[Console] = None,
    status_ctx: Optional[Any] = None,
    cancel_event: Optional[threading.Event] = None,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single stock with AI

    Args:
        symbol: Stock ticker symbol
        config: Configuration dictionary
        progress_callback: Optional callable invoked with human-readable status updates
        investment_horizon: User's investment time horizon (e.g., "short-term", "medium-term", "long-term")
        user_context: Optional user-provided context (e.g., budget, experience level, goals)

    Returns:
        Analysis results or None if failed
    """
    operation_start = log_operation_start(
        logger,
        f"Single Stock Analysis: {symbol}",
        {
            "symbol": symbol,
            "horizon": investment_horizon,
            "has_user_context": bool(user_context),
        },
    )

    def _check_cancel() -> None:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelled(f"Analysis cancelled for {symbol}")

    def _notify(message: str) -> None:
        _check_cancel()
        if progress_callback:
            try:
                progress_callback(message)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Progress callback failed", exc_info=True)
        else:
            # Only log if no progress callback (to avoid duplicate messages)
            logger.info("[%s] %s", symbol, message)

    if not force_refresh:
        cached_result = _read_daily_analysis_cache(
            symbol, investment_horizon, user_context
        )
        if cached_result:
            logger.info(
                "[%s] Daily analysis cache hit; reusing completed result.", symbol
            )
            _notify("⚡ Using cached analysis result for today.")
            log_operation_end(
                logger,
                f"Single Stock Analysis: {symbol}",
                operation_start,
                success=True,
                details={"cache": "daily_analysis_hit"},
            )
            return cached_result

    # ══════════════════════════════════════════════════════════════
    # WORKFLOW PHASES:
    # Phase 1: Data Collection
    #   - 1.1: YFinance data and news (detailed scraping with content extraction)
    #   - 1.2: Google News scraping (MANDATORY)
    # Phase 2: Data Processing
    #   - 2.1: Processing and enriching financial data
    # Phase 3: AI Analysis
    #   - 3.1: News and market analysis (yfinance + Google News scraping)
    #   - 3.2: Investment decision generation
    # Phase 4: Results
    #   - 4.1: Compiling and presenting results
    # ══════════════════════════════════════════════════════════════

    # Google News scraping is now mandatory
    # use_google_news_scrape = True

    def _step(phase: int, step: int, description: str) -> None:
        step_msg = f"Phase {phase} - Step {phase}.{step}: {description}"
        _notify(step_msg)

    try:
        _check_cancel()
        # ══════════════════════════════════════════════════════════════
        # PHASE 1: DATA COLLECTION
        # ══════════════════════════════════════════════════════════════

        # Step 1.1: Gathering YFinance data
        _step(1, 1, "📊 Gathering market data from YFinance...")
        step1_start = datetime.now()

        financial_service = get_financial_service(config)
        company_data = financial_service.get_company_data(symbol)

        # Sanitize immediately after fetching so downstream phases and returned raw data are consistent
        if company_data:
            company_data = sanitize_raw_company_data(company_data)

        step1_duration = (datetime.now() - step1_start).total_seconds()

        if not company_data:
            logger.error(
                "[%s] Phase 1.1 - Company data could not be retrieved - Duration: %.2fs",
                symbol,
                step1_duration,
            )
            _notify(
                "❌ Unable to find this stock. Please verify the ticker symbol is correct."
            )
            log_operation_end(
                logger,
                f"Single Stock Analysis: {symbol}",
                operation_start,
                success=False,
            )
            return None

        logger.info(
            "[%s] Phase 1.1 completed - Company data retrieved - Duration: %.2fs, Data keys: %d",
            symbol,
            step1_duration,
            len(company_data.keys()),
        )

        # Show company profile information
        company_profile = company_data.get("company_profile", {})
        company_name = company_profile.get("name") or company_data.get(
            "longName", "N/A"
        )
        sector = company_profile.get("sector") or company_data.get("sector", "N/A")
        industry = company_profile.get("industry") or company_data.get(
            "industry", "N/A"
        )
        market_cap = company_data.get("marketCap", 0)

        # Log company details without excessive terminal output
        logger.info(
            "[%s] Phase 1.1 - %s | Sector: %s | Industry: %s",
            symbol,
            company_name,
            sector,
            industry,
        )

        # Initialize AIService early for reuse throughout the workflow
        ai_service = AIService(config)

        # ══════════════════════════════════════════════════════════════
        # PHASE 1 - Step 1.2 (MANDATORY): Agentic AI News Discovery
        # ══════════════════════════════════════════════════════════════
        _step(1, 2, "🧭 Starting Agentic AI news discovery and analysis loop...")
        step_agentic_start = datetime.now()
        agentic_news_analysis = None

        try:
            agentic_news_analysis = _run_agentic_news_discovery(
                ai_service=ai_service,
                company_data=company_data,
                symbol=symbol,
                is_batch=is_batch,
                console=console,
                check_cancel=_check_cancel,
            )

            if agentic_news_analysis.get("step_limit_reached"):
                logger.warning(
                    "[%s] Agentic AI reached max steps - returning partial results",
                    symbol,
                )
                if console:
                    warning = agentic_news_analysis.get(
                        "warning",
                        "Maximum step limit reached - returning partial results",
                    )
                    console.print(f"[yellow]⚠️  {warning}[/yellow]")

            if agentic_news_analysis.get("error"):
                raise RuntimeError(
                    agentic_news_analysis.get("error", "Agentic AI analysis failed")
                )

            step_agentic_duration = (
                datetime.now() - step_agentic_start
            ).total_seconds()
            logger.info(
                "[%s] Phase 1.2 completed - Agentic AI news analysis generated - Duration: %.2fs",
                symbol,
                step_agentic_duration,
            )
        except Exception as e:
            step_agentic_duration = (
                datetime.now() - step_agentic_start
            ).total_seconds()
            logger.error(
                "[%s] Phase 1.2 - Agentic AI news discovery failed - Duration: %.2fs, Error: %s",
                symbol,
                step_agentic_duration,
                str(e)[:100],
            )
            _notify("❌ Agentic AI news discovery failed. Analysis stopped.")
            log_operation_end(
                logger,
                f"Single Stock Analysis: {symbol}",
                operation_start,
                success=False,
            )
            return None

        logger.info(
            "[%s] Phase 1 completed - Agentic AI source coverage: %d",
            symbol,
            _count_news_sources(agentic_news_analysis),
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 2: DATA PROCESSING
        # ══════════════════════════════════════════════════════════════

        # Step 2.1: Processing and enriching financial data
        _step(2, 1, "📈 Processing company fundamentals and financial metrics...")
        step2_start = datetime.now()

        processed_company_data = None
        try:
            processed = process_and_enrich_company_data(
                company_data,
                {symbol.upper(): company_data},
            )
            processed_company_data = (
                asdict(processed) if is_dataclass(processed) else processed
            )

            step2_duration = (datetime.now() - step2_start).total_seconds()

            # Info about processed data
            data_keys = (
                len(processed_company_data.keys())
                if isinstance(processed_company_data, dict)
                else 0
            )

            logger.info(
                "[%s] Phase 2.1 completed - Metrics: %d, Duration: %.2fs",
                symbol,
                data_keys,
                step2_duration,
            )
        except Exception as exc:
            step2_duration = (datetime.now() - step2_start).total_seconds()
            logger.warning(
                "[%s] Phase 2.1 - Company data processing failed (raw data will be used) - Duration: %.2fs, Error: %s",
                symbol,
                step2_duration,
                str(exc)[:100],
            )
            _notify("⚠️ Advanced metrics unavailable; using basic financial data.")

        # ══════════════════════════════════════════════════════════════
        # PHASE 3: AI ANALYSIS
        # ══════════════════════════════════════════════════════════════

        # Step 3.1-3.2: AI news analysis and Investment decision
        _step(3, 1, "🤖 Starting AI-powered analysis workflow...")
        step4_start = datetime.now()

        logger.info(
            "[%s] Phase 3 - AI workflow starting - News: %d, Metrics: %d",
            symbol,
            _count_news_sources(agentic_news_analysis),
            len(company_data.keys()),
        )

        workflow_result = ai_service.full_ai_google_workflow(
            company_data=company_data,
            articles=[],
            has_news_permission=True,
            investment_horizon=investment_horizon,
            user_context=user_context,
            console=console,
            news_analysis_override=agentic_news_analysis,
            skip_news_analysis=True,
        )

        step4_duration = (datetime.now() - step4_start).total_seconds()

        if not isinstance(workflow_result, dict):
            logger.error(
                "[%s] Phase 3 - AI workflow unexpected result type: %s - Duration: %.2fs",
                symbol,
                type(workflow_result).__name__,
                step4_duration,
            )
            _notify("❌ AI analysis encountered an unexpected error. Please try again.")
            log_operation_end(
                logger,
                f"Single Stock Analysis: {symbol}",
                operation_start,
                success=False,
            )
            return None

        if workflow_result.get("error"):
            logger.error(
                "[%s] Phase 3 - AI workflow error: %s - Duration: %.2fs",
                symbol,
                workflow_result["error"],
                step4_duration,
            )
            _notify("❌ AI analysis could not complete. Check your API configuration.")
            log_operation_end(
                logger,
                f"Single Stock Analysis: {symbol}",
                operation_start,
                success=False,
            )
            return None

        logger.info(
            "[%s] Phase 3 completed - AI workflow duration: %.2fs",
            symbol,
            step4_duration,
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 4: RESULTS
        # ══════════════════════════════════════════════════════════════

        # Step 4.1: Compiling and presenting results
        _step(4, 1, "✨ Compiling results and preparing investment insights...")

        decision_summary = _normalize_decision_summary(workflow_result)
        news_analysis = _normalize_news_analysis(
            workflow_result,
            fallback=agentic_news_analysis,
        )
        source_count = _count_news_sources(news_analysis or agentic_news_analysis)

        result = {
            "symbol": symbol,
            "company_data": (
                processed_company_data if processed_company_data else company_data
            ),  # Prefer processed data for consistency
            "raw_company_data": company_data,  # Keep raw data available if needed
            "processed_company_data": processed_company_data,
            "workflow": workflow_result,
            "ai_analysis": decision_summary,
            "news_analysis": news_analysis,
            "news_count": source_count,
            "news_sources": {
                "agentic_ai": True,
                "sources_consulted": source_count,
            },
            "investment_horizon": investment_horizon,
            "user_context": user_context,
        }

        # Phase 4 summary
        total_duration = (datetime.now() - operation_start).total_seconds()
        decision = decision_summary.get("decision", "N/A")
        risk_score = decision_summary.get("risk_score", "N/A")

        # Log final summary
        logger.info(
            "[%s] Analysis completed - Decision: %s, Risk: %s, News: %d, Duration: %.2fs",
            symbol,
            decision,
            risk_score,
            source_count,
            total_duration,
        )

        # Log total operation duration
        log_operation_end(
            logger,
            f"Single Stock Analysis: {symbol}",
            operation_start,
            success=True,
            details={
                "news_count": source_count,
                "decision": decision_summary.get("decision", "N/A"),
            },
        )

        try:
            _write_daily_analysis_cache(
                symbol,
                result,
                investment_horizon=investment_horizon,
                user_context=user_context,
            )
        except Exception as cache_exc:
            logger.debug(
                "[%s] Daily analysis cache write skipped: %s", symbol, cache_exc
            )

        try:
            news_output = None
            final_output = None

            workflow_node = (
                result.get("workflow", {}) if isinstance(result, dict) else {}
            )
            if isinstance(workflow_node, dict):
                news_node = workflow_node.get("news_analysis")
                decision_node = workflow_node.get("investment_decision")

                if isinstance(news_node, dict):
                    news_output = news_node.get("analysis")
                elif isinstance(news_node, str):
                    news_output = news_node

                if isinstance(decision_node, dict):
                    final_output = decision_node.get("content")
                elif isinstance(decision_node, str):
                    final_output = decision_node

            if not final_output and isinstance(result.get("ai_analysis"), dict):
                final_output = result["ai_analysis"].get("analysis")

            exchange_name = (
                str(company_data.get("exchange") or "")
                or str(company_data.get("fullExchangeName") or "")
                or str(company_profile.get("exchange") or "")
            )

            rag = get_rag_service(config)
            indexed_count = rag.index_analysis(
                symbol=symbol,
                news_output=news_output,
                final_output=final_output,
                timestamp=datetime.now(timezone.utc).isoformat(),
                exchange=exchange_name,
            )
            if indexed_count:
                logger.info(
                    "[%s] RAG indexed analysis chunks: %d", symbol, indexed_count
                )
        except Exception as rag_exc:
            logger.debug("[%s] RAG analysis indexing skipped: %s", symbol, rag_exc)

        return result

    except AnalysisCancelled:
        logger.info("[%s] Analysis cancelled by user request.", symbol)
        log_operation_end(
            logger, f"Single Stock Analysis: {symbol}", operation_start, success=False
        )
        return None
    except BorsaException as e:
        logger.error("[%s] BorsaException: %s", symbol, e.error_id)
        log_operation_end(
            logger, f"Single Stock Analysis: {symbol}", operation_start, success=False
        )
        return None
    except Exception as e:
        log_exception(logger, f"Analysis failed: {symbol}", e, {"symbol": symbol})
        log_operation_end(
            logger, f"Single Stock Analysis: {symbol}", operation_start, success=False
        )
        return None


def analyze_multiple_stocks(
    symbols: List[str],
    max_workers: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    quiet: bool = False,
    investment_horizon: Optional[str] = None,
    user_context: Optional[str] = None,
    wait_time: float = 0.0,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Analyze multiple stocks in parallel

    Everything is the same as single stock analysis, except multiple stocks
    are analyzed in parallel at the same time. For each stock:
    - Gathering financial data (YFinance)
    - Google News scraping (mandatory)
    - AI analysis (news analysis + investment decision)
    - Full output (same structure as single analysis)

    Worker count defaults to the number of symbols.

    Args:
        symbols: List of stock ticker symbols
        max_workers: Number of parallel workers (default: len(symbols))
        config: Configuration dictionary
        console: Optional Rich console instance to reuse for progress output
        quiet: Suppress progress output when True
        investment_horizon: Investment time horizon for all symbols
        user_context: Optional user-provided context for all symbols
        wait_time: Time to wait (in seconds) between analyses in each thread
        cancel_event: Optional threading.Event to cancel ongoing analyses

    Returns:
        Dictionary mapping symbol to full analysis results (same structure as analyze_single_stock)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
    )

    max_workers = _resolve_batch_worker_limit(symbols, max_workers, config)

    operation_start = log_operation_start(
        logger,
        "Batch Stock Analysis",
        {
            "symbol_count": len(symbols),
            "max_workers": max_workers,
            "wait_time": wait_time,
        },
    )

    logger.info(
        "Batch analysis starting - Symbols: %s (Workers: %d, Wait: %.1fs)",
        ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else ""),
        max_workers,
        wait_time,
    )

    results = {}
    _console = console or core_console

    def _run_single_analysis(symbol: str) -> tuple:
        """
        Run full analysis for a single symbol.

        This function does exactly the same things as analyze_single_stock:
        - Gathering financial data
        - Google News scraping
        - AI analysis
        - Full output structure
        """
        try:
            if cancel_event and cancel_event.is_set():
                return (symbol, None)

            # Progress callback - show if not quiet
            def progress_cb(msg: str) -> None:
                if not quiet:
                    _console.print(f"[dim][{symbol}][/dim] {msg}")

            # Call analyze_single_stock - pass ALL parameters
            result = analyze_single_stock(
                symbol=symbol,
                config=config,
                progress_callback=progress_cb if not quiet else None,
                investment_horizon=investment_horizon,
                user_context=user_context,
                is_batch=True,
                console=_console,
                cancel_event=cancel_event,
            )

            # Wait time (rest break) between analyses
            if wait_time > 0:
                time.sleep(wait_time)

            return (symbol, result)
        except Exception as e:
            logger.error("Analysis error (%s): %s", symbol, e)
            return (symbol, None)

    try:
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=_console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Batch analysis: {len(symbols)} symbols...",
                    total=len(symbols),
                )

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_symbol = {
                        executor.submit(_run_single_analysis, symbol): symbol
                        for symbol in symbols
                        if not (cancel_event and cancel_event.is_set())
                    }

                    for future in as_completed(future_to_symbol):
                        if cancel_event and cancel_event.is_set():
                            break
                        symbol, result = future.result()
                        results[symbol] = result
                        status = "[green]✓[/green]" if result else "[red]✗[/red]"
                        progress.update(
                            task, advance=1, description=f"{status} {symbol} completed"
                        )
        else:
            # Quiet mode - without progress bar
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(_run_single_analysis, symbol): symbol
                    for symbol in symbols
                    if not (cancel_event and cancel_event.is_set())
                }

                for future in as_completed(future_to_symbol):
                    if cancel_event and cancel_event.is_set():
                        break
                    symbol, result = future.result()
                    results[symbol] = result

        success_count = sum(1 for v in results.values() if v is not None)
        failure_count = len(results) - success_count

        log_operation_end(
            logger,
            "Batch Stock Analysis",
            operation_start,
            success=True,
            details={
                "total": len(results),
                "success": success_count,
                "failed": failure_count,
            },
        )

        logger.info(
            "Batch analysis completed - Success: %d, Failed: %d",
            success_count,
            failure_count,
        )

        return results

    except BorsaException as e:
        logger.error("Batch analysis BorsaException: %s", e.error_id)
        log_operation_end(
            logger, "Batch Stock Analysis", operation_start, success=False
        )
        return {symbol: None for symbol in symbols}
    except Exception as e:
        log_exception(logger, "Batch analysis failed", e, {"symbols": symbols[:5]})
        log_operation_end(
            logger, "Batch Stock Analysis", operation_start, success=False
        )
        return {symbol: None for symbol in symbols}
