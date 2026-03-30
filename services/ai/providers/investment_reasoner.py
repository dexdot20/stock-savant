import json
import copy
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from config import get_config
from core import APIError
from domain.confidence import calculate_company_confidence
from services.utils.calculations import make_json_serializable
from services.ai.memory_formatter import (
    format_working_memory_evidence_pack,
    format_working_memory_for_llm,
)

from .response_parser import ResponseParser


class InvestmentReasoner:
    """Final investment recommendation stage."""

    def __init__(
        self,
        logger,
        prompt_resolver,
        request_executor,
        response_parser: ResponseParser,
    ) -> None:
        self.logger = logger
        self._prompt_resolver = prompt_resolver
        self._request_executor = request_executor
        self._response_parser = response_parser

    @staticmethod
    def _build_working_memory_context(news_analysis: Optional[Dict[str, Any]]) -> str:
        """Builds a compact, structured Working Memory section for the reasoner prompt."""
        if not isinstance(news_analysis, dict):
            return ""

        working_memory = news_analysis.get("working_memory")
        if not isinstance(working_memory, dict):
            return ""
        return format_working_memory_for_llm(working_memory, style="reasoner")

    @staticmethod
    def _build_evidence_pack_context(news_analysis: Optional[Dict[str, Any]]) -> str:
        if not isinstance(news_analysis, dict):
            return ""

        working_memory = news_analysis.get("working_memory")
        if not isinstance(working_memory, dict):
            return ""
        return format_working_memory_evidence_pack(working_memory)

    @staticmethod
    def _resolve_confidence_bundle(
        company_data: Dict[str, Any],
        news_analysis: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        company_report = company_data.get("confidence_report")
        if not isinstance(company_report, dict):
            company_report = calculate_company_confidence(company_data)

        company_conf_pct = float(company_report.get("confidence_pct") or 0.0)
        news_conf = 0.0
        news_level = "very_low"
        news_factors: Dict[str, Any] = {}
        news_warnings = []
        if isinstance(news_analysis, dict):
            try:
                news_conf = float(news_analysis.get("confidence") or 0.0) * 100.0
            except (TypeError, ValueError):
                news_conf = 0.0
            news_level = str(news_analysis.get("confidence_level") or "very_low")
            news_factors = news_analysis.get("confidence_factors") or {}
            news_warnings = list(news_analysis.get("confidence_warnings") or [])

        has_news = isinstance(news_analysis, dict) and bool(news_analysis)
        final_pct = (
            (company_conf_pct * 0.65) + (news_conf * 0.35)
            if has_news
            else (company_conf_pct * 0.85)
        )
        final_pct = max(0.0, min(100.0, round(final_pct, 2)))

        def _level(score_pct: float) -> str:
            if score_pct >= 85:
                return "very_high"
            if score_pct >= 70:
                return "high"
            if score_pct >= 55:
                return "moderate"
            if score_pct >= 35:
                return "low"
            return "very_low"

        warnings = list(company_report.get("warnings") or [])
        warnings.extend(item for item in news_warnings if item not in warnings)

        return {
            "company_confidence_pct": round(company_conf_pct, 2),
            "company_confidence_level": str(
                company_report.get("confidence_level") or "very_low"
            ),
            "news_confidence_pct": round(news_conf, 2),
            "news_confidence_level": news_level,
            "confidence": round(final_pct / 100.0, 4),
            "confidence_pct": final_pct,
            "confidence_level": _level(final_pct),
            "confidence_factors": {
                "company": company_report.get("factors") or {},
                "news": news_factors,
                "blend": {
                    "company_weight": 0.65 if has_news else 0.85,
                    "news_weight": 0.35 if has_news else 0.0,
                },
            },
            "confidence_warnings": warnings,
        }

    def decide(
        self,
        company_data: Dict[str, Any],
        news_analysis: Optional[Dict[str, Any]],
        time_getter: Optional[Callable[[float], Optional[float]]] = None,
        investment_horizon: Optional[str] = None,
        user_context: Optional[str] = None,
        console: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate investment decision based on company data and news analysis.

        Args:
            company_data: Company financial data from yfinance (FULL DATA)
            news_analysis: AI-generated news analysis (News Analyzer FULL OUTPUT)
            time_getter: Optional function that takes min_seconds and returns remaining time or None
            investment_horizon: User's investment time horizon
            user_context: Optional user-provided context (e.g., budget, experience level, goals)

        Returns:
            Tuple of (decision_response, decision_summary)
        """
        try:
            today = datetime.now().strftime("%d %B %Y")
            system_prompt = f"Today's Date: {today}\n\n" + self._prompt_resolver(
                "reasoner_system"
            )

            # Get config once; extract all needed values up-front.
            config = get_config()
            ai_cfg = config.get("ai", {})
            output_lang = ai_cfg.get("output_language", "English")
            optimize_shares = ai_cfg.get("optimize_shares_history", True)
            system_prompt = system_prompt.replace("{output_language}", output_lang)

            confidence_bundle = self._resolve_confidence_bundle(
                company_data,
                news_analysis,
            )

            data_quality_level = str(
                company_data.get("data_quality")
                or company_data.get("confidence_level")
                or "unknown"
            )
            data_quality_score = confidence_bundle.get("company_confidence_pct", 0.0)
            news_confidence = confidence_bundle.get("news_confidence_pct", 0.0)

            final_company_data = company_data
            if optimize_shares and "sharesFull" in company_data:
                # Deep copy to avoid mutating original data
                final_company_data = copy.deepcopy(company_data)
                shares_full = final_company_data.get("sharesFull")
                if isinstance(shares_full, dict) and "history" in shares_full:
                    history_count = len(shares_full.get("history", []))
                    shares_full.pop("history", None)
                    self.logger.debug(
                        "Token optimization: Removed %d sharesFull history entries for AI payload",
                        history_count,
                    )

            decision_payload = {
                "company_data": make_json_serializable(final_company_data),
                "news_analysis": (
                    make_json_serializable(news_analysis) if news_analysis else {}
                ),
                "investment_horizon": investment_horizon or "not specified",
                "user_context": user_context or None,
                "meta_data": {
                    "calculated_data_quality_score": data_quality_score,
                    "calculated_news_confidence_score": news_confidence,
                    "data_quality_level": data_quality_level,
                    "company_confidence_level": confidence_bundle.get(
                        "company_confidence_level", "very_low"
                    ),
                    "news_confidence_level": confidence_bundle.get(
                        "news_confidence_level", "very_low"
                    ),
                    "confidence_warnings": confidence_bundle.get(
                        "confidence_warnings", []
                    ),
                },
            }

            decision_payload = make_json_serializable(decision_payload)
            # Compact JSON serialization: separators used instead of indent=2.
            # indent=2 produces %20-35 extra characters in large payloads; model understands flat JSON
            # perfectly, so human-readable whitespace is unnecessary.
            json_context = json.dumps(
                decision_payload, separators=(",", ":"), ensure_ascii=False
            )

            self.logger.debug(
                "Decision analysis JSON payload: %d characters, user_context: %s",
                len(json_context),
                bool(user_context),
            )

            # Build investment horizon context
            horizon_context = ""
            if investment_horizon:
                horizon_descriptions = {
                    "short-term": "The investor is focused on SHORT-TERM gains (days to weeks). Prioritize momentum, technical signals, and near-term catalysts.",
                    "medium-term": "The investor is focused on MEDIUM-TERM returns (months). Balance growth potential with risk management.",
                    "long-term": "The investor is focused on LONG-TERM value (years). Prioritize fundamental strength, sustainable growth, and compounding potential.",
                }
                horizon_context = f"\n\nINVESTMENT HORIZON CONTEXT:\n{horizon_descriptions.get(investment_horizon, f'Investment horizon: {investment_horizon}')}"
            else:
                # Default to multi-horizon instruction if not specified
                horizon_context = (
                    "\n\nINVESTMENT HORIZON CONTEXT:\n"
                    "No specific horizon provided. You MUST provide a nuanced recommendation that distinguishes between:\n"
                    "- Short-term (Trading/Momentum)\n"
                    "- Medium-term (Swing/Growth)\n"
                    "- Long-term (Value/Investing)\n"
                    "Explicitly state if the decision differs by horizon (e.g., 'BUY for long-term, but WAIT for short-term due to volatility')."
                )

            # Build user context section if provided
            user_context_section = ""
            if user_context:
                user_context_section = (
                    f"\n\nINVESTOR PROFILE CONTEXT:\n"
                    f"The investor has provided the following personal context that should be considered in your recommendation:\n"
                    f'"{user_context}"\n'
                    f"Tailor your advice to be relevant and actionable based on this context. "
                    f"Consider their experience level, budget constraints, and stated goals when making recommendations."
                )

            working_memory_section = self._build_working_memory_context(news_analysis)
            evidence_pack_section = self._build_evidence_pack_context(news_analysis)

            user_content = (
                f"Investment Analysis Data (JSON):\n\n```json\n{json_context}\n```\n\n"
                "Notes:\n"
                "- `company_data` contains the complete Yahoo Finance dataset (all metrics, statements, ownership, insider, ESG).\n"
                "- `news_analysis` contains the complete output from News Analyzer AI (sentiment, themes, risks, opportunities).\n"
                "- `meta_data` contains system-calculated quality scores. Use these to weight your confidence.\n"
                "- `STRUCTURED RESEARCH FINDINGS` contains persisted working memory extracted by the autonomous agent.\n"
                f"{horizon_context}"
                f"{user_context_section}\n\n"
                f"{working_memory_section}\n\n"
                f"{evidence_pack_section}\n\n"
                "Based on the provided financial data, news analysis, investment horizon, and investor profile context, "
                "deliver an investment recommendation with "
                "category scores, total score, decision (BUY/WAIT/SELL), decision_strength (STRONG/MODERATE/WEAK), and detailed "
                "justification consistent with the scoring rubric."
            )

            decision_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            timeout_override = time_getter(5.0) if time_getter else None
            try:
                decision_response = self._request_executor.send(
                    decision_prompt,
                    "reasoner",
                    timeout_override=timeout_override,
                    console=console,
                )
            except Exception as exc:
                self.logger.error("Reasoner AI request failed: %s", exc)
                raise APIError(f"Investment Reasoner failed: {exc}") from exc

            content = decision_response.get("content", "")
            if not content.strip():
                self.logger.error("Reasoner returned empty response")
                raise APIError("Investment Reasoner returned empty response")

        except Exception as exc:
            self.logger.error(
                "Critical error in investment reasoning: %s",
                exc,
                exc_info=True,
            )
            # Re-raise to let the command handler deal with it (show error UI)
            raise

        decision_length = len(decision_response.get("content", ""))
        self.logger.info(
            "Investment decision completed: %s characters", decision_length
        )

        # Determine strict fallback state
        is_fallback = False

        # For display purposes (internal logic only), set defaults if tracking needed
        parsed_decision = {
            "reasoning": decision_response.get("content", ""),
            "decision": "CHECK_REPORT",
            "risk_score": 0,
        }

        # Legacy confidence tracking removed - using static defaults

        decision_summary = {
            "analysis": decision_response.get("content", ""),
            "decision": parsed_decision.get("decision", "N/A"),
            "decision_strength": parsed_decision.get("decision_strength", "MODERATE"),
            "risk_score": parsed_decision.get("risk_score", 50),
            "reasoning": parsed_decision.get(
                "reasoning", decision_response.get("content", "")
            ),
            "structured": parsed_decision.get("structured"),
            "scores": parsed_decision.get("scores"),
            "analysis_steps": parsed_decision.get("analysis_steps"),
            "total_score": parsed_decision.get("total_score"),
            "thesis": parsed_decision.get("thesis"),
            "confidence": confidence_bundle.get("confidence", 0.0),
            "confidence_level": confidence_bundle.get(
                "confidence_level", "very_low"
            ),
            "confidence_factors": confidence_bundle.get("confidence_factors", {}),
            "confidence_warnings": confidence_bundle.get(
                "confidence_warnings", []
            ),
            "model_used": decision_response.get("model", "unknown"),
            "usage": decision_response.get("usage"),
            "fallback_used": is_fallback,
        }

        return decision_response, decision_summary

    async def decide_async(
        self,
        company_data: Dict[str, Any],
        news_analysis: Optional[Dict[str, Any]],
        time_getter: Optional[Callable[[float], Optional[float]]] = None,
        investment_horizon: Optional[str] = None,
        user_context: Optional[str] = None,
        console: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate investment decision based on company data and news analysis (ASYNC).

        Args:
            company_data: Company financial data from yfinance (FULL DATA)
            news_analysis: AI-generated news analysis (News Analyzer FULL OUTPUT)
            time_getter: Optional function that takes min_seconds and returns remaining time or None
            investment_horizon: User's investment time horizon
            user_context: Optional user-provided context (e.g., budget, experience level, goals)

        Returns:
            Tuple of (decision_response, decision_summary)
        """
        try:
            today = datetime.now().strftime("%d %B %Y")
            system_prompt = f"Today's Date: {today}\n\n" + self._prompt_resolver(
                "reasoner_system"
            )

            # Get config once; extract all needed values up-front.
            config = get_config()
            ai_cfg = config.get("ai", {})
            output_lang = ai_cfg.get("output_language", "English")
            optimize_shares = ai_cfg.get("optimize_shares_history", True)
            system_prompt = system_prompt.replace("{output_language}", output_lang)

            confidence_bundle = self._resolve_confidence_bundle(
                company_data,
                news_analysis,
            )
            data_quality_level = str(
                company_data.get("data_quality")
                or company_data.get("confidence_level")
                or "unknown"
            )
            data_quality_score = confidence_bundle.get("company_confidence_pct", 0.0)
            news_confidence = confidence_bundle.get("news_confidence_pct", 0.0)

            final_company_data = company_data
            if optimize_shares and "sharesFull" in company_data:
                final_company_data = copy.deepcopy(company_data)
                shares_full = final_company_data.get("sharesFull")
                if isinstance(shares_full, dict) and "history" in shares_full:
                    shares_full.pop("history", None)

            decision_payload = {
                "company_data": make_json_serializable(final_company_data),
                "news_analysis": (
                    make_json_serializable(news_analysis) if news_analysis else {}
                ),
                "investment_horizon": investment_horizon or "not specified",
                "user_context": user_context or None,
                "meta_data": {
                    "calculated_data_quality_score": data_quality_score,
                    "calculated_news_confidence_score": news_confidence,
                    "data_quality_level": data_quality_level,
                    "company_confidence_level": confidence_bundle.get(
                        "company_confidence_level", "very_low"
                    ),
                    "news_confidence_level": confidence_bundle.get(
                        "news_confidence_level", "very_low"
                    ),
                    "confidence_warnings": confidence_bundle.get(
                        "confidence_warnings", []
                    ),
                },
            }

            decision_payload = make_json_serializable(decision_payload)
            json_context = json.dumps(
                decision_payload, separators=(",", ":"), ensure_ascii=False
            )

            # Build investment horizon context
            horizon_context = ""
            if investment_horizon:
                horizon_descriptions = {
                    "short-term": "The investor is focused on SHORT-TERM gains (days to weeks). Prioritize momentum, technical signals, and near-term catalysts.",
                    "medium-term": "The investor is focused on MEDIUM-TERM returns (months). Balance growth potential with risk management.",
                    "long-term": "The investor is focused on LONG-TERM value (years). Prioritize fundamental strength, sustainable growth, and compounding potential.",
                }
                horizon_context = f"\n\nINVESTMENT HORIZON CONTEXT:\n{horizon_descriptions.get(investment_horizon, f'Investment horizon: {investment_horizon}')}"
            else:
                horizon_context = (
                    "\n\nINVESTMENT HORIZON CONTEXT:\n"
                    "No specific horizon provided. You MUST provide a nuanced recommendation that distinguishes between horizons."
                )

            # Build user context section if provided
            user_context_section = ""
            if user_context:
                user_context_section = (
                    f"\n\nINVESTOR PROFILE CONTEXT:\n"
                    f'The investor context:\n"{user_context}"'
                )

            working_memory_section = self._build_working_memory_context(news_analysis)

            user_content = (
                f"Investment Analysis Data (JSON):\n\n```json\n{json_context}\n```\n\n"
                f"{horizon_context}"
                f"{user_context_section}\n"
                f"{working_memory_section}\n\n"
                "Based on the data, deliver an investment recommendation."
            )

            decision_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            timeout_override = time_getter(15.0) if time_getter else None

            decision_response = await self._request_executor.send_async(
                decision_prompt,
                "reasoner",
                timeout_override=timeout_override,
                console=console,
            )

            content = decision_response.get("content", "")
            if not content.strip():
                raise APIError("Investment Reasoner returned empty response")

            decision_summary = {
                "analysis": content,
                "decision": "CHECK_REPORT",
                "confidence": confidence_bundle.get("confidence", 0.0),
                "confidence_level": confidence_bundle.get(
                    "confidence_level", "very_low"
                ),
                "confidence_factors": confidence_bundle.get(
                    "confidence_factors", {}
                ),
                "confidence_warnings": confidence_bundle.get(
                    "confidence_warnings", []
                ),
                "model_used": decision_response.get("model", "unknown"),
                "usage": decision_response.get("usage"),
            }

            return decision_response, decision_summary

        except Exception as exc:
            self.logger.error("ASYNC Investment reasoning failed: %s", exc)
            return {"error": str(exc)}, {"recommendation": "ERROR", "score": 0}
