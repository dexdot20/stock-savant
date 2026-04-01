import time
from typing import Any, Callable, Dict, List, Optional

from config import get_config, is_configured_secret
from core import APIError, get_standard_logger
from services.network import retry_smart as retry_api, retry_smart_async
from services.utils.calculations import make_json_serializable

from .prompt_store import PromptStore
from .provider_metadata import get_provider_display_name
from .provider_manager import ProviderManager
from .response_parser import ResponseParser
from .investment_reasoner import InvestmentReasoner
from .pre_research_agent import PreResearchAgent
from .comparison_agent import ComparisonAgent
from .system_prompt_utils import augment_system_prompt
from news_scraper.async_utils import run_async


class ProviderRequestExecutor:
    """Retries outbound provider requests with a minimal wrapper."""

    def __init__(self, provider_manager: ProviderManager) -> None:
        self._provider_manager = provider_manager

    @retry_api(max_attempts=3, base_delay=3.0)
    def send(
        self,
        prompt: List[Dict[str, str]],
        request_type: str = "news",
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._provider_manager.send_request(
            prompt,
            request_type,
            timeout_override=timeout_override,
            **kwargs,
        )

    @retry_smart_async(max_attempts=3, base_delay=3.0)
    async def send_async(
        self,
        prompt: List[Dict[str, str]],
        request_type: str = "news",
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send an asynchronous AI request."""
        return await self._provider_manager.send_request_async(
            prompt,
            request_type,
            timeout_override=timeout_override,
            **kwargs,
        )


class AIOrchestrator:
    """Coordinates the full multi-provider investment workflow."""

    @staticmethod
    def _display_provider_name(provider_name: str) -> str:
        return get_provider_display_name(provider_name)

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or get_config()
        self.logger = get_standard_logger(__name__)
        self.prompt_store = PromptStore(self.config, self.logger)

        ai_config = self.config.get("ai", {})
        self.ai_config = ai_config

        # Simplified output language initialization
        raw_lang = (
            ai_config.get("output_language") if isinstance(ai_config, dict) else None
        )
        self.output_language = (
            raw_lang.strip() if isinstance(raw_lang, str) else None
        ) or "English"

        self.providers = (
            ai_config.get("providers", {}) if isinstance(ai_config, dict) else {}
        )
        if not isinstance(self.providers, dict):
            self.providers = {}
        models_cfg = ai_config.get("models", {})
        self.model_configs = models_cfg if isinstance(models_cfg, dict) else {}
        self.provider_order = self._build_provider_order()

        # Read timeout from network configuration.
        network_cfg = self.config.get("network", {})
        self.request_timeout = network_cfg.get("request_timeout_seconds", 30)

        # Log provider status.
        self.logger.info("=" * 60)
        self.logger.info("AI Orchestrator starting")
        self.logger.info("Output language: %s", self.output_language)
        self.logger.info("Request timeout: %s seconds", self.request_timeout)
        self.logger.info("Provider order: %s", self.provider_order)

        for provider_name, provider_cfg in self.providers.items():
            api_key = provider_cfg.get("api_key", "")
            has_key = is_configured_secret(api_key)
            status = "✅ ACTIVE" if has_key else "❌ INACTIVE"
            self.logger.info(
                "  Provider %s: %s (API key: %s)",
                self._display_provider_name(provider_name),
                status,
                "[SET]" if has_key else "[MISSING]",
            )

        self.logger.info("Model configurations:")
        for model_type, model_cfg in self.model_configs.items():
            provider = model_cfg.get("provider") or (
                self.provider_order[0] if self.provider_order else "unknown"
            )
            model = model_cfg.get("model", "unknown")
            self.logger.info(
                "  %s: %s @ %s",
                model_type,
                model,
                self._display_provider_name(provider),
            )
        self.logger.info("=" * 60)

        self.provider_manager = ProviderManager(
            logger=self.logger,
            providers_cfg=self.providers,
            model_configs=self.model_configs,
            provider_order=self.provider_order,
            request_timeout=self.request_timeout,
        )

        self.request_executor = ProviderRequestExecutor(self.provider_manager)
        self.response_parser = ResponseParser()

        self._autonomous_agent = None
        self.investment_reasoner = InvestmentReasoner(
            logger=self.logger,
            prompt_resolver=self._get_system_prompt,
            request_executor=self.request_executor,
            response_parser=self.response_parser,
        )
        self.pre_research_agent = PreResearchAgent(
            logger=self.logger,
            prompt_resolver=self._get_system_prompt,
            request_executor=self.request_executor,
        )
        self.comparison_agent = ComparisonAgent(
            logger=self.logger,
            prompt_resolver=self._get_system_prompt,
            request_executor=self.request_executor,
        )

    def _build_provider_order(self) -> List[str]:
        order = [
            name
            for name, provider_cfg in self.providers.items()
            if isinstance(provider_cfg, dict)
        ]
        return order or ["deepseek"]

    def _get_system_prompt(self, key: str) -> str:
        prompt = self.prompt_store.get(key)
        return self._augment_system_prompt(prompt)

    def _augment_system_prompt(self, prompt: str) -> str:
        return augment_system_prompt(
            prompt,
            config=self.config,
            output_language=self.output_language,
        )

    def _get_autonomous_agent(self):
        if self._autonomous_agent is None:
            from .autonomous_news_agent import AutonomousNewsAgent

            self._autonomous_agent = AutonomousNewsAgent(
                logger=self.logger,
                prompt_resolver=self._get_system_prompt,
                request_executor=self.request_executor,
            )
        return self._autonomous_agent

    def _run_news_agent(
        self,
        company_name: str,
        company_data: Dict[str, Any],
        *,
        initial_news: Optional[List[Dict[str, Any]]] = None,
        console: Optional[Any] = None,
        session_id: Optional[str] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        on_max_steps_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        depth_mode: str = "standard",
    ) -> Dict[str, Any]:
        return run_async(
            self._get_autonomous_agent().run(
                company_name,
                company_data,
                initial_news=initial_news,
                console=console,
                session_id=session_id,
                resume_state=resume_state,
                on_max_steps_callback=on_max_steps_callback,
                depth_mode=depth_mode,
            )
        )

    def analyze_company_with_ai(
        self,
        company_data: Dict[str, Any],
        news_context: Optional[List[Dict[str, Any]]] = None,
        company_summary: Optional[str] = None,
        request_timeout: Optional[float] = None,
        has_news_permission: bool = True,
        console: Optional[Any] = None,
        session_id: Optional[str] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        on_max_steps_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Compatibility layer delegating to the autonomous agent."""

        company_name = (
            company_data.get("longName")
            or company_data.get("symbol")
            or "Unknown Company"
        )

        try:
            return self._run_news_agent(
                company_name,
                company_data,
                initial_news=news_context,
                console=console,
                session_id=session_id,
                resume_state=resume_state,
                on_max_steps_callback=on_max_steps_callback,
            )
        except Exception as e:
            self.logger.error("Autonomous Agent Failed: %s", e)
            return {"error": str(e), "analysis": "Detailed analysis failed."}

    def pre_research_exchange(
        self,
        exchange: str,
        criteria: Optional[str] = None,
        console: Optional[Any] = None,
        depth_mode: str = "standard",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run exchange-focused pre-research and list discovered companies."""
        try:
            return run_async(
                self.pre_research_agent.run(
                    exchange=exchange,
                    criteria=criteria,
                    console=console,
                    depth_mode=depth_mode,
                    session_id=session_id,
                )
            )
        except KeyboardInterrupt:
            self.logger.info("Pre-research cancelled by user")
            return {
                "cancelled": True,
                "error": "Cancelled by user",
                "analysis": "",
            }
        except Exception as e:
            self.logger.error("Pre-research agent failed: %s", e)
            return {"error": str(e), "analysis": "Pre-research failed."}

    def compare_companies(
        self,
        tickers: List[str],
        criteria: Optional[str] = None,
        initial_data: Optional[List[Dict[str, Any]]] = None,
        console: Optional[Any] = None,
        depth_mode: str = "standard",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Kullanici secimiyle birden fazla sirketi agentic sekilde karsilastirir."""
        try:
            return run_async(
                self.comparison_agent.run(
                    tickers=tickers,
                    criteria=criteria,
                    initial_data=initial_data,
                    console=console,
                    depth_mode=depth_mode,
                    session_id=session_id,
                )
            )
        except Exception as e:
            self.logger.error("Karsilastirma ajani basarisiz: %s", e)
            return {"error": str(e), "analysis": "Comparison failed."}

    def full_ai_google_workflow(
        self,
        company_data: Dict[str, Any],
        articles: List[Dict[str, Any]],
        ai_prompt_text: Optional[str] = None,
        remaining_time: Optional[float] = None,
        has_news_permission: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
        investment_horizon: Optional[str] = None,
        user_context: Optional[str] = None,
        console: Optional[Any] = None,
        news_analysis_override: Optional[Dict[str, Any]] = None,
        skip_news_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full AI workflow including news analysis and investment reasoning.

        News analysis uses yfinance news and Google News scraping (no Google Search API).

        Args:
            company_data: Company financial data
            articles: News articles for analysis (from yfinance + Google News scraping)
            ai_prompt_text: Optional AI prompt text
            remaining_time: Optional deadline for workflow
            has_news_permission: Whether news analysis is permitted
            progress_callback: Optional callback for progress updates
            investment_horizon: User's investment time horizon (e.g., "short-term", "medium-term", "long-term")
            user_context: Optional user-provided context (e.g., budget, experience level, goals)
        """

        def _notify(message: str) -> None:
            if progress_callback:
                try:
                    progress_callback(message)
                except Exception as exc:
                    self.logger.debug("Progress callback failed: %s", exc)
            self.logger.info(message)

        workflow_start = time.time()
        articles = articles or []
        try:
            symbol = company_data.get("symbol", "N/A")
            _notify(f"🚀 Starting AI workflow: {symbol}")
            _notify(f"📈 Available data: {len(articles)} news articles")
            if user_context:
                _notify(f"👤 User context: Present")
            if investment_horizon:
                _notify(f"⏱️ Investment horizon: {investment_horizon}")

            self.logger.info(
                "AI workflow starting - company: %s, user_context: %s",
                symbol,
                bool(user_context),
            )

            deadline = None
            if remaining_time is not None:
                deadline = time.time() + max(0.0, remaining_time)
                self.logger.debug("Workflow deadline: %.1fs", remaining_time)
                _notify(f"⏰ Time limit: {remaining_time:.1f}s")

            def _time_left(min_seconds: float = 1.0) -> Optional[float]:
                if deadline is None:
                    return None
                left = deadline - time.time()
                if left <= 0:
                    raise APIError("AI workflow timeout")
                return max(left, min_seconds)

            news_analysis: Optional[Dict[str, Any]] = None

            if news_analysis_override is not None:
                news_analysis = news_analysis_override
                skip_news_analysis = True
                self.logger.info(
                    "Phase 3 - Step 3.1 skipped: Using pre-generated news analysis"
                )
                _notify(
                    "✅ [Phase 3.1] Pre-generated Agentic news analysis used"
                )

            if has_news_permission and not skip_news_analysis:
                news_provider = (
                    self.provider_manager.get_primary_enabled_provider("news")
                    or self.provider_manager.get_default_provider()
                )
                news_display = self.provider_manager.display_name(news_provider)
                self.logger.info(
                    "Phase 3 - Step 3.1: Haber ve piyasa analizi (%s)", news_display
                )
                _notify(f"\n🔍 [Phase 3.1] News and market analysis starting...")
                _notify(f"🤖 AI Provider: {news_display}")
                _notify(f"📰 Number of articles to analyze: {len(articles)}")

                news_start = time.time()

                try:
                    # Delegate market/news analysis to the autonomous agent.
                    company_name = (
                        company_data.get("longName")
                        or company_data.get("symbol")
                        or "Unknown Company"
                    )

                    news_analysis = self._run_news_agent(
                        company_name,
                        company_data,
                        initial_news=articles,
                        console=console,
                    )

                    if "error" in news_analysis:
                        error_msg = news_analysis["error"]
                        self.logger.error("News analysis error: %s", error_msg)
                        raise APIError(f"News analysis failed: {error_msg}")

                    # ══════════════════════════════════════════════════════════════
                    # SUFFICIENCY CHECK - Yeterlilik Kontrolü
                    # ══════════════════════════════════════════════════════════════
                    news_analysis = self._perform_sufficiency_check(
                        news_analysis=news_analysis,
                        company_name=company_name,
                        company_data=company_data,
                        articles=articles,
                        console=console,
                        notify_fn=_notify,
                    )

                    news_duration = time.time() - news_start
                    analysis_length = len(news_analysis.get("analysis", ""))
                    model_used = news_analysis.get("model_used", "Unknown")

                    confidence = news_analysis.get("confidence", 0.0)
                    if confidence is None:
                        confidence = 0.0

                    confidence_level = news_analysis.get("confidence_level", "Unknown")
                    if confidence_level is None:
                        confidence_level = "Unknown"

                    _notify(f"✅ [Phase 3.1] News analysis completed!")
                    _notify(f"📝 Analysis length: {analysis_length:,} characters")
                    _notify(f"🤖 Model used: {model_used}")
                    _notify(f"⏱️ Duration: {news_duration:.2f}s")

                    # Log working memory stats if available
                    working_memory = news_analysis.get("working_memory")
                    if working_memory and isinstance(working_memory, dict):
                        depth_score = working_memory.get("research_depth_score", 0)
                        sources_count = len(working_memory.get("sources_consulted", []))
                        facts_count = len(working_memory.get("facts_learned", []))
                        _notify(
                            f"🧠 Research Depth: Score={depth_score}, Sources={sources_count}, Findings={facts_count}"
                        )

                    # Log usage info if available
                    usage = news_analysis.get("usage")
                    if usage and isinstance(usage, dict):
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        if total_tokens > 0:
                            _notify(
                                f"🔢 Token Usage: {total_tokens:,} (Input: {input_tokens:,}, Output: {output_tokens:,})"
                            )

                    self.logger.info(
                        "News analysis completed: %s characters, model: %s, duration: %.2fs",
                        analysis_length,
                        model_used,
                        news_duration,
                    )

                except APIError:
                    raise
                except Exception as exc:
                    self.logger.error(
                        "News analysis unexpected error: %s",
                        exc,
                        exc_info=True,
                    )
                    raise APIError(f"News analysis error: {exc}")

            elif not skip_news_analysis:
                self.logger.info(
                    "Skipping Phase 3 - Step 3.1 (News and market analysis) due to missing web scraping permission",
                )

                news_analysis = {
                    "analysis": "News analysis not performed because user web scraping permission is not available.",
                    "confidence": 0.0,
                    "confidence_level": "very_low",
                    "model_used": "none",
                    "timestamp": time.time(),
                    "has_news_data": False,
                    "news_permission_available": False,
                }

            # Phase 2 (Google Search API) kaldırıldı - artık sadece yfinance ve Google News scraping kullanılıyor

            reasoner_provider = (
                self.provider_manager.get_primary_enabled_provider("reasoner")
                or self.provider_manager.get_default_provider()
            )
            reasoner_display = self.provider_manager.display_name(reasoner_provider)
            self.logger.info(
                "Phase 3 - Step 3.2: Investment decision and risk assessment (%s)",
                reasoner_display,
            )

            _notify(
                f"\n💡 [Phase 3.2] Investment decision and risk assessment starting..."
            )
            _notify(f"🤖 AI Provider: {reasoner_display}")
            if investment_horizon:
                _notify(f"⏱️ Investment horizon: {investment_horizon}")

            reasoner_start = time.time()

            decision_response, decision_summary = self.investment_reasoner.decide(
                company_data,
                news_analysis,
                time_getter=_time_left,
                investment_horizon=investment_horizon,
                user_context=user_context,
                console=console,
            )

            reasoner_duration = time.time() - reasoner_start
            decision_length = (
                len(decision_response.get("content", ""))
                if isinstance(decision_response, dict)
                else 0
            )
            decision_model = (
                decision_response.get("model", "Unknown")
                if isinstance(decision_response, dict)
                else "Unknown"
            )

            _notify(f"✅ [Phase 3.2] Investment decision completed!")
            _notify(f"📝 Decision length: {decision_length:,} characters")
            _notify(f"🤖 Model used: {decision_model}")
            _notify(f"⏱️ Duration: {reasoner_duration:.2f}s")

            # Log usage info for decision
            if isinstance(decision_response, dict):
                decision_usage = decision_response.get("usage")
                if decision_usage and isinstance(decision_usage, dict):
                    input_tokens = decision_usage.get("prompt_tokens", 0)
                    output_tokens = decision_usage.get("completion_tokens", 0)
                    total_tokens = decision_usage.get("total_tokens", 0)
                    if total_tokens > 0:
                        _notify(
                            f"🔢 Token usage: {total_tokens:,} (Input: {input_tokens:,}, Output: {output_tokens:,})"
                        )

            result = {
                "news_analysis": news_analysis,
                "news_articles": articles,
                "investment_decision": decision_response,
                "decision_summary": decision_summary,
                "ai_prompt": ai_prompt_text,
                "user_context": user_context,
                "timestamp": time.time(),
            }

            result = make_json_serializable(result)

            workflow_duration = time.time() - workflow_start

            _notify(f"\n🎉 AI Workflow Completed Successfully!")
            _notify(f"⏱️ Total Time: {workflow_duration:.2f}s")
            _notify(f"📈 Data Processed: {len(articles)} news articles")
            _notify(f"✅ All stages completed successfully")

            self.logger.info(
                "AI workflow completed successfully: %.2fs, %d news processed",
                workflow_duration,
                len(articles),
            )

            self._display_workflow_results(result)
            return result

        except APIError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            error_msg = f"AI workflow data processing error: {exc}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
        except Exception as exc:
            error_msg = f"AI workflow unexpected error: {exc}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

    def _perform_sufficiency_check(
        self,
        news_analysis: Dict[str, Any],
        company_name: str,
        company_data: Dict[str, Any],
        articles: List[Dict[str, Any]],
        console: Optional[Any],
        notify_fn: Callable[[str], None],
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Performs a sufficiency check on the news analysis.
        If the analysis is deemed shallow, re-runs the agent in deep mode.

        Sufficiency Criteria:
        - Analysis length >= 1500 characters
        - Research depth score >= 3 (if working memory available)
        - At least 2 sources consulted

        Returns:
            Enhanced news_analysis if re-run was needed, otherwise original.
        """
        MIN_ANALYSIS_LENGTH = 1500
        MIN_DEPTH_SCORE = 3
        MIN_SOURCES = 2

        analysis_text = news_analysis.get("analysis", "")
        working_memory = news_analysis.get("working_memory", {})

        depth_score = (
            working_memory.get("research_depth_score", 0) if working_memory else 0
        )
        sources_count = (
            len(working_memory.get("sources_consulted", [])) if working_memory else 0
        )

        is_sufficient = True
        insufficiency_reasons = []

        if len(analysis_text) < MIN_ANALYSIS_LENGTH:
            is_sufficient = False
            insufficiency_reasons.append(
                f"Analysis too short ({len(analysis_text)}/{MIN_ANALYSIS_LENGTH} characters)"
            )

        if depth_score < MIN_DEPTH_SCORE:
            is_sufficient = False
            insufficiency_reasons.append(
                f"Research depth insufficient (score: {depth_score}/{MIN_DEPTH_SCORE})"
            )

        if sources_count < MIN_SOURCES:
            is_sufficient = False
            insufficiency_reasons.append(
                f"Insufficient number of sources ({sources_count}/{MIN_SOURCES})"
            )

        if is_sufficient:
            self.logger.info(
                "Sufficiency check passed: Analysis meets quality criteria"
            )
            return news_analysis

        # Analysis is insufficient - log reasons and re-run if retries available
        self.logger.warning(
            "Sufficiency check FAILED: %s", "; ".join(insufficiency_reasons)
        )
        notify_fn(f"\n⚠️ [Sufficiency Check] Analysis deemed insufficient:")
        for reason in insufficiency_reasons:
            notify_fn(f"   ❌ {reason}")

        if max_retries <= 0:
            self.logger.info("No retries remaining, returning current analysis")
            notify_fn("⏭️ No retries remaining, using current analysis")
            return news_analysis

        notify_fn(
            f"\n🔄 [Deep Research Mode] Agent restarting in deep research mode..."
        )

        try:
            deep_analysis = self._run_news_agent(
                company_name,
                company_data,
                initial_news=articles,
                console=console,
                depth_mode="deep",
            )

            if "error" not in deep_analysis:
                new_length = len(deep_analysis.get("analysis", ""))
                new_depth = deep_analysis.get("working_memory", {}).get(
                    "research_depth_score", 0
                )
                notify_fn(
                    f"✅ Deep research completed! New length: {new_length}, New depth: {new_depth}"
                )

                # Recursive check with decremented retries
                return self._perform_sufficiency_check(
                    news_analysis=deep_analysis,
                    company_name=company_name,
                    company_data=company_data,
                    articles=articles,
                    console=console,
                    notify_fn=notify_fn,
                    max_retries=max_retries - 1,
                )
            else:
                self.logger.error(
                    "Deep research failed: %s", deep_analysis.get("error")
                )
                notify_fn(f"❌ Deep research failed: {deep_analysis.get('error')}")
                return news_analysis

        except Exception as exc:
            self.logger.error("Deep research exception: %s", exc, exc_info=True)
            notify_fn(f"❌ Deep research error: {exc}")
            return news_analysis

    def _display_workflow_results(self, result: Dict[str, Any]) -> None:
        try:
            analysis = result.get("news_analysis", {})
            analysis_text = (
                analysis.get("analysis") if isinstance(analysis, dict) else None
            )

            decision = result.get("investment_decision", {})
            decision_text = (
                decision.get("content") if isinstance(decision, dict) else None
            )

            news_length = len(analysis_text or "")
            decision_length = len(decision_text or "")

            self.logger.info(
                "AI workflow summary | news_analysis_chars=%d | decision_chars=%d | has_news=%s | has_decision=%s",
                news_length,
                decision_length,
                bool(analysis_text),
                bool(decision_text),
            )

            if analysis_text:
                self.logger.debug(
                    "AI workflow news preview: %s",
                    analysis_text[:500],
                )
            if decision_text:
                self.logger.debug(
                    "AI workflow decision preview: %s",
                    decision_text[:500],
                )
        except Exception as exc:  # pragma: no cover
            self.logger.error("Result display error: %s", exc)
