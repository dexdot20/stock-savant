"""Exposes multi-provider AI service package."""

from .orchestrator import AIOrchestrator
from .investment_reasoner import InvestmentReasoner
from .response_parser import ResponseParser
from .prompt_store import PromptStore
from .provider_manager import ProviderManager
from .comparison_agent import ComparisonAgent

# Backward-compatible alias: legacy imports expect AIService symbol
AIService = AIOrchestrator

__all__ = [
    "AIService",
    "AIOrchestrator",
    "InvestmentReasoner",
    "ResponseParser",
    "PromptStore",
    "ProviderManager",
    "ComparisonAgent",
]
