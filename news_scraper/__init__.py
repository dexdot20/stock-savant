"""
News Scraper Package - News Search and Content Extraction
=====================================================

This package contains Google News scraping and article content extraction operations.
Organized with modular structure.

Modules:
- search_engine: Google News search operations
- content_extractor: Content extraction operations
- article_processor: Article processing and quality control
"""

from .search_engine import SearchAPIError
from .search_engine import GoogleNewsSearchEngine
from .content_extractor import ContentExtractor
from .article_processor import ArticleProcessor

__all__ = [
    "SearchAPIError",
    "GoogleNewsSearchEngine",
    "ContentExtractor",
    "ArticleProcessor",
]
