"""
Article Processor Module - Article Processing and Quality Control
=================================================================

This module contains article processing and quality control operations.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from core import get_standard_logger
from .text_preprocessor import clean_news_text
from .async_utils import run_async


class ArticleProcessor:
    """Specialized class for article processing and quality control."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, content_extractor=None):
        self.config = config or {}
        self.logger = get_standard_logger(__name__)
        self.content_extractor = content_extractor

        processing_cfg = self.config.get("news_processing", {})
        self.max_workers = max(1, int(processing_cfg.get("max_workers", 4)))

    def smart_article_collection(
        self, articles_raw: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Advanced article collection - async wrapper."""
        try:
            return run_async(self.smart_article_collection_async(articles_raw))
        except RuntimeError as exc:
            self.logger.warning("Async execution failed, falling back to thread pool: %s", exc)
            return self._smart_article_collection_sync(articles_raw)

    async def smart_article_collection_async(
        self,
        articles_raw: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Advanced article collection with async content extraction and quality control."""
        try:
            articles: List[Dict[str, Any]] = []
            total_articles = len(articles_raw)

            self.logger.info("🚀 Article collection started: %d articles", total_articles)

            indexed_articles: List[Tuple[int, Dict[str, Any]]] = []
            for idx, art in enumerate(articles_raw, 1):
                if not art.get("link"):
                    self.logger.warning("⚠️  Article %d: No link, skipping", idx)
                    continue
                indexed_articles.append((idx, art))

            total_valid = len(indexed_articles)
            if not indexed_articles:
                self.logger.info("📭 No valid articles to process")
                return []

            semaphore = asyncio.Semaphore(min(self.max_workers, total_valid))
            results: Dict[int, Dict[str, Any]] = {}
            results_lock = asyncio.Lock()

            async def _run_one(original_idx: int, article: Dict[str, Any]) -> None:
                async with semaphore:
                    processed_article, _ = await self._process_single_article_async(
                        original_idx,
                        total_valid,
                        deepcopy(article),
                    )
                async with results_lock:
                    results[original_idx] = processed_article

            tasks = [
                asyncio.create_task(_run_one(original_idx, article))
                for original_idx, article in indexed_articles
            ]

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)

            for idx, _ in indexed_articles:
                if idx in results:
                    articles.append(results[idx])

            successful_count = len(articles)
            success_rate = (
                (successful_count / total_articles * 100) if total_articles > 0 else 0
            )

            self.logger.info("📊 Article collection completed")
            self.logger.info(
                "   📈 Success rate: %.1f%% (%d/%d)",
                success_rate,
                successful_count,
                total_articles,
            )

            return articles

        except Exception as e:
            self.logger.error("❌ Article collection critical error: %s", e)
            return articles_raw

    def _smart_article_collection_sync(
        self,
        articles_raw: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """ThreadPool-based synchronous fallback handler."""
        try:
            articles: List[Dict[str, Any]] = []
            total_articles = len(articles_raw)

            self.logger.info("🚀 Article collection started: %d articles", total_articles)

            indexed_articles: List[Tuple[int, Dict[str, Any]]] = []
            for idx, art in enumerate(articles_raw, 1):
                if not art.get("link"):
                    self.logger.warning("⚠️  Article %d: No link, skipping", idx)
                    continue
                indexed_articles.append((idx, art))

            total_valid = len(indexed_articles)
            if not indexed_articles:
                self.logger.info("📭 No valid articles to process")
                return []

            max_workers = min(self.max_workers, total_valid)
            results: Dict[int, Dict[str, Any]] = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_article,
                        original_idx,
                        total_valid,
                        deepcopy(article),
                    ): original_idx
                    for original_idx, article in indexed_articles
                }

                for future in as_completed(futures):
                    original_idx = futures[future]
                    try:
                        processed_article, _ = future.result()
                        results[original_idx] = processed_article
                    except Exception as exc:  # pragma: no cover - beklenmeyen hatalar
                        self.logger.error(
                            "❌ Parallel processing error (article %d): %s",
                            original_idx,
                            exc,
                        )
                        continue

            for idx, _ in indexed_articles:
                if idx in results:
                    articles.append(results[idx])

            successful_count = len(articles)
            success_rate = (
                (successful_count / total_articles * 100) if total_articles > 0 else 0
            )

            self.logger.info("📊 Article collection completed")
            self.logger.info(
                "   📈 Success rate: %.1f%% (%d/%d)",
                success_rate,
                successful_count,
                total_articles,
            )

            return articles

        except Exception as e:
            self.logger.error("❌ Article collection critical error: %s", e)
            return articles_raw

    def _process_single_article(
        self, index: int, total: int, article: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Process a single article and return it with full_text and source_info."""

        url = article.get("link")
        self.logger.info(
            f"📄 Article {index}/{total}: {article.get('title', 'Untitled')[:50]}..."
        )

        if not url:
            article["full_text"] = article.get("title", "Content could not be extracted")
            article["source_info"] = self._format_source_for_ai(article)
            return article, "failed"

        try:
            full_txt = None
            meta_desc = ""
            meta_keys = ""

            if self.content_extractor:
                extraction_result = self.content_extractor.extract_article_content(url)
                if extraction_result:
                    full_txt = extraction_result.get("content")
                    meta_desc = extraction_result.get("description", "")
                    meta_keys = extraction_result.get("keywords", "")

            if full_txt and len(full_txt.strip()) > 100:
                cleaned_txt = clean_news_text(full_txt, preserve_newlines=True)

                article["full_text"] = cleaned_txt
                article["meta_description"] = meta_desc
                article["meta_keywords"] = meta_keys
                article["source_info"] = self._format_source_for_ai(article)
                self.logger.info(
                    f"✅ Success: extracted {len(full_txt)} characters of content"
                )
                return article, "successful"

            self.logger.debug(f"Scraping failed or content is too short: {url}")
            fallback_content = article.get("title", "Content could not be extracted")
            article["full_text"] = fallback_content
            article["source_info"] = self._format_source_for_ai(article)
            self.logger.info(
                f"📝 Title fallback used: {len(fallback_content)} characters of title"
            )
            return article, "fallback"

        except Exception as exc:
            error_msg = f"Scraping error ({url}): {exc}"
            self.logger.error(f"❌ {error_msg}")

            article["full_text"] = article.get("title", "An error occurred")
            article["source_info"] = self._format_source_for_ai(article)
            return article, "failed"

    async def _process_single_article_async(
        self,
        index: int,
        total: int,
        article: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """Process a single article asynchronously."""

        url = article.get("link")
        self.logger.info(
            "📄 Makale %d/%d: %s...",
            index,
            total,
            article.get("title", "Untitled")[:50],
        )

        if not url:
            article["full_text"] = article.get("title", "Content could not be extracted")
            article["source_info"] = self._format_source_for_ai(article)
            return article, "failed"

        try:
            full_txt = None
            meta_desc = ""
            meta_keys = ""

            if self.content_extractor:
                if hasattr(self.content_extractor, "extract_article_content_async"):
                    extraction_result = (
                        await self.content_extractor.extract_article_content_async(url)
                    )
                else:
                    extraction_result = await asyncio.to_thread(
                        self.content_extractor.extract_article_content, url
                    )

                if extraction_result:
                    full_txt = extraction_result.get("content")
                    meta_desc = extraction_result.get("description", "")
                    meta_keys = extraction_result.get("keywords", "")

            if full_txt and len(full_txt.strip()) > 100:
                cleaned_txt = clean_news_text(full_txt, preserve_newlines=True)

                article["full_text"] = cleaned_txt
                article["meta_description"] = meta_desc
                article["meta_keywords"] = meta_keys
                article["source_info"] = self._format_source_for_ai(article)
                self.logger.info(
                    "✅ Success: extracted %d characters of content", len(full_txt)
                )
                return article, "successful"

            self.logger.debug("Scraping failed or content is too short: %s", url)
            fallback_content = article.get("title", "Content could not be extracted")
            article["full_text"] = fallback_content
            article["source_info"] = self._format_source_for_ai(article)
            self.logger.info(
                "📝 Title fallback used: %d characters of title",
                len(fallback_content),
            )
            return article, "fallback"

        except Exception as exc:
            error_msg = f"Scraping error ({url}): {exc}"
            self.logger.error("❌ %s", error_msg)

            article["full_text"] = article.get("title", "An error occurred")
            article["source_info"] = self._format_source_for_ai(article)
            return article, "failed"

    def _format_source_for_ai(self, article: Dict[str, Any]) -> str:
        """Format source information clearly for AI - only the domain in parentheses."""
        # Priority: use publisher information if available
        publisher = article.get("publisher")
        if publisher and isinstance(publisher, str) and publisher.lower() != "unknown":
            return f"({publisher})"

        url = article.get("link", "")

        if url:
            try:
                # Extract the domain from the URL
                parsed_url = urlparse(url)
                domain = parsed_url.netloc

                # Remove the port if it is a default port
                if ":" in domain:
                    domain_part, port_part = domain.rsplit(":", 1)
                    # 80 for HTTP and 443 for HTTPS are the default ports
                    if (parsed_url.scheme == "http" and port_part == "80") or (
                        parsed_url.scheme == "https" and port_part == "443"
                    ):
                        domain = domain_part

                # Remove the www. prefix
                if domain.startswith("www."):
                    domain = domain[4:]

                # Handle subdomains - usually keep the registrable domain
                parts = domain.split(".")
                if len(parts) > 2:
                    # Handle country-code TLDs (e.g. .com.tr, .co.uk)
                    if len(parts[-1]) == 2 and len(parts[-2]) <= 3:  # .com.tr gibi
                        domain = ".".join(parts[-3:])  # keep the last 3 parts
                    else:
                        domain = ".".join(
                            parts[-2:]
                        )  # keep the last 2 parts for a normal domain

                # Return the domain in parentheses if it is valid
                if domain and "." in domain:  # Must contain at least one dot
                    return f"({domain})"
            except Exception:
                # Return empty string if URL parsing fails
                pass

            # Return an empty string if the URL is missing or the domain cannot be extracted
        return ""
