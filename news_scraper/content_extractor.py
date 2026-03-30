"""
Content Extractor Module - Content Extraction Operations
===================================================

This module contains article content extraction operations.
"""

import time
import random
import asyncio
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple

import aiohttp
from aiohttp_socks import ProxyConnector
from bs4 import BeautifulSoup

try:  # readability-lxml optional
    from readability import Document  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Document = None

from core import get_standard_logger
from .text_preprocessor import decode_bytes_content
from .async_utils import run_async
from config import DEFAULT_USER_AGENTS


class ContentExtractor:
    """Specialized class for article content extraction operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_standard_logger(__name__)

        # Configuration extraction
        network_cfg = self.config.get("network", {})
        api_config = self.config.get("api", {})
        web_search_config = self.config.get("web_search", {})
        extraction_config = self.config.get("content_extraction", {})
        proxy_config = self.config.get("proxy", {})

        # Network parameters
        self.request_timeout = network_cfg.get("request_timeout_seconds", 30)
        self.verify_ssl = bool(network_cfg.get("verify_ssl", True))
        if not self.verify_ssl:
            self.logger.warning(
                "SSL verification disabled for content extraction; set network.verify_ssl=True for production safety."
            )

        # API settings
        self.user_agents = api_config.get("user_agents", DEFAULT_USER_AGENTS)

        # Extraction settings
        self.min_article_length = web_search_config.get("min_article_length", 100)
        self.use_readability = extraction_config.get("use_readability", True)

        # Cache settings
        self.cache: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()
        self.cache_ttl = extraction_config.get("cache_ttl_seconds", 3600)
        self.cache_max_entries = extraction_config.get("cache_max_entries", 256)

        # Proxy manager (for content_extraction step)
        from services.network import ProxyManager, PROXY_STEP_CONTENT_EXTRACTION

        self.proxy_manager = ProxyManager(
            proxy_config, step=PROXY_STEP_CONTENT_EXTRACTION
        )

    def extract_article_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Extracts comprehensive content and metadata from article URL (async wrapper)."""
        try:
            return run_async(self.extract_article_content_async(url))
        except RuntimeError as exc:
            self.logger.debug("Async execution failed: %s", exc)
            return None

    async def extract_article_content_async(self, url: str) -> Optional[Dict[str, Any]]:
        """Extracts comprehensive content and metadata from article URL (async)."""
        try:
            current_time = time.time()
            cached_entry = self.cache.get(url)
            if cached_entry:
                cached_data, timestamp = cached_entry
                if current_time - timestamp < self.cache_ttl:
                    self.logger.debug("Cache hit for URL: %s", url)
                    self.cache.move_to_end(url)

                    if isinstance(cached_data, str):
                        return {
                            "content": cached_data,
                            "description": "",
                            "keywords": "",
                        }
                    return cached_data
                del self.cache[url]

            raw_bytes = None
            response_charset = None
            last_exception = None

            # Retry loop for 403/429 errors or timeouts
            for attempt in range(3):
                try:
                    current_ua = random.choice(self.user_agents)
                    from urllib.parse import urlparse

                    domain = urlparse(url).netloc

                    headers = {
                        "User-Agent": current_ua,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "same-origin",
                        "Sec-Fetch-User": "?1",
                        "Referer": (
                            f"https://{domain}/"
                            if domain
                            else "https://www.google.com/"
                        ),
                    }

                    if "Chrome" in current_ua:
                        headers.update(
                            {
                                "Sec-Ch-Ua": '"Google Chrome";v="130", "Chromium";v="130", "Not?A_Brand";v="99"',
                                "Sec-Ch-Ua-Mobile": "?0",
                                "Sec-Ch-Ua-Platform": (
                                    '"Windows"'
                                    if "Windows" in current_ua
                                    else '"macOS"'
                                ),
                            }
                        )

                    proxy = self.proxy_manager.get_proxy()
                    proxy_url = None
                    connector = None

                    if proxy:
                        self.logger.info(
                            "📰 [Content Extraction] URL: %s | Proxy: %s | Attempt: %d",
                            url,
                            proxy,
                            attempt + 1,
                        )
                        from services.network import ProxyManager

                        proxy_normalized = ProxyManager.normalize_proxy_for_aiohttp(
                            proxy
                        )
                        if proxy_normalized.startswith("socks"):
                            connector = ProxyConnector.from_url(proxy_normalized)
                        else:
                            proxy_url = proxy_normalized
                    else:
                        self.logger.debug(
                            "[Content Extraction] URL: %s | Proxy disabled (Attempt: %d)",
                            url,
                            attempt + 1,
                        )

                    timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                    async with aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector,
                        max_line_size=32768,
                        max_field_size=32768,
                    ) as session:
                        async with session.get(
                            url,
                            headers=headers,
                            proxy=proxy_url,
                            ssl=self.verify_ssl,
                        ) as response:
                            response.raise_for_status()
                            raw_bytes = await response.read()
                            response_charset = response.charset

                    if proxy:
                        self.proxy_manager.mark_proxy_success(proxy)
                    self.logger.debug("Content fetch successful: %s", url)
                    break  # Success

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e
                    self.logger.warning(
                        "Content fetch attempt %d failed (%s): %s", attempt + 1, url, e
                    )
                    if proxy:
                        self.proxy_manager.mark_proxy_failed(proxy)

                    if attempt < 2:
                        wait_time = 1.0 + random.random()
                        await asyncio.sleep(wait_time)
                    else:
                        raise e

            html_text = decode_bytes_content(raw_bytes, response_charset)

            soup = BeautifulSoup(html_text, "html.parser")

            meta_description = ""
            description_tag = soup.find(
                "meta", attrs={"name": "description"}
            ) or soup.find("meta", attrs={"property": "og:description"})
            if description_tag:
                meta_description = description_tag.get("content", "")

            meta_keywords = ""
            keywords_tag = soup.find("meta", attrs={"name": "keywords"}) or soup.find(
                "meta", attrs={"name": "news_keywords"}
            )
            if keywords_tag:
                meta_keywords = keywords_tag.get("content", "")

            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "noscript",
                    "iframe",
                    "form",
                    "input",
                    "button",
                    "advertisement",
                    "ads",
                    "promo",
                ]
            ):
                element.decompose()

            content = ""
            used_method = ""

            if self.use_readability and Document is not None:
                try:
                    doc = Document(html_text)
                    summary_html = doc.summary()

                    summary_soup = BeautifulSoup(summary_html, "html.parser")
                    text_content = summary_soup.get_text(strip=True)

                    if len(text_content) > 100:
                        content = summary_html
                        used_method = "readability"
                        self.logger.debug(
                            "Readability content used (%d characters)",
                            len(text_content),
                        )
                except Exception as exc:
                    self.logger.debug("Readability failed: %s", exc)

            if not content:
                paragraphs = soup.find_all("p")
                if paragraphs:
                    valid_paragraphs = []
                    for p in paragraphs:
                        if len(p.get_text(strip=True)) > 20:
                            valid_paragraphs.append(str(p))

                    if valid_paragraphs:
                        content = "\n".join(valid_paragraphs)
                        used_method = "paragraph_fallback"
                        self.logger.debug(
                            "Paragraph fallback used (%d characters)", len(content)
                        )

            if content:
                if len(content) < self.min_article_length:
                    self.logger.warning(
                        "Content too short: %d characters", len(content)
                    )
                    return None

                result = {
                    "content": content,
                    "description": meta_description,
                    "keywords": meta_keywords,
                }

                self._prune_cache(current_time)
                self.cache[url] = (result, current_time)
                self.cache.move_to_end(url)
                self._prune_cache(current_time)
                self.logger.info(
                    "Content extracted: %s (%d characters)", used_method, len(content)
                )
                return result

            self.logger.warning("No content extracted: %s", url)
            return None

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.logger.debug("HTTP error (%s): %s", url, e)
            return None
        except Exception as e:
            self.logger.debug("Article content extraction error (%s): %s", url, e)
            return None

    @staticmethod
    def _normalize_content(text: str) -> str:
        """
        Content cleaning - simplified (KISS principle).
        Only performs whitespace normalization while preserving line endings.
        """
        if not text:
            return ""
        # Process line by line and remove empty lines
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(filter(None, lines))

    def _prune_cache(self, current_time: float) -> None:
        if not self.cache:
            return

        expired_keys = [
            key
            for key, (_, ts) in self.cache.items()
            if current_time - ts >= self.cache_ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)

        if self.cache_max_entries and self.cache_max_entries > 0:
            while len(self.cache) > self.cache_max_entries:
                self.cache.popitem(last=False)

    # REMOVED: _format_proxy() method - now using centralized ProxyManager.format_proxy_for_requests()
    # DRY principle: Single source of truth for proxy formatting (15 lines removed)
