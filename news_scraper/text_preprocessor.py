"""
Text Preprocessing Tools - Cleaning Functions for News Texts
===============================================================

This module contains helper functions to normalize news texts and remove
unnecessary noise.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional
import requests
from bs4 import BeautifulSoup

__all__ = [
    "clean_news_text",
    "squash_whitespace",
    "decode_response_content",
    "decode_bytes_content",
    "strip_html_tags",
    "convert_html_to_markdown",
    "split_text_semantically",
]

_CONTROL_CHARS = {chr(c) for c in range(0, 32)} - {"\t", "\n", "\r"}


def decode_bytes_content(content: bytes, encoding: Optional[str] = None) -> str:
    """Safely converts bytes content to text.

    Args:
        content: Raw byte content
        encoding: Server-specified encoding (optional)

    Returns:
        Content converted to text
    """
    safe_encoding = (encoding or "utf-8").lower()

    # Some sites incorrectly use ISO-8859-1 but actually send UTF-8
    if safe_encoding in ("iso-8859-1", "latin-1", "latin1"):
        try:
            return content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return content.decode(safe_encoding, errors="replace")

    # Other cases: use specified encoding, fallback to replace on failure
    try:
        return content.decode(safe_encoding, errors="strict")
    except (UnicodeDecodeError, LookupError):
        return content.decode("utf-8", errors="replace")


def decode_response_content(response: requests.Response) -> str:
    """
    Safely converts a requests response to text.
    Resolves encoding issues to prevent REPLACEMENT CHARACTER warnings.

    Args:
        response: requests.Response object

    Returns:
        Content converted to text
    """
    return decode_bytes_content(response.content, response.encoding)


def clean_news_text(
    text: str, preserve_newlines: bool = False, to_markdown: bool = True
) -> str:
    """Normalize news texts and remove unnecessary noise.

    Args:
        text: Text to clean.
        preserve_newlines: If True, line breaks are not collapsed.
        to_markdown: If True, HTML content is converted to Markdown format.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _strip_control_characters(normalized)
    normalized = normalized.replace("\xa0", " ")

    # Clean HTML tags or convert to Markdown
    if to_markdown:
        normalized = convert_html_to_markdown(normalized)
    else:
        normalized = strip_html_tags(normalized)

    # Noise reduction
    normalized = remove_common_noise(normalized)

    # Simplify multiple spaces
    # In Markdown mode, preserve blank lines (for paragraph separation)
    normalized = squash_whitespace(
        normalized,
        preserve_newlines=preserve_newlines or to_markdown,
        preserve_blank_lines=to_markdown,
    )

    # Clean up leading/trailing unnecessary punctuation and spaces
    normalized = normalized.strip(" \t\n\r-•·")

    return normalized


def remove_common_noise(text: str) -> str:
    """Remove common news noise (ads, social media, etc.)."""
    if not text:
        return ""

    # Split text into lines for line-based cleanup
    lines = text.split("\n")
    cleaned_lines = []

    # Noise patterns (case insensitive)
    noise_patterns = [
        r"read more:",
        r"also read:",
        r"click here to",
        r"follow us on",
        r"subscribe to our",
        r"sign up for",
        r"^advertisement$",
        r"^sponsored content$",
        r"share this article",
        r"photo credit:",
        r"image source:",
    ]

    # Regex pattern'i derle
    pattern = re.compile("|".join(noise_patterns), re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        # Skip very short lines and lines containing noise
        if len(stripped) < 3 or pattern.search(stripped):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def convert_html_to_markdown(html_content: str) -> str:
    """Convert HTML content to Markdown format."""
    if not html_content or not ("<" in html_content and ">" in html_content):
        return html_content

    try:
        soup = BeautifulSoup(html_content, "lxml")

        # Remove unnecessary tags
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
                "link",
                "meta",
            ]
        ):
            element.decompose()

        # Headers (H1-H6)
        for i in range(1, 7):
            for tag in soup.find_all(f"h{i}"):
                text = tag.get_text().strip()
                if text:
                    tag.replace_with(f"\n\n{'#' * i} {text}\n\n")

        # Lists (UL/OL) - Process in reverse to support nested lists
        lists = soup.find_all(["ul", "ol"])
        for list_tag in reversed(lists):
            is_ordered = list_tag.name == "ol"
            for i, li in enumerate(list_tag.find_all("li", recursive=False), 1):
                prefix = f"\n{i}. " if is_ordered else "\n* "
                # Add prefix to the beginning
                li.insert(0, prefix)
                li.unwrap()

            # Remove list tag and add whitespace
            list_tag.insert_before("\n")
            list_tag.insert_after("\n")
            list_tag.unwrap()

        # Bold (Strong/B)
        for tag in soup.find_all(["strong", "b"]):
            text = tag.get_text().strip()
            if text:
                tag.replace_with(f"**{text}**")

        # Italic (Em/I)
        for tag in soup.find_all(["em", "i"]):
            text = tag.get_text().strip()
            if text:
                tag.replace_with(f"*{text}*")

        # Paragraphs (P)
        for tag in soup.find_all("p"):
            tag.insert_before("\n\n")
            tag.insert_after("\n\n")
            tag.unwrap()

        # Line breaks
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Get text
        text = soup.get_text()

        # Clean up multiple consecutive newlines
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    except Exception:
        return strip_html_tags(html_content)


def squash_whitespace(
    text: str, preserve_newlines: bool = False, preserve_blank_lines: bool = False
) -> str:
    """Reduce multiple spaces to a single space.

    Args:
        text: Text to process.
        preserve_newlines: If True, line breaks are preserved.
        preserve_blank_lines: If True, blank lines are preserved (required for Markdown).
    """
    if preserve_newlines:
        # Normalize each line separately
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        if preserve_blank_lines:
            # Preserve blank lines but reduce consecutive blank lines to one
            result = []
            prev_empty = False
            for line in lines:
                if not line:
                    if not prev_empty:
                        result.append("")
                        prev_empty = True
                else:
                    result.append(line)
                    prev_empty = False
            return "\n".join(result)
        return "\n".join(filter(None, lines))

    return re.sub(r"\s+", " ", text).strip()


def strip_html_tags(text: str) -> str:
    """Remove HTML tags."""
    if not text or not ("<" in text and ">" in text):
        return text
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator="\n")
    except Exception:
        # Fallback to regex if BeautifulSoup fails
        return re.sub(r"<[^>]+>", " ", text)


def _strip_control_characters(text: str) -> str:
    return "".join(ch for ch in text if ch not in _CONTROL_CHARS)


def split_text_semantically(
    text: str,
    max_chunk_size: int = 20000,
    min_chunk_size: Optional[int] = None,
) -> list[str]:
    """
    Split text into semantic chunks based on paragraph and sentence boundaries.
    """
    if not text:
        return []

    if max_chunk_size <= 0:
        return [text]

    min_size = (
        min_chunk_size if min_chunk_size is not None else max(200, max_chunk_size // 4)
    )

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    def _push_current() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def _add_text_piece(piece: str) -> None:
        nonlocal current
        if not current:
            current = piece
            return

        if len(current) + 2 + len(piece) <= max_chunk_size:
            current = f"{current}\n\n{piece}"
        else:
            _push_current()
            current = piece

    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_size:
            _add_text_piece(paragraph)
            continue

        _push_current()
        for sentence in _split_sentences(paragraph):
            if len(sentence) <= max_chunk_size:
                _add_text_piece(sentence)
            else:
                for start in range(0, len(sentence), max_chunk_size):
                    _add_text_piece(sentence[start : start + max_chunk_size])

    _push_current()

    if min_size and len(chunks) > 1:
        merged: list[str] = []
        idx = 0
        while idx < len(chunks):
            current_chunk = chunks[idx]
            if len(current_chunk) < min_size and idx + 1 < len(chunks):
                next_chunk = chunks[idx + 1]
                if len(current_chunk) + 2 + len(next_chunk) <= max_chunk_size:
                    merged.append(f"{current_chunk}\n\n{next_chunk}")
                    idx += 2
                    continue
            merged.append(current_chunk)
            idx += 1
        chunks = merged

    return chunks


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
