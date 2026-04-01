from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import get_config
from core.cache_manager import get_unified_cache
from core import get_standard_logger
from core.paths import get_runtime_dir
from domain.utils import safe_int_strict as _safe_int

logger = get_standard_logger(__name__)


class _SuppressUnauthenticatedHFHubWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.getMessage() or "")
        return "unauthenticated requests to the HF Hub" not in message


try:
    import chromadb
    from chromadb import Documents
    from chromadb.utils.embedding_functions import register_embedding_function
except Exception:  # pragma: no cover - optional dependency
    chromadb = None
    Documents = list
    def register_embedding_function(ef_class=None):  # type: ignore
        if ef_class is None:
            def _decorator(cls):
                return cls

            return _decorator
        return ef_class

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None
    SentenceTransformer = None


@dataclass(frozen=True)
class _StructuredBlock:
    text: str
    section: str
    block_type: str


@dataclass(frozen=True)
class _StructuredChunk:
    content: str
    section: str
    content_type: str


@register_embedding_function
class CachedSentenceTransformerEmbeddingFunction:
    """SentenceTransformer wrapper with process-local TTL caching."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_namespace: str,
        batch_size: int,
        normalize_embeddings: bool,
        document_prefix: str,
        query_prefix: str,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available")

        self._model_name = str(model_name).strip()
        self._cache_namespace = str(cache_namespace).strip().lower()
        self._batch_size = max(1, int(batch_size))
        self._normalize_embeddings = bool(normalize_embeddings)
        self._document_prefix = str(document_prefix or "")
        self._query_prefix = str(query_prefix or "")
        self._cache_manager = get_unified_cache()
        self._model = SentenceTransformer(self._model_name)

    @staticmethod
    def name() -> str:
        return "sentence_transformer"

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "cache_namespace": self._cache_namespace,
            "batch_size": self._batch_size,
            "normalize_embeddings": self._normalize_embeddings,
            "document_prefix": self._document_prefix,
            "query_prefix": self._query_prefix,
        }

    @classmethod
    def build_from_config(
        cls,
        config: Dict[str, Any],
    ) -> "CachedSentenceTransformerEmbeddingFunction":
        model_name = str(config.get("model_name") or "").strip()
        if not model_name:
            raise ValueError("CachedSentenceTransformerEmbeddingFunction requires model_name")

        return cls(
            model_name=model_name,
            cache_namespace=str(config.get("cache_namespace") or "rag_embeddings").strip().lower(),
            batch_size=max(1, _safe_int(config.get("batch_size", 32), 32)),
            normalize_embeddings=bool(config.get("normalize_embeddings", True)),
            document_prefix=str(config.get("document_prefix") or ""),
            query_prefix=str(config.get("query_prefix") or ""),
        )

    def is_legacy(self) -> bool:
        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _prefix_for_kind(self, kind: str) -> str:
        return self._query_prefix if kind == "query" else self._document_prefix

    def _prepare_text(self, text: str, kind: str) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return ""

        prefix = self._prefix_for_kind(kind)
        if prefix and not normalized.startswith(prefix):
            return f"{prefix}{normalized}"
        return normalized

    def _cache_key(self, text: str, kind: str) -> str:
        payload = f"{self._model_name}|{kind}|{self._normalize_embeddings}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _coerce_texts(input_texts: Any) -> List[str]:
        if isinstance(input_texts, str):
            return [input_texts]
        return list(input_texts)

    def encode(self, texts: Sequence[str], *, kind: str) -> List[List[float]]:
        normalized_inputs = [self._prepare_text(text, kind) for text in texts]
        cache_keys = [self._cache_key(text, kind) if text else "" for text in normalized_inputs]
        cached = self._cache_manager.get_many(
            self._cache_namespace,
            [key for key in cache_keys if key],
        )

        missing_positions: List[int] = []
        missing_payloads: List[str] = []
        resolved: List[Optional[List[float]]] = [None] * len(normalized_inputs)

        for index, (prepared, cache_key) in enumerate(zip(normalized_inputs, cache_keys)):
            if not prepared or not cache_key:
                resolved[index] = []
                continue

            cached_value = cached.get(cache_key)
            if isinstance(cached_value, list) and cached_value:
                resolved[index] = [float(value) for value in cached_value]
                continue

            missing_positions.append(index)
            missing_payloads.append(prepared)

        if missing_payloads:
            encoded_batches: List[List[float]] = []
            for start in range(0, len(missing_payloads), self._batch_size):
                batch = missing_payloads[start : start + self._batch_size]
                batch_vectors = self._model.encode(
                    batch,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize_embeddings,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                encoded_batches.extend(batch_vectors.tolist())

            cache_updates: Dict[str, List[float]] = {}
            for position, vector in zip(missing_positions, encoded_batches):
                normalized_vector = [float(value) for value in vector]
                resolved[position] = normalized_vector
                cache_key = cache_keys[position]
                if cache_key:
                    cache_updates[cache_key] = normalized_vector

            if cache_updates:
                self._cache_manager.set_many(self._cache_namespace, cache_updates)

        return [list(vector or []) for vector in resolved]

    def __call__(self, input: Documents) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: Documents) -> List[List[float]]:
        return self.encode(self._coerce_texts(input), kind="document")

    def embed_query(self, input: Documents) -> List[List[float]]:
        return self.encode(self._coerce_texts(input), kind="query")


class _RAGCrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder is not available")
        self._model = CrossEncoder(model_name)

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(value) for value in scores]


class RAGMemory:
    """Persistent vector memory service based on ChromaDB."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or get_config()
        rag_cfg = self.config.get("ai", {}).get("rag", {})
        cache_cfg = self.config.get("cache", {})
        hf_cfg = rag_cfg.get("huggingface", {}) or {}
        reranker_cfg = rag_cfg.get("reranker", {}) or {}
        query_expansion_cfg = rag_cfg.get("query_expansion", {}) or {}
        embedding_cache_cfg = cache_cfg.get("rag_embeddings", {}) or {}

        self.enabled = bool(rag_cfg.get("enabled", True))
        self.top_k = _safe_int(rag_cfg.get("top_k", 5), 5)
        self._candidate_pool = max(
            self.top_k,
            _safe_int(rag_cfg.get("candidate_pool", 18), 18),
        )
        self.embedding_model = str(
            rag_cfg.get("embedding_model", "intfloat/multilingual-e5-large")
        )
        self._embedding_batch_size = max(
            1,
            _safe_int(rag_cfg.get("embedding_batch_size", 32), 32),
        )
        self._normalize_embeddings = bool(rag_cfg.get("normalize_embeddings", True))
        self._analysis_chunk_size = _safe_int(
            rag_cfg.get("analysis_chunk_size", 1800), 1800
        )
        self._analysis_chunk_overlap = _safe_int(
            rag_cfg.get("analysis_chunk_overlap", 220), 220
        )
        self._pre_research_chunk_size = _safe_int(
            rag_cfg.get("pre_research_chunk_size", 1800), 1800
        )
        self._pre_research_chunk_overlap = _safe_int(
            rag_cfg.get("pre_research_chunk_overlap", 220), 220
        )
        self._news_chunk_size = _safe_int(rag_cfg.get("news_chunk_size", 1800), 1800)
        self._news_chunk_overlap = _safe_int(
            rag_cfg.get("news_chunk_overlap", 220), 220
        )
        self._default_confidence_score = float(
            rag_cfg.get("default_confidence_score", 0.6)
        )
        self._confidence_weight = float(rag_cfg.get("confidence_weight", 0.2))
        self._data_gap_penalty = float(rag_cfg.get("data_gap_penalty", 0.03))
        self._reranker_enabled = bool(reranker_cfg.get("enabled", False))
        self._reranker_model = str(
            reranker_cfg.get("model", "BAAI/bge-reranker-v2-m3")
        )
        self._rerank_weight = max(
            0.0,
            min(1.0, float(reranker_cfg.get("weight", 0.65) or 0.65)),
        )
        self._rerank_candidate_pool = max(
            self.top_k,
            _safe_int(reranker_cfg.get("candidate_pool", self._candidate_pool), self._candidate_pool),
        )
        self._query_expansion_enabled = bool(query_expansion_cfg.get("enabled", True))
        self._query_expansion_weight = max(
            0.0,
            min(1.0, float(query_expansion_cfg.get("weight", 0.35) or 0.35)),
        )
        self._query_expansion_template = str(
            query_expansion_cfg.get(
                "fallback_template",
                "Factual research notes about {query}. Include dated catalysts, financial metrics, management commentary, risks, and prior analysis context.",
            )
        ).strip()
        self._query_expansion_max_chars = max(
            200,
            _safe_int(query_expansion_cfg.get("hypothesis_max_chars", 900), 900),
        )

        self._cache_manager = get_unified_cache()
        self._embedding_cache_namespace = "rag_embeddings"
        self._cache_manager.get_ttl_cache(
            namespace=self._embedding_cache_namespace,
            maxsize=_safe_int(embedding_cache_cfg.get("max_entries", 2048), 2048),
            ttl_seconds=_safe_int(embedding_cache_cfg.get("ttl_seconds", 43200), 43200),
        )

        self._document_prefix, self._query_prefix = self._resolve_embedding_prefixes(
            self.embedding_model,
            rag_cfg,
        )

        default_collections = {
            "analysis": "analysis_history",
            "pre_research": "pre_research_reports",
            "news": "news_articles",
        }
        self.collection_names = {
            **default_collections,
            **(rag_cfg.get("collections", {}) or {}),
        }

        self._client = None
        self._collections: Dict[str, Any] = {}
        self._parent_collections: Dict[str, Any] = {}
        self._parent_collection_names = {
            key: f"{name}__parents" for key, name in self.collection_names.items()
        }
        self._warmed_up = False
        self._embedding_function: Optional[CachedSentenceTransformerEmbeddingFunction] = None
        self._reranker: Optional[_RAGCrossEncoderReranker] = None
        self._reranker_failed = False

        self._hf_suppress_model_load_report = bool(
            hf_cfg.get("suppress_model_load_report", True)
        )

        self._configure_embedding_runtime(hf_cfg)

        if not self.enabled:
            logger.info("RAG disabled by config.")
            return
        if chromadb is None or SentenceTransformer is None:
            logger.warning(
                "RAG dependencies are missing (chromadb/sentence-transformers)."
            )
            self.enabled = False
            return

        try:
            memory_dir = get_runtime_dir() / "instance" / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)

            with self._suppress_embedding_load_noise(
                self._hf_suppress_model_load_report
            ):
                self._embedding_function = CachedSentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model,
                    cache_namespace=self._embedding_cache_namespace,
                    batch_size=self._embedding_batch_size,
                    normalize_embeddings=self._normalize_embeddings,
                    document_prefix=self._document_prefix,
                    query_prefix=self._query_prefix,
                )

            self._client = chromadb.PersistentClient(path=str(memory_dir))
            for key, name in self.collection_names.items():
                self._collections[key] = self._client.get_or_create_collection(
                    name=name,
                    embedding_function=self._embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                self._parent_collections[key] = self._client.get_or_create_collection(
                    name=self._parent_collection_names[key],
                    embedding_function=self._embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )

            logger.info(
                "RAG initialized. Collections=%s model=%s reranker=%s",
                self.collection_names,
                self.embedding_model,
                "enabled" if self._reranker_enabled else "disabled",
            )
        except Exception as exc:
            logger.error("RAG initialization failed: %s", exc)
            self.enabled = False

    def is_ready(self) -> bool:
        return self.enabled and bool(self._collections)

    def warmup(self) -> bool:
        """Force embedding model initialization early to avoid first-use stalls."""
        if self._warmed_up:
            return self.is_ready()
        if not self.is_ready():
            return False

        collection_key = next(iter(self._collections.keys()), None)
        if not collection_key:
            return False

        with self._suppress_embedding_load_noise(
            self._hf_suppress_model_load_report
        ):
            self._query_collection(collection_key, "startup warmup", 1)
        self._warmed_up = True
        logger.info("RAG warmup completed. model=%s", self.embedding_model)
        return True

    @staticmethod
    def _resolve_embedding_prefixes(
        model_name: str,
        rag_cfg: Dict[str, Any],
    ) -> Tuple[str, str]:
        explicit_document_prefix = str(rag_cfg.get("document_prefix") or "").strip()
        explicit_query_prefix = str(rag_cfg.get("query_prefix") or "").strip()
        if explicit_document_prefix or explicit_query_prefix:
            return explicit_document_prefix, explicit_query_prefix

        lowered = str(model_name or "").strip().lower()
        if "e5" in lowered:
            return "passage: ", "query: "
        return "", ""

    def _configure_embedding_runtime(self, hf_cfg: Dict[str, Any]) -> None:
        hf_token = str(hf_cfg.get("hf_token") or os.getenv("HF_TOKEN") or "").strip()
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        if bool(hf_cfg.get("suppress_hub_warnings", True)):
            warnings.filterwarnings(
                "ignore",
                message=r".*unauthenticated requests to the HF Hub.*",
            )
            hf_logger = logging.getLogger("huggingface_hub.utils._http")
            hf_logger.addFilter(_SuppressUnauthenticatedHFHubWarning())

        if bool(hf_cfg.get("suppress_http_logs", True)):
            logging.getLogger("httpx").setLevel(logging.WARNING)

        logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(
            logging.WARNING
        )

        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
        except Exception:
            pass

    @staticmethod
    @contextlib.contextmanager
    def _suppress_embedding_load_noise(enabled: bool):
        if not enabled:
            yield
            return

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            yield

    @staticmethod
    def _normalize_multiline_text(text: str) -> str:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _extract_heading(line: str) -> Optional[Tuple[int, str]]:
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$", str(line or ""))
        if not match:
            return None
        level = len(match.group(1))
        title = re.sub(r"\s+#*$", "", match.group(2)).strip()
        if not title:
            return None
        return level, title

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = str(line or "").strip()
        return stripped.startswith("|") and stripped.count("|") >= 2

    @staticmethod
    def _is_list_line(line: str) -> bool:
        return bool(re.match(r"^\s*(?:[-*+]|\d+[.)])\s+\S", str(line or "")))

    @classmethod
    def _split_structured_sections(cls, text: str) -> List[Tuple[str, List[str]]]:
        normalized = cls._normalize_multiline_text(text)
        if not normalized:
            return []

        lines = normalized.split("\n")
        sections: List[Tuple[str, List[str]]] = []
        current_path: List[str] = []
        current_lines: List[str] = []

        def flush_current() -> None:
            if not current_lines:
                return
            section_name = " / ".join(current_path) if current_path else "General"
            sections.append((section_name, list(current_lines)))
            current_lines.clear()

        for line in lines:
            heading = cls._extract_heading(line)
            if heading:
                flush_current()
                level, title = heading
                current_path = current_path[: level - 1] + [title]
                continue
            current_lines.append(line)

        flush_current()
        return sections or [("General", lines)]

    @classmethod
    def _section_blocks(cls, section: str, lines: List[str]) -> List[_StructuredBlock]:
        blocks: List[_StructuredBlock] = []
        index = 0
        total = len(lines)

        while index < total:
            line = lines[index]
            stripped = line.strip()
            if not stripped:
                index += 1
                continue

            if cls._is_table_line(line):
                table_lines = [line.rstrip()]
                index += 1
                while index < total and cls._is_table_line(lines[index]):
                    table_lines.append(lines[index].rstrip())
                    index += 1
                blocks.append(
                    _StructuredBlock(
                        text="\n".join(table_lines).strip(),
                        section=section,
                        block_type="table",
                    )
                )
                continue

            if cls._is_list_line(line):
                list_lines = [line.rstrip()]
                index += 1
                while index < total and cls._is_list_line(lines[index]):
                    list_lines.append(lines[index].rstrip())
                    index += 1
                blocks.append(
                    _StructuredBlock(
                        text="\n".join(list_lines).strip(),
                        section=section,
                        block_type="list",
                    )
                )
                continue

            paragraph_lines = [stripped]
            index += 1
            while index < total:
                next_line = lines[index]
                if not next_line.strip():
                    index += 1
                    break
                if cls._is_table_line(next_line) or cls._is_list_line(next_line):
                    break
                paragraph_lines.append(next_line.strip())
                index += 1

            blocks.append(
                _StructuredBlock(
                    text=" ".join(paragraph_lines).strip(),
                    section=section,
                    block_type="paragraph",
                )
            )

        return blocks

    @staticmethod
    def _split_sentences(text: str) -> Optional[List[str]]:
        normalized = (text or "").strip()
        if not normalized:
            return []

        paragraphs = [
            part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()
        ]
        if len(paragraphs) <= 1:
            paragraphs = [normalized]

        sentences: List[str] = []
        for paragraph in paragraphs:
            parts = [
                piece.strip()
                for piece in re.split(r"(?<=[.!?])\s+", paragraph)
                if piece.strip()
            ]
            if not parts:
                continue
            sentences.extend(parts)

        if len(sentences) <= 1:
            return None
        return sentences

    @staticmethod
    def _chunk_text_semantic(
        sentences: List[str], chunk_size: int, overlap: int
    ) -> List[str]:
        if not sentences:
            return []

        avg_sentence_len = max(1, int(sum(len(s) for s in sentences) / len(sentences)))
        overlap_sentences = max(
            0, min(len(sentences) - 1, int(overlap / avg_sentence_len))
        )

        chunks: List[str] = []
        start_idx = 0
        total = len(sentences)
        while start_idx < total:
            end_idx = start_idx
            current: List[str] = []
            current_size = 0
            while end_idx < total:
                sentence = sentences[end_idx]
                projected = current_size + (1 if current else 0) + len(sentence)
                if current and projected > chunk_size:
                    break
                current.append(sentence)
                current_size = projected
                end_idx += 1

            if not current:
                current = [sentences[start_idx][:chunk_size]]
                end_idx = start_idx + 1

            chunks.append(" ".join(current).strip())
            if end_idx >= total:
                break

            if overlap_sentences > 0:
                start_idx = max(start_idx + 1, end_idx - overlap_sentences)
            else:
                start_idx = end_idx

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _join_blocks(blocks: Sequence[_StructuredBlock]) -> str:
        return "\n\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()

    @staticmethod
    def _block_window_length(blocks: Sequence[_StructuredBlock]) -> int:
        if not blocks:
            return 0
        return sum(len(block.text) for block in blocks) + (2 * max(0, len(blocks) - 1))

    @classmethod
    def _overlap_blocks(
        cls,
        blocks: Sequence[_StructuredBlock],
        overlap: int,
    ) -> List[_StructuredBlock]:
        if overlap <= 0 or not blocks:
            return []

        retained: List[_StructuredBlock] = []
        current_size = 0
        for block in reversed(blocks):
            projected = current_size + len(block.text) + (2 if retained else 0)
            if retained and projected > overlap:
                break
            retained.insert(0, block)
            current_size = projected
            if current_size >= overlap:
                break
        return retained

    def _split_oversized_block(
        self,
        block: _StructuredBlock,
        chunk_size: int,
        overlap: int,
    ) -> List[_StructuredChunk]:
        if block.block_type in {"table", "list"}:
            return [
                _StructuredChunk(
                    content=block.text.strip(),
                    section=block.section,
                    content_type=block.block_type,
                )
            ]

        sentences = self._split_sentences(block.text)
        semantic_chunks = (
            self._chunk_text_semantic(sentences, chunk_size, overlap)
            if sentences
            else []
        )
        if semantic_chunks:
            return [
                _StructuredChunk(
                    content=chunk,
                    section=block.section,
                    content_type=block.block_type,
                )
                for chunk in semantic_chunks
            ]

        chunks: List[_StructuredChunk] = []
        step = max(1, chunk_size - max(0, overlap))
        start = 0
        while start < len(block.text):
            end = min(len(block.text), start + chunk_size)
            chunk = block.text[start:end].strip()
            if chunk:
                chunks.append(
                    _StructuredChunk(
                        content=chunk,
                        section=block.section,
                        content_type=block.block_type,
                    )
                )
            if end >= len(block.text):
                break
            start += step
        return chunks

    def _build_structured_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[_StructuredChunk]:
        sections = self._split_structured_sections(text)
        if not sections:
            return []

        chunks: List[_StructuredChunk] = []
        for section_name, lines in sections:
            blocks = self._section_blocks(section_name, lines)
            if not blocks:
                continue

            current_blocks: List[_StructuredBlock] = []
            current_size = 0

            def emit_current() -> None:
                nonlocal current_blocks, current_size
                if not current_blocks:
                    return
                combined = self._join_blocks(current_blocks)
                if not combined:
                    current_blocks = []
                    current_size = 0
                    return
                content_types = {block.block_type for block in current_blocks}
                chunk_type = content_types.pop() if len(content_types) == 1 else "mixed"
                chunks.append(
                    _StructuredChunk(
                        content=combined,
                        section=section_name,
                        content_type=chunk_type,
                    )
                )
                current_blocks = self._overlap_blocks(current_blocks, overlap)
                current_size = self._block_window_length(current_blocks)

            for block in blocks:
                if len(block.text) > chunk_size and block.block_type == "paragraph":
                    emit_current()
                    chunks.extend(self._split_oversized_block(block, chunk_size, overlap))
                    current_blocks = []
                    current_size = 0
                    continue

                projected = current_size + len(block.text) + (2 if current_blocks else 0)
                if current_blocks and projected > chunk_size:
                    emit_current()

                if len(block.text) > chunk_size and block.block_type in {"table", "list"}:
                    chunks.append(
                        _StructuredChunk(
                            content=block.text,
                            section=block.section,
                            content_type=block.block_type,
                        )
                    )
                    current_blocks = []
                    current_size = 0
                    continue

                current_blocks.append(block)
                current_size = self._block_window_length(current_blocks)

            if current_blocks:
                emit_current()

        return [chunk for chunk in chunks if chunk.content.strip()]

    def _build_chunk_records(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[Dict[str, str]]:
        normalized = self._normalize_multiline_text(text)
        if not normalized:
            return []
        if len(normalized) <= chunk_size:
            return [
                {
                    "content": normalized,
                    "section": "General",
                    "content_type": "plain",
                }
            ]

        structured_chunks = self._build_structured_chunks(normalized, chunk_size, overlap)
        if structured_chunks:
            return [
                {
                    "content": chunk.content,
                    "section": chunk.section,
                    "content_type": chunk.content_type,
                }
                for chunk in structured_chunks
                if chunk.content.strip()
            ]

        sentences = self._split_sentences(normalized)
        if sentences:
            semantic_chunks = self._chunk_text_semantic(sentences, chunk_size, overlap)
            if semantic_chunks:
                return [
                    {
                        "content": chunk,
                        "section": "General",
                        "content_type": "plain",
                    }
                    for chunk in semantic_chunks
                ]

        chunks: List[Dict[str, str]] = []
        step = max(1, chunk_size - max(0, overlap))
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_size)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(
                    {
                        "content": chunk,
                        "section": "General",
                        "content_type": "plain",
                    }
                )
            if end >= len(normalized):
                break
            start += step
        return chunks

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        return [
            record["content"]
            for record in self._build_chunk_records(text, chunk_size, overlap)
            if str(record.get("content") or "").strip()
        ]

    @staticmethod
    def _doc_id(prefix: str, text: str, metadata: Dict[str, Any]) -> str:
        meta_key = "|".join(f"{k}={metadata.get(k)}" for k in sorted(metadata.keys()))
        digest = hashlib.sha256(
            f"{prefix}|{meta_key}|{text}".encode("utf-8")
        ).hexdigest()
        return f"{prefix}_{digest[:24]}"

    @staticmethod
    def _doc_group_id(*parts: str) -> str:
        payload = "|".join(str(part or "") for part in parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _to_unix_ts(value: Any) -> float:
        """Convert ISO string, datetime, or numeric value to a Unix timestamp float.

        ChromaDB $gte/$lte operators require int or float operands.
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                pass
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def _clamp_confidence(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(fallback)
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _safe_int(value: Any, fallback: int = 0) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in (meta or {}).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    def _get_reranker(self) -> Optional[_RAGCrossEncoderReranker]:
        if not self._reranker_enabled or self._reranker_failed:
            return None
        if self._reranker is not None:
            return self._reranker

        try:
            with self._suppress_embedding_load_noise(
                self._hf_suppress_model_load_report
            ):
                self._reranker = _RAGCrossEncoderReranker(self._reranker_model)
            return self._reranker
        except Exception as exc:
            self._reranker_failed = True
            logger.warning("RAG reranker disabled after initialization failure: %s", exc)
            return None

    @staticmethod
    def _normalize_model_scores(scores: Sequence[float]) -> List[float]:
        numeric = [float(score) for score in scores]
        if not numeric:
            return []
        min_score = min(numeric)
        max_score = max(numeric)
        if max_score == min_score:
            return [1.0 for _ in numeric]
        return [
            (float(score) - min_score) / (max_score - min_score)
            for score in numeric
        ]

    def _embed_query(self, query_text: str) -> Optional[List[float]]:
        if not self._embedding_function:
            return None
        normalized = str(query_text or "").strip()
        if not normalized:
            return None
        vectors = self._embedding_function.encode([normalized], kind="query")
        if not vectors:
            return None
        return vectors[0]

    def semantic_similarity_scores(
        self,
        query_text: str,
        documents: Sequence[str],
    ) -> List[float]:
        if not self._embedding_function:
            return [0.0 for _ in documents]

        normalized_query = str(query_text or "").strip()
        normalized_docs = [str(doc or "").strip() for doc in documents]
        if not normalized_query or not normalized_docs:
            return [0.0 for _ in normalized_docs]

        query_vector = self._embed_query(normalized_query)
        if not query_vector:
            return [0.0 for _ in normalized_docs]

        doc_vectors = self._embedding_function.encode(normalized_docs, kind="document")
        scores: List[float] = []
        for vector in doc_vectors:
            if not vector:
                scores.append(0.0)
                continue
            dot = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
            query_norm = sum(float(value) * float(value) for value in query_vector) ** 0.5
            vector_norm = sum(float(value) * float(value) for value in vector) ** 0.5
            if query_norm <= 0.0 or vector_norm <= 0.0:
                scores.append(0.0)
                continue
            scores.append(max(-1.0, min(1.0, dot / (query_norm * vector_norm))))

        return scores

    def has_document_url(self, collection_key: str, url: str) -> bool:
        if collection_key not in self._collections:
            return False

        normalized_url = str(url or "").strip()
        if not normalized_url:
            return False

        try:
            existing = self._collections[collection_key].get(
                where={"url": normalized_url},
                limit=1,
            )
        except Exception as exc:
            logger.debug("URL lookup failed in %s for %s: %s", collection_key, normalized_url, exc)
            return False

        return bool((existing or {}).get("ids"))

    def _build_query_variants(
        self,
        query: str,
        query_hypothesis: Optional[str] = None,
        query_variants: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []

        variants: List[Dict[str, Any]] = [
            {"label": "query", "text": normalized_query, "weight": 1.0}
        ]

        explicit_hypothesis = str(query_hypothesis or "").strip()
        if self._query_expansion_enabled and explicit_hypothesis:
            variants.append(
                {
                    "label": "hyde",
                    "text": explicit_hypothesis[: self._query_expansion_max_chars],
                    "weight": self._query_expansion_weight,
                }
            )

        for index, variant in enumerate(query_variants or []):
            text = str(variant or "").strip()
            if not text:
                continue
            variants.append(
                {
                    "label": f"variant_{index + 1}",
                    "text": text[: self._query_expansion_max_chars],
                    "weight": self._query_expansion_weight,
                }
            )

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for variant in variants:
            fingerprint = str(variant.get("text") or "").strip().lower()
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(variant)
        return deduped

    @staticmethod
    def _result_identity(item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        parent_id = str(metadata.get("parent_id") or "").strip()
        collection_name = str(item.get("collection") or "").strip()
        doc_id = str(item.get("id") or "").strip()
        return f"{collection_name}:{parent_id or doc_id}"

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        for item in results:
            retrieval_score = item.get("retrieval_score")
            if not isinstance(retrieval_score, (int, float)):
                retrieval_score = item.get("quality_adjusted_score")
            item["final_score"] = float(retrieval_score or 0.0)

        if len(results) < 2:
            return results

        reranker = self._get_reranker()
        if reranker is None:
            return results

        ranked = sorted(
            results,
            key=lambda item: float(item.get("final_score") or 0.0),
            reverse=True,
        )
        candidate_count = min(len(ranked), self._rerank_candidate_pool)
        candidates = ranked[:candidate_count]
        candidate_docs = [str(item.get("content") or "") for item in candidates]

        try:
            raw_scores = reranker.score(query, candidate_docs)
        except Exception as exc:
            logger.warning("RAG rerank skipped after scoring failure: %s", exc)
            self._reranker_failed = True
            return results

        normalized_scores = self._normalize_model_scores(raw_scores)
        for item, raw_score, normalized_score in zip(
            candidates, raw_scores, normalized_scores
        ):
            retrieval_score = float(item.get("final_score") or 0.0)
            item["rerank_score"] = float(normalized_score)
            item["rerank_score_raw"] = float(raw_score)
            item["final_score"] = (
                (1.0 - self._rerank_weight) * retrieval_score
                + self._rerank_weight * float(normalized_score)
            )

        return results

    def _add_documents(
        self,
        collection_key: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        id_prefix: str,
    ) -> int:
        if not self.is_ready() or collection_key not in self._collections:
            return 0

        clean_docs: List[str] = []
        clean_meta: List[Dict[str, Any]] = []
        clean_ids: List[str] = []

        for doc, meta in zip(documents, metadatas):
            if not isinstance(doc, str) or not doc.strip():
                continue
            safe_meta = self._sanitize_metadata(meta or {})
            clean_docs.append(doc.strip())
            clean_meta.append(safe_meta)
            clean_ids.append(self._doc_id(id_prefix, doc.strip(), safe_meta))

        if not clean_docs:
            return 0

        try:
            self._collections[collection_key].upsert(
                ids=clean_ids,
                documents=clean_docs,
                metadatas=clean_meta,
            )
            return len(clean_docs)
        except Exception as exc:
            logger.warning("RAG upsert failed for %s: %s", collection_key, exc)
            return 0

    def _add_parent_document(
        self,
        collection_key: str,
        parent_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> bool:
        if (
            not self.is_ready()
            or collection_key not in self._parent_collections
            or not parent_id
            or not str(content or "").strip()
        ):
            return False

        try:
            self._parent_collections[collection_key].upsert(
                ids=[parent_id],
                documents=[str(content).strip()],
                metadatas=[self._sanitize_metadata(metadata)],
            )
            return True
        except Exception as exc:
            logger.warning("RAG parent upsert failed for %s: %s", collection_key, exc)
            return False

    def _get_parent_document(
        self,
        collection_key: str,
        parent_id: str,
    ) -> Optional[Dict[str, Any]]:
        if collection_key not in self._parent_collections or not parent_id:
            return None

        try:
            payload = self._parent_collections[collection_key].get(ids=[parent_id])
        except Exception as exc:
            logger.debug("Parent fetch failed (%s): %s", collection_key, exc)
            return None

        docs = payload.get("documents") or []
        metas = payload.get("metadatas") or []
        ids = payload.get("ids") or []
        if not docs:
            return None

        metadata = metas[0] if metas and isinstance(metas[0], dict) else {}
        return {
            "id": ids[0] if ids else parent_id,
            "content": str(docs[0] or ""),
            "metadata": metadata,
            "collection": collection_key,
        }

    def _index_parent_child_document(
        self,
        collection_key: str,
        content: str,
        *,
        metadata: Dict[str, Any],
        chunk_size: int,
        overlap: int,
        id_prefix: str,
    ) -> int:
        normalized = self._normalize_multiline_text(content)
        if not normalized:
            return 0

        base_metadata = dict(metadata or {})
        parent_metadata = dict(base_metadata)
        parent_metadata["doc_role"] = "parent"
        parent_id = self._doc_id(f"{id_prefix}_parent", normalized, parent_metadata)
        parent_metadata["parent_id"] = parent_id

        parent_saved = self._add_parent_document(
            collection_key,
            parent_id,
            normalized,
            parent_metadata,
        )

        chunk_records = self._build_chunk_records(normalized, chunk_size, overlap)
        if not chunk_records:
            return 0

        total_chunks = len(chunk_records)
        child_documents: List[str] = []
        child_metadatas: List[Dict[str, Any]] = []

        for index, chunk in enumerate(chunk_records):
            child_documents.append(str(chunk.get("content") or "").strip())
            child_metadata = dict(base_metadata)
            child_metadata.update(
                {
                    "doc_role": "child",
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "section": str(chunk.get("section") or "General"),
                    "content_type": str(chunk.get("content_type") or "plain"),
                }
            )
            if parent_saved:
                child_metadata["parent_id"] = parent_id
            child_metadatas.append(child_metadata)

        return self._add_documents(
            collection_key,
            child_documents,
            child_metadatas,
            id_prefix=id_prefix,
        )

    def index_analysis(
        self,
        symbol: str,
        news_output: Optional[str],
        final_output: Optional[str],
        timestamp: Optional[str] = None,
        exchange: Optional[str] = None,
        confidence_score: Optional[float] = None,
        data_gaps_count: Optional[int] = None,
    ) -> int:
        if not self.is_ready():
            return 0

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        confidence = self._clamp_confidence(
            confidence_score, self._default_confidence_score
        )
        gap_count = self._safe_int(data_gaps_count, 0)
        total_indexed = 0

        if news_output:
            news_group_id = self._doc_group_id(
                "analysis", symbol.upper(), ts, "news_analysis"
            )
            total_indexed += self._index_parent_child_document(
                "analysis",
                news_output,
                metadata={
                    "symbol": symbol.upper(),
                    "exchange": exchange or "",
                    "timestamp": self._to_unix_ts(ts),
                    "doc_type": "news_analysis",
                    "confidence_score": confidence,
                    "data_gaps_count": gap_count,
                    "doc_group_id": news_group_id,
                },
                chunk_size=self._analysis_chunk_size,
                overlap=self._analysis_chunk_overlap,
                id_prefix=f"analysis_news_{symbol.upper()}",
            )

        if final_output:
            final_group_id = self._doc_group_id(
                "analysis", symbol.upper(), ts, "final_analysis"
            )
            total_indexed += self._index_parent_child_document(
                "analysis",
                final_output,
                metadata={
                    "symbol": symbol.upper(),
                    "exchange": exchange or "",
                    "timestamp": self._to_unix_ts(ts),
                    "doc_type": "final_analysis",
                    "confidence_score": confidence,
                    "data_gaps_count": gap_count,
                    "doc_group_id": final_group_id,
                },
                chunk_size=self._analysis_chunk_size,
                overlap=self._analysis_chunk_overlap,
                id_prefix=f"analysis_final_{symbol.upper()}",
            )

        return total_indexed

    def index_pre_research(
        self,
        exchange: str,
        markdown_content: str,
        timestamp: Optional[str] = None,
        confidence_score: Optional[float] = None,
        data_gaps_count: Optional[int] = None,
        symbol: Optional[str] = None,
        url: Optional[str] = None,
        doc_type: str = "pre_research",
    ) -> int:
        if not self.is_ready() or not markdown_content:
            return 0

        normalized_url = str(url or "").strip()
        if normalized_url and self.has_document_url("pre_research", normalized_url):
            return 0

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        confidence = self._clamp_confidence(
            confidence_score, self._default_confidence_score
        )
        gap_count = self._safe_int(data_gaps_count, 0)
        group_id = self._doc_group_id(
            "pre_research", normalized_url or exchange.upper(), ts, doc_type
        )
        return self._index_parent_child_document(
            "pre_research",
            markdown_content,
            metadata={
                "exchange": exchange.upper(),
                "symbol": (symbol or "").upper(),
                "url": normalized_url,
                "timestamp": self._to_unix_ts(ts),
                "doc_type": str(doc_type or "pre_research").strip() or "pre_research",
                "confidence_score": confidence,
                "data_gaps_count": gap_count,
                "doc_group_id": group_id,
            },
            chunk_size=self._pre_research_chunk_size,
            overlap=self._pre_research_chunk_overlap,
            id_prefix=f"pre_research_{exchange.upper()}",
        )

    def index_news_article(
        self,
        url: str,
        content: str,
        symbol: Optional[str] = None,
        timestamp: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> int:
        if not self.is_ready() or not content:
            return 0

        try:
            existing = self._collections["news"].get(where={"url": url}, limit=1)
            if (existing or {}).get("ids"):
                return 0
        except Exception as exc:
            logger.debug("News dedupe lookup failed for %s: %s", url, exc)

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        confidence = self._clamp_confidence(
            confidence_score, self._default_confidence_score
        )
        group_id = self._doc_group_id("news", url, ts, "news_article")
        return self._index_parent_child_document(
            "news",
            content,
            metadata={
                "url": url,
                "symbol": (symbol or "").upper(),
                "timestamp": self._to_unix_ts(ts),
                "doc_type": "news_article",
                "confidence_score": confidence,
                "doc_group_id": group_id,
            },
            chunk_size=self._news_chunk_size,
            overlap=self._news_chunk_overlap,
            id_prefix=f"news_{(symbol or 'NA').upper()}",
        )

    def _fetch_adjacent_chunks(
        self,
        collection_key: str,
        doc_group_id: str,
        center_index: int,
        window: int = 1,
    ) -> List[Dict[str, Any]]:
        if collection_key not in self._collections or not doc_group_id:
            return []

        try:
            payload = self._collections[collection_key].get(
                where={"doc_group_id": doc_group_id}
            )
        except Exception as exc:
            logger.debug("Adjacent chunk fetch failed (%s): %s", collection_key, exc)
            return []

        docs = payload.get("documents") or []
        metas = payload.get("metadatas") or []
        ids = payload.get("ids") or []

        results: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            meta = (
                metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            )
            chunk_index = meta.get("chunk_index")
            if not isinstance(chunk_index, int):
                continue
            if (
                chunk_index < center_index - window
                or chunk_index > center_index + window
            ):
                continue
            results.append(
                {
                    "id": ids[idx] if idx < len(ids) else None,
                    "content": str(doc or ""),
                    "metadata": meta,
                    "chunk_index": chunk_index,
                }
            )

        results.sort(key=lambda item: int(item.get("chunk_index", 0)))
        return results

    def _expand_with_context_windows(
        self, items: List[Dict[str, Any]], context_window: int
    ) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        cache: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for item in items:
            metadata = (
                item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            )
            collection_key = str(item.get("collection") or "")
            parent_id = str(metadata.get("parent_id") or "").strip()

            if collection_key and parent_id:
                parent_doc = self._get_parent_document(collection_key, parent_id)
                if parent_doc:
                    updated = dict(item)
                    updated_metadata = dict(metadata)
                    parent_metadata = (
                        parent_doc.get("metadata")
                        if isinstance(parent_doc.get("metadata"), dict)
                        else {}
                    )
                    for key, value in parent_metadata.items():
                        updated_metadata.setdefault(key, value)
                    updated_metadata["parent_id"] = parent_id
                    updated_metadata["matched_chunk_index"] = metadata.get("chunk_index")
                    updated_metadata["matched_section"] = metadata.get("section")
                    updated["metadata"] = updated_metadata
                    updated["content"] = str(parent_doc.get("content") or "")
                    updated["matched_chunk_content"] = str(item.get("content") or "")
                    updated["parent_document"] = parent_doc
                    expanded.append(updated)
                    continue

            if context_window <= 0:
                expanded.append(item)
                continue

            doc_group_id = str(metadata.get("doc_group_id") or "").strip()
            chunk_index = metadata.get("chunk_index")

            if (
                not doc_group_id
                or not isinstance(chunk_index, int)
                or not collection_key
            ):
                expanded.append(item)
                continue

            cache_key = (collection_key, doc_group_id)
            if cache_key not in cache:
                cache[cache_key] = self._fetch_adjacent_chunks(
                    collection_key=collection_key,
                    doc_group_id=doc_group_id,
                    center_index=chunk_index,
                    window=context_window,
                )
            adjacent = cache.get(cache_key, [])
            if not adjacent:
                expanded.append(item)
                continue

            previous = [
                entry["content"]
                for entry in adjacent
                if entry.get("chunk_index", -1) < chunk_index
            ]
            matched = [
                entry["content"]
                for entry in adjacent
                if entry.get("chunk_index") == chunk_index
            ]
            next_items = [
                entry["content"]
                for entry in adjacent
                if entry.get("chunk_index", -1) > chunk_index
            ]

            composed_parts: List[str] = []
            if previous:
                composed_parts.append("[PREV_CONTEXT]\n" + "\n".join(previous))
            if matched:
                composed_parts.append("[MATCH]\n" + "\n".join(matched))
            else:
                composed_parts.append("[MATCH]\n" + str(item.get("content") or ""))
            if next_items:
                composed_parts.append("[NEXT_CONTEXT]\n" + "\n".join(next_items))

            updated = dict(item)
            updated["content"] = "\n\n".join(composed_parts).strip()
            expanded.append(updated)

        return expanded

    @staticmethod
    def _merge_where_clauses(clauses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        valid = [c for c in clauses if isinstance(c, dict) and c]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        return {"$and": valid}

    @staticmethod
    def _normalize_collection_scores(results: List[Dict[str, Any]]) -> None:
        finite_distances = [
            float(item.get("distance"))
            for item in results
            if isinstance(item.get("distance"), (int, float))
        ]

        if not finite_distances:
            for item in results:
                item["normalized_score"] = 0.0
            return

        min_dist = min(finite_distances)
        max_dist = max(finite_distances)

        if max_dist == min_dist:
            for item in results:
                item["normalized_score"] = (
                    1.0 if isinstance(item.get("distance"), (int, float)) else 0.0
                )
            return

        for item in results:
            distance = item.get("distance")
            if not isinstance(distance, (int, float)):
                item["normalized_score"] = 0.0
                continue
            item["normalized_score"] = (max_dist - float(distance)) / (
                max_dist - min_dist
            )

    def _apply_confidence_weighting(self, results: List[Dict[str, Any]]) -> None:
        weight = max(0.0, min(1.0, self._confidence_weight))
        gap_penalty_unit = max(0.0, self._data_gap_penalty)

        for item in results:
            base = (
                float(item.get("normalized_score"))
                if isinstance(item.get("normalized_score"), (int, float))
                else 0.0
            )
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            confidence = self._clamp_confidence(
                metadata.get("confidence_score", self._default_confidence_score),
                self._default_confidence_score,
            )
            data_gaps = self._safe_int(metadata.get("data_gaps_count", 0), 0)
            weighted = base * ((1.0 - weight) + (weight * confidence))
            weighted -= min(0.5, data_gaps * gap_penalty_unit)
            item["quality_adjusted_score"] = max(0.0, weighted)

    def _query_collection(
        self,
        collection_key: str,
        query: str,
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if collection_key not in self._collections:
            return []

        try:
            query_kwargs: Dict[str, Any] = {
                "n_results": max(1, n_results),
                "where": where or None,
            }
            if query_embedding:
                query_kwargs["query_embeddings"] = [query_embedding]
            else:
                query_kwargs["query_texts"] = [query]
            payload = self._collections[collection_key].query(**query_kwargs)
        except Exception as exc:
            logger.warning("RAG query failed in %s: %s", collection_key, exc)
            return []

        docs = (payload.get("documents") or [[]])[0]
        metas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        ids = (payload.get("ids") or [[]])[0]

        results: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            results.append(
                {
                    "id": ids[idx] if idx < len(ids) else None,
                    "content": doc,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                    "collection": collection_key,
                }
            )
        return results

    def fetch_hits(
        self,
        hit_ids: List[str],
        *,
        context_window: int = 0,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready() or not hit_ids:
            return []

        grouped: Dict[str, List[str]] = {}
        requested_order: Dict[str, int] = {}
        for index, raw_hit_id in enumerate(hit_ids):
            hit_id = str(raw_hit_id or "").strip()
            if not hit_id or ":" not in hit_id:
                continue
            collection_key, doc_id = hit_id.split(":", 1)
            collection_key = collection_key.strip()
            doc_id = doc_id.strip()
            if not collection_key or not doc_id or collection_key not in self._collections:
                continue
            grouped.setdefault(collection_key, []).append(doc_id)
            requested_order[f"{collection_key}:{doc_id}"] = index

        if not grouped:
            return []

        hydrated: List[Dict[str, Any]] = []
        for collection_key, doc_ids in grouped.items():
            try:
                payload = self._collections[collection_key].get(ids=doc_ids)
            except Exception as exc:
                logger.warning("RAG hydrate failed in %s: %s", collection_key, exc)
                continue

            docs = payload.get("documents") or []
            metas = payload.get("metadatas") or []
            ids = payload.get("ids") or []

            for idx, doc in enumerate(docs):
                doc_id = ids[idx] if idx < len(ids) else None
                metadata = metas[idx] if idx < len(metas) else {}
                hydrated.append(
                    {
                        "id": doc_id,
                        "content": str(doc or ""),
                        "metadata": metadata if isinstance(metadata, dict) else {},
                        "collection": collection_key,
                    }
                )

        hydrated.sort(
            key=lambda item: requested_order.get(
                f"{item.get('collection')}:{item.get('id')}", 10**9
            )
        )
        return self._expand_with_context_windows(
            hydrated, context_window=max(0, int(context_window))
        )

    def search(
        self,
        query: str,
        collection: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        recent_days: Optional[int] = None,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        context_window: int = 0,
        query_hypothesis: Optional[str] = None,
        query_variants: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready() or not query or not query.strip():
            return []

        limit = int(top_k or self.top_k or 5)
        retrieval_variants = self._build_query_variants(
            query.strip(),
            query_hypothesis=query_hypothesis,
            query_variants=query_variants,
        )
        if not retrieval_variants:
            return []

        where: Dict[str, Any] = {}
        if symbol_filter:
            where["symbol"] = symbol_filter.upper()

        effective_from = from_timestamp
        effective_to = to_timestamp
        if recent_days is not None and recent_days > 0:
            effective_from = (
                datetime.now(timezone.utc) - timedelta(days=int(recent_days))
            ).timestamp()

        time_clause: Optional[Dict[str, Any]] = None
        if effective_from is not None and effective_to is not None:
            time_clause = {
                "timestamp": {
                    "$gte": self._to_unix_ts(effective_from),
                    "$lte": self._to_unix_ts(effective_to),
                }
            }
        elif effective_from is not None:
            time_clause = {"timestamp": {"$gte": self._to_unix_ts(effective_from)}}
        elif effective_to is not None:
            time_clause = {"timestamp": {"$lte": self._to_unix_ts(effective_to)}}

        selected_collections = []
        if collection:
            mapped = collection.strip().lower()
            if mapped in self._collections:
                selected_collections = [mapped]
            else:
                return []
        else:
            selected_collections = ["analysis", "pre_research", "news"]

        aggregated_by_identity: Dict[str, Dict[str, Any]] = {}
        per_collection_limit = max(
            limit,
            self.top_k,
            self._candidate_pool,
            self._rerank_candidate_pool,
        )
        for collection_key in selected_collections:
            clauses: List[Dict[str, Any]] = []
            if where:
                clauses.append(where)
            if time_clause:
                clauses.append(time_clause)
            collection_where = self._merge_where_clauses(clauses)

            for variant in retrieval_variants:
                query_embedding = self._embed_query(str(variant.get("text") or ""))
                collection_results = self._query_collection(
                    collection_key=collection_key,
                    query=str(variant.get("text") or query).strip(),
                    query_embedding=query_embedding,
                    n_results=per_collection_limit,
                    where=collection_where,
                )
                self._normalize_collection_scores(collection_results)
                self._apply_confidence_weighting(collection_results)

                for item in collection_results:
                    variant_weight = float(variant.get("weight") or 1.0)
                    retrieval_score = float(item.get("quality_adjusted_score") or 0.0)
                    retrieval_score *= variant_weight
                    item["retrieval_score"] = retrieval_score
                    item["retrieval_query"] = str(variant.get("label") or "query")
                    item["query_variant_weight"] = variant_weight
                    item["matched_queries"] = [str(variant.get("label") or "query")]

                    identity = self._result_identity(item)
                    existing = aggregated_by_identity.get(identity)
                    if existing is None:
                        aggregated_by_identity[identity] = item
                        continue

                    existing_matches = list(existing.get("matched_queries") or [])
                    label = str(variant.get("label") or "query")
                    if label not in existing_matches:
                        existing_matches.append(label)
                    existing["matched_queries"] = existing_matches

                    if retrieval_score > float(existing.get("retrieval_score") or 0.0):
                        item["matched_queries"] = existing_matches
                        aggregated_by_identity[identity] = item

        aggregated = list(aggregated_by_identity.values())
        if not aggregated:
            return []

        self._rerank_results(query.strip(), aggregated)
        aggregated.sort(
            key=lambda item: (
                -(float(item.get("final_score") or 0.0)),
                -(float(item.get("retrieval_score") or 0.0)),
                item.get("distance") if item.get("distance") is not None else 999.0,
            )
        )
        final_items = aggregated[:limit]
        return self._expand_with_context_windows(
            final_items, context_window=max(0, int(context_window))
        )


__all__ = ["RAGMemory"]
