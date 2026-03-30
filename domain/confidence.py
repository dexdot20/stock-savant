from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from config import NA_VALUE

from .quality import assess_data_quality


_COMPLETION_RULES: Dict[str, tuple[str, ...]] = {
    "currentPrice": ("regularMarketPrice", "price"),
    "regularMarketPrice": ("currentPrice", "price"),
    "price": ("currentPrice", "regularMarketPrice"),
    "marketCap": ("mktCap", "marketCapitalization", "market_cap"),
    "longName": ("shortName", "displayName"),
    "shortName": ("longName",),
    "revenue": ("totalRevenue",),
    "volume": ("regularMarketVolume", "averageVolume"),
    "regularMarketVolume": ("volume", "averageVolume"),
    "fiftyDayAverage": ("avg50", "movingAverage50"),
    "twoHundredDayAverage": ("avg200", "movingAverage200"),
}

_CRITICAL_FIELDS: tuple[str, ...] = (
    "symbol",
    "longName",
    "sector",
    "industry",
    "currentPrice",
    "regularMarketPrice",
    "marketCap",
)


def _is_missing(value: Any) -> bool:
    if value is None or value == NA_VALUE:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip() in {"null", "None"}
    return False


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _confidence_level(score_pct: float) -> str:
    if score_pct >= 85:
        return "very_high"
    if score_pct >= 70:
        return "high"
    if score_pct >= 55:
        return "moderate"
    if score_pct >= 35:
        return "low"
    return "very_low"


def auto_complete_company_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply low-risk alias-based completion for missing company fields."""
    if not isinstance(data, dict):
        return {
            "completed_count": 0,
            "completed_fields": {},
            "remaining_missing_fields": list(_CRITICAL_FIELDS),
        }

    completed_fields: Dict[str, str] = {}
    for target, candidates in _COMPLETION_RULES.items():
        if not _is_missing(data.get(target)):
            continue
        for candidate in candidates:
            candidate_value = data.get(candidate)
            if _is_missing(candidate_value):
                continue
            data[target] = candidate_value
            completed_fields[target] = candidate
            break

    remaining_missing = [field for field in _CRITICAL_FIELDS if _is_missing(data.get(field))]

    return {
        "completed_count": len(completed_fields),
        "completed_fields": completed_fields,
        "remaining_missing_fields": remaining_missing,
    }


def calculate_company_confidence(
    data: Dict[str, Any],
    *,
    quality_report: Optional[Dict[str, Any]] = None,
    completion_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a normalized confidence envelope for company data."""
    if not isinstance(data, dict) or not data:
        return {
            "confidence": 0.0,
            "confidence_pct": 0.0,
            "confidence_level": "very_low",
            "factors": {},
            "warnings": ["Company data is missing."],
            "missing_fields": list(_CRITICAL_FIELDS),
            "completed_fields": {},
        }

    quality_report = quality_report if isinstance(quality_report, dict) else {}
    completion_report = completion_report if isinstance(completion_report, dict) else {}

    overall_score = _to_float(quality_report.get("overall_score"), 0.0)
    if overall_score <= 0:
        quality_map = {
            "excellent": 95.0,
            "good": 80.0,
            "fair": 60.0,
            "poor": 35.0,
            "missing": 0.0,
        }
        overall_score = quality_map.get(assess_data_quality(data).value, 0.0)

    field_completeness = _to_float(
        quality_report.get("field_completeness"),
        100.0 - (len(completion_report.get("remaining_missing_fields") or []) * 10.0),
    )
    numeric_validity = _to_float(quality_report.get("numeric_validity"), overall_score)

    data_sources = data.get("data_sources")
    if not isinstance(data_sources, list):
        data_sources = []
    source_count = len([item for item in data_sources if item])
    source_score = min(100.0, 55.0 + (source_count * 15.0)) if source_count else 50.0
    if any(str(item).lower() == "kap" for item in data_sources):
        source_score = min(100.0, source_score + 10.0)

    data_source_mode = str(data.get("data_source") or "").lower()
    freshness_score = 100.0 if data_source_mode == "live" else 72.0 if data_source_mode == "cache" else 60.0

    completed_count = int(completion_report.get("completed_count") or 0)
    missing_fields = list(completion_report.get("remaining_missing_fields") or [])
    inconsistency_count = int(
        _to_float(quality_report.get("numeric_inconsistencies"), 0.0)
        + _to_float(quality_report.get("relationship_issues"), 0.0)
    )

    missing_penalty = min(28.0, len(missing_fields) * 6.0)
    completion_penalty = min(8.0, completed_count * 1.5)
    inconsistency_penalty = min(18.0, inconsistency_count * 3.0)

    score_pct = (
        overall_score * 0.55
        + field_completeness * 0.15
        + numeric_validity * 0.15
        + source_score * 0.10
        + freshness_score * 0.05
        - missing_penalty
        - completion_penalty
        - inconsistency_penalty
    )
    score_pct = round(_clamp(score_pct), 2)
    confidence = round(score_pct / 100.0, 4)

    warnings = []
    if missing_fields:
        warnings.append(
            "Missing critical fields: " + ", ".join(sorted(set(missing_fields)))
        )
    if completed_count > 0:
        warnings.append(
            "Some fields were auto-completed using low-risk fallback aliases."
        )
    if inconsistency_count > 0:
        warnings.append(
            f"{inconsistency_count} validation inconsistency issue(s) detected."
        )
    if score_pct < 55.0:
        warnings.append("Confidence is below preferred threshold for strong recommendations.")

    return {
        "confidence": confidence,
        "confidence_pct": score_pct,
        "confidence_level": _confidence_level(score_pct),
        "factors": {
            "quality_score": round(overall_score, 2),
            "field_completeness": round(field_completeness, 2),
            "numeric_validity": round(numeric_validity, 2),
            "source_score": round(source_score, 2),
            "freshness_score": round(freshness_score, 2),
            "completed_count": completed_count,
            "inconsistency_count": inconsistency_count,
        },
        "warnings": warnings,
        "missing_fields": missing_fields,
        "completed_fields": dict(completion_report.get("completed_fields") or {}),
    }


def calculate_news_confidence(
    company_data: Dict[str, Any],
    *,
    article_count: int,
    source_domains: Optional[Iterable[str]] = None,
    has_news_permission: bool = True,
) -> Dict[str, Any]:
    """Estimate confidence for news synthesis based on coverage and source diversity."""
    source_domains = list(source_domains or [])
    unique_sources = len({str(item).strip().lower() for item in source_domains if str(item).strip()})

    company_confidence = _to_float(company_data.get("confidence") or 0.0, 0.0) * 100.0
    coverage_score = min(100.0, article_count * 18.0)
    diversity_score = min(100.0, unique_sources * 22.0)
    permission_score = 100.0 if has_news_permission else 65.0

    score_pct = (
        company_confidence * 0.35
        + coverage_score * 0.30
        + diversity_score * 0.25
        + permission_score * 0.10
    )
    if article_count == 0 and has_news_permission:
        score_pct *= 0.45

    score_pct = round(_clamp(score_pct), 2)
    warnings = []
    if article_count == 0 and has_news_permission:
        warnings.append("No news articles were available despite news access being enabled.")
    if unique_sources < 2 and article_count > 0:
        warnings.append("News confidence is constrained by limited source diversity.")
    if score_pct < 50.0:
        warnings.append("News confidence is below preferred threshold.")

    return {
        "confidence": round(score_pct / 100.0, 4),
        "confidence_pct": score_pct,
        "confidence_level": _confidence_level(score_pct),
        "factors": {
            "company_confidence": round(company_confidence, 2),
            "coverage_score": round(coverage_score, 2),
            "source_diversity_score": round(diversity_score, 2),
            "article_count": int(article_count),
            "unique_sources": int(unique_sources),
        },
        "warnings": warnings,
    }