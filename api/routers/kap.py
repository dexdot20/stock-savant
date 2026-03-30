"""
KAP Bildirim Router
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from api.models import KapBatchDetailRequest, KapBatchDetailResponse, KapListRequest
from core import get_standard_logger
from services.kap import (
    KapLookupError,
    batch_get_disclosure_details,
    get_disclosure_detail,
    get_proxy_manager,
    search_disclosures,
)

router = APIRouter(prefix="/kap", tags=["KAP"])
logger = get_standard_logger(__name__)


@router.post(
    "/disclosures/list",
    summary="Disclosure list",
    description="Return a KAP disclosure list filtered by ticker symbols.",
)
def disclosure_list(req: KapListRequest):
    try:
        return search_disclosures(
            req.stock_codes,
            category=req.category,
            days=req.days,
            from_date=req.from_date,
            to_date=req.to_date,
        )
    except KapLookupError as exc:
        raise HTTPException(status_code=422, detail=exc.details)
    except httpx.HTTPStatusError as exc:
        logger.error("KAP list HTTP error: %s", exc.response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"KAP API error: {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        logger.error("KAP list connection error: %s", exc)
        raise HTTPException(status_code=503, detail=f"Connection error: {exc}")


@router.get(
    "/disclosures/{disclosure_index}/detail",
    summary="Disclosure details",
    description="Fetch and parse the detail page for a single disclosure.",
)
def disclosure_detail(disclosure_index: int):
    try:
        detail = get_disclosure_detail(disclosure_index)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "KAP disclosure #%d HTTP error: %s", disclosure_index, exc.response.status_code
        )
        raise HTTPException(
            status_code=502,
            detail=f"KAP page error: {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        logger.error("KAP disclosure #%d connection error: %s", disclosure_index, exc)
        raise HTTPException(status_code=503, detail=f"Connection error: {exc}")

    if detail is None:
        raise HTTPException(status_code=422, detail="Disclosure detail could not be parsed.")
    return detail


@router.post(
    "/disclosures/batch-details",
    response_model=KapBatchDetailResponse,
    summary="Batch disclosure details",
    description="Fetch and parse multiple disclosures in parallel.",
)
def disclosure_batch_details(req: KapBatchDetailRequest):
    results, errors = batch_get_disclosure_details(
        req.disclosure_indexes, max_workers=req.max_workers
    )
    return KapBatchDetailResponse(results=results, errors=errors)


@router.get(
    "/proxies/status",
    summary="Proxy pool status",
    description="Return the status of each proxy in the pool.",
)
def proxy_status():
    mgr = get_proxy_manager()
    statuses = mgr.status()
    return {
        "proxies": statuses,
        "total": len(statuses),
        "active": sum(1 for p in statuses if p["active"]),
    }
