"""Investor profile router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.models import (
    InvestorProfileResponse,
    InvestorProfileUpdateRequest,
    PlaybookListResponse,
    PlaybookOptionResponse,
)
from core import get_standard_logger
from services.investor_profile import (
    DEFAULT_PLAYBOOKS,
    get_playbook_choices,
    get_playbook_summary,
    load_investor_profile,
    save_investor_profile,
)

router = APIRouter(prefix="/profile", tags=["Investor Profile"])
logger = get_standard_logger(__name__)


def _serialize_profile(profile: dict) -> InvestorProfileResponse:
    playbook = str(profile.get("active_playbook") or "balanced")
    return InvestorProfileResponse(
        profile_name=str(profile.get("profile_name") or "Default"),
        risk_tolerance=str(profile.get("risk_tolerance") or "medium"),
        investment_horizon=str(profile.get("investment_horizon") or "long-term"),
        market_focus=str(profile.get("market_focus") or "BIST"),
        preferred_sectors=list(profile.get("preferred_sectors") or []),
        avoided_sectors=list(profile.get("avoided_sectors") or []),
        max_single_position_pct=float(profile.get("max_single_position_pct") or 25.0),
        alert_sensitivity=str(profile.get("alert_sensitivity") or "medium"),
        active_playbook=playbook,
        playbook_summary=get_playbook_summary(playbook),
    )


@router.get(
    "/",
    response_model=InvestorProfileResponse,
    summary="Get investor profile",
    description="Return the persisted investor profile used by analysis workflows.",
)
async def get_investor_profile_endpoint():
    logger.info("API Request: investor profile")
    profile = await run_in_threadpool(load_investor_profile)
    return _serialize_profile(profile)


@router.put(
    "/",
    response_model=InvestorProfileResponse,
    summary="Update investor profile",
    description="Update and persist the investor profile used by analysis workflows.",
)
async def update_investor_profile_endpoint(payload: InvestorProfileUpdateRequest):
    logger.info("API Request: update investor profile")
    current_profile = await run_in_threadpool(load_investor_profile)
    updates = payload.model_dump(exclude_none=True)

    if updates.get("active_playbook"):
        available_playbooks = set(get_playbook_choices())
        if updates["active_playbook"] not in available_playbooks:
            raise HTTPException(status_code=422, detail="Unknown playbook.")

    merged_profile = {**current_profile, **updates}
    saved_profile = await run_in_threadpool(save_investor_profile, merged_profile)
    return _serialize_profile(saved_profile)


@router.get(
    "/playbooks",
    response_model=PlaybookListResponse,
    summary="List playbooks",
    description="Return the available investor playbooks and their summaries.",
)
async def list_playbooks_endpoint():
    playbooks = [
        PlaybookOptionResponse(
            key=key,
            label=str(value.get("label") or key.title()),
            summary=str(value.get("summary") or ""),
        )
        for key, value in DEFAULT_PLAYBOOKS.items()
    ]
    return PlaybookListResponse(playbooks=playbooks)