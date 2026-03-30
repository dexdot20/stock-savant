"""
API Entry Point
"""

import asyncio
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to sys.path for direct execution support
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn
import uuid
import time
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.routers import (
    alerts,
    analysis,
    compare,
    favorites,
    finance,
    history,
    kap,
    portfolio,
    preresearch,
    profile,
)
from api.state import purge_expired_tasks, reconcile_incomplete_tasks
from config import get_config
from core import get_standard_logger
from core.exceptions import BorsaException
import ipaddress

# Initialize Logging
logger = get_standard_logger("api")


def _load_allowed_networks() -> list[ipaddress._BaseNetwork]:
    config = get_config()
    raw_networks = (
        config.get("api", {}).get("allowed_networks")
        if isinstance(config.get("api", {}), dict)
        else None
    )
    if not isinstance(raw_networks, list) or not raw_networks:
        raw_networks = ["127.0.0.0/8", "::1/128", "192.168.0.0/24"]

    parsed: list[ipaddress._BaseNetwork] = []
    for value in raw_networks:
        try:
            parsed.append(ipaddress.ip_network(str(value)))
        except ValueError:
            logger.warning("Skipped invalid allowed network: %s", value)

    return parsed or [ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128")]


ALLOWED_NETWORKS = _load_allowed_networks()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    reconciled = reconcile_incomplete_tasks()
    if reconciled > 0:
        logger.warning("API restart reconciled %s unfinished task(s)", reconciled)

    async def _cleanup_loop():
        interval_seconds = 3600
        ttl_seconds = 24 * 60 * 60
        while not stop_event.is_set():
            removed = purge_expired_tasks(ttl_seconds=ttl_seconds)
            if removed > 0:
                logger.info("API task cleanup removed %s expired task(s)", removed)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                continue

    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        stop_event.set()
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Struct API",
    description=(
        "REST API for AI-powered stock analysis, financial data access, "
        "portfolio and watchlist management, alert monitoring, company comparison, "
        "history access, and autonomous research workflows."
    ),
    version="2.2.0",
    lifespan=lifespan,
)


# X-Request-ID and processing-time middleware
@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{process_time_ms:.2f}"
    return response


# IP access control middleware
@app.middleware("http")
async def limit_ip_access(request: Request, call_next):
    client_ip = request.client.host
    try:
        ip = ipaddress.ip_address(client_ip)
        if not any(ip in net for net in ALLOWED_NETWORKS):
            logger.warning("Unauthorized access attempt from %s", client_ip)
            return JSONResponse(
                status_code=403, content={"detail": "Access denied: IP not allowed"}
            )
    except ValueError:
        return JSONResponse(
            status_code=403, content={"detail": "Access denied: invalid IP"}
        )

    return await call_next(request)


# Central error handling
@app.exception_handler(BorsaException)
async def borsa_exception_handler(request: Request, exc: BorsaException):
    logger.error("Application error: %s", exc.message, extra={"details": exc.details})
    return JSONResponse(
        status_code=400,
        content={"success": False, "message": exc.message, "error_code": "APP_ERROR"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error", extra={"errors": exc.errors()})
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error.",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected server error occurred.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


# CORS (Cross-Origin Resource Sharing)
# In production, replace allow_origins with specific domains.
# NOTE: when allow_credentials=True, allow_origins=["*"] violates the CORS spec;
# browsers reject the preflight request. Because IP filtering already restricts
# access, credentials=False is used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(finance.router)
app.include_router(analysis.router)
app.include_router(preresearch.router)
app.include_router(compare.router)
app.include_router(kap.router)
app.include_router(favorites.router)
app.include_router(portfolio.router)
app.include_router(alerts.router)
app.include_router(profile.router)
app.include_router(history.router)


@app.get("/")
async def root():
    return {
        "message": "Struct API is running.",
        "version": app.version,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "ticker_search": "/finance/search?query=AAPL",
            "market_summary": "/finance/market/summary?region=US",
            "screener": "/finance/screener",
            "company_data": "/finance/{symbol}/company",
            "index_etf_data": "/finance/{symbol}/index",
            "overview": "/finance/{symbol}/overview",
            "financials": "/finance/{symbol}/financials",
            "price_history": "/finance/{symbol}/price?period=1y&interval=1d",
            "dividends": "/finance/{symbol}/dividends",
            "analyst": "/finance/{symbol}/analyst",
            "earnings": "/finance/{symbol}/earnings",
            "ownership": "/finance/{symbol}/ownership",
            "sustainability": "/finance/{symbol}/sustainability",
            "ticker_news": "/finance/{symbol}/news",
            "raw_data": "/finance/{symbol}/raw",
            "single_analysis": "/analyze/stock",
            "batch_analysis": "/analyze/batch",
            "analysis_status": "/analyze/status/{task_id}",
            "pre_research": "/preresearch/",
            "pre_research_status": "/preresearch/{task_id}",
            "comparison": "/compare/",
            "kap_disclosures": "/kap/disclosures/list",
            "favorites": "/favorites/",
            "portfolio_positions": "/portfolio/positions",
            "portfolio_snapshot": "/portfolio/snapshot",
            "portfolio_risk": "/portfolio/risk",
            "alerts": "/alerts/",
            "alert_center": "/alerts/center",
            "investor_profile": "/profile/",
            "playbooks": "/profile/playbooks",
            "history": "/history/",
        },
    }


def start_api(host: str = "0.0.0.0", port: int = 8001):
    """Start the API server programmatically."""
    logger.info("Starting API server at %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api()
