# Struct API Guide

## Overview

Struct API exposes the same core workflows available in the CLI through a FastAPI surface. It covers market data, AI-driven analysis, pre-research, KAP disclosures, favorites, portfolio state, alerts, investor profile settings, and persisted analysis history.

## Run Locally

```bash
python -m api.main
```

Interactive docs are available at:

- `/docs` for Swagger UI
- `/redoc` for ReDoc
- `/openapi.json` for the OpenAPI schema

## Endpoint Groups

### Finance

- `GET /finance/search` searches symbols, ETFs, indices, or crypto assets.
- `GET /finance/market/summary` returns a regional market snapshot.
- `POST /finance/screener` runs an equity or fund screener query.
- `GET /finance/{symbol}/...` exposes company, index, overview, price, dividends, analyst, earnings, financials, ownership, sustainability, news, and raw-data views.

### Asynchronous Research Workflows

- `POST /analyze/stock` starts a single-symbol analysis job.
- `POST /analyze/batch` starts a multi-symbol analysis job.
- `GET /analyze/status/{task_id}` polls analysis job state.
- `POST /preresearch/` starts exchange-level pre-research.
- `GET /preresearch/{task_id}` polls pre-research state.
- `POST /compare/` runs a direct company comparison.

### Persistent User State

- `GET|POST|DELETE /favorites/...` manages the saved watchlist.
- `GET|POST|DELETE /portfolio/positions...` manages portfolio positions.
- `GET /portfolio/snapshot` returns valuation and P/L.
- `GET /portfolio/risk` returns the portfolio risk cockpit snapshot.
- `GET /alerts/` lists saved price alerts.
- `POST /alerts/price` creates a price alert.
- `GET /alerts/center` evaluates price, risk, and KAP alerts together.
- `GET|PUT /profile/` reads or updates the investor profile.
- `GET /profile/playbooks` lists available investor playbooks.
- `GET /history/` lists persisted analysis history.
- `GET /history/{analysis_id}` returns one history record.

### KAP

- `POST /kap/disclosures/list` searches KAP disclosures.
- `GET /kap/disclosures/{disclosure_index}/detail` returns a parsed disclosure page.
- `POST /kap/disclosures/batch-details` fetches multiple disclosure details in parallel.
- `GET /kap/proxies/status` returns proxy pool health.

## Example Requests

```bash
curl "http://127.0.0.1:8001/finance/search?query=AAPL"

curl -X POST "http://127.0.0.1:8001/portfolio/positions" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AKBNK","quantity":10,"average_cost":42.5}'

curl -X POST "http://127.0.0.1:8001/analyze/stock" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"THYAO.IS","investment_horizon":"long-term"}'
```

## Notes

- API access is filtered by `api.allowed_networks` in the application configuration.
- Long-running analysis and pre-research endpoints return `202 Accepted`; use the returned `task_id` to poll for completion.
- Favorites, portfolio positions, alerts, profile data, and history records are persisted under the runtime `instance/` directory.