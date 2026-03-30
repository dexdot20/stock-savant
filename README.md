# Struct

Struct is a CLI and FastAPI toolset for stock analysis, financial data access, company comparison, and pre-research workflows.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

- CLI: `python cli.py`
- API: `python -m api.main`
- API documentation: `/docs`

## Configuration

- Copy .env.example to `.env` and fill in your secrets and environment variables.
- Put one real `User-Agent` string per line in `user-agents.txt`.
- Local runtime data and cache output are stored under `instance/`.

## Notes

- See api/API_GUIDE.md for API endpoints and example requests.
- The project is designed for both interactive use and automation.