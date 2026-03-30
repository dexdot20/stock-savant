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

## About

`struct-savant` is an opinionated finance tooling framework:

- multi-source market data collection (yfinance adapter)
- integrated AI-driven analysis and preset comparison flows
- personal portfolio tracking and Kap membership insight integration
- optional web API + CLI workflow modes

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-improvement`.
3. Add tests under `tests/` and run `pytest`.
4. Open a pull request with a concise description and rationale.

## License

This project does not include a license file yet; add one (MIT, Apache-2.0, etc.) before downstream usage.