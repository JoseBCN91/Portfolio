# Finance Stationality

Interactive Streamlit dashboard for analyzing monthly seasonality, momentum patterns, win rates, correlations, and performance across single or multiple tickers.

## Features
- Single-ticker monthly analysis: weekly progression, 3-day momentum, daily win rates, and monthly summaries.
- Return distribution with fitted probability models (AIC/BIC ranking) and KS goodness-of-fit.
- Multi-ticker comparison: correlation matrix, relative cumulative performance, and stats (annualized return, volatility, Sharpe, max drawdown).
- Fast cached downloads via yfinance with robust preprocessing.

## Project Structure
- `original.py`: Streamlit app entrypoint and UI orchestration.
- `helpers/`: Stateless utilities (validation, preprocessing, stats, charts, style, config) and the facade `app_helpers.py`.
- `services/`: Stateful integrations and orchestration (yfinance provider, multi-ticker analyzer).
- `components/`: UI pieces (sidebar, metrics).
- `tests/`: Unit tests.

## Setup (uv)

We use `uv` for fast, reproducible Python package management.

### 1) Install uv (Windows)

PowerShell (recommended):

```pwsh
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
```

Winget (alternative):

```pwsh
winget install Astral.UV
```

### 2) Create and sync environment

```pwsh
uv sync
```

This reads `pyproject.toml`, creates `.venv/`, and installs dependencies.

### 3) Activate (optional)

```pwsh
.\.venv\Scripts\activate
```

You can also run commands without activating by using `uv run`.

## Run

Start the app with uv-managed environment:

```pwsh
uv run streamlit run original.py
```

Or, if activated:

```pwsh
streamlit run original.py
```

## Tests

```pwsh
uv run -m pytest -q
```

## Notes
- Some tickers (FX/crypto/indices) may not provide `Volume`; the app only requires `Close` for analyses.
- Fitted distributions use SciPy; ensure `scipy` is listed in `pyproject.toml` (it is).
