# 📊 Seasonality Insights Pro

**Advanced Market Seasonality Analysis Platform**

A professional-grade Streamlit dashboard for analyzing monthly seasonality patterns, momentum trends, win rates, statistical distributions, correlations, and comparative performance across financial markets. Built with modern Python practices and optimized for speed with intelligent caching.

**Live Demo:** https://marketseasonalityanalytics.streamlit.app

---

## 🎯 Key Features

### Single-Asset Analysis
- **Monthly Pattern Recognition**: Comprehensive weekly progression analysis with historical win rates
- **Momentum Tracking**: 3-day rolling momentum indicators with daily granularity
- **Win Rate Analytics**: Day-by-day success metrics across all trading days in each month
- **Monthly Summaries**: Average returns, median performance, volatility, and positive year counts

### Statistical Distribution Analysis
- **Automated Distribution Fitting**: Tests 50+ probability distributions (Normal, Student's t, Skewed Normal, etc.)
- **Model Ranking**: AIC/BIC criteria for best-fit identification
- **Goodness-of-Fit Testing**: Kolmogorov-Smirnov tests with p-values
- **Visual Comparisons**: Distribution overlay charts with theoretical vs. empirical data

### Multi-Asset Comparison
- **Correlation Matrix**: Dynamic cross-asset correlation heatmaps
- **Relative Performance**: Cumulative return tracking across multiple securities
- **Portfolio Metrics**: Annualized returns, volatility, Sharpe ratios, and maximum drawdown
- **Batch Analysis**: Compare up to 10 tickers simultaneously

### Technical Capabilities
- **Fast Data Pipeline**: Cached downloads via yfinance with intelligent TTL management
- **Robust Preprocessing**: Automatic handling of missing data, multi-index columns, and edge cases
- **Responsive UI**: Modern dark theme with Inter font, gradient accents, and professional styling
- **Comprehensive Asset Database**: 100+ pre-configured tickers (stocks, indices, ETFs, crypto, forex)

---

## 📁 Project Structure

```
Finance_stationality/
│
├── app.py                          # Main Streamlit application entry point
├── pyproject.toml                  # Project metadata and dependencies (uv-compatible)
├── requirements.txt                # Legacy dependency file (deprecated)
├── README.md                       # This file
│
├── components/                     # Reusable UI components
│   ├── __init__.py
│   ├── metrics.py                  # Metric cards and visual elements
│   └── sidebar.py                  # Sidebar controls and ticker selection
│
├── helpers/                        # Core business logic and utilities
│   ├── __init__.py
│   ├── analyzer.py                 # Monthly pattern analysis engine
│   ├── app_helpers.py              # Facade for all helper functions
│   ├── chart_builder.py            # Plotly chart generation (heatmaps, distributions)
│   ├── chart_style.py              # Chart theming and color schemes
│   ├── config.py                   # Application constants and ticker database
│   ├── data_preprocessing.py       # Data cleaning and feature engineering
│   ├── data_validation.py          # Input validation and sanity checks
│   ├── stats.py                    # Statistical analysis and distribution fitting
│   └── ui_style.py                 # CSS styling and page configuration
│
├── services/                       # External integrations and orchestration
│   ├── __init__.py
│   ├── data_provider_interface.py  # Abstract data provider contract
│   ├── data_provider.py            # YFinance implementation with caching
│   └── multi_ticker_analyzer.py    # Cross-asset analysis and correlation engine
│
└── tests/                          # Unit and integration tests
    ├── test_analyzer.py            # Tests for pattern analysis
    └── test_helpers.py             # Tests for helper functions
```

### Architecture Overview

**Modular Design Philosophy:**
- **Components**: Stateless UI rendering (sidebar, metrics, cards)
- **Helpers**: Pure functions for calculations, validation, and transformations
- **Services**: Stateful external integrations (data fetching, caching)
- **App**: Orchestration layer that ties everything together

**Key Design Patterns:**
- **Facade Pattern**: `app_helpers.py` provides a single import point for all helper functions
- **Repository Pattern**: `data_provider.py` abstracts data source (easy to swap yfinance for alternatives)
- **Caching Strategy**: Streamlit `@st.cache_data` with 1-hour TTL for optimal performance
- **Separation of Concerns**: UI, business logic, and data access are strictly separated

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or 3.13
- Windows, macOS, or Linux

### Installation (Using `uv` - Recommended)

[`uv`](https://github.com/astral-sh/uv) is a fast, modern Python package manager that's 10-100x faster than pip.

#### 1. Install `uv`

**Windows (PowerShell):**
```powershell
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
```

**Windows (Winget):**
```powershell
winget install Astral.UV
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create Virtual Environment & Install Dependencies

```powershell
uv sync
```

This command:
- Reads `pyproject.toml`
- Creates a `.venv/` directory
- Installs all dependencies with locked versions

#### 3. Activate Virtual Environment (Optional)

**Windows:**
```powershell
.\.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

> **Note**: You can skip activation and use `uv run` to execute commands directly.

---

## ▶️ Running the Application

### Option 1: Using `uv run` (No activation needed)

```powershell
uv run streamlit run app.py
```

### Option 2: After activating virtual environment

```powershell
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Testing

### Run All Tests

```powershell
uv run -m pytest -q
```

### Run with Verbose Output

```powershell
uv run -m pytest -v
```

### Run Specific Test File

```powershell
uv run -m pytest tests/test_analyzer.py
```

---

## 📊 Usage Guide

### Analysis Modes

1. **Single Ticker Mode**
   - Select an asset from the dropdown or enter a custom ticker
   - View monthly breakdown with weekly returns, momentum, and win rates
   - Analyze return distributions with statistical model fitting

2. **Multi-Ticker Mode**
   - Select 2-10 tickers for comparison
   - View correlation heatmap to identify relationships
   - Compare relative cumulative performance
   - Review portfolio-level statistics (Sharpe, volatility, drawdown)

### Key Tabs

- **Overview**: High-level summary of all 12 months
- **Month Deep Dive**: Detailed analysis for a specific month
- **Distribution Analysis**: Statistical distribution fitting and visualization
- **Multi-Ticker Comparison**: Cross-asset correlation and performance metrics

---

## 🛠️ Configuration

### Adding Custom Tickers

Edit [`helpers/config.py`](helpers/config.py) to add tickers to the database:

```python
TICKER_DATABASE = {
    "Custom Asset": "TICKER",
    # ... more tickers
}
```

### Adjusting Cache Duration

Modify TTL in [`services/data_provider.py`](services/data_provider.py):

```python
CACHE_TTL_SECONDS = 3600  # Default: 1 hour
```

### Customizing UI Theme

Update CSS variables in [`helpers/ui_style.py`](helpers/ui_style.py):

```python
:root {
    --primary: #4ECDC4;  # Accent color
    --positive: #26de81; # Positive returns
    --negative: #fc5c65; # Negative returns
}
```

---

## 📦 Dependencies

**Core Libraries:**
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `yfinance`: Yahoo Finance data downloader
- `plotly`: Interactive charting
- `scipy`: Statistical distributions and tests

**Development:**
- `pytest`: Testing framework
- `ruff`: Fast Python linter (optional)

See [`pyproject.toml`](pyproject.toml) for complete dependency list with version constraints.

---

## 💡 Technical Notes

### Data Limitations
- Some assets (forex, crypto, indices) may not provide volume data
- Analysis requires only `Close` prices; volume is optional
- Historical data availability varies by ticker (yfinance limitation)

### Performance Optimization
- All data fetches are cached for 1 hour (`@st.cache_data`)
- Analysis results are memoized per session
- Multi-ticker downloads are batched for efficiency

### Statistical Methods
- Distribution fitting uses Maximum Likelihood Estimation (MLE)
- Goodness-of-fit assessed via Kolmogorov-Smirnov test
- Model selection via AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run tests: `uv run -m pytest`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

**Code Style:**
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions (Google style)
- Keep functions focused and testable

---

## 📝 License

This project is private and proprietary. All rights reserved.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data via `yfinance`
- **Streamlit** for the excellent web framework
- **Astral** for the blazing-fast `uv` package manager
- **Plotly** for interactive visualization capabilities

---

## 📞 Support

For issues, questions, or feature requests, please open an issue in the repository.

---

**Built with ❤️ for financial analysis enthusiasts**
