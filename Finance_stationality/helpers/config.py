"""Configuration and constants for the Finance Stationality app."""

# ============================================================================
# Asset Configuration - Comprehensive Ticker Database
# ============================================================================
TICKER_DATABASE = {
    # Indices
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "Dow Jones Industrial": "^DJI",
    "Russell 2000": "^RUT",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai Composite": "000001.SS",
    
    # Mega-cap Tech
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google/Alphabet": "GOOGL",
    "Meta (Facebook)": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Intel": "INTC",
    
    # Other Large Caps
    "Berkshire Hathaway B": "BRK.B",
    "Johnson & Johnson": "JNJ",
    "Visa": "V",
    "Mastercard": "MA",
    "Coca-Cola": "KO",
    "Nike": "NKE",
    
    # Cryptocurrencies
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Cardano": "ADA-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    
    # Commodities & Futures
    "Gold": "GC=F",
    "Crude Oil (WTI)": "CL=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Silver": "SI=F",
    
    # Foreign Exchange
    "USD/EUR": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPYUSD=X",
}

# Popular assets (for quick selection)
POPULAR_ASSETS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "Bitcoin": "BTC-USD",
    "Gold": "GC=F",
}

# Default tickers for multi-ticker mode
DEFAULT_MULTI_TICKERS = ["AAPL", "MSFT", "GOOGL"]

# ============================================================================
# Date Range Presets
# ============================================================================
PRESET_DATE_RANGES = {
    "All Available Data": ("1950-01-01", "2024-12-31"),
    "Last 10 Years": ("2014-01-01", "2024-12-31"),
    "Last 5 Years": ("2019-01-01", "2024-12-31"),
    "Last 3 Years": ("2021-01-01", "2024-12-31"),
    "COVID Era": ("2020-01-01", "2024-12-31"),
    "Post-2008 Crisis": ("2010-01-01", "2024-12-31"),
    "Custom Range": None
}

# Default date range option
DEFAULT_DATE_RANGE = "Last 5 Years"

# ============================================================================
# Analysis Thresholds
# ============================================================================
WIN_RATE_THRESHOLDS = {
    'excellent': 55,  # >= 55%
    'good': 50,       # 50-55%
    'poor': 0         # < 50%
}

# Win rate color mapping
WIN_RATE_COLORS = {
    'excellent': '#06D6A0',  # Green
    'good': '#2E86AB',       # Blue
    'poor': '#A23B72'        # Purple/Red
}

# ============================================================================
# Month Configuration
# ============================================================================
TRADING_MONTHS = list(range(1, 13))

# ============================================================================
# UI Styling Constants
# ============================================================================
# Font sizes
FONT_SIZE_TITLE = 16
FONT_SIZE_SUBTITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_METRIC = 12

# Chart styling
CHART_HEIGHT = 450
SIDEBAR_HEADER_FONTSIZE = 14

# Colors (dark theme)
COLOR_PRIMARY = '#2E86AB'
COLOR_SECONDARY = '#A23B72'
COLOR_POSITIVE = '#06D6A0'
COLOR_NEGATIVE = '#F18F01'
COLOR_TEXT = '#FFFFFF'
COLOR_TEXT_SECONDARY = '#B8BCC8'
COLOR_GRID = 'rgba(255,255,255,0.1)'
COLOR_BORDER = 'rgba(255,255,255,0.2)'

# ============================================================================
# Formatting
# ============================================================================
# Date format for display
DATE_FORMAT = "%Y-%m-%d"

# Number formatting
PERCENTAGE_DECIMALS = 2
MOMENTUM_DECIMALS = 3
WIN_RATE_DECIMALS = 1

# ============================================================================
# Analysis Configuration
# ============================================================================
# First/second half split
MONTH_SPLIT_RATIO = 0.5

# ============================================================================
# Default Settings
# ============================================================================
DEFAULT_TICKER = "^GSPC"
DEFAULT_SHOW_TRENDS = True
DEFAULT_SHOW_STATS = True
DEFAULT_ASSET_METHOD = "Popular Assets"
