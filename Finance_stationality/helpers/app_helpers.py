"""Facade module exposing helper functions and constants used by original.py.
This aggregates imports to reduce coupling and centralize access.
"""

# UI styling
from helpers.ui_style import apply_style

# Data access
from services.data_provider import download_data, download_multiple_tickers

# Single-ticker analysis
from helpers.analyzer import (
    analyze_month_patterns,
    get_month_summary,
    compute_return_distribution,
    fit_distribution_to_returns,
)

# Charting
from helpers.chart_builder import (
    create_monthly_chart,
    create_correlation_heatmap,
    create_relative_performance_chart,
    create_return_distribution_chart,
    get_ticker_display_name,
)

# Multi-ticker analysis
from services.multi_ticker_analyzer import (
    compute_correlation_matrix,
    calculate_relative_performance,
    calculate_statistics,
)

# Sidebar
from components.sidebar import render_sidebar

# Config constants
from helpers.config import (
    TRADING_MONTHS,
    DEFAULT_TICKER,
    POPULAR_ASSETS,
    WIN_RATE_THRESHOLDS,
    PERCENTAGE_DECIMALS,
    WIN_RATE_DECIMALS,
    MONTH_SPLIT_RATIO,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_TEXT_SECONDARY,
    FONT_SIZE_TITLE,
    CHART_HEIGHT,
)

# Metric components
from components.metrics import (
    create_metric_card,
    get_win_rate_color,
    get_win_rate_status,
    get_return_color,
    create_insight_box,
    create_metric_cards_row,
)

# Stats helpers
from helpers.stats import (
    get_distribution_stats,
    get_goodness_of_fit,
)

__all__ = [
    # ui
    "apply_style",
    # data
    "download_data",
    "download_multiple_tickers",
    # analyzer
    "analyze_month_patterns",
    "get_month_summary",
    "compute_return_distribution",
    "fit_distribution_to_returns",
    # charts
    "create_monthly_chart",
    "create_correlation_heatmap",
    "create_relative_performance_chart",
    "create_return_distribution_chart",
    "get_ticker_display_name",
    # multi-ticker
    "compute_correlation_matrix",
    "calculate_relative_performance",
    "calculate_statistics",
    # sidebar
    "render_sidebar",
    # config
    "TRADING_MONTHS",
    "DEFAULT_TICKER",
    "POPULAR_ASSETS",
    "WIN_RATE_THRESHOLDS",
    "PERCENTAGE_DECIMALS",
    "WIN_RATE_DECIMALS",
    "MONTH_SPLIT_RATIO",
    "COLOR_POSITIVE",
    "COLOR_NEGATIVE",
    "COLOR_TEXT_SECONDARY",
    "FONT_SIZE_TITLE",
    "CHART_HEIGHT",
    # metrics
    "create_metric_card",
    "get_win_rate_color",
    "get_win_rate_status",
    "get_return_color",
    "create_insight_box",
    "create_metric_cards_row",
    # stats helpers
    "get_distribution_stats",
    "get_goodness_of_fit",
]