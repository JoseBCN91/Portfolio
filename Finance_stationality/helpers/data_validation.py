import logging
import re
from typing import Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# Validation patterns
TICKER_PATTERN = re.compile(r'^[\^A-Z0-9_\-\.=/\$]{1,20}$')
MIN_DATA_POINTS = 20


def validate_ticker(ticker: str) -> bool:
    """Validate ticker format."""
    if not isinstance(ticker, str):
        return False
    return bool(TICKER_PATTERN.match(ticker.upper()))


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range logic."""
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
        return pd.to_datetime(start_date) < pd.to_datetime(end_date)
    except Exception as e:
        logger.warning(f"Invalid date range: {start_date} to {end_date}: {e}")
        return False


def validate_dataframe(data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Validate DataFrame structure and content."""
    if data is None:
        return False, "DataFrame is None"

    if data.empty:
        return False, "DataFrame is empty"

    if len(data) < MIN_DATA_POINTS:
        return False, f"Insufficient data: {len(data)} rows (min {MIN_DATA_POINTS})"

    # Only 'Close' is required for analyses
    required_cols = ['Close']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    nan_ratio = data[['Close']].isna().sum().sum() / len(data)
    if nan_ratio > 0.1:
        return False, f"Excessive NaN values in Close: {nan_ratio*100:.1f}%"

    return True, None
