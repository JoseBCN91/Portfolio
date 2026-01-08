import logging
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess downloaded data: flatten columns, validate, add derived fields."""
    # Handle MultiIndex columns (from multi-ticker downloads)
    if isinstance(data.columns, pd.MultiIndex):
        logger.debug("Detected MultiIndex columns, flattening")
        data.columns = data.columns.droplevel(1)

    # Sort and ensure clean index
    data = data.sort_index()

    # Calculate technical indicators
    data['Daily_Return'] = data['Close'].pct_change()
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

    # Remove NaN from pct_change
    data['Daily_Return'] = data['Daily_Return'].fillna(0)

    logger.info(f"Data preprocessed: {len(data)} rows, date range {data.index.min()} to {data.index.max()}")
    return data
