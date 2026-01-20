import logging
from typing import Optional, Tuple
import streamlit as st
import yfinance as yf
import pandas as pd
from services.data_provider_interface import DataProvider

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 3600  # 1 hour

# Helpers: validation and preprocessing
from helpers.data_validation import (
    validate_ticker as _validate_ticker,
    validate_date_range as _validate_date_range,
    validate_dataframe as _validate_dataframe,
)
from helpers.data_preprocessing import preprocess_data as _preprocess_data_helper


class DataProviderError(Exception):
    """Custom exception for data provider issues."""
    pass




class YFinanceProvider:
    """Yahoo Finance data provider."""
    
    @staticmethod
    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataProviderError: If fetch fails
        """
        if not _validate_ticker(ticker):
            raise DataProviderError(f"Invalid ticker format: {ticker}")
        
        if not _validate_date_range(start, end):
            raise DataProviderError(f"Invalid date range: {start} to {end}")
        
        try:
            logger.info(f"Fetching {ticker} from {start} to {end}")
            data = yf.download(ticker, start=start, end=end, progress=False)
            
            if data is None or data.empty:
                raise DataProviderError(f"No data returned for {ticker}")
            
            return data
        except Exception as e:
            logger.error(f"yfinance fetch failed: {e}")
            raise DataProviderError(f"Failed to fetch {ticker}: {str(e)}")


def _preprocess_and_validate(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and preprocess downloaded data using helpers."""
    # Handle MultiIndex columns (from multi-ticker downloads)
    if isinstance(data.columns, pd.MultiIndex):
        logger.debug("Detected MultiIndex columns, flattening")
        data.columns = data.columns.droplevel(1)

    # Validate structure
    is_valid, error_msg = _validate_dataframe(data)
    if not is_valid:
        raise DataProviderError(f"Data validation failed: {error_msg}")

    # Preprocess via helper
    return _preprocess_data_helper(data)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def download_data(
    ticker: str,
    start_date: str,
    end_date: str,
    _provider: DataProvider | None = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Download and preprocess data using a provider.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Tuple of (DataFrame, error_message).
        If successful: (DataFrame, None)
        If failed: (None, error_message)
    """
    try:
        # Validate inputs
        ticker = ticker.upper().strip()
        if not ticker:
            return None, "Ticker symbol cannot be empty"
        
        # Fetch data
        provider = _provider or YFinanceProvider()
        raw_data = provider.fetch(ticker, start_date, end_date)
        
        # Preprocess
        data = _preprocess_and_validate(raw_data)
        
        logger.info(f"Successfully downloaded and processed {ticker}")
        return data, None
        
    except DataProviderError as e:
        error_msg = str(e)
        logger.error(f"Data provider error: {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def download_multiple_tickers(
    tickers: list,
    start_date: str,
    end_date: str,
    _provider: DataProvider | None = None,
) -> Tuple[Optional[dict], Optional[str]]:
    """Download and align data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Tuple of (dict of DataFrames, error_message).
        dict structure: {ticker_str: DataFrame with aligned Close, Daily_Return}
        If any fetch fails, returns (None, error_message)
    """
    if not tickers or not isinstance(tickers, list):
        return None, "Tickers must be a non-empty list"
    
    ticker_data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            data, error = download_data(ticker, start_date, end_date, _provider=_provider)
            if error or data is None:
                failed_tickers.append(f"{ticker}: {error}")
                logger.warning(f"Failed to fetch {ticker}: {error}")
                continue
            
            # Keep only Date, Close, Daily_Return for correlation/comparison
            ticker_data[ticker] = data[['Close', 'Daily_Return']].copy()
            logger.info(f"Fetched {ticker}: {len(data)} rows")
        
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            failed_tickers.append(f"{ticker}: {str(e)}")
    
    if not ticker_data:
        error_msg = f"Failed to fetch any tickers: {', '.join(failed_tickers)}"
        return None, error_msg
    
    # Align data on common dates (inner join on index)
    try:
        aligned_data = {}
        # Join all dataframes on index (date)
        all_data = None
        for ticker, data in ticker_data.items():
            data = data.rename(columns={'Close': f'{ticker}_Close', 'Daily_Return': f'{ticker}_Return'})
            if all_data is None:
                all_data = data
            else:
                all_data = all_data.join(data, how='inner')
        
        if all_data is None or all_data.empty:
            return None, "No common trading dates across tickers"
        
        logger.info(f"Aligned data: {len(all_data)} common trading days across {len(tickers)} tickers")
        return all_data, None
    
    except Exception as e:
        error_msg = f"Error aligning ticker data: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
