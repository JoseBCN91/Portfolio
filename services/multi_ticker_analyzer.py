"""Multi-ticker analysis: correlation, relative performance, beta."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 30  # 30-day rolling correlation


def compute_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix of returns across tickers.
    
    Args:
        data: DataFrame with columns like 'AAPL_Return', 'MSFT_Return', etc.
        
    Returns:
        Correlation matrix (symmetric, values -1 to 1)
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to compute_correlation_matrix")
        return pd.DataFrame()
    
    # Extract return columns
    return_cols = [col for col in data.columns if '_Return' in col]
    if not return_cols:
        logger.warning("No return columns found in data")
        return pd.DataFrame()
    
    corr = data[return_cols].corr()
    
    # Rename columns to remove '_Return' suffix for readability
    corr.columns = [col.replace('_Return', '') for col in corr.columns]
    corr.index = [idx.replace('_Return', '') for idx in corr.index]
    
    logger.info(f"Computed correlation for {len(corr)} tickers")
    return corr


def calculate_relative_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative returns (normalized) for each ticker.
    
    Args:
        data: DataFrame with columns like 'AAPL_Return', 'MSFT_Return', etc.
        
    Returns:
        DataFrame with cumulative returns (%) for each ticker
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to calculate_relative_performance")
        return pd.DataFrame()
    
    # Extract return columns
    return_cols = [col for col in data.columns if '_Return' in col]
    if not return_cols:
        logger.warning("No return columns found in data")
        return pd.DataFrame()
    
    # Calculate cumulative returns
    cumulative = (1 + data[return_cols]).cumprod() - 1
    
    # Rename columns
    cumulative.columns = [col.replace('_Return', '') for col in cumulative.columns]
    
    logger.info(f"Calculated relative performance for {len(cumulative.columns)} tickers")
    return cumulative * 100  # Convert to percentage


def rolling_correlation(data: pd.DataFrame, ticker1: str, ticker2: str, window: int = ROLLING_WINDOW) -> pd.Series:
    """Calculate rolling correlation between two tickers.
    
    Args:
        data: DataFrame with return columns
        ticker1: First ticker symbol
        ticker2: Second ticker symbol
        window: Rolling window size (default 30 days)
        
    Returns:
        Series of rolling correlation values
    """
    col1 = f'{ticker1}_Return'
    col2 = f'{ticker2}_Return'
    
    if col1 not in data.columns or col2 not in data.columns:
        logger.warning(f"Columns {col1} or {col2} not found in data")
        return pd.Series()
    
    rolling_corr = data[col1].rolling(window=window).corr(data[col2])
    logger.info(f"Computed {window}-day rolling correlation between {ticker1} and {ticker2}")
    return rolling_corr


def calculate_statistics(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate performance statistics for each ticker.
    
    Args:
        data: DataFrame with columns like 'AAPL_Return', 'MSFT_Return', etc.
        
    Returns:
        Dict mapping ticker -> {annualized_return, volatility, sharpe, max_dd}
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to calculate_statistics")
        return {}
    
    return_cols = [col for col in data.columns if '_Return' in col]
    if not return_cols:
        logger.warning("No return columns found in data")
        return {}
    
    stats = {}
    for col in return_cols:
        ticker = col.replace('_Return', '')
        returns = data[col].dropna()
        
        if len(returns) < 2:
            logger.warning(f"Insufficient data for {ticker}")
            continue
        
        # Annualized return (252 trading days)
        annual_return = returns.mean() * 252 * 100
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (annual_return / volatility) if volatility > 0 else 0
        
        # Max drawdown
        cumsum = (1 + returns).cumprod()
        running_max = cumsum.expanding().max()
        drawdown = (cumsum - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        stats[ticker] = {
            'annualized_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }
    
    logger.info(f"Calculated statistics for {len(stats)} tickers")
    return stats
