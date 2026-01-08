import pandas as pd
import numpy as np
from helpers.app_helpers import analyze_month_patterns


def make_sample_month(year=2020, month=1, days=20):
    # Create business-day-like index within the same month
    rng = pd.date_range(start=f"{year}-{month:02d}-01", periods=days, freq='B')
    prices = 100 + np.cumsum(np.random.randn(len(rng)) * 0.5)
    df = pd.DataFrame({'Close': prices}, index=rng)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    return df


def test_analyze_month_patterns_basic():
    df = make_sample_month(year=2021, month=1, days=22)
    weekly_avg_returns, momentum_avg, win_rates, years = analyze_month_patterns(1, df)

    # Basic sanity checks
    assert isinstance(weekly_avg_returns, dict)
    assert isinstance(momentum_avg, dict)
    assert isinstance(win_rates, dict)
    assert isinstance(years, list)

    # If enough days provided, we should have at least one year processed
    assert (len(years) == 1) or (len(years) == 0 and len(df) < 10)
