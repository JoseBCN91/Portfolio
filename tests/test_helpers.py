import pandas as pd
import numpy as np

from helpers.data_validation import validate_ticker, validate_date_range, validate_dataframe
from helpers.data_preprocessing import preprocess_data
from helpers.chart_style import hex_to_rgba
from helpers.stats import get_goodness_of_fit
from helpers.app_helpers import download_data

class FakeProvider:
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        rng = pd.date_range(start="2024-01-01", periods=30, freq="B")
        prices = 100 + np.cumsum(np.random.randn(len(rng)) * 0.5)
        df = pd.DataFrame({"Close": prices}, index=rng)
        # Include Volume optionally
        df["Volume"] = 1000
        return df


def test_validate_ticker_basic():
    assert validate_ticker("AAPL")
    assert validate_ticker("^GSPC")
    assert validate_ticker("BTC-USD")
    assert not validate_ticker(123)  # non-string


def test_validate_date_range():
    assert validate_date_range("2020-01-01", "2020-02-01")
    assert not validate_date_range("2020-02-01", "2020-01-01")


def test_validate_and_preprocess():
    rng = pd.date_range(start="2024-01-01", periods=25, freq="B")
    prices = 100 + np.cumsum(np.random.randn(len(rng)))
    df = pd.DataFrame({"Close": prices}, index=rng)
    ok, err = validate_dataframe(df)
    assert ok and err is None
    out = preprocess_data(df.copy())
    assert "Daily_Return" in out.columns
    assert "Year" in out.columns


def test_chart_style_hex_to_rgba():
    rgba = hex_to_rgba("#2E86AB", 0.3)
    assert rgba.startswith("rgba(") and rgba.endswith(")")


def test_download_data_with_fake_provider():
    data, err = download_data("FAKE", "2024-01-01", "2024-02-01", _provider=FakeProvider())
    assert err is None and data is not None
    assert "Daily_Return" in data.columns


def test_gof_on_synthetic_normal():
    # Generate synthetic normal data
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(loc=0, scale=0.01, size=500))
    dist_info = {"best_dist": "norm", "fitted_loc": float(returns.mean()), "fitted_scale": float(returns.std(ddof=1)), "fitted_shape": {}}
    gof = get_goodness_of_fit(returns, dist_info)
    assert gof["ks_stat"] is not None
    assert gof["ks_pvalue"] is not None
