from typing import Protocol
import pandas as pd

class DataProvider(Protocol):
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
