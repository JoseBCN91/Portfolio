"""Data loading and preprocessing for superconducting materials analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
FORMULA_FILE = DATA_DIR / "unique_m.csv"
CRITICAL_TEMP_THRESHOLD = 77.0


def load_raw_data(train_path: Path = TRAIN_FILE, formula_path: Path = FORMULA_FILE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    formula_df = pd.read_csv(formula_path)
    return train_df, formula_df


def preprocess_data(train_df: pd.DataFrame, formula_df: pd.DataFrame) -> pd.DataFrame:
    props = train_df[
        [
            "critical_temp",
            "mean_atomic_mass",
            "mean_ThermalConductivity",
            "mean_ElectronAffinity",
            "mean_Density",
            "mean_FusionHeat",
            "mean_Valence",
        ]
    ].copy()

    elems = formula_df.copy()
    cols = [c for c in elems.columns if c not in ("critical_temp", "material")]
    for col in cols:
        elems[col] = elems[col].astype(float)
    elems = elems.replace(0.0, np.nan)
    elems["Number of elements"] = elems.count(axis=1) - 2
    elems = elems.fillna(0.0)
    elems = elems.drop(columns=["critical_temp"])

    merged = props.merge(elems, left_index=True, right_index=True, how="inner")
    merged["critical_temp"] = merged["critical_temp"].astype(float)
    return merged


def split_temperature_buckets(df: pd.DataFrame, threshold: float = CRITICAL_TEMP_THRESHOLD):
    lts = df[df["critical_temp"] <= threshold]
    hts = df[df["critical_temp"] > threshold]
    return lts, hts


def select_families(df: pd.DataFrame, threshold: float = CRITICAL_TEMP_THRESHOLD):
    ybco = df[
        (df.get("Cu", 0.0) > 0)
        & (df.get("Y", 0.0) > 0)
        & (df.get("Ba", 0.0) > 0)
        & (df.get("O", 0.0) > 0)
        & (df["critical_temp"] > threshold)
    ]
    if "material" in ybco.columns:
        ybco = ybco[ybco["material"].str.startswith("Y", na=False)]

    iron = df[(df.get("Fe", 0.0) > 0) & ((df.get("As", 0.0) > 0) | (df.get("P", 0.0) > 0))]
    return ybco, iron
