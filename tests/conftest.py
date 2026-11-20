"""Общие настройки тестов: индекс цен (инфляция) никогда не скачивается из интернета.

Ряд synthetic: цены растут ровно на 2.5% в год, месячные значения с 2000 года.
"""
import numpy as np
import pandas as pd
import pytest

import data

CPI_YEARLY = 0.025


def synthetic_cpi() -> pd.DataFrame:
    idx = pd.date_range("2000-01-01", "2026-10-01", freq="MS")
    years = np.array([(d - idx[0]).days / 365.25 for d in idx])
    return pd.DataFrame({data.CPI_SERIES: 100 * (1 + CPI_YEARLY) ** years}, index=idx)


@pytest.fixture(autouse=True)
def no_cpi_download(monkeypatch):
    monkeypatch.setattr(data, "_download_cpi", synthetic_cpi)
