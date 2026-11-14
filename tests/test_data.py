"""Тесты для data.py: офлайн-режим и разбор ошибок. Интернет НЕ нужен: загрузка подменяется.

Все ряды здесь synthetic (искусственные). Проверяем поведение загрузчика, а не формулы.
"""
from datetime import date

import pandas as pd
import pytest
import requests

import data

TODAY = date.today().isoformat()
START, END = date(2024, 1, 1), date(2024, 12, 31)


def synthetic_prices(ticker, days=60, first="2024-01-02"):
    # synthetic: цены растут на 1 в день, только рабочие дни
    index = pd.bdate_range(first, periods=days)
    return pd.DataFrame({ticker: [100.0 + i for i in range(days)]}, index=index)


def synthetic_rate(days=60):
    # synthetic: ставка 4.5% годовых каждый день
    index = pd.bdate_range("2024-01-02", periods=days)
    return pd.DataFrame({data.RF_SERIES: [4.5] * days}, index=index)


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    # Подменяем папку data/raw на временную, чтобы тесты не трогали настоящие файлы
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    return tmp_path


def fail_with(error):
    def _raise(*args, **kwargs):
        raise error
    return _raise


# ---------- Свежая загрузка и файлы ----------

def test_fresh_download_is_saved_with_pull_date_in_name(raw_dir):
    df, info = data._load_with_fallback("prices_BRK-B", "Prices of BRK-B",
                                        lambda: synthetic_prices("BRK-B"), START, END)
    assert (raw_dir / f"prices_BRK-B_{TODAY}.csv").exists()  # тикер с дефисом работает
    assert info["offline"] is False
    assert info["pull_date"] == TODAY
    assert len(df) == 60


def test_second_run_on_same_day_overwrites_file(raw_dir):
    for _ in range(2):
        data._load_with_fallback("prices_SPY", "Prices of SPY",
                                 lambda: synthetic_prices("SPY"), START, END)
    assert len(list(raw_dir.glob("prices_SPY_*.csv"))) == 1
    assert not list(raw_dir.glob(".*.tmp"))  # временных файлов не осталось


def test_disk_write_error_is_not_treated_as_network_error(raw_dir, monkeypatch):
    monkeypatch.setattr(data, "_save_raw", fail_with(PermissionError("read-only disk")))
    df, info = data._load_with_fallback("prices_SPY", "Prices of SPY",
                                        lambda: synthetic_prices("SPY"), START, END)
    assert info["offline"] is False          # данные свежие
    assert "could not be saved" in info["save_warning"]
    assert len(df) == 60


# ---------- Три причины, по которым скачать не удалось ----------

def saved_file(raw_dir, name, df, day="2026-09-01"):
    df.to_csv(raw_dir / f"{name}_{day}.csv", index_label="date")


def test_no_network_uses_saved_file_and_says_so(raw_dir):
    saved_file(raw_dir, "prices_SPY", synthetic_prices("SPY"))
    df, info = data._load_with_fallback("prices_SPY", "Prices of SPY",
                                        fail_with(data.NetworkError("x")), START, END)
    assert info["offline"] is True and info["cause"] == "no_network"
    assert info["pull_date"] == "2026-09-01"
    assert "No internet connection" in info["message"] and "2026-09-01" in info["message"]


def test_api_error_message_says_it_is_not_a_network_problem(raw_dir):
    saved_file(raw_dir, "rf_DGS3MO", synthetic_rate())
    df, info = data._load_with_fallback("rf_DGS3MO", "Risk-free rate DGS3MO",
                                        fail_with(data.ApiError("FRED request limit reached (HTTP 429).")),
                                        START, END)
    assert info["cause"] == "api_error"
    assert "limit reached" in info["message"]
    assert "No internet" not in info["message"]


def test_missing_key_message_mentions_env_file(raw_dir):
    saved_file(raw_dir, "rf_DGS3MO", synthetic_rate())
    df, info = data._load_with_fallback("rf_DGS3MO", "Risk-free rate DGS3MO",
                                        fail_with(data.MissingKeyError("x")), START, END)
    assert info["cause"] == "missing_key"
    assert "FRED_API_KEY" in info["message"] and ".env" in info["message"]
    assert "No internet" not in info["message"]


def test_no_saved_file_and_missing_key_gives_instruction(raw_dir):
    with pytest.raises(data.DataError, match="Add your FRED key to .env and run once with internet"):
        data._load_with_fallback("rf_DGS3MO", "Risk-free rate DGS3MO",
                                 fail_with(data.MissingKeyError("x")), START, END)


def test_no_saved_file_and_no_network_gives_instruction(raw_dir):
    with pytest.raises(data.DataError, match="run once so the data can be saved"):
        data._load_with_fallback("prices_SPY", "Prices of SPY",
                                 fail_with(data.NetworkError("x")), START, END)


def test_latest_saved_file_is_chosen_by_date_in_name(raw_dir):
    saved_file(raw_dir, "prices_SPY", synthetic_prices("SPY"), day="2026-08-01")
    saved_file(raw_dir, "prices_SPY", synthetic_prices("SPY"), day="2026-09-15")
    (raw_dir / "prices_SPY_notes.csv").write_text("not a data file")  # лишний файл не мешает
    assert data._find_latest_raw("prices_SPY").name == "prices_SPY_2026-09-15.csv"


# ---------- Ключ FRED не должен утекать ----------

def test_fred_errors_never_contain_the_key(monkeypatch):
    secret = "SECRETKEY123456"
    monkeypatch.setenv("FRED_API_KEY", secret)

    def boom(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError(f"Max retries exceeded with url: ?api_key={secret}")
    monkeypatch.setattr(data.requests, "get", boom)
    with pytest.raises(data.NetworkError) as err:
        data._download_risk_free()
    assert secret not in str(err.value)


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_fred_bad_key_and_rate_limit_are_api_errors(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123456")
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(400))
    with pytest.raises(data.ApiError, match="probably invalid"):
        data._download_risk_free()
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(429))
    with pytest.raises(data.ApiError, match="limit"):
        data._download_risk_free()


def test_fred_missing_or_placeholder_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(data.MissingKeyError):
        data._download_risk_free()
    monkeypatch.setenv("FRED_API_KEY", "your_fred_api_key_here")
    with pytest.raises(data.MissingKeyError):
        data._download_risk_free()


def test_fred_parses_dots_as_missing_values(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123456")
    payload = {"observations": [{"date": "2024-01-01", "value": "."},
                                {"date": "2024-01-02", "value": "5.25"}]}
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(200, payload))
    df = data._download_risk_free()
    assert list(df[data.RF_SERIES]) == [5.25]


def test_fred_empty_answer_is_api_error(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123456")
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(200, {"observations": []}))
    with pytest.raises(data.ApiError):
        data._download_risk_free()


# ---------- Очистка цен и объединение ----------

def test_clean_prices_removes_timezone_duplicates_and_bad_values():
    # synthetic: ответ Yahoo с часовым поясом, дублем последнего дня, NaN и нулевой ценой
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05"],
                             tz="America/New_York")
    raw = pd.DataFrame({"Close": [100.0, 101.0, 102.0, None, 0.0]}, index=index)
    cleaned = data._clean_prices(raw, "BRK-B")
    assert cleaned.index.tz is None
    assert list(cleaned.columns) == ["BRK-B"]
    assert list(cleaned["BRK-B"]) == [100.0, 102.0]  # дубль: оставлен последний; NaN и 0 убраны


def test_get_dataset_counts_dates_dropped_in_join(raw_dir, monkeypatch):
    # synthetic: у бенчмарка на 3 даты больше, чем у фонда -> 3 даты отброшены при объединении
    fund = synthetic_prices("ARKK", days=60)
    bench = synthetic_prices("SPY", days=63)
    monkeypatch.setattr(data, "_download_prices",
                        lambda ticker: fund if ticker == "ARKK" else bench)
    monkeypatch.setattr(data, "_download_risk_free", lambda: synthetic_rate(63))
    result = data.get_dataset("ARKK", "SPY", START, END)
    info = result["info"]
    assert info["n_days"] == 60 and info["n_returns"] == 59
    assert info["dropped_dates"] == 3
    assert info["offline"] is False
    assert [s["ticker"] for s in info["series"]] == ["ARKK", "SPY", "DGS3MO"]
    assert all(s["pull_date"] == TODAY for s in info["series"])
    assert info["start"] == fund.index.min().date()
    assert len(result["risk_free"]) == 60  # ставка обрезана до окна цен


def test_get_dataset_mixed_sources_reports_only_saved_ones(raw_dir, monkeypatch):
    # цены свежие, а ставка из сохранённого файла (нет ключа) -> offline=True и одно сообщение
    saved_file(raw_dir, "rf_DGS3MO", synthetic_rate(63), day="2026-09-01")
    monkeypatch.setattr(data, "_download_prices", lambda ticker: synthetic_prices(ticker, days=63))
    monkeypatch.setattr(data, "_download_risk_free", fail_with(data.MissingKeyError("x")))
    info = data.get_dataset("ARKK", "SPY", START, END)["info"]
    assert info["offline"] is True
    assert len(info["messages"]) == 1 and "FRED_API_KEY" in info["messages"][0]
    assert info["pull_date"] == "2026-09-01"


def test_get_dataset_rejects_same_ticker_and_few_days(raw_dir, monkeypatch):
    with pytest.raises(data.DataError, match="different"):
        data.get_dataset("SPY", "SPY", START, END)
    monkeypatch.setattr(data, "_download_prices", lambda ticker: synthetic_prices(ticker, days=10))
    monkeypatch.setattr(data, "_download_risk_free", lambda: synthetic_rate(10))
    with pytest.raises(data.DataError, match="Too few common trading days"):
        data.get_dataset("ARKK", "SPY", START, END)


# ---------- История с 2000 года: список фондов и сокращённое окно ----------

# Ручной расчёт: с 2005-01-01 до 2014-10-31 ровно 3590 дней (9 лет по 365 + 2 високосных дня + 303 дня 2014 года).
# С 2005-01-01 до 2015-01-02: 10 лет по 365 + 2 високосных дня (2008, 2012) + 1 день = 3653 дня.
DAYS_2005_TO_2014_10_31 = 3590
DAYS_2005_TO_2015_01_02 = 3653


def long_rate(first, days):
    # synthetic: ставка 4.5% годовых каждый рабочий день
    return pd.DataFrame({data.RF_SERIES: [4.5] * days}, index=pd.bdate_range(first, periods=days))


def test_download_start_and_fund_list():
    assert data.DOWNLOAD_START == "2000-01-01"
    assert data.FUNDS == ["ARKK", "QQQ", "ICLN", "ARKW", "ARKG", "SMH", "TAN", "LIT", "GLD",
                          "SCHD", "EEM", "VTI", "VUG", "XLV", "VNQ", "TLT"]
    assert len(set(data.FUNDS)) == len(data.FUNDS)  # без повторов


def test_young_fund_gets_shortened_window_warning_online(raw_dir, monkeypatch):
    # synthetic: ARKK начинается 2014-10-31, SPY торгуется с 2004 года; пользователь выбрал старт 2005-01-01
    young = synthetic_prices("ARKK", days=60, first="2014-10-31")
    old = synthetic_prices("SPY", days=3000, first="2004-12-31")
    monkeypatch.setattr(data, "_download_prices", lambda ticker: young if ticker == "ARKK" else old)
    monkeypatch.setattr(data, "_download_risk_free", lambda: long_rate("2004-12-31", 3000))
    info = data.get_dataset("ARKK", "SPY", date(2005, 1, 1), date(2014, 12, 31))["info"]
    assert info["warnings"] == [
        "Data for ARKK starts on 2014-10-31, later than the start date you chose (2005-01-01); "
        f"the window was shortened by {DAYS_2005_TO_2014_10_31} days."]
    shortened = {s["ticker"]: s["shortened_days"] for s in info["series"]}
    assert shortened == {"ARKK": DAYS_2005_TO_2014_10_31, "SPY": 0, "DGS3MO": 0}
    assert info["requested_start"] == date(2005, 1, 1) and info["start"] == date(2014, 10, 31)
    assert info["window_shortened_days"] == DAYS_2005_TO_2014_10_31
    assert info["offline"] is False and "saved file" not in info["warnings"][0]


def test_old_saved_file_gets_shortened_window_warning_offline(raw_dir, monkeypatch):
    # synthetic: на диске только старые файлы с началом 2015-01-02; интернета нет; выбран старт 2005-01-01
    for name, frame in [("prices_ARKK", synthetic_prices("ARKK", days=60, first="2015-01-02")),
                        ("prices_SPY", synthetic_prices("SPY", days=60, first="2015-01-02")),
                        ("rf_DGS3MO", long_rate("2015-01-02", 60))]:
        saved_file(raw_dir, name, frame, day="2026-09-01")
    monkeypatch.setattr(data, "_download_prices", fail_with(data.NetworkError("x")))
    monkeypatch.setattr(data, "_download_risk_free", fail_with(data.NetworkError("x")))
    info = data.get_dataset("ARKK", "SPY", date(2005, 1, 1), date(2015, 3, 31))["info"]
    assert info["offline"] is True
    spy_warning = [w for w in info["warnings"] if w.startswith("Data for SPY")][0]
    assert f"starts on 2015-01-02, later than the start date you chose (2005-01-01); " in spy_warning
    assert f"shortened by {DAYS_2005_TO_2015_01_02} days" in spy_warning
    assert "saved file pulled 2026-09-01" in spy_warning and "download the full history" in spy_warning
    assert len(info["warnings"]) == 3          # старый файл не используется молча: предупреждение у каждого ряда


def test_weekend_or_holiday_start_is_not_a_shortened_window(raw_dir, monkeypatch):
    # synthetic: ряд начинается 2024-01-02, а выбран старт 2024-01-01 (праздник): разница в день - не сокращение
    monkeypatch.setattr(data, "_download_prices", lambda ticker: synthetic_prices(ticker, days=63))
    monkeypatch.setattr(data, "_download_risk_free", lambda: synthetic_rate(63))
    info = data.get_dataset("ARKK", "SPY", START, END)["info"]
    assert info["warnings"] == [] and info["window_shortened_days"] == 0


def test_same_ticker_error_names_the_ticker():
    with pytest.raises(data.DataError, match="must be different tickers: you chose QQQ for both"):
        data.get_dataset("QQQ", "QQQ", START, END)
