"""data.py - вся загрузка данных (Yahoo Finance и FRED). Никакой математики здесь.

Что делает модуль:
1. Скачивает цены фондов (Adjusted Close, auto_adjust=True) через yfinance.
2. Скачивает risk-free rate (серия DGS3MO) через FRED API.
3. Каждый скачанный файл сохраняет в data/raw/ с датой загрузки (pull date) в имени.
4. Если скачать не удалось, читает последний сохранённый файл и честно говорит,
   ПОЧЕМУ скачать не удалось: нет сети, ошибка API (неверный ключ, лимит, пустой ответ)
   или не задан FRED_API_KEY.

Главная функция для остального проекта: get_dataset(...).
"""
import os
import re
import socket
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

import core  # здесь только проверка дат окна (calendar), не доходности

# Читаем файл .env (там лежит FRED_API_KEY). В коде ключа нет.
# Путь задан явно, чтобы приложение находило .env, даже если запущено из другой папки.
load_dotenv(Path(__file__).parent / ".env")

# Папка для сырых файлов. Мы их только создаём, руками не правим.
RAW_DIR = Path(__file__).parent / "data" / "raw"

# Всегда скачиваем историю с этой даты, а нужный период вырезаем уже потом.
# Так один сохранённый файл подходит для любого периода (и в офлайн-режиме тоже).
DOWNLOAD_START = "2000-01-01"

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
RF_SERIES = "DGS3MO"  # 3-Month Treasury, значения в ПРОЦЕНТАХ годовых
CPI_SERIES = "CPIAUCSL"  # индекс потребительских цен США (инфляция), месячный

# Список для выбора на экране. Многие тикеры появились позже 2000 года: тогда окно данных
# начинается с их первой даты, и приложение говорит об этом (info["warnings"]).
# BRK-B (одна акция, а не фонд) и BITO (фьючерсы на биткоин, короткая история, высокий риск) убраны:
# сайт подбирает фонды, и в списке должны быть только обычные фонды с достаточной историей.
# Добавлены VTI, VUG, XLV, VNQ, TLT: широкий рынок США, рост, здравоохранение, недвижимость, облигации
# (у всех история с 2004 года или раньше).
FUNDS = ["ARKK", "QQQ", "ICLN", "ARKW", "ARKG", "SMH", "TAN", "LIT", "GLD", "SCHD", "EEM",
         "VTI", "VUG", "XLV", "VNQ", "TLT"]
DEFAULT_BENCHMARK = "SPY"

MIN_COMMON_DAYS = 30  # меньше общих торговых дней считать нечего

YAHOO_HOST = "query1.finance.yahoo.com"
FRED_HOST = "api.stlouisfed.org"


# ---------- Ошибки ----------
# Все они наследуют DataError, поэтому app.py может показать любую одним блоком except.

class DataError(Exception):
    """Понятная ошибка данных: её текст можно показывать пользователю."""


class NetworkError(DataError):
    """Нет сети: до источника данных не достучаться."""


class ApiError(DataError):
    """Источник ответил, но с ошибкой (неверный ключ, лимит запросов, пустой ответ)."""


class MissingKeyError(DataError):
    """Не задан FRED_API_KEY."""


class NoDataInPeriodError(DataError):
    """За выбранный период у ряда нет данных. Текст тот же, что и раньше; даты ряда - отдельными полями."""

    def __init__(self, message: str, first_date: date, last_date: date):
        super().__init__(message)
        self.first_date = first_date
        self.last_date = last_date


def _redact(text: str) -> str:
    """Убирает значение ключа FRED из текста. Страховка: ключ не должен попасть на экран или в лог."""
    key = (os.getenv("FRED_API_KEY") or "").strip()
    if key:
        text = text.replace(key, "***")
    return re.sub(r"api_key=[^&\s'\"]+", "api_key=***", text)


def _host_reachable(host: str) -> bool:
    """Проверяет, есть ли вообще сеть: пробуем открыть соединение с сервером источника."""
    try:
        with socket.create_connection((host, 443), timeout=3):
            return True
    except OSError:
        return False


# ---------- Работа с файлами в data/raw/ ----------

def _save_raw(df: pd.DataFrame, name: str) -> Path:
    """Сохраняет таблицу как data/raw/<name>_<дата загрузки>.csv.

    Повторный запуск в тот же день просто перезаписывает файл того же дня.
    Пишем во временный файл и переименовываем, чтобы сбой записи не испортил старый файл.
    Ошибка записи (OSError) пробрасывается наверх: это проблема диска, а не сети.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pull_date = date.today().isoformat()  # например 2026-10-01
    path = RAW_DIR / f"{name}_{pull_date}.csv"
    tmp = RAW_DIR / f".{name}_{pull_date}.tmp"
    try:
        df.to_csv(tmp, index_label="date")
        os.replace(tmp, path)  # атомарная замена (работает и на Windows, если файл не открыт)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _saved_files(name: str) -> list:
    """Все сохранённые файлы для name, от старых к новым. Формат имени: <name>_ГГГГ-ММ-ДД.csv."""
    pattern = re.compile(rf"^{re.escape(name)}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
    found = [(m.group(1), p) for p in RAW_DIR.glob(f"{name}_*.csv")
             if (m := pattern.match(p.name))]
    return [p for _, p in sorted(found)]


def _find_latest_raw(name: str):
    """Ищет самый свежий сохранённый файл для name. Если файлов нет, вернёт None."""
    files = _saved_files(name)
    return files[-1] if files else None


def _pull_date_from_path(path: Path) -> str:
    """Достаёт дату загрузки из имени файла: prices_ARKK_2026-10-01.csv -> 2026-10-01."""
    return path.stem.rsplit("_", 1)[1]


def _read_raw(path: Path) -> pd.DataFrame:
    """Читает сохранённый файл обратно в таблицу с датами в индексе."""
    try:
        return pd.read_csv(path, index_col="date", parse_dates=True)
    except Exception as err:
        raise DataError(
            f"The saved file {path.name} cannot be read ({type(err).__name__}). "
            "Delete it from data/raw/ and run once with internet to download it again."
        ) from err


# ---------- Скачивание из интернета ----------

def _clean_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Приводит ответ Yahoo к одному столбцу Adjusted Close с датами без часового пояса."""
    prices = df[["Close"]].copy()
    # Убираем часовой пояс (оставляя местную дату биржи), чтобы даты разных бирж можно было сравнивать
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices.columns = [ticker]
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()  # дубли дат (бывают у последнего дня)
    prices = prices.dropna()
    prices = prices[prices[ticker] > 0]  # нулевые и отрицательные цены - это ошибки данных
    return prices


def _download_prices(ticker: str) -> pd.DataFrame:
    """Скачивает Adjusted Close одного тикера (auto_adjust=True) с DOWNLOAD_START до сегодня."""
    import yfinance as yf  # импорт здесь, чтобы модуль загружался и без yfinance (например, в тестах)

    try:
        df = yf.Ticker(ticker).history(
            start=DOWNLOAD_START,
            end=(date.today() + timedelta(days=1)).isoformat(),  # end в yfinance не включается
            auto_adjust=True,  # цены уже учитывают дивиденды и сплиты
        )
    except Exception as err:  # типы ошибок yfinance зависят от версии, поэтому смотрим на сеть напрямую
        if not _host_reachable(YAHOO_HOST):
            raise NetworkError("No internet connection (Yahoo Finance is unreachable).") from err
        raise ApiError(f"Yahoo Finance request for {ticker} failed ({type(err).__name__}); "
                       "it may be rate-limiting, try again later.") from err
    if df is None or df.empty:
        # yfinance часто возвращает пустую таблицу и при отсутствии сети, и при неверном тикере
        if not _host_reachable(YAHOO_HOST):
            raise NetworkError("No internet connection (Yahoo Finance is unreachable).")
        raise ApiError(f"Yahoo Finance returned no data for {ticker} "
                       "(wrong ticker or a temporary Yahoo problem).")
    prices = _clean_prices(df, ticker)
    if prices.empty:
        raise ApiError(f"Yahoo Finance returned only empty prices for {ticker}.")
    return prices


def _download_risk_free() -> pd.DataFrame:
    """Скачивает ставку DGS3MO с FRED. Значения остаются в процентах (как на сайте FRED)."""
    return _download_fred(RF_SERIES)


def _download_cpi() -> pd.DataFrame:
    """Скачивает индекс потребительских цен CPIAUCSL с FRED (месячный, для доходности после инфляции)."""
    return _download_fred(CPI_SERIES)


def _download_fred(series_id: str) -> pd.DataFrame:
    """Скачивает один ряд FRED (тот же запрос и те же ошибки, что раньше были у ставки DGS3MO).

    Тексты ошибок строим сами и НЕ берём str(исключения): в адресе запроса есть api_key,
    и он мог бы попасть на экран.
    """
    api_key = (os.getenv("FRED_API_KEY") or "").strip()
    if not api_key or api_key == "your_fred_api_key_here":
        raise MissingKeyError("FRED_API_KEY is not set in the .env file.")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": DOWNLOAD_START,
    }
    try:
        response = requests.get(FRED_URL, params=params, timeout=15)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise NetworkError("No internet connection (FRED is unreachable).") from None
    except requests.exceptions.RequestException as err:  # тип можно показать, текст исключения нельзя (там ключ)
        raise ApiError(f"The FRED request failed ({type(err).__name__}).") from None

    status = response.status_code
    if status in (400, 401, 403):
        raise ApiError(f"FRED rejected the request (HTTP {status}): the FRED_API_KEY in .env "
                       "is probably invalid.")
    if status == 429:
        raise ApiError("FRED request limit reached (HTTP 429): try again later.")
    if status != 200:
        raise ApiError(f"FRED answered with an error (HTTP {status}).")

    try:
        observations = response.json()["observations"]
        df = pd.DataFrame(observations)[["date", "value"]]
    except (ValueError, KeyError, TypeError):
        raise ApiError("FRED returned an unexpected answer (no observations).") from None
    # В выходные и праздники FRED пишет "." вместо числа: превращаем в NaN и выбрасываем
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = [series_id]
    if df.empty:
        raise ApiError(f"FRED returned no data for {series_id}.")
    return df


# ---------- Скачивание с офлайн-запасным вариантом ----------

def _fallback_message(err: DataError, label: str, path: Path) -> str:
    """Честное сообщение: что случилось и что мы используем вместо свежих данных."""
    saved = f"Using the saved file {path.name} (pulled {_pull_date_from_path(path)})."
    if isinstance(err, NetworkError):
        return f"No internet connection: {label} could not be downloaded. {saved}"
    if isinstance(err, MissingKeyError):
        return (f"{label} was not downloaded because FRED_API_KEY is not set in .env "
                f"(the internet may be available). {saved}")
    return f"{label} could not be downloaded: {_redact(str(err))} {saved}"


def _no_saved_message(err: DataError, label: str) -> str:
    """Ошибка, если скачать нельзя и сохранённых файлов нет: с инструкцией, что делать."""
    if isinstance(err, MissingKeyError):
        return (f"{label} cannot be loaded: FRED_API_KEY is not set and there is no saved file in "
                "data/raw/. Add your FRED key to .env and run once with internet.")
    if isinstance(err, NetworkError):
        return (f"{label} cannot be loaded: there is no internet connection and no saved file in "
                "data/raw/. Connect to the internet and run once so the data can be saved.")
    return (f"{label} cannot be loaded: {_redact(str(err))} There is no saved file in data/raw/ "
            "either. Fix the problem above and run once with internet.")


def _load_with_fallback(name: str, label: str, download_fn, start: date, end: date):
    """Пробует скачать данные. Если не вышло, берёт последний сохранённый файл.

    Возвращает (таблица за период start..end, info про источник).
    Сбой записи на диск НЕ считается сетевой ошибкой: данные остаются свежими,
    а в info добавляется предупреждение save_warning.
    """
    save_warning = ""
    try:
        df = download_fn()
    except DataError as err:  # нет сети / ошибка API / нет ключа
        path = _find_latest_raw(name)
        if path is None:
            raise DataError(_no_saved_message(err, label)) from None
        df = _read_raw(path)
        info = {
            "offline": True,  # True = данные не свежие, взяты из сохранённого файла
            "cause": ("no_network" if isinstance(err, NetworkError)
                      else "missing_key" if isinstance(err, MissingKeyError) else "api_error"),
            "pull_date": _pull_date_from_path(path),
            "file": path.name,
            "message": _fallback_message(err, label, path),
            "save_warning": "",
        }
    else:
        try:
            path = _save_raw(df, name)  # сохраняем свежую загрузку
            file_name = path.name
        except OSError as err:
            file_name = ""
            save_warning = (f"The fresh {label} data could not be saved to data/raw/ "
                            f"({type(err).__name__}); they were still downloaded and used.")
        info = {"offline": False, "cause": "", "pull_date": date.today().isoformat(),
                "file": file_name, "message": "", "save_warning": save_warning}

    info["first_date"] = df.index.min().date()  # весь ряд, до вырезания периода
    info["last_date"] = df.index.max().date()

    # Ряд (фонд младше выбранной даты или старый сохранённый файл) может начаться позже даты начала:
    # это не ошибка, но молча использовать такое окно нельзя. Допуск на выходные и праздники в core.
    short_name = label.split()[-1]  # "Prices of ARKK" -> "ARKK"
    info["shortened_days"] = core.window_shortening_days(start, info["first_date"])
    info["warning"] = ""
    if info["shortened_days"]:
        info["warning"] = (f"Data for {short_name} starts on {info['first_date']}, later than the start date "
                           f"you chose ({start}); the window was shortened by {info['shortened_days']} days.")
        if info["offline"]:
            info["warning"] += (f" It comes from the saved file pulled {info['pull_date']}; "
                                "run once with internet to download the full history.")

    df = df.loc[str(start):str(end)]  # оставляем только выбранный период
    if df.empty:
        raise NoDataInPeriodError(f"No data for {label} between {start} and {end}. "
                                  f"The data cover {info['first_date']} to {info['last_date']}.",
                                  info["first_date"], info["last_date"])
    return df, info


def get_prices(ticker: str, start: date, end: date):
    """Цены одного тикера за период. Возвращает (таблица, info)."""
    return _load_with_fallback(f"prices_{ticker}", f"Prices of {ticker}",
                               lambda: _download_prices(ticker), start, end)


def get_risk_free(start: date, end: date):
    """Risk-free rate (в процентах годовых) за период. Возвращает (таблица, info)."""
    return _load_with_fallback(f"rf_{RF_SERIES}", f"Risk-free rate {RF_SERIES}",
                               _download_risk_free, start, end)


def get_cpi(start: date, end: date):
    """Индекс потребительских цен CPIAUCSL за период. Возвращает (таблица, info). Тот же офлайн-режим."""
    return _load_with_fallback(f"cpi_{CPI_SERIES}", f"Inflation index {CPI_SERIES}", _download_cpi, start, end)


# ---------- Главная функция ----------

def get_dataset(fund: str, benchmark: str, start: date, end: date) -> dict:
    """Собирает всё, что нужно приложению: цены фонда и бенчмарка + risk-free rate.

    Возвращает словарь:
      prices     - таблица Adjusted Close, 2 столбца (фонд, бенчмарк), только общие даты
      risk_free  - таблица со ставкой DGS3MO в ПРОЦЕНТАХ годовых (перевод в доли делает core.py),
                   только за окно реально использованных цен
      info       - источники, даты загрузки, режим, окно, счётчики (для Honesty panel и DATA.md)
    """
    if fund == benchmark:
        raise DataError(f"Fund and benchmark must be different tickers: you chose {fund} for both. "
                        "Pick another benchmark.")
    if start >= end:
        raise DataError("Start date must be before end date.")

    fund_prices, fund_info = get_prices(fund, start, end)
    bench_prices, bench_info = get_prices(benchmark, start, end)
    risk_free, rf_info = get_risk_free(start, end)

    # Оставляем только даты, где есть цены и у фонда, и у бенчмарка (inner join).
    # Это выравнивание данных, а не математика.
    prices = fund_prices.join(bench_prices, how="inner")
    all_dates = fund_prices.index.union(bench_prices.index)
    dropped_dates = len(all_dates) - len(prices)  # даты, которые есть только у одного из двух рядов
    if len(prices) < MIN_COMMON_DAYS:
        raise DataError(f"Too few common trading days: {len(prices)} "
                        f"(at least {MIN_COMMON_DAYS} are required). Choose a longer period.")

    window_start, window_end = prices.index.min(), prices.index.max()
    # Ставку усредняем за то же окно, что и цены (а не за запрошенные даты, которые могут быть выходными)
    risk_free = risk_free.loc[window_start:window_end]
    if risk_free.empty:
        raise DataError(f"No risk-free rate data between {window_start.date()} and {window_end.date()}.")

    series = [
        {"role": "Fund prices", "ticker": fund, "source": "Yahoo Finance (yfinance)",
         "field": "Adjusted Close (Close with auto_adjust=True)", "currency": "USD",
         "frequency": "daily", "adjusted": True, "unit": "USD per share", **fund_info},
        {"role": "Benchmark prices", "ticker": benchmark, "source": "Yahoo Finance (yfinance)",
         "field": "Adjusted Close (Close with auto_adjust=True)", "currency": "USD",
         "frequency": "daily", "adjusted": True, "unit": "USD per share", **bench_info},
        {"role": "Risk-free rate", "ticker": RF_SERIES, "source": "FRED, St. Louis Fed (API)",
         "field": "3-Month Treasury constant maturity rate", "currency": "percent per year (USD)",
         "frequency": "daily", "adjusted": None, "unit": "% per year", **rf_info},
    ]
    used_saved = [s for s in series if s["offline"]]
    return {
        "prices": prices,
        "risk_free": risk_free,
        "info": {
            "series": series,
            "tickers": [fund, benchmark, RF_SERIES],
            "price_source": "Yahoo Finance (yfinance), column Adjusted Close (auto_adjust=True)",
            "rate_source": "FRED API, series DGS3MO (3-Month Treasury, % per year)",
            "pull_date": min(s["pull_date"] for s in series),  # самая старая загрузка из трёх
            "offline": bool(used_saved),
            "messages": [s["message"] for s in used_saved],
            "save_warnings": [s["save_warning"] for s in series if s["save_warning"]],
            "requested_start": start,           # окно, которое выбрал пользователь
            "requested_end": end,
            "start": window_start.date(),       # окно, которое реально использовано
            "end": window_end.date(),
            "window_shortened_days": core.window_shortening_days(start, window_start.date()),
            "warnings": [s["warning"] for s in series if s["warning"]],  # окно сокращено: ряд начался позже
            "n_days": len(prices),             # торговых дней после inner join
            "n_returns": len(prices) - 1,      # из них дневных доходностей
            "dropped_dates": dropped_dates,    # дат отброшено при объединении
            "n_rf_observations": len(risk_free),
            "currency": "USD",
            "frequency": "daily",
            "adjusted": True,
        },
    }
