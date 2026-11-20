"""app.py - только экран. Здесь нет формул: все расчёты делает core.py,
все данные приносит data.py. Этот файл лишь соединяет их и показывает результат.
Весь текст, который видит пользователь, - на английском.

Запуск:  python3 app.py   ->   открыть http://127.0.0.1:5000
"""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from urllib.parse import urlencode

from flask import Flask, jsonify, render_template, request
from markupsafe import escape

import core
import data
import forecast
import make_evidence
import rule

# ---------- Скорость: загрузки помним 1 час и скачиваем параллельно ----------
# Источники данных и офлайн-режим не меняются: это те же функции data.py, просто их ответ запоминается.
# Ошибка загрузки не запоминается, поэтому запасной вариант с сохранённым файлом работает как раньше.

CACHE_SECONDS = 60 * 60      # свежая загрузка используется повторно в течение часа
DOWNLOAD_WORKERS = 4         # сколько тикеров скачивать одновременно
_cache = {}
_cache_lock = threading.Lock()


def _remember(download_fn, key_fn):
    """Обёртка над функцией загрузки: удачный ответ хранится в памяти CACHE_SECONDS секунд."""
    def wrapper(*args):
        key = key_fn(*args)
        with _cache_lock:
            hit = _cache.get(key)
        if hit is not None and time.monotonic() - hit[0] < CACHE_SECONDS:
            return hit[1].copy()
        df = download_fn(*args)              # ошибка (DataError) уходит дальше и не запоминается
        with _cache_lock:
            _cache[key] = (time.monotonic(), df)
        return df.copy()
    wrapper.__wrapped__ = download_fn
    return wrapper


def enable_download_cache() -> None:
    """Включается только при запуске сайта (python app.py), чтобы тесты всегда шли без памяти."""
    data._download_prices = _remember(data._download_prices, lambda ticker: ("prices", ticker))
    data._download_risk_free = _remember(data._download_risk_free, lambda: ("risk_free",))


def load_prices_parallel(tickers, start: date, end: date) -> dict:
    """data.get_prices для нескольких тикеров одновременно. Возвращает {тикер: (prices, info) или DataError}."""
    def one(ticker):
        try:
            return ticker, data.get_prices(ticker, start, end)
        except data.DataError as err:
            return ticker, err
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        return dict(pool.map(one, tickers))


def warm_up() -> None:
    """При запуске сайта заранее скачивает ставку FRED и цены всех фондов и бенчмарков (в фоне)."""
    start, end = core.default_period(date.today())
    try:
        data.get_risk_free(start, end)
    except data.DataError:
        pass
    load_prices_parallel(sorted(set(data.FUNDS) | set(BENCHMARKS)), start, end)

app = Flask(__name__)

# Только бенчмарки, разные по смыслу. VTI и IVV убраны (почти повторяют SPY), VT убран (почти повторяет ACWI).
BENCHMARKS = ["SPY", "QQQ", "IWM", "ACWI", "AGG"]  # SPY - по умолчанию

# Тексты вердикта по коду из core.verdict_code: (подпись, css-класс)
VERDICT_TEXT = {
    "justifies": ("Fund justifies its risk (its Sharpe ratio is higher than the benchmark's)", "good"),
    "does_not_justify": ("Fund does not justify its risk (its Sharpe ratio is not higher than the benchmark's)", "bad"),
    "unknown": ("No conclusion: Sharpe ratio could not be calculated", "neutral"),
}


# ---------- Форматирование чисел в текст (только внешний вид) ----------

def fmt_pct(value) -> str:
    """0.1234 -> '12.34%'. Если значения нет (None), показываем тире."""
    return "-" if value is None else f"{core.to_percent(value):.2f}%"


def fmt_num(value) -> str:
    """1.23456 -> '1.23'. Если значения нет (None), показываем тире."""
    return "-" if value is None else f"{value:.2f}"


def fmt_pp(points: float) -> str:
    """27.9812 -> '27.98' (процентные пункты уже посчитаны в core.py)."""
    return f"{points:.2f}"


# Тексты главного числа и пояснения по коду направления из core.gap_headline: {код: (текст, css-класс)}
HEADLINE_TEXT = {
    "overstated": ("Advertised return overstated by {pts} percentage points per year", "bad"),
    "beat": ("Fund beat the advertised return by {pts} percentage points per year", "good"),
    "matched": ("Fund matched the advertised return (difference of 0.00 percentage points per year)", "neutral"),
}
SENTENCE_TEXT = {
    "overstated": "The advertised return is higher than the actual return over the selected period.",
    "beat": "Over the selected period the fund delivered more than the advertisement claims.",
    "matched": "Over the selected period the fund delivered about what the advertisement claims.",
}


def headline_text(headline: dict) -> dict:
    """Собирает текст главного числа из результата core.gap_headline (только подстановка в шаблон текста)."""
    text, css = HEADLINE_TEXT[headline["direction"]]
    return {"text": text.format(pts=fmt_pp(headline["points_abs"])), "css": css,
            "number": f"{fmt_pp(headline['points_signed'])} pp per year"}


# Фразы панели "Why this number?": какую брать, решает core.explain_gap (по знаку gap, бете и альфе)
WHY_READING = {
    "overstated": "A positive gap means the ad is overstated: it promised more than the fund delivered.",
    "beat": "A negative gap means the fund beat the ad: it delivered more than was promised.",
    "matched": "The gap is zero: the fund delivered what the ad promised.",
}
WHY_BETA = {
    "higher": "Beta is {beta}, so {fund} tends to move more than {bench}.",
    "lower": "Beta is {beta}, so {fund} tends to move less than {bench}.",
    "similar": "Beta is {beta}, so {fund} tends to move about as much as {bench}.",
    "opposite": "Beta is {beta}, so {fund} tends to move against {bench}.",
    "unknown": "Beta could not be calculated for this period.",
}
WHY_ALPHA = {
    "above": "Jensen's alpha is {alpha} per year: the fund earned that much above the price of its risk.",
    "below": "Jensen's alpha is {alpha} per year: the fund earned less than the price of its risk.",
    "zero": "Jensen's alpha is {alpha} per year: the fund earned about the price of its risk.",
    "unknown": "Jensen's alpha needs beta, so it is not available either.",
}


def why_panel(why: dict, fund: str, benchmark: str) -> dict:
    """Собирает фразы панели из результата core.explain_gap: только подстановка чисел и форматирование."""
    risk = " ".join([WHY_BETA[why["beta_level"]], WHY_ALPHA[why["alpha_level"]]]).format(
        beta=fmt_num(why["beta"]), alpha=fmt_pct(why["alpha"]), fund=fund, bench=benchmark)
    advertised, actual = fmt_pct(why["advertised"]), fmt_pct(why["actual"])
    return {
        "promised": f"The ad promised {advertised} per year.",
        "actual": ("Over the selected period the fund actually returned "
                   f"{actual} per year (compound annual growth, from adjusted prices)."),
        "equation": f"{advertised} minus {actual} = {fmt_pp(why['points_signed'])} percentage points per year.",
        "reading": WHY_READING[why["direction"]],
        "risk": risk,
    }


# ---------- Deep analysis: тексты панелей (числа и выбор вариантов приходят из core.deep_analysis) ----------

CHERRY_PICKING_WARNING = ("These are historical what-ifs, not a forecast. Choosing a window only because it "
                          "turns the verdict green is cherry-picking; use this to see how fragile the verdict is, "
                          "not to pick a result.")

DEEP_SHARPE = ("The Sharpe ratio is the return a fund earned above the risk-free rate (US 3-month Treasury bills) "
               "for each unit of risk it took (volatility). Higher is better. Over {period}, {fund} scored {sf} "
               "and {bench} scored {sb}.")
DEEP_SHARPE_UNKNOWN = ("The Sharpe ratio is the return a fund earned above the risk-free rate for each unit of risk. "
                       "Over {period} it could not be calculated for this pair, so there is no verdict.")
DEEP_GAP = {
    "overstated": "So the ad looks {pts} percentage points per year too high for this period.",
    "beat": "So the fund beat the ad by {pts} percentage points per year in this period.",
    "matched": "So the ad matches what the fund delivered in this period.",
}
DEEP_VERDICT = {
    ("justifies", "clear"): ("{fund}'s Sharpe ratio is {margin} higher than {bench}'s: for the risk it took, "
                             "the fund earned more than simply holding {bench}."),
    ("justifies", "too_close"): ("{fund}'s Sharpe ratio is only {margin} higher than {bench}'s. That is too close "
                                 "to call: a small change in the dates or in the data could reverse it."),
    ("does_not_justify", "clear"): ("{fund}'s Sharpe ratio is {margin} lower than {bench}'s: for the risk it took, "
                                    "the fund earned less than simply holding {bench}."),
    ("does_not_justify", "too_close"): ("{fund}'s Sharpe ratio is only {margin} lower than {bench}'s. That is too "
                                        "close to call: a small change in the dates or in the data could reverse it."),
    ("unknown", "unknown"): "Without a Sharpe ratio for both, the app cannot say whether the fund justified its risk.",
}
DEEP_PERIOD = ("All of this describes {period} ({n_days} trading days). A different window can give a different "
               "answer, so treat the verdict as a description of this period only.")

SHORTFALL_REASON = {
    "not_enough_data": "There is too little data in this period to calculate what is missing.",
    "no_volatility": "The fund's price did not vary at all in this period, so there is no risk figure to compare.",
    "sharpe_missing": "A Sharpe ratio could not be calculated, so there is nothing to compare.",
    "benchmark_sharpe_not_positive": ("{bench} itself earned no more than the risk-free rate in this period "
                                      "(its Sharpe ratio is zero or negative), so the fund has no positive bar to "
                                      "reach. Compare it over another period or with another benchmark."),
}
SHORTFALL_VOL_REASON = {
    "return_not_above_risk_free": ("Lower volatility alone could not fix this: the fund's average return was not "
                                   "above the risk-free rate."),
}
WINDOW_DIRECTION = {"later_start": "starting {days} days later", "earlier_end": "ending {days} days earlier"}


def try_link(params: dict, **changes) -> str:
    """Ссылка на эту же страницу с теми же параметрами формы, но с изменёнными значениями.

    Значения проходят ту же проверку (parse_form), что и обычный ввод: ссылка просто открывает форму.
    """
    merged = dict(params, **changes)
    merged["run"] = "1"
    return "/?" + urlencode(merged)


def window_item(window: dict, params: dict) -> dict:
    """Одна строка списка окон: текст и ссылка "Try this window" (даты уже посчитаны в core)."""
    direction = WINDOW_DIRECTION[window["direction"]].format(days=window["shift_days"])
    return {"text": (f"From {window['start']} to {window['end']} ({direction}): "
                     f"Sharpe ratio {fmt_num(window['sharpe_fund'])} vs {fmt_num(window['sharpe_bench'])}."),
            "link": try_link(params, start=window["start"].isoformat(), end=window["end"].isoformat())}


def deep_panels(analysis: dict, claim: dict, fund: str, benchmark: str, period: str,
                n_days: int, params: dict) -> dict:
    """Собирает раздел "Deep analysis": какие панели показать, решает core.deep_analysis; здесь только тексты."""
    sf, sb = fmt_num(claim["sharpe_fund"]), fmt_num(claim["sharpe_bench"])
    kind, close = analysis["verdict"], analysis["closeness"]
    margin = fmt_num(analysis["sharpe_margin_abs"])
    names = {"fund": fund, "bench": benchmark, "period": period, "sf": sf, "sb": sb, "margin": margin}

    sharpe_text = DEEP_SHARPE_UNKNOWN if kind == "unknown" else DEEP_SHARPE
    gap_text = DEEP_GAP[claim["headline"]["direction"]].format(pts=fmt_pp(claim["headline"]["points_abs"]))
    meaning = [
        sharpe_text.format(**names),
        (f"The ad promised {fmt_pct(claim['advertised'])} per year. Over the same dates the fund actually grew "
         f"{fmt_pct(claim['actual'])} per year (CAGR, compounded). {gap_text}"),
        DEEP_VERDICT[(kind, close if kind != "unknown" else "unknown")].format(**names),
        DEEP_PERIOD.format(period=period, n_days=n_days),
    ]

    honest = None
    if analysis["overstated"]:
        limit = analysis["honest_limit_pct"]
        if limit is None:
            honest = {"text": ("Over this period the fund lost money each year on average "
                               f"(CAGR {fmt_pct(claim['actual'])}), so no advertised return of 0% or more "
                               "would match what it delivered."), "link": "", "link_text": ""}
        else:
            honest = {"text": (f"An advertised return of up to {limit:.1f}% per year would match what this fund "
                               f"actually delivered (CAGR). You entered {params['advertised']}%."),
                      "link_text": f"Try {limit:.1f}%", "link": try_link(params, advertised=f"{limit:.1f}")}

    missing = None
    if kind == "does_not_justify":
        short = analysis["shortfall"]
        lines = [f"{fund}'s Sharpe ratio is {sf}; {benchmark}'s is {sb}."]
        if short["status"] == "ok":
            lines.append(f"Sharpe gap: {fmt_num(short['sharpe_gap'])}.")
            lines.append(f"To match {benchmark} at its own volatility ({fmt_pct(short['volatility'])}), {fund} would "
                         f"have needed an arithmetic return of at least {fmt_pct(short['required_arith_return'])} "
                         f"per year instead of {fmt_pct(short['fund_arith'])}: about "
                         f"{fmt_pp(short['return_shortfall_pp'])} percentage points more per year. At exactly that "
                         "figure the Sharpe ratios would only tie; slightly above it would justify the risk.")
            if short["max_volatility"] is not None:
                lines.append(f"Or, with the same return, volatility would have had to be "
                             f"{fmt_pct(short['max_volatility'])} or slightly lower instead of "
                             f"{fmt_pct(short['volatility'])} (about {fmt_pp(short['volatility_cut_pp'])} "
                             "percentage points lower).")
            else:
                lines.append(SHORTFALL_VOL_REASON[short["max_volatility_reason"]])
        else:
            lines.append(SHORTFALL_REASON[short["reason"]].format(bench=benchmark))
        if close == "too_close":
            lines.append("The gap is small: this is too close to call.")
        missing = {
            "lines": lines,
            "heading": "Nearest date windows where the verdict is green",
            "windows": [window_item(w, params) for w in analysis["windows"]],
            "empty": (f"No window inside the selected period (at least {analysis['min_window_days']} trading "
                      "days) turns the verdict green."),
            "caption": "Nearest = the smallest change of the date window that turns the verdict green.",
        }

    solid = None
    if kind == "justifies":
        lines = [f"{fund}'s Sharpe ratio is {margin} above {benchmark}'s ({sf} vs {sb})."]
        if close == "too_close":
            lines.append("That is too close to call.")
        else:
            lines.append("A margin of 0.10 or more is treated as clear (a threshold chosen by the team, "
                         "not a rule from the course).")
        solid = {
            "lines": lines,
            "heading": "Nearest date windows where the verdict would turn red",
            "windows": [window_item(w, params) for w in analysis["windows"]],
            "empty": (f"No window inside the selected period (at least {analysis['min_window_days']} trading "
                      "days) turns the verdict red."),
            "caption": "Nearest = the smallest change of the date window that flips the verdict.",
        }

    return {"meaning": meaning, "honest": honest, "missing": missing, "solid": solid,
            "warning": CHERRY_PICKING_WARNING}


# ---------- Чтение формы ----------

def read_form() -> dict:
    """Берёт значения из адресной строки (форма отправляется методом GET).

    Если значения нет, подставляем значения по умолчанию: последние 3 года и реклама 35%.
    """
    default_start, default_end = core.default_period(date.today())
    return {
        "fund": request.args.get("fund", data.FUNDS[0]),
        "benchmark": request.args.get("benchmark", data.DEFAULT_BENCHMARK),
        "start": request.args.get("start", default_start.isoformat()),
        "end": request.args.get("end", default_end.isoformat()),
        "advertised": request.args.get("advertised", str(core.DEFAULT_ADVERTISED_PCT)),
    }


def parse_form(form: dict):
    """Проверяет значения формы. Возвращает (fund, benchmark, start, end, advertised_pct).

    Любая проблема - ValueError с понятным текстом, он показывается на странице.
    """
    if form["fund"] not in data.FUNDS:
        raise ValueError("Please choose a fund from the list.")
    if form["benchmark"] not in BENCHMARKS:
        raise ValueError("Please choose a benchmark from the list.")
    try:
        start = date.fromisoformat(form["start"])
        end = date.fromisoformat(form["end"])
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format, e.g. 2023-10-01.")
    # Правила дат лежат в core.py; самую раннюю дату передаём из data.DOWNLOAD_START (одно место для константы)
    core.validate_period(start, end, date.today(), min_date=date.fromisoformat(data.DOWNLOAD_START))
    advertised_pct = core.parse_advertised_pct(form["advertised"])
    return form["fund"], form["benchmark"], start, end, advertised_pct


# ---------- Сборка результата для страницы ----------

def build_result(fund, benchmark, start, end, advertised_pct) -> dict:
    """Вызывает data.py и core.py и складывает всё нужное для страницы в словарь."""
    dataset = data.get_dataset(fund, benchmark, start, end)
    prices = dataset["prices"]
    info = dataset["info"]

    # Дневные доходности и безрисковая ставка (формулы в core.py)
    fund_ret = core.daily_returns(prices[fund])
    bench_ret = core.daily_returns(prices[benchmark])
    rf = core.risk_free_annual(dataset["risk_free"].iloc[:, 0])

    # Сравнение рекламы с реальностью (тоже в core.py)
    claim = core.compare_claim(core.percent_to_decimal(advertised_pct), fund_ret, bench_ret, rf)

    # Панель "Why this number?": выбор формулировок делает core.explain_gap
    why = why_panel(core.explain_gap(claim["advertised"], claim["actual"], claim["gap"],
                                     claim["beta"], claim["alpha"]), fund, benchmark)

    # Deep analysis: что не хватает до зелёного (вся математика в core.deep_analysis)
    analysis = core.deep_analysis(claim, fund_ret, bench_ret, dataset["risk_free"].iloc[:, 0], rf,
                                  start_date=info["start"])
    form_params = {"fund": fund, "benchmark": benchmark, "start": start.isoformat(), "end": end.isoformat(),
                   "advertised": f"{advertised_pct:g}"}
    deep = deep_panels(analysis, claim, fund, benchmark, f"{info['start']} to {info['end']}",
                       info["n_days"], form_params)

    # Таблица сравнения: строки - показатели, столбцы - фонд и бенчмарк
    table = [
        ("Cumulative return over the period",
         fmt_pct(core.cumulative_return(prices[fund])), fmt_pct(core.cumulative_return(prices[benchmark]))),
        ("Annual return, arithmetic (average daily return x 252)",
         fmt_pct(claim["actual_arith"]), fmt_pct(claim["bench_arith"])),
        ("Annual return, geometric (CAGR, compounded)",
         fmt_pct(claim["actual"]), fmt_pct(claim["bench_actual"])),
        ("Annualized volatility",
         fmt_pct(core.annualized_volatility(fund_ret)), fmt_pct(core.annualized_volatility(bench_ret))),
        ("Sharpe ratio (annualized, arithmetic)",
         fmt_num(claim["sharpe_fund"]), fmt_num(claim["sharpe_bench"])),
        ("Beta vs benchmark",
         fmt_num(claim["beta"]), fmt_num(core.BENCHMARK_BETA)),
    ]

    verdict, verdict_class = VERDICT_TEXT[claim["verdict"]]
    headline = headline_text(claim["headline"])
    fund_chart = core.chart_values(prices[fund])
    bench_chart = core.chart_values(prices[benchmark])
    y_min, y_max = core.chart_range(fund_chart, bench_chart)

    return {
        "fund": fund,
        "benchmark": benchmark,
        "period": f"{info['start']} to {info['end']}",
        "headline": headline,
        "why": why,
        "deep": deep,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "sharpe_fund": fmt_num(claim["sharpe_fund"]),
        "sharpe_bench": fmt_num(claim["sharpe_bench"]),
        "alpha": fmt_pct(claim["alpha"]),
        "table": table,
        "rf": fmt_pct(rf),
        "claim": {
            "advertised": fmt_pct(claim["advertised"]),
            "actual": fmt_pct(claim["actual"]),
            "gap": f"{fmt_pp(claim['headline']['points_signed'])} pp",
            "capm_expected": fmt_pct(claim["capm_expected"]),
            "alpha": fmt_pct(claim["alpha"]),
            "overstated": claim["overstated"],
            "sentence": SENTENCE_TEXT[claim["headline"]["direction"]],
        },
        "chart": {
            "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
            "fund": fund_chart,
            "bench": bench_chart,
            "y_min": y_min,
            "y_max": y_max,
        },
        "info": info,
    }


# ---------- Главная страница: подбор фондов под желаемую доходность ----------

# Короткие описания простыми словами (только текст для экрана)
FUND_DESCRIPTIONS = {
    "ARKK": "Actively managed fund that bets on new technologies (high risk)",
    "QQQ": "The 100 largest companies on the Nasdaq exchange, mostly tech",
    "ICLN": "Clean energy companies around the world",
    "ARKW": "Internet and digital economy companies, actively managed",
    "ARKG": "Genomics and biotech companies, actively managed",
    "SMH": "Companies that design and make computer chips",
    "TAN": "Solar energy companies",
    "LIT": "Lithium mining and battery makers",
    "GLD": "Physical gold",
    "SCHD": "Large US companies that pay steady dividends",
    "EEM": "Large companies in emerging markets (China, India, Brazil and others)",
    "VTI": "The whole US stock market in one fund (thousands of companies)",
    "VUG": "Large US companies that grow fast (growth stocks)",
    "XLV": "US healthcare companies: drug makers, hospitals, insurers",
    "VNQ": "US real estate companies (REITs): offices, flats, warehouses",
    "TLT": "Long-term US government bonds (20 years and more)",
}
BENCHMARK_LABELS = {
    "SPY": "SPY: 500 large US companies (S&P 500)",
    "QQQ": "QQQ: 100 large Nasdaq companies",
    "IWM": "IWM: small US companies",
    "ACWI": "ACWI: stocks from the whole world",
    "AGG": "AGG: US bonds",
}
NO_FUND = ""  # значение варианта "Without a fund: show all that fit"
TARGET_ERROR = "Enter a number from 0 to {max:.0f} (% per year), e.g. 12."

# Описание рядов для панели Honesty: те же подписи, что в data.get_dataset
PRICE_SERIES = {"source": "Yahoo Finance (yfinance)", "field": "Adjusted Close (Close with auto_adjust=True)",
                "currency": "USD", "frequency": "daily", "adjusted": True}
RF_SERIES_INFO = {"ticker": data.RF_SERIES, "role": "Risk-free rate", "source": "FRED, St. Louis Fed (API)",
                  "field": "3-Month Treasury constant maturity rate", "currency": "percent per year (USD)",
                  "frequency": "daily", "adjusted": None}

# Тексты Deep analysis на странице подбора (числа приходят из core.py)
PICK_DEEP_GENERAL = [
    ("Return per year (CAGR) is the steady yearly growth that turns the price on the first day into the price on "
     "the last day: (last price / first price) ^ (1 / number of years) - 1. Prices are adjusted for dividends and "
     "splits, so dividends count as growth. One year = 252 trading days."),
    ("How much it swings (volatility) is how strongly the price went up and down: the standard deviation of daily "
     "returns times the square root of 252. A bigger number means a bumpier ride and deeper falls along the way."),
    ("The Sharpe ratio is the return a fund earned above the risk-free rate (US 3-month Treasury bills, from FRED) "
     "for each unit of risk it took (volatility): (average yearly return - risk-free rate) / volatility. "
     "Higher is better. The average yearly return here is arithmetic (average daily return x 252), so it is "
     "usually a little higher than CAGR."),
]
PICK_DEEP_WARNING = ("All of this describes the selected period only. A different period can give a different "
                     "answer. Past returns are not a forecast and not a promise of future returns.")


def read_pick_form() -> dict:
    """Значения главной формы из адресной строки. Период по умолчанию - последние 3 года."""
    default_start, default_end = core.default_period(date.today())
    return {
        "target": request.args.get("target", ""),
        "my_fund": request.args.get("my_fund", NO_FUND),
        "benchmark": request.args.get("benchmark", data.DEFAULT_BENCHMARK),
        "start": request.args.get("start", default_start.isoformat()),
        "end": request.args.get("end", default_end.isoformat()),
    }


def parse_pick_form(form: dict):
    """Проверяет главную форму. Возвращает (target_pct, my_fund, benchmark, start, end) или бросает ValueError."""
    try:
        target_pct = core.parse_advertised_pct(form["target"])  # число от 0 до MAX_ADVERTISED_PCT (1000)
    except ValueError:
        raise ValueError(TARGET_ERROR.format(max=core.MAX_ADVERTISED_PCT)) from None
    if form["my_fund"] != NO_FUND and form["my_fund"] not in data.FUNDS:
        raise ValueError("Please choose a fund from the list.")
    if form["benchmark"] not in BENCHMARKS:
        raise ValueError("Please choose a benchmark from the list.")
    try:
        start = date.fromisoformat(form["start"])
        end = date.fromisoformat(form["end"])
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format, e.g. 2023-10-01.")
    core.validate_period(start, end, date.today(), min_date=date.fromisoformat(data.DOWNLOAD_START))
    return target_pct, form["my_fund"], form["benchmark"], start, end


def no_data_text(err: data.DataError, end: date) -> str:
    """Понятный текст для блока "Funds without data": тексты ошибок те же, что в data.py."""
    if isinstance(err, data.NoDataInPeriodError) and err.first_date > end:
        return f"No data: the fund's history starts on {err.first_date}. {err}"
    return str(err)


def fund_row(fund: str, prices, info: dict, risk_free, target: float, target_text: str, link):
    """Показатели одного фонда за период (все формулы из core.py). Нечего считать - DataError."""
    series = prices[fund]
    late = f"No data: the fund's history starts on {info['first_date']}. " if info["shortened_days"] else ""
    if len(series) < data.MIN_COMMON_DAYS:
        raise data.DataError(f"{late}Too few trading days: {len(series)} (at least {data.MIN_COMMON_DAYS} "
                             "are required). Choose a longer period.")
    first, last = series.index.min(), series.index.max()
    rf_window = risk_free.loc[first:last].iloc[:, 0]  # ставка за то же окно, что и цены
    if rf_window.empty:
        raise data.DataError(f"No risk-free rate data between {first.date()} and {last.date()}.")
    returns = core.daily_returns(series)
    cagr = core.annualized_return(returns)
    volatility = core.annualized_volatility(returns)
    rf = core.risk_free_annual(rf_window)
    sharpe = core.sharpe_ratio(returns, rf)
    meets = core.gap_headline(target - cagr)["direction"] != "overstated"  # та же логика, что у подробной страницы
    years = len(returns) / core.TRADING_DAYS  # число лет торговли: так CAGR приводит рост к году
    p0, p1 = float(series.iloc[0]), float(series.iloc[-1])
    worst_fall = rule.max_drawdown(returns)  # худшее падение от вершины за период (для карточки-итога)

    explain = [
        (f"Price on {first.date()}: {p0:.2f} USD. Price on {last.date()}: {p1:.2f} USD (adjusted). "
         f"Growth over the whole period: {fmt_pct(core.cumulative_return(series))}."),
        (f"The period has {len(returns)} daily returns, that is {years:.2f} years of trading. "
         f"CAGR = ({p1:.2f} / {p0:.2f}) ^ (1 / {years:.2f}) - 1 = {fmt_pct(cagr)} per year."),
        f"Volatility: {fmt_pct(volatility)} per year.",
        (f"Sharpe ratio = (average yearly return {fmt_pct(core.arithmetic_annual_return(returns))} - risk-free "
         f"rate {fmt_pct(rf)}) / volatility {fmt_pct(volatility)} = {fmt_num(sharpe)}." if sharpe is not None else
         "The Sharpe ratio could not be calculated: the price did not vary at all in this period."),
        (f"In the list: {fmt_pct(cagr)} per year is at least the {target_text} you want." if meets else
         f"Not in the list: {fmt_pct(cagr)} per year is less than the {target_text} you want."),
    ]
    if info["shortened_days"]:
        explain.insert(0, f"Data only from {info['first_date']}; the calculation starts on that date.")
    return {
        "fund": fund,
        "description": FUND_DESCRIPTIONS.get(fund, ""),
        "cagr_value": cagr,
        "meets": meets,
        "cagr": fmt_pct(cagr),
        "volatility": fmt_pct(volatility),
        "sharpe": fmt_num(sharpe),
        "first_date": info["first_date"] if info["shortened_days"] else None,
        "link": link,
        "explain": explain,
        "used": (first.date(), last.date()),
        "volatility_value": volatility,
        "max_drawdown": fmt_pct(worst_fall),
    }


def pick_funds(target_pct: float, my_fund: str, benchmark: str, start: date, end: date) -> dict:
    """Считает показатели всех фондов и оставляет те, что дали желаемую доходность или больше."""
    risk_free, rf_info = data.get_risk_free(start, end)
    target = core.percent_to_decimal(target_pct)
    target_text = f"{target_pct:g}%"
    # Ссылка открывает подробную страницу: желаемая доходность подставляется вместо "advertised"
    link_params = {"benchmark": benchmark, "start": start.isoformat(), "end": end.isoformat(),
                   "advertised": f"{target_pct:g}"}
    rows, no_data, skipped, series = [], [], [], []
    wanted = [f for f in data.FUNDS if not (f == benchmark and f != my_fund)]
    loaded = load_prices_parallel(wanted, start, end)   # все фонды скачиваются одновременно
    for fund in data.FUNDS:
        if fund == benchmark and fund != my_fund:
            skipped.append(f"{fund} is the chosen benchmark, so it is not in the list.")
            continue
        try:
            result = loaded[fund]
            if isinstance(result, data.DataError):
                raise result
            prices, info = result
            series.append({"role": "Fund prices", "ticker": fund, **PRICE_SERIES, **info})
            # фонд не может быть своим бенчмарком: тогда подробной страницы нет и ссылки тоже
            link = None if fund == benchmark else try_link(dict(link_params, fund=fund))
            rows.append(fund_row(fund, prices, info, risk_free, target, target_text, link))
        except data.DataError as err:
            no_data.append({"fund": fund, "description": FUND_DESCRIPTIONS.get(fund, ""),
                            "text": no_data_text(err, end)})

    selected = None
    if my_fund != NO_FUND:
        row = next((r for r in rows if r["fund"] == my_fund), None)
        if row is not None:
            selected = dict(row, status=(
                f"This fund returned {row['cagr']} a year: as much as or more than the {target_text} you want."
                if row["meets"] else
                f"This fund returned {row['cagr']} a year: less than the {target_text} you want."))
            if row["link"] is None:
                selected["status"] += (f" {my_fund} is the chosen benchmark: choose another benchmark in "
                                       "Additional settings to open its detailed check.")
        else:
            missing = next(n for n in no_data if n["fund"] == my_fund)
            selected = {"fund": my_fund, "description": missing["description"], "status": missing["text"],
                        "no_data": True}

    others = [r for r in rows if r["fund"] != my_fund]
    matches = sorted((r for r in others if r["meets"]), key=lambda r: r["cagr_value"])  # ближе к цели - выше
    best = max(others, key=lambda r: r["cagr_value"]) if others else None
    shown = ([selected] if selected and not selected.get("no_data") else []) + matches
    infos = [s for s in series] + [rf_info]

    return {
        "target": target_text,
        "period": f"{start} to {end}",
        "used_start": start.isoformat(),
        "used_end": end.isoformat(),
        "selected": selected,
        "matches": matches,
        "best": best,
        "all_rows": rows,
        "no_data": [n for n in no_data if n["fund"] != my_fund],
        "skipped": skipped,
        "deep": {
            "general": PICK_DEEP_GENERAL + [
                f"Risk-free rate: average of FRED {data.RF_SERIES} over the selected period, "
                f"{fmt_pct(core.risk_free_annual(risk_free.iloc[:, 0]))} per year "
                "(each fund uses the average over its own dates)."],
            "why": [f"{r['fund']}: {r['cagr']} per year, "
                    + ("in the list." if r["meets"] else f"less than {target_text}, not in the list.")
                    for r in sorted(rows, key=lambda r: -r["cagr_value"])],
            "funds": shown,
            "warning": PICK_DEEP_WARNING,
        },
        "honesty": {
            "series": series + [{**RF_SERIES_INFO, **rf_info}],
            "price_source": "Yahoo Finance (yfinance), column Adjusted Close (auto_adjust=True)",
            "rate_source": "FRED API, series DGS3MO (3-Month Treasury, % per year)",
            "requested_start": start, "requested_end": end,
            "used": [(r["fund"], *r["used"]) for r in rows],
            "n_rf_observations": len(risk_free),
            "offline": any(i["offline"] for i in infos),
        },
        "offline_messages": [i["message"] for i in infos if i["offline"]],
        "save_warnings": [i["save_warning"] for i in infos if i["save_warning"]],
    }


# ---------- Маршрут ----------

@app.route("/")
def index():
    # Если в адресе есть fund, это подробная проверка одного фонда (страница и её ссылки "Try ..." не менялись)
    if "fund" in request.args:
        return check_page()
    return one_page()


def check_page():
    """Подробная страница одного фонда (без изменений)."""
    form = read_form()
    result, error = None, None
    if request.args.get("run"):  # считаем только после нажатия кнопки
        try:
            result = build_result(*parse_form(form))
        except (ValueError, data.DataError) as err:
            error = str(err)  # понятный текст ошибки показываем на странице
        except Exception as err:  # непредвиденная ошибка: без traceback на экране
            # В лог пишем только тип ошибки: текст исключения мог бы содержать лишнее
            app.logger.error("Unexpected error: %s", type(err).__name__)
            error = ("Something unexpected went wrong while calculating. "
                     "Please check the inputs and try again.")
    return render_template("index.html", form=form, funds=data.FUNDS, benchmarks=BENCHMARKS,
                           result=result, error=error, min_date=data.DOWNLOAD_START,
                           max_date=date.today().isoformat(),
                           max_advertised=int(core.MAX_ADVERTISED_PCT))


# ---------- Rule Lab (Assignment 2): правило против бенчмарка ----------
# Все числа берутся из evidence/evidence.json, который пишет make_evidence.py. Поэтому числа на странице
# и в evidence/evidence_table.md всегда одни и те же. Здесь ничего не считается, только выбор и показ.

for _name, _fn in [("pct", make_evidence.fmt_pct), ("num", make_evidence.fmt_num), ("sr", make_evidence.fmt_sr),
                   ("interval", make_evidence.fmt_interval), ("cost", make_evidence.fmt_cost)]:
    app.add_template_filter(_fn, _name)

RESULT_CARD_PATH = make_evidence.ROOT / "RESULT_CARD.md"


def team_verdict():
    """Строка 'Verdict:' из RESULT_CARD.md, если команда её уже написала (без '...' и '[...]')."""
    if not RESULT_CARD_PATH.exists():
        return None
    for line in RESULT_CARD_PATH.read_text(encoding="utf-8").splitlines():
        text = line.strip().lstrip("-* ").replace("**", "")
        if text.lower().startswith("verdict:"):
            body = text[len("verdict:"):].strip()
            return None if not body or "…" in body or "..." in body or "[" in body else body
    return None


# ---------- Forecast: сценарии на 3, 5 и 10 лет по истории (forecast.py) ----------
# Здесь ничего не считается: загрузка через data.py, расчёт в forecast.py, тут только выбор и показ.

FORECAST_RULE = "RULE_MA10"


def forecast_assets() -> list:
    return [rule.ASSET] + [f for f in data.FUNDS if f != rule.ASSET] + [FORECAST_RULE]


def forecast_labels() -> dict:
    labels = {rule.ASSET: BENCHMARK_LABELS.get(rule.ASSET, rule.ASSET)}
    labels.update({f: f"{f}: {FUND_DESCRIPTIONS.get(f, '')}" for f in data.FUNDS})
    labels[FORECAST_RULE] = (f"Rule MA{rule.PARAM_MAIN} on {rule.ASSET} (the Rule Lab rule, "
                             f"net of {make_evidence.fmt_cost(rule.COST_DEFAULT)} per switch)")
    return labels


def read_forecast_form() -> dict:
    return {"asset": request.args.get("asset", rule.ASSET),
            "amount": request.args.get("amount", f"{forecast.AMOUNT_DEFAULT:.0f}")}


def parse_forecast_form(form: dict):
    """Проверяет форму Forecast. Возвращает (asset, amount) или бросает ValueError."""
    if form["asset"] not in forecast_assets():
        raise ValueError("Please choose a fund from the list.")
    text = form["amount"].replace(" ", "").replace(",", "").replace("$", "")
    try:
        amount = float(text)
    except ValueError:
        raise ValueError(f"Enter the amount as a number from 1 to {forecast.AMOUNT_MAX:,.0f}, e.g. 1000.") from None
    if not (1 <= amount <= forecast.AMOUNT_MAX) or math.isnan(amount):
        raise ValueError(f"Enter the amount as a number from 1 to {forecast.AMOUNT_MAX:,.0f}, e.g. 1000.")
    return form["asset"], amount


def forecast_returns(asset: str, today: date):
    """Месячные доходности для сценариев и сведения об источниках (те же функции data.py и rule.py)."""
    start = date.fromisoformat(data.DOWNLOAD_START)
    if asset == FORECAST_RULE:
        prices, price_info = data.get_prices(rule.ASSET, start, today)
        rates, rate_info = data.get_risk_free(start, today)
        table = rule.backtest(rule.monthly_prices(prices[rule.ASSET]), rule.monthly_rate(rates.iloc[:, 0]),
                              rule.PARAM_MAIN)
        return rule.net_returns(table, rule.COST_DEFAULT), [price_info, rate_info]
    prices, info = data.get_prices(asset, start, today)
    return forecast.monthly_returns(prices[asset]), [info]


def fmt_money(value: float) -> str:
    return f"{value:,.0f} USD"


# ---------- Одна страница: подбор, итог, Forecast, Rule Lab и "How we calculate" ----------
# Расчёты - в core.py, rule.py и forecast.py; данные - data.py; числа Rule Lab - из evidence/evidence.json.
# Здесь только сборка данных для экрана и проверка формы.

ADD_MAX = 10_000_000.0
WARN_TARGET_PCT = 20.0          # от этого желаемого % в год показываем предупреждение
FIVE_YEARS = 5


def parse_money(text: str, low: float, high: float, what: str) -> float:
    cleaned = str(text).replace(" ", "").replace(",", "").replace("$", "")
    try:
        value = float(cleaned) if cleaned else 0.0
    except ValueError:
        value = float("nan")
    if math.isnan(value) or not (low <= value <= high):
        raise ValueError(f"Enter the {what} as a number from {low:,.0f} to {high:,.0f} USD, e.g. "
                         f"{'1000' if low >= 1 else '100'}.")
    return value


def read_one_form() -> dict:
    form = read_pick_form()
    form["amount"] = request.args.get("amount", f"{forecast.AMOUNT_DEFAULT:,.0f}")
    form["add"] = request.args.get("add", "0")
    return form


def forecast_block(assets: list, pct: float, amount: float, add: float, today: date) -> dict:
    """Сценарии на 3, 5 и 10 лет для каждого фонда из списка (и правила MA10), с шансом получить pct% в год."""
    labels, items, errors = forecast_labels(), [], []
    for asset in assets:
        try:
            returns, infos = forecast_returns(asset, today)
            if len(returns) < forecast.MIN_MONTHS:
                raise data.DataError(f"too little history ({len(returns)} months)")
            sc = forecast.scenarios(returns, amount, add, pct)
        except data.DataError as err:
            errors.append(f"{asset}: {err}")
            continue
        items.append({"asset": asset, "name": "Rule MA10" if asset == FORECAST_RULE else asset,
                      "label": labels.get(asset, asset), "history": forecast.history_summary(returns),
                      "sources": [{k: str(i[k]) for k in ("file", "pull_date", "offline", "message")} for i in infos],
                      **sc})
    return {"pct": pct, "amount": amount, "add": add, "items": items, "errors": errors,
            "horizons": forecast.HORIZONS, "paths": forecast.PATHS, "block": forecast.BLOCK, "seed": forecast.SEED}


def summary_block(pick: dict, fc: dict, pct: float) -> dict:
    """Карточка-итог и предупреждение о слишком высоком проценте."""
    found = pick["matches"]
    chance5 = {i["asset"]: next(r["chance_target"] for r in i["rows"] if r["years"] == FIVE_YEARS) for i in fc["items"]}
    chance10 = {i["asset"]: next(r["chance_target"] for r in i["rows"] if r["years"] == max(forecast.HORIZONS))
                for i in fc["items"]}
    best = found[0] if found else None
    summary = None
    if best is not None:
        summary = {"fund": best["fund"], "description": best["description"], "cagr": best["cagr"],
                   "chance5": chance5.get(best["fund"]), "max_drawdown": best["max_drawdown"]}
    warning = None
    all_rows = pick["all_rows"]
    if pct >= WARN_TARGET_PCT and all_rows:
        vols = sorted(r["volatility_value"] for r in all_rows)
        median_vol = vols[len(vols) // 2]
        risky = found and all(r["volatility_value"] >= median_vol for r in found)
        n = len(found)
        best10 = max((chance10[r["fund"]] for r in found if r["fund"] in chance10), default=None)
        warning = {
            "pct": f"{pct:g}%", "n": n, "total": len(all_rows),
            "risky": bool(risky), "plural": n != 1,
            "best10": None if best10 is None else make_evidence.fmt_pct(best10, 0),
        }
    return {"summary": summary, "warning": warning}


def rule_lab_block():
    """Числа Rule Lab из evidence/evidence.json (тот же файл, что evidence_table.md)."""
    evidence, error = make_evidence.load_evidence(), None
    if evidence is None:  # первый запуск: доказательства генерирует тот же скрипт
        try:
            evidence = make_evidence.run()
        except (make_evidence.EvidenceError, data.DataError) as err:
            error = str(err)
        except Exception as err:  # непредвиденная ошибка: без traceback на экране
            app.logger.error("Unexpected error: %s", type(err).__name__)
            error = "Something unexpected went wrong while generating the evidence. Run python make_evidence.py to see why."
    return evidence, error


@app.route("/api/forecast")
def api_forecast():
    """Пересчёт Forecast при вводе суммы и пополнения (без перезагрузки страницы)."""
    try:
        pct = core.parse_advertised_pct(request.args.get("target", ""))
        amount = parse_money(request.args.get("amount", ""), 1, forecast.AMOUNT_MAX, "amount")
        add = parse_money(request.args.get("add", "0"), 0, ADD_MAX, "monthly top-up")
        assets = [a for a in request.args.get("assets", "").split(",") if a in forecast_assets()]
        if not assets:
            raise ValueError("Please choose a fund from the list.")
        return jsonify(forecast_block(assets, pct, amount, add, date.today()))
    except (ValueError, data.DataError) as err:
        return jsonify({"error": str(err)}), 400


@app.route("/api/fund")
def api_fund():
    """Подробная проверка одного фонда для раскрывающейся строки таблицы (HTML-фрагмент)."""
    try:
        fund = request.args.get("fund", "")
        benchmark = request.args.get("benchmark", data.DEFAULT_BENCHMARK)
        if fund not in data.FUNDS:
            raise ValueError("Please choose a fund from the list.")
        if benchmark not in BENCHMARKS:
            raise ValueError("Please choose a benchmark from the list.")
        start, end = date.fromisoformat(request.args["start"]), date.fromisoformat(request.args["end"])
        target = core.parse_advertised_pct(request.args.get("target", "0"))
        result = build_result(fund, benchmark, start, end, target)
        return render_template("_fund_detail.html", r=result, benchmarks=BENCHMARKS, labels=BENCHMARK_LABELS)
    except (KeyError, ValueError, data.DataError) as err:
        return f'<div class="error">{escape(str(err))}</div>', 200


def one_page():
    form = read_one_form()
    pick = fc = top = None
    error = None
    if request.args.get("target") is not None:  # считаем только после нажатия кнопки
        try:
            target_pct, my_fund, benchmark, start, end = parse_pick_form(form)
            amount = parse_money(form["amount"], 1, forecast.AMOUNT_MAX, "amount")
            add = parse_money(form["add"], 0, ADD_MAX, "monthly top-up")
            pick = pick_funds(target_pct, my_fund, benchmark, start, end)
            shown = ([pick["selected"]["fund"]] if pick["selected"] and not pick["selected"].get("no_data") else []) \
                + [r["fund"] for r in pick["matches"]]
            fc = forecast_block(shown + [FORECAST_RULE], target_pct, amount, add, date.today())
            top = summary_block(pick, fc, target_pct)
        except (ValueError, data.DataError) as err:
            error = str(err)
        except Exception as err:  # непредвиденная ошибка: без traceback на экране
            app.logger.error("Unexpected error: %s", type(err).__name__)
            error = ("Something unexpected went wrong while calculating. "
                     "Please check the inputs and try again.")
    evidence, evidence_error = rule_lab_block()
    return render_template(
        "one.html", form=form, funds=data.FUNDS, fund_labels=FUND_DESCRIPTIONS, no_fund=NO_FUND,
        result=pick, fc=fc, top=top, error=error, min_date=data.DOWNLOAD_START,
        max_date=date.today().isoformat(), max_target=int(core.MAX_ADVERTISED_PCT),
        ev=evidence, ev_error=evidence_error, rule=rule, cost_keys=[make_evidence.cost_key(c) for c in rule.COSTS],
        costs=rule.COSTS, cost_key=make_evidence.cost_key, team_verdict=team_verdict(),
        default_cost=make_evidence.cost_key(rule.COST_DEFAULT), forecast=forecast, benchmarks=BENCHMARKS,
        benchmark_labels=BENCHMARK_LABELS, default_benchmark=data.DEFAULT_BENCHMARK, deep_general=PICK_DEEP_GENERAL)


if __name__ == "__main__":
    enable_download_cache()
    print("Preloading prices in the background: the first search will be fast in about a minute.")
    threading.Thread(target=warm_up, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
