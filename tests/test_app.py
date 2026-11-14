"""Тесты страницы (app.py) через тестовый клиент Flask. Интернет НЕ нужен.

Загрузка данных подменена: вместо Yahoo и FRED отдаются synthetic (искусственные) ряды, а папка data/raw
временная. Каждый сценарий сначала проверяет вердикт через core, поэтому понятно, что именно тестируется.
Здесь проверяем, что панели Deep analysis появляются при нужном результате, что ссылки "Try ..." рабочие,
что предупреждение про cherry-picking есть всегда и что пустые значения (None) не роняют страницу.
"""
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

import app as appmod
import core
import data

N = 400                                              # synthetic: 400 торговых доходностей
INDEX = pd.bdate_range("2023-01-02", periods=N + 1)
END = INDEX[-1].date().isoformat()
RF_PERCENT = 4.5


def make_bench():
    # synthetic: среднее ровно 0.001 в день (25.2% в год), шум с нулевым средним: Sharpe бенчмарка положительный
    noise = np.random.RandomState(3).normal(0, 0.006, N)
    return 0.001 + noise - noise.mean()


BENCH = make_bench()


def prices_from(returns):
    # synthetic: цены из доходностей, старт 100
    return pd.DataFrame({"p": 100 * np.concatenate([[1.0], np.cumprod(1 + np.asarray(returns))])}, index=INDEX)


def fund_returns(kind):
    """synthetic: доходности фонда для сценариев (бенчмарк всегда BENCH)."""
    if kind == "red_with_green_windows":      # плохое начало, хороший конец: общий итог красный
        return BENCH + np.r_[np.full(240, -0.003), np.full(N - 240, 0.003)]
    if kind == "green_with_red_windows":      # хорошее начало, плохой конец: общий итог зелёный
        return BENCH + np.r_[np.full(240, 0.003), np.full(N - 240, -0.003)]
    if kind == "red_no_windows":              # чуть хуже бенчмарка всё время: зелёных окон нет
        return BENCH - 0.0003
    if kind == "green_too_close":             # чуть лучше бенчмарка: разница Sharpe меньше 0.10
        return BENCH + 0.00002
    raise ValueError(kind)


def install(monkeypatch, tmp_path, fund, bench=BENCH):
    """Подменяет загрузку: ARKK = fund, всё остальное = bench; ставка 4.5%."""
    def download(ticker):
        returns = fund if ticker == "ARKK" else bench
        return prices_from(returns).rename(columns={"p": ticker})
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data, "_download_prices", download)
    monkeypatch.setattr(data, "_download_risk_free",
                        lambda: pd.DataFrame({data.RF_SERIES: np.full(N + 1, RF_PERCENT)}, index=INDEX))


def verdict_of(fund, bench=BENCH):
    # вердикт считаем напрямую через core: так сценарий проверяет сам себя
    f, b = pd.Series(np.asarray(fund)), pd.Series(np.asarray(bench))
    rf = RF_PERCENT / 100
    return core.verdict_code(core.justifies_risk(core.sharpe_ratio(f, rf), core.sharpe_ratio(b, rf)))


def get_page(**changes):
    query = {"fund": "ARKK", "benchmark": "SPY", "start": "2023-01-02", "end": END, "advertised": "35", "run": "1"}
    query.update(changes)
    response = appmod.app.test_client().get("/", query_string=query)
    assert response.status_code == 200
    return response.get_data(as_text=True)


def text_of(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


CHERRY = ("These are historical what-ifs, not a forecast. Choosing a window only because it turns the verdict "
          "green is cherry-picking; use this to see how fragile the verdict is, not to pick a result.")


# ---------- Красный результат ----------

def test_red_result_shows_honest_and_missing_panels_with_links(tmp_path, monkeypatch):
    fund = fund_returns("red_with_green_windows")
    assert verdict_of(fund) == "does_not_justify"
    install(monkeypatch, tmp_path, fund)
    html = get_page()
    assert "What this means" in html and "What would be honest" in html
    assert "What is missing to justify the risk" in html and "How solid is this" not in html
    text = text_of(html)
    assert "An advertised return of up to" in text and "You entered 35%" in text
    assert "to match SPY" in text.replace("To match", "to match") and "percentage points more per year" in text
    assert "Nearest = the smallest change of the date window that turns the verdict green." in text
    # ссылка "Try X%": те же параметры формы, но advertised изменён
    honest_link = re.search(r'href="(/\?[^"]*advertised=(?!35&)[^"&]+[^"]*)">Try [\d.]+%', html.replace("&amp;", "&"))
    assert honest_link and "fund=ARKK" in honest_link.group(1) and "start=2023-01-02" in honest_link.group(1)
    # ссылка "Try this window": меняются start/end, реклама прежняя
    window_links = [l.replace("&amp;", "&") for l in re.findall(r'href="(/\?[^"]+)">Try this window', html)]
    assert window_links and all("advertised=35" in l and "run=1" in l for l in window_links)
    assert all("start=2023-01-02&end=" + END not in l for l in window_links)   # окно действительно другое


def test_try_links_open_valid_result_pages(tmp_path, monkeypatch):
    fund = fund_returns("red_with_green_windows")
    install(monkeypatch, tmp_path, fund)
    html = get_page().replace("&amp;", "&")
    links = re.findall(r'href="(/\?[^"]+)">Try', html)
    assert len(links) >= 2
    client = appmod.app.test_client()
    for link in links:
        page = client.get(link).get_data(as_text=True)
        assert 'class="error"' not in page and "Deep analysis" in page     # ссылка проходит обычную проверку формы
    # ссылка на окно даёт ровно ту картину, что обещана: вердикт в этом окне зелёный
    window_link = re.findall(r'href="(/\?[^"]+)">Try this window', html)[0]
    assert "Fund justifies its risk" in client.get(window_link).get_data(as_text=True)


def test_red_without_green_windows_says_so_in_plain_text(tmp_path, monkeypatch):
    fund = fund_returns("red_no_windows")
    assert verdict_of(fund) == "does_not_justify"
    install(monkeypatch, tmp_path, fund)
    text = text_of(get_page())
    assert "No window inside the selected period (at least 126 trading days) turns the verdict green." in text


# ---------- Зелёный результат ----------

def test_green_result_shows_how_solid_panel_with_red_windows(tmp_path, monkeypatch):
    fund = fund_returns("green_with_red_windows")
    assert verdict_of(fund) == "justifies"
    install(monkeypatch, tmp_path, fund)
    html = get_page(advertised="5")
    text = text_of(html)
    assert "How solid is this" in html and "What is missing to justify the risk" not in html
    assert "What would be honest" not in html          # реклама 5% ниже реальной доходности: приукрашивания нет
    assert "A margin of 0.10 or more is treated as clear" in text
    assert "Nearest date windows where the verdict would turn red" in text
    assert re.search(r'href="/\?[^"]+">Try this window', html)


def test_too_close_green_is_marked_too_close_to_call(tmp_path, monkeypatch):
    fund = fund_returns("green_too_close")
    assert verdict_of(fund) == "justifies"
    install(monkeypatch, tmp_path, fund)
    text = text_of(get_page())
    assert "too close to call" in text.lower()


# ---------- Всегда ----------

def test_cherry_picking_warning_is_always_present(tmp_path, monkeypatch):
    flat = np.full(N, 0.001)                            # synthetic: бенчмарк не меняется -> Sharpe не посчитать
    cases = [("red_with_green_windows", BENCH), ("green_with_red_windows", BENCH),
             ("red_with_green_windows", flat)]
    for kind, bench in cases:
        install(monkeypatch, tmp_path, fund_returns(kind), bench=bench)
        assert CHERRY in text_of(get_page()), kind


def test_none_values_do_not_crash_the_page(tmp_path, monkeypatch):
    # synthetic: бенчмарк не меняется: Sharpe и бета не определены (None) -> вердикта нет, но страница работает
    flat = np.full(N, 0.001)
    install(monkeypatch, tmp_path, fund_returns("red_with_green_windows"), bench=flat)
    html = get_page()
    text = text_of(html)
    assert "Deep analysis" in html and "could not be calculated" in text
    assert "What is missing to justify the risk" not in html and "How solid is this" not in html
    assert "Traceback" not in html and 'class="error"' not in html
    # реклама приукрашена, но при отрицательном CAGR честной цифры нет: обычный текст без ссылки
    install(monkeypatch, tmp_path, np.full(N, -0.001), bench=BENCH)
    text = text_of(get_page())
    assert "no advertised return of 0% or more would match what it delivered" in text


def test_analysis_panels_are_neutral_not_red():
    # красный цвет остаётся только у главного числа и вердикта: у панелей анализа свои нейтральные цвета
    css = open("templates/index.html", encoding="utf-8").read()
    block = css[css.index(".panel {"): css.index("/* ---- Honesty panel")]
    assert "var(--bad" not in block and "var(--info" in block


# ---------- Фонды, бенчмарки, даты ----------

def test_fund_and_benchmark_lists_and_hint():
    for ticker in ["ARKK", "QQQ", "ICLN", "ARKW", "ARKG", "SMH", "TAN", "LIT", "GLD", "SCHD", "EEM"]:
        assert ticker in data.FUNDS
    assert appmod.BENCHMARKS == ["SPY", "QQQ", "IWM", "ACWI", "AGG"]
    html = appmod.app.test_client().get("/?fund=ARKK").get_data(as_text=True)   # подробная страница фонда
    assert "Pick a benchmark that fits the fund (e.g. GLD vs SPY gives a beta near zero)." in html


def test_same_ticker_in_both_lists_gives_clear_error_before_any_download(tmp_path, monkeypatch):
    assert "QQQ" in data.FUNDS and "QQQ" in appmod.BENCHMARKS      # QQQ есть в обоих списках

    def must_not_download(ticker):
        raise AssertionError("no download is expected: the tickers are the same")
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data, "_download_prices", must_not_download)
    html = get_page(fund="QQQ", benchmark="QQQ")
    assert "must be different tickers: you chose QQQ for both" in text_of(html)


def test_date_minimum_comes_from_download_start(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path, fund_returns("red_no_windows"))
    html = appmod.app.test_client().get("/").get_data(as_text=True)
    assert f'min="{data.DOWNLOAD_START}"' in html and data.DOWNLOAD_START == "2000-01-01"
    # раньше минимума: понятная ошибка с этой датой (старый минимум 2015 больше не действует)
    assert "cannot be earlier than 2000-01-01" in text_of(get_page(start="1999-12-31"))
    assert "cannot be earlier than 2000-01-01" not in text_of(get_page(start="2005-03-01"))


def test_shortened_window_is_shown_on_page_and_in_honesty_panel(tmp_path, monkeypatch):
    # synthetic: данные начинаются 2023-01-02, а пользователь выбрал старт 2005-03-01
    install(monkeypatch, tmp_path, fund_returns("red_no_windows"))
    html = get_page(start="2005-03-01")
    text = text_of(html)
    assert "Shorter window than you chose." in text
    assert "Data for ARKK starts on 2023-01-02, later than the start date you chose (2005-03-01)" in text
    assert "Window you chose: 2005-03-01 to " in text and "Window actually used: 2023-01-02 to " in text
    assert "The window was shortened by" in text


# ---------- Главная страница: подбор фондов ----------

# synthetic: у каждого фонда своя постоянная средняя доходность в день (плюс общий шум бенчмарка)
PICK_DAILY = {"ARKK": 0.0012, "SMH": 0.0009, "QQQ": 0.0006, "GLD": 0.0001}   # остальные: 0.00005 в день
LATE_START = "2026-01-05"    # synthetic: этот фонд появился после конца периода (данных за период нет)


def install_pick(monkeypatch, tmp_path, late_fund="TLT"):
    def download(ticker):
        if ticker == late_fund:
            return pd.DataFrame({ticker: np.linspace(100, 110, 50)}, index=pd.bdate_range(LATE_START, periods=50))
        returns = BENCH - BENCH.mean() + PICK_DAILY.get(ticker, 0.00005)
        return prices_from(returns).rename(columns={"p": ticker})
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data, "_download_prices", download)
    monkeypatch.setattr(data, "_download_risk_free",
                        lambda: pd.DataFrame({data.RF_SERIES: np.full(N + 1, RF_PERCENT)}, index=INDEX))


def get_pick(**changes):
    query = {"target": "15", "start": "2023-01-02", "end": END}
    query.update(changes)
    response = appmod.app.test_client().get("/", query_string=query)
    assert response.status_code == 200
    return response.get_data(as_text=True)


def pick_text(html):
    return text_of(html).replace("&#39;", "'")      # апостроф в HTML экранирован


def matched_funds(html):
    table = html.split('id="matches"')[1].split("</table>")[0] if 'id="matches"' in html else ""
    return re.findall(r'<a href="[^"]+">([A-Z]+)</a>', table)


def test_target_limit_0_to_1000_on_server_and_in_html(tmp_path, monkeypatch):
    install_pick(monkeypatch, tmp_path)
    html = appmod.app.test_client().get("/").get_data(as_text=True)
    assert 'min="0"' in html and 'max="1000"' in html and 'step="any"' in html
    for bad in ["-1", "1000.5", "1500", "abc", ""]:
        assert "Enter a number from 0 to 1000 (% per year)" in text_of(get_pick(target=bad)), bad
    for good in ["0", "1000", "12.5", "12,5"]:
        assert 'class="error"' not in get_pick(target=good), good


def test_without_fund_shows_all_matches_sorted_closest_first(tmp_path, monkeypatch):
    install_pick(monkeypatch, tmp_path)
    html = get_pick(my_fund="")
    assert "Your selected fund" not in html
    assert matched_funds(html) == ["QQQ", "SMH", "ARKK"]           # от ближней к цели доходности к дальней
    assert "Deep analysis" in html and "Honesty panel" in html


def test_selected_fund_goes_first_and_is_not_repeated(tmp_path, monkeypatch):
    install_pick(monkeypatch, tmp_path)
    html = get_pick(my_fund="SMH")
    assert html.index('id="selected-fund"') < html.index('id="matches"')
    card = html.split('id="selected-fund"')[1].split("</section>")[0]
    assert "Your selected fund" in card and ">SMH</a>" in card
    assert matched_funds(html) == ["QQQ", "ARKK"]                  # SMH в общем списке не повторяется


def test_selected_fund_below_target_is_shown_with_clear_text(tmp_path, monkeypatch):
    install_pick(monkeypatch, tmp_path)
    card = text_of(get_pick(my_fund="GLD").split('id="selected-fund"')[1].split("</section>")[0])
    assert "Your selected fund" in card and "GLD" in card and "less than the 15% you want" in card


def test_unknown_selected_fund_gives_clear_error(tmp_path, monkeypatch):
    install_pick(monkeypatch, tmp_path)
    assert "Please choose a fund from the list." in text_of(get_pick(my_fund="BITO"))


def test_fund_without_data_in_period_is_listed_with_its_start_date(tmp_path, monkeypatch):
    install_pick(monkeypatch, tmp_path, late_fund="TLT")
    html = get_pick()
    block = pick_text(html.split('id="no-data"')[1].split("</section>")[0])
    assert "Funds without data for this period" in html
    assert "TLT" in block and f"No data: the fund's history starts on {LATE_START}." in block
    # выбранный фонд без данных всё равно показан первым, с тем же текстом
    card = pick_text(get_pick(my_fund="TLT").split('id="selected-fund"')[1].split("</section>")[0])
    assert f"No data: the fund's history starts on {LATE_START}." in card


def test_fund_with_late_start_is_calculated_from_its_first_date(tmp_path, monkeypatch):
    install_pick(monkeypatch, tmp_path)
    text = text_of(get_pick(start="2022-06-01", target="0", my_fund="ARKK"))
    assert "Data only from 2023-01-02; the calculation starts on that date." in text
