# Fund Claim Fact-Checker

This app checks whether a fund's advertised return (16 funds to choose from, see "Series offered") is real once it is compared with a benchmark (SPY by default, 5 to choose from) and adjusted for risk through the Sharpe ratio, beta and CAPM alpha.
It runs locally on your computer: you enter the advertised return, and the page shows one headline number (advertised minus actual return, in percentage points per year), a chart, a comparison table, a "Deep analysis" of what is missing to turn a red result green, and an honesty panel.

## Install and run (about 10 minutes)

Python 3.9 or newer is required. In the Terminal, go to the project folder and run:

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Open the `.env` file and replace `your_fred_api_key_here` with your free FRED API key
(get one at https://fred.stlouisfed.org: My Account -> API Keys). The `.env` file is listed in `.gitignore`, so the key is never committed. Never paste a key into the code or into this README.

Then start the app (one command):

```
python3 app.py
```

Open http://127.0.0.1:5000 in your browser. While the site runs, it preloads prices in the background, downloads up to 4 tickers at a time and reuses a download for one hour, so after the first minute searches take seconds (sources and offline mode are unchanged). Everything is on **one page** (dark design), split into sections that open when you click their heading. At the top you enter the yearly return you want (a number from 0 to 1000, % per year, decimals allowed; checked in the browser and on the server), the amount in USD (any number) and an optional monthly top-up, then click "Find funds". Under "Additional settings" you can choose the period and, optionally, one fund: it is shown first as "Your selected fund" even if it gave less than you want. The sections are: **Funds** (open by default; click a fund to open its full check right in the list, with its own "Compare with" benchmark), **Forecast** (the chance of keeping your % for 3, 5 and 10 years for every fund found and for the Rule MA10, plus a fan chart; it recalculates as you type the amount or top-up), **Rule Lab** (Assignment 2, see below) and **How we calculate** (tabs "Deep analysis" and "Honesty panel"). All charts are interactive: hover or tap to see the values, click a legend item to hide or show a line. A summary card "Best match" shows the fund closest to your target, and a warning appears for targets of 20% a year or more. Funds with no data for the period are listed under "Funds without data for this period" with the date their history starts. **Extra tools on the page:** the fund list also shows the return **after inflation** (US consumer prices from FRED, series CPIAUCSL, over the same dates) and each fund's **yearly fee** (expense ratio; it is already inside the price, so returns are after it). The Forecast section has **"How much to add every month?"**: enter a goal and a number of years, and the site gives the monthly top-up for a 50%, 75% and 90% chance, using the same 10,000 scenarios. **Mix funds** compares a mix of 2-3 funds (shares in %, back to those shares at the start of every month) with each fund alone. **Download (CSV)** saves the fund table and the forecast for Excel or Google Sheets. If the inflation data cannot be loaded, the page still works and says so.

The **first run needs internet** (it downloads and saves the data); after that the app can also work offline. To stop the app press Ctrl+C.

## Rule Lab (Assignment 2): would a simple rule have helped?

The rule: at the end of each month, if SPY closes above its average of the last 10 monthly closes, hold SPY next month;
otherwise hold cash that earns the 3-month Treasury rate (FRED). Benchmark: buy and hold SPY, same months.
Settings, samples and costs are the constants at the top of `rule.py`; the full description is in `RULE_CARD.md`.

```
python make_evidence.py               # evidence for the development sample (train + validation); test stays closed
python make_evidence.py --open-test   # ONCE: open the test period 2019-01 to 2026-10 (recorded in DECISIONS.md)
python app.py                         # then open http://127.0.0.1:5000 and open the "Rule Lab" section
```

`make_evidence.py` writes `evidence/evidence.json`, `evidence/evidence_table.md` (E1–E8), `evidence/result_card_numbers.md`
and `evidence/figures/`, and appends one line per configuration and sample to `trials.csv` (lines are never deleted).
The Rule Lab section reads `evidence/evidence.json`, so its numbers are always the same as in the evidence table;
if the evidence has never been generated, the site runs the script itself on the first visit.
Opening the test a second time needs `--open-test --reopen`; it is recorded and counted.

Files for the assignment: `RULE_CARD.md`, `RESULT_CARD.md`, `trials.csv`, `evidence/`, `REUSE_CHANGES.md`,
`DECISIONS.md`, `AI_USE.md`, `CONTRIBUTIONS.md`.

## Run the tests

```
pytest -v
```

155 tests (58 in `tests/test_core.py`, 23 in `tests/test_data.py`, 19 in `tests/test_app.py`, 16 in `tests/test_rule.py`, 9 in `tests/test_evidence.py`, 9 in `tests/test_forecast.py`, 21 in `tests/test_extras.py`). No internet is needed: `tests/conftest.py` replaces the inflation download with a synthetic series.
The toy test in `tests/test_rule.py` is skipped until the team writes its hand-computed values into `TOY_EXPECTED`; after that all 155 must pass.
`tests/test_rule.py` tests the rule core: the toy example, two canary tests for look-ahead (future prices must not change past positions;
on a random walk the honest rule has no edge, while a version without the one-month lag does), and known answers for costs, drawdown, cash, CAGR,
the standard error of Sharpe, the bootstrap, break-even and the expected best-of-noise Sharpe.
`tests/test_evidence.py` runs `make_evidence.py` on synthetic data: the test period stays closed until opened, opens once and is recorded,
`trials.csv` only grows, and the Rule Lab shows the same numbers as the evidence table.
`tests/test_core.py` tests the formulas of `core.py`: 20 tests compare the result with hand-calculated values (the `HAND_...` constants at the top, with the calculation steps in the comments) and the rest use data labelled `synthetic`. It includes the Deep analysis functions (honest advertised limit, Sharpe shortfall, closeness, nearest date windows) and the date-window rules.
`tests/test_data.py` tests offline behaviour, error handling and the shortened-window warnings of `data.py` with replaced (fake) downloads.
`tests/test_app.py` opens the page with Flask's test client on synthetic data: red and green results show the right Deep analysis panels, the "Try ..." links work, the cherry-picking warning is always there and empty (None) values do not crash the page.

## How the project is organized

| File | Responsibility |
|---|---|
| `data.py` | All data loading (Yahoo Finance, FRED), saving to `data/raw/`, offline fallback, join counts. No math. |
| `core.py` | Pure functions: returns, volatility, Sharpe, beta, CAPM, alpha, gap, verdict, input validation. No internet, no files, no screen. |
| `app.py` | The screen only (Flask). Calls `data.py` and `core.py`, calculates nothing itself; only turns numbers into text. |
| `extras.py` | Extra calculations: return after inflation (FRED CPIAUCSL), the monthly top-up needed to reach a goal, a mix of 2-3 funds rebalanced every month, fund fees (expense ratios). |
| `forecast.py` | Forecast scenarios: block bootstrap of past monthly returns (10,000 paths, 12-month blocks, fixed seed), monthly top-ups, chance to reach the target. |
| `templates/one.html`, `templates/_fund_detail.html` | The one-page site and the full check of one fund that opens inside the list. |
| `static/one.css`, `static/one.js` | Dark design; interactive charts and live forecast (the JavaScript only draws and asks the server, it calculates nothing). |
| `templates/index.html` | The detailed check page (opened with `?fund=...`): form, headline with the "Why this number?" panel, chart, tables, Deep analysis, honesty panel. |
| `tests/` | Tests for `core.py`, `data.py` and the page (`app.py`). |
| `data/raw/` | Saved downloads with the pull date in the file name (never edit by hand). |

## Series offered

Funds (the "Fund" list in `data.py`, `FUNDS`). The descriptions are short labels, check the issuer's page for details. The first date of each series is not listed here: it is read from the data, see the Honesty panel (and the "Shorter window" message, see "History from 2000").

| Ticker | What it is | Benchmark hint | Data start |
|---|---|---|---|
| ARKK | ARK Innovation ETF (actively managed, thematic) | QQQ or SPY | see Honesty panel |
| QQQ | Invesco QQQ (Nasdaq-100 index fund) | SPY | see Honesty panel |
| ICLN | iShares Global Clean Energy ETF | ACWI | see Honesty panel |
| ARKW | ARK Next Generation Internet ETF | QQQ | see Honesty panel |
| ARKG | ARK Genomic Revolution ETF | SPY or IWM | see Honesty panel |
| SMH | VanEck Semiconductor ETF | QQQ | see Honesty panel |
| TAN | Invesco Solar ETF | SPY or IWM | see Honesty panel |
| LIT | Global X Lithium & Battery Tech ETF | SPY or ACWI | see Honesty panel |
| GLD | SPDR Gold Shares (gold) | SPY: beta will be near zero, gold is a diversifier, not a rival of stocks | see Honesty panel |
| SCHD | Schwab U.S. Dividend Equity ETF | SPY | see Honesty panel |
| EEM | iShares MSCI Emerging Markets ETF | ACWI | see Honesty panel |
| VTI | Vanguard Total Stock Market ETF (the whole US market) | SPY | see Honesty panel |
| VUG | Vanguard Growth ETF (large US growth companies) | SPY or QQQ | see Honesty panel |
| XLV | Health Care Select Sector SPDR Fund (US healthcare) | SPY | see Honesty panel |
| VNQ | Vanguard Real Estate ETF (US REITs) | SPY | see Honesty panel |
| TLT | iShares 20+ Year Treasury Bond ETF (long US government bonds) | AGG | see Honesty panel |

Benchmarks (`BENCHMARKS` in `app.py`): SPY (default, S&P 500), QQQ (Nasdaq-100), IWM (US small companies), ACWI (world stocks), AGG (US bonds). A benchmark should fit the fund: a bad match (for example a gold fund against the S&P 500) still gives numbers, but the comparison says little. A fund cannot be its own benchmark (QQQ is in both lists): the app asks you to pick another one.

## History from 2000

The app downloads the whole history from 2000-01-01 (`DOWNLOAD_START` in `data.py`, the single place for this date: the validation of dates and the `min` attribute of the date fields read it) and cuts the window you choose. Many tickers are younger than 2000. This is not an error: the window starts on the first available date of the series, and the app says so: a yellow "Shorter window than you chose" message with the real first date and the number of days lost, and the Honesty panel shows the window you chose and the window actually used. The same message appears when an old saved file starts later than the date you chose (for example a file saved before the history was extended: run once with internet to download the full history). A difference of up to 7 days between your start date and the first date is treated as weekends and holidays, not as a shortened window. With fewer than 30 common trading days the app shows an error instead.

## Formulas

All returns inside `core.py` are fractions (0.05 = 5%). There are two annual returns, on purpose:

- **Arithmetic annual return** = `mean(daily returns) * 252`. Used for Sharpe, CAPM expected return and Jensen's alpha, because these formulas are built on average returns.
- **Geometric annual return (CAGR)** = `(product of (1 + daily return)) ^ (252 / number of days) - 1`. Used only for the comparison with the advertised figure, because an advertised return is normally a compounded yearly growth rate.

For a volatile fund (for example ARKK) the arithmetic number is higher than the geometric one, so both are shown on the screen with labels.

- Daily return: `P_t / P_(t-1) - 1`; cumulative return: `P_end / P_start - 1`
- Annualized volatility: `std(daily returns, ddof=1) * sqrt(252)`
- Risk-free rate: average FRED DGS3MO (3-Month Treasury, in % per year, divided by 100) over the window actually used
- Sharpe ratio (annualized): `(mean(return - rf/252) * 252) / annualized volatility` = `(arithmetic annual return - rf) / volatility`
- Beta: `covariance(fund, benchmark) / variance(benchmark)`; the benchmark's beta to itself is 1 by definition
- CAPM expected return: `rf + beta * (benchmark arithmetic annual return - rf)`
- Alpha (Jensen): `fund arithmetic annual return - CAPM expected return`
- **Headline number (gap)**: `(advertised return - fund CAGR) * 100`, in percentage points per year over the selected period. Positive: the advertisement overstates the return. Negative: the fund beat the advertisement.
- Verdict (caption): the fund "justifies its risk" if its Sharpe ratio is higher than the benchmark's

## Deep analysis (history only, not a forecast)

Under "Advertised vs reality" the page explains what is missing to turn a red result green (or how solid a green one is). All numbers come from `core.py`; `app.py` only puts them into texts and builds the links.

- **What this means**: four short paragraphs in plain English (what the Sharpe ratio is, what the gap means, why the verdict is what it is, for which period).
- **What would be honest** (when the ad is overstated): `honest_advertised_limit_pct` gives the largest advertised figure (step 0.1% per year, rounded down) that would not overstate, which is the fund's CAGR; "Try X%" reopens the page with it. If the CAGR is negative, no figure of 0% or more would be honest and the page says so.
- **What is missing to justify the risk** (red verdict): `sharpe_shortfall` shows the Sharpe gap, the arithmetic return the fund would have needed at its own volatility (`rf + Sharpe_benchmark * volatility`, "at least": at exactly that figure the Sharpe ratios only tie) and the volatility that would have been enough at the same return (`(return - rf) / Sharpe_benchmark`). `nearest_green_windows` lists the nearest date windows inside the loaded period where the verdict is green, with "Try this window" links.
- **How solid is this** (green verdict): the Sharpe margin, a "too close to call" mark (`closeness`), and the nearest windows where the verdict would turn red.
- A warning is always shown: *these are historical what-ifs, not a forecast; choosing a window only because it turns the verdict green is cherry-picking; use this to see how fragile the verdict is, not to pick a result.*

Three thresholds are **our own decisions, not formulas from the course**: a Sharpe difference below **0.10** is "too close to call" (`FRAGILE_SHARPE_DIFF`); windows shorter than **126 trading days** (about half a year) are not considered (`WINDOW_MIN_DAYS`); the window moves in steps of **21 trading days** (about a month, `WINDOW_STEP_DAYS`). In the window search the Sharpe ratio is recalculated for every window, with the risk-free rate averaged over that window; the search only moves the start forward or the end backward inside the period you already loaded, and never downloads anything.
The red colour stays on the headline number and the verdict; the analysis panels are neutral blue.

## Offline mode

Every successful download is saved to `data/raw/` as `prices_<ticker>_<date>.csv` and `rf_DGS3MO_<date>.csv` (the date is the pull date; a second run on the same day overwrites that day's file). Prices are Adjusted Close (`auto_adjust=True`) with the time zone removed; nothing else is changed.

When a download fails, the app uses the latest saved file and the page says **why** and **which file (and pull date)** it used. The three cases are reported separately:

1. No internet connection: "No internet connection: ... Using the saved file ...".
2. The source answered with an error (invalid FRED key, request limit, empty answer): the message names the problem and says that saved data are used.
3. `FRED_API_KEY` is not set in `.env`: the message says so (the internet may be fine).

If there is no saved file and the download is impossible, the app shows an instruction, for example: "Add your FRED key to .env and run once with internet." A failure to write a file to disk is not treated as a network problem: the fresh data are still used and a warning is shown.

**To test offline mode:** run one check with internet, switch the network off, restart `python3 app.py` and click "Check": the yellow "Using saved data" box appears.

## What is new vs. the Week 4 X-Ray starter

*(placeholder: to be filled in by the team)*

- Reused from the Week 4 Portfolio Risk X-Ray: ... (for example the layout of three layers, the volatility and returns functions, the data-saving idea)
- New in this project: ... (Sharpe ratio against a risk-free rate, beta, CAPM alpha, advertised-vs-actual gap, input validation, offline-mode handling, honesty panel)

## Limitations

- Data starts on 2000-01-01; an earlier period cannot be selected. Many funds are younger than 2000: the window then starts on the fund's first date and the page says so (see "History from 2000"); the results only describe that shorter window.
- Deep analysis windows are what-ifs on the history you loaded; they depend on the team's thresholds (0.10, 126 days, 21-day step) and show how fragile a verdict is, not which period to pick.
- The advertised return is compared with the return over the period you select; the ad may refer to a different period.
- The funds in the list exist today (survivorship): funds closed during the period are not included.
- Yahoo Finance is an unofficial source; the saved raw files let you re-check the numbers.
- Historical data. Not a forecast. Not investment advice.
