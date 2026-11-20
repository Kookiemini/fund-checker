# Reuse and changes (Assignment 1 → Assignment 2)

## Reused from Assignment 1
- Assignment 1 was not kept in Git; this repository was created for Assignment 2 from the Assignment 1 code (fund-checker_1), so the history starts here.
- **Data loader `data.py`**: `get_prices` (Yahoo Finance, Adjusted Close) and `get_risk_free` (FRED DGS3MO),
  the saved files in `data/raw/`, the offline fallback and its messages, `DataError`. Used as is by `make_evidence.py`.
- **Flask app `app.py`** and the page style (colours, cards, the honesty panel layout): the Rule Lab is a new page of the same app.
- **Test set-up**: `pytest` with one command, synthetic data, the replaced-download pattern from `tests/test_data.py`.
- **Ideas from `core.py`**: CAGR, volatility and Sharpe definitions. `core.py` works with daily returns (252 days) and is
  unchanged; the rule works with monthly returns, so `rule.py` has its own monthly versions (12 periods a year).

## New
- `rule.py` — the rule core: monthly prices, signal, lagged position, cash at the risk-free rate, costs,
  Sharpe ± standard error, paired intervals (bootstrap and analytic), drawdown, VaR, break-even cost, expected best-of-noise Sharpe.
- `make_evidence.py` — the script that generates `evidence/` (E1–E8, figures) and appends to `trials.csv`;
  the test period opens only with `--open-test` and the opening is written to `DECISIONS.md`.
- The Rule Lab section of the site (first a separate `/rule-lab` page, since 2026-11-19 a section of the one-page site in `templates/one.html`): rule vs. benchmark, window, trial counter, costs toggle, verdict, E1–E8.
- `forecast.py` and the Forecast section: scenarios for 3, 5 and 10 years from past monthly returns (circular block bootstrap, 10,000 paths, 12-month blocks, fixed seed), with an amount and a monthly top-up; the chance to keep the yearly return the user entered, for every fund found and for the rule. Added because the instructor asked for a forecast.
- `templates/_fund_detail.html`, `static/one.css`, `static/one.js`, `tests/test_forecast.py`: the full check of one fund that opens inside the list, the dark design, interactive charts, tests for the one-page site and the forecast.
- Tests: `tests/test_rule.py` (toy test, canary tests, known answers for costs, drawdown, cash, SE, bootstrap, break-even)
  and `tests/test_evidence.py` (trial ledger, test opened once, page numbers equal the evidence table).
- `RULE_CARD.md`, `RESULT_CARD.md`, `trials.csv`, this file, `DECISIONS.md`, `AI_USE.md`, `CONTRIBUTIONS.md`.

## Changed
- Main page: the fund finder (pick a fund by the yearly return you want) replaced the advertised-claim form as the start page;
  the detailed fund check is still reachable from every fund in the list. Why: a user usually starts from the yearly return they want, not from an advert to check, so the finder answers that question first; the claim check stays one click away for every fund.
- Fund list: removed BRK-B (a single stock) and BITO (bitcoin futures, short history); added VTI, VUG, XLV, VNQ, TLT.
  Benchmarks: removed VTI, IVV (almost the same as SPY) and VT (almost the same as ACWI).
- One-page site (2026-11-19): the fund finder, Forecast, Rule Lab and "How we calculate" (Deep analysis, Honesty panel) are sections of one page that open on click; a fund in the list opens its full check in place; the Forecast uses the yearly return, amount and monthly top-up entered at the top; charts show values on hover or tap. Why: the instructor asked for a forecast connected to the user's target, and one page is easier for a beginner than several pages. The rule, its parameters, windows and costs did not change; the Rule Lab numbers come from the same `evidence/evidence.json`. The old pages `/rule-lab`, `/forecast` and the separate finder template were removed; the detailed check of one fund (`?fund=...`) is still available.
- `requirements.txt`: added `matplotlib` for the evidence figures.
- Speed (`app.py`): fund prices are downloaded 4 at a time, kept in memory for one hour and preloaded when the site starts; the data sources, saved files and offline mode in `data.py` are unchanged.
- Changed after feedback on Assignment 1: the instructor asked us to turn the product into a fund finder by the yearly return a user wants, to make the start page clear to a beginner within seconds, to remove texts about adverts, to clean the fund and benchmark lists, to keep the Deep analysis and the Honesty panel visible, to let the user pick one fund, to limit the input to 0–1000% and to show funds without data instead of hiding them. In response we changed:
  (1) the product idea: the start page is now a fund finder (enter the yearly return you want, see the funds that gave as much or more in the chosen period, sorted closest first, with CAGR, volatility and Sharpe); the original claim check is kept as the detailed page of each fund;
  (2) the start page design: a new title and subtitle in plain words, one main field "Desired return, % per year" (0 to 1000, checked in the browser and on the server), period, benchmark and an optional fund under "Additional settings", texts about adverts removed, the "not a forecast, not investment advice" warning kept at the top;
  (3) honesty on the start page: a Deep analysis that explains where each number comes from (CAGR from the first and last price, volatility, Sharpe with the FRED rate) and why a fund is in the list or not, plus the Honesty panel with sources, dates actually used and offline status;
  (4) funds without data in the period are listed with the date their history starts instead of being hidden; funds whose history starts later are calculated from their first date and marked;
  (5) the lists: BRK-B (a single stock) and BITO (bitcoin futures, short history) removed; VTI, VUG, XLV, VNQ, TLT added (broad market, growth, healthcare, real estate, bonds); benchmarks VTI, IVV and VT removed as near-duplicates of SPY and ACWI.

## Dropped
- Removed with the one-page redesign: the separate pages `templates/rule_lab.html` (`/rule-lab`), `templates/forecast.html` (`/forecast`) and `templates/pick.html`; their content is now in `templates/one.html`. No calculation code was deleted. The advertised-claim form is no longer the start page; BRK-B, BITO, VTI, IVV and VT left the lists for the reasons above.
