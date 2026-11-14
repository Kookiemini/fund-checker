# Reuse and changes (Assignment 1 → Assignment 2)

## Reused from Assignment 1
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
- `templates/rule_lab.html` and the `/rule-lab` route — the Rule Lab page: rule vs. benchmark, window, trial counter, costs toggle, verdict.
- Tests: `tests/test_rule.py` (toy test, canary tests, known answers for costs, drawdown, cash, SE, bootstrap, break-even)
  and `tests/test_evidence.py` (trial ledger, test opened once, page numbers equal the evidence table).
- `RULE_CARD.md`, `RESULT_CARD.md`, `trials.csv`, this file, `DECISIONS.md`, `AI_USE.md`, `CONTRIBUTIONS.md`.

## Changed
- Main page: the fund finder (pick a fund by the yearly return you want) replaced the advertised-claim form as the start page;
  the detailed fund check is still reachable from every fund in the list. Why: a user usually starts from the yearly return they want, not from an advert to check, so the finder answers that question first; the claim check stays one click away for every fund.
- Fund list: removed BRK-B (a single stock) and BITO (bitcoin futures, short history); added VTI, VUG, XLV, VNQ, TLT.
  Benchmarks: removed VTI, IVV (almost the same as SPY) and VT (almost the same as ACWI).
- Main page: a link to the Rule Lab.
- `requirements.txt`: added `matplotlib` for the evidence figures.
- Changed after feedback on Assignment 1: the feedback was. In response we changed:
  (1) the product idea: the start page is now a fund finder (enter the yearly return you want, see the funds that gave as much or more in the chosen period, sorted closest first, with CAGR, volatility and Sharpe); the original claim check is kept as the detailed page of each fund;
  (2) the start page design: a new title and subtitle in plain words, one main field "Desired return, % per year" (0 to 1000, checked in the browser and on the server), period, benchmark and an optional fund under "Additional settings", texts about adverts removed, the "not a forecast, not investment advice" warning kept at the top;
  (3) honesty on the start page: a Deep analysis that explains where each number comes from (CAGR from the first and last price, volatility, Sharpe with the FRED rate) and why a fund is in the list or not, plus the Honesty panel with sources, dates actually used and offline status;
  (4) funds without data in the period are listed with the date their history starts instead of being hidden; funds whose history starts later are calculated from their first date and marked;
  (5) the lists: BRK-B (a single stock) and BITO (bitcoin futures, short history) removed; VTI, VUG, XLV, VNQ, TLT added (broad market, growth, healthcare, real estate, bonds); benchmarks VTI, IVV and VT removed as near-duplicates of SPY and ACWI.

## Dropped
- No code was deleted. The advertised-claim form is no longer the start page; BRK-B, BITO, VTI, IVV and VT left the lists for the reasons above.
