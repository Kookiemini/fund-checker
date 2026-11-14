git add .
git commit -m "AI_USE: Rule Card fields reviewed by the team"
git push
# Rule Card — MA10 on SPY (rule version v1)

Status: **draft v1** (G0). Revise after feedback (G1), then freeze with the git tag `freeze-w13` (G2).
After the freeze, every change goes into `DECISIONS.md`, gets a new `RULE_VERSION` in `rule.py`, and counts as a trial.

> Fields marked **[TEAM — RED]** must be written by the team in its own words, without AI (assignment, Part 7).
> They must agree with the code: the functions to read are named next to each field.

**R1. Question**
If an investor had held SPY only when it closed above its 10-month moving average, and cash otherwise,
would they have been better off than buying and holding SPY in 2001–2026, after costs, and how sure can we be?

**R2. Universe and data**
- Asset: SPY (SPDR S&P 500 ETF Trust), one asset.
- Prices: Yahoo Finance via `yfinance`, Adjusted Close (`auto_adjust=True`, includes dividends and splits), daily,
  converted to the close of the last trading day of each complete month (`rule.monthly_prices`).
- Cash: FRED series DGS3MO (3-Month Treasury constant maturity, % per year), last value of each month (`rule.monthly_rate`).
- Dates: data from 2000-01-01 (the first 12 months warm up the moving average); evaluation from 2001-01 to 2026-10.
- Frequency: monthly. Currency: USD. Loaded through `data.py` (same loader, cache and offline mode as Assignment 1).

**R3. Signal (formula at close of t)** — [TEAM — RED]
At the close of month t we compare the SPY month-end price with the simple average of the last N = 10 month-end prices, including month t itself. The signal is 1 if the price is strictly above that average and 0 if it is equal to or below it; during the first N − 1 months there is no average yet, so there is no signal.

**R4. Position held in t + 1 for each signal value, including the "out" position** — [TEAM — RED]
If the signal at the close of month t is 1, we hold 100% SPY during month t + 1 and earn its total return (price change plus dividends). If the signal is 0, we hold no SPY at all: the money sits in cash and earns the 3-month Treasury rate divided by 12 for that month.

**R5. Timing (information → decision → return)** — [TEAM — RED]
At the close of month t we know only prices up to and including month t and the Treasury rate published at the end of month t; the decision uses nothing else. The position chosen then is held through month t + 1, so the return we book is the month t + 1 return (SPY or the cash rate known at the end of t), never the return of month t that produced the signal.

**R6. Benchmark and why**
Buy and hold SPY over exactly the same months, with no costs. It is the investment the rule tries to improve on:
the same asset, so the only difference is the rule's decision to be in or out.

**R7. Parameters and ranges allowed**
- N (months in the moving average): main value **10**, taken from the literature before looking at our data (not optimised).
- Allowed range: 6 to 14 months. Looked at: 8, 9, 10, 11, 12 (neighbours ± 1 and ± 2, evidence E5). Every one is a trial.
- Costs: c = 0%, 0.10%, 0.25%, 0.50% per unit of position change; 0.10% is the default shown.
- Samples: train 2001-01 to 2012-12, validation 2013-01 to 2018-12, test 2019-01 to 2026-10 (opened once).
- Bootstrap: paired circular block, block 12 months, 5000 repetitions, seed 20261114.
All of these live in one place: the constants at the top of `rule.py`.

**R8. When we drop the rule (decided now)** — [TEAM — RED]
We say the rule does not work if, on the full window 2001-01 to 2026-10 net of 0.10% costs, the 95% bootstrap interval for the Sharpe difference (rule minus buy-and-hold) contains 0 and the rule's maximum drawdown is not at least 10 percentage points smaller than the benchmark's. We also drop it if the break-even cost is below 0.25% per switch, or if on the test period alone the rule has both a lower Sharpe and a deeper drawdown than buy-and-hold. (Written 2026-11-14, before opening the test.)
