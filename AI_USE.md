# AI use

One entry per use: date, tool, task, what we kept or changed. The team fills the last column after reading the code.
Policy colours from the assignment (Part 7): GREEN allowed, YELLOW allowed after our own hand-computed values, RED never.

| Date | Tool | Task | Colour | What we kept or changed |
|---|---|---|---|---|
| 2026-11-13 | Claude (Anthropic), chat | Turned the start page into a fund finder by desired yearly return; cleaned the fund and benchmark lists; README and tests for removed tickers. | GREEN (app code) | Kept; checked the start page by hand and with `pytest`. |
| 2026-11-14 | Claude | Fund finder: Deep analysis and honesty panels, optional fund selection, 0–1000 limit, "funds without data" block, added VTI, VUG, XLV, VNQ, TLT, smaller input field, tests. | GREEN (app code) | Kept; checked the new blocks in the browser. |
| 2026-11-14 | Claude | Checked the site against the Assignment 2 criteria and listed what was missing. | GREEN | Used the list as our to-do for GA2. |
| 2026-11-14 | Claude | Drafted `rule.py` (signal, lagged position, cash, costs, statistics, intervals, break-even). | YELLOW: must be checked against our toy example | Yes: with our toy values (signal 0,0,1,1; position 0,0,1; gross 0.001, 0.001, 0.048544) `pytest` gives 125 passed. |
| 2026-11-14 | Claude | Drafted `make_evidence.py` (E1–E8, trials ledger, test opened once, figures), the Rule Lab page, `tests/test_rule.py` (canary and known-answer tests), `tests/test_evidence.py`. | GREEN / YELLOW | Kept; ran it and compared the page with `evidence/evidence_table.md`. |
| 2026-11-14 | Claude | Templates of RULE_CARD (R1, R2, R6, R7 only), RESULT_CARD, REUSE_CHANGES, DECISIONS, CONTRIBUTIONS. | GREEN | Kept the structure, filled our parts. |
| 2026-11-14 | Claude | Drafted the wording of Rule Card R3–R5 and R8, checked the `TOY_EXPECTED` values against a hand calculation, filled lines in AI_USE / REUSE_CHANGES / DECISIONS. | RED fields — declared here | Team: reread, rewrite in your own words, note what you changed. |

Planned as not done by AI (RED; see the last row above for what was drafted with AI anyway): the toy example and its expected values, the canary-test explanation, Rule Card R3–R5 and R8,
the verdict and limits wording, the explanation of the numbers in the evidence table.
