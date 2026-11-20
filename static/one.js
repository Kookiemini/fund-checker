// one.js - interactivity of the one-page site. No calculations of finance here:
// all numbers come ready from the server (core.py, rule.py, forecast.py, evidence.json).
"use strict";

const money = v => Math.round(v).toLocaleString("en-US");
const pctText = (v, d = 0) => (v * 100).toFixed(d) + "%";
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ---------- Interactive SVG line chart: hover/tap tooltip, crosshair, legend toggles ----------
function lineChart(el, opt) {
  let W = 900; const H = opt.height || 280, P = {l: 64, r: 14, t: 12, b: 26};
  const n = opt.x.length, hidden = new Set();
  el.classList.add("chart");
  el.innerHTML = `<div class="legend"></div><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:${H}px"></svg>
    <div class="hint">Hover or tap the chart to see exact values · click a legend item to hide or show it</div><div class="tip"></div>`;
  const svg = $("svg", el), tip = $(".tip", el), leg = $(".legend", el);
  opt.series.forEach((s, i) => {
    const sp = document.createElement("span");
    sp.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
    sp.onclick = () => { hidden.has(i) ? hidden.delete(i) : hidden.add(i); sp.classList.toggle("off"); draw(); };
    leg.appendChild(sp);
  });
  let X, Y;
  function draw() {
    W = Math.max(300, Math.round(el.clientWidth - 20) || 900);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const vis = opt.series.filter((s, i) => !hidden.has(i));
    const vals = vis.flatMap(s => s.band ? [...s.lo, ...s.hi] : s.y).filter(v => v != null && isFinite(v));
    if (opt.base != null) vals.push(opt.base);
    let ymin = Math.min(...vals), ymax = Math.max(...vals);
    if (!(ymax > ymin)) { ymin -= 1; ymax += 1; }
    const pad = (ymax - ymin) * 0.04; const nonNeg = ymin >= 0; ymin -= pad; ymax += pad; if (nonNeg) ymin = Math.max(0, ymin);
    X = i => P.l + (n === 1 ? 0 : i / (n - 1)) * (W - P.l - P.r);
    Y = v => P.t + (1 - (v - ymin) / (ymax - ymin)) * (H - P.t - P.b);
    let g = "";
    for (let k = 0; k <= 4; k++) {
      const v = ymin + (ymax - ymin) * k / 4, y = Y(v);
      g += `<line x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" stroke="#26324f"/>` +
           `<text x="${P.l - 8}" y="${y + 4}" fill="#95a3bf" font-size="11" text-anchor="end">${opt.fmt(v)}</text>`;
    }
    [0, Math.floor((n - 1) / 2), n - 1].forEach(i =>
      g += `<text x="${X(i)}" y="${H - 6}" fill="#95a3bf" font-size="11" text-anchor="${i === 0 ? "start" : i === n - 1 ? "end" : "middle"}">${opt.x[i]}</text>`);
    if (opt.base != null) g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(opt.base)}" y2="${Y(opt.base)}" stroke="#95a3bf" stroke-dasharray="4 4"/>`;
    opt.series.forEach((s, i) => {
      if (hidden.has(i)) return;
      if (s.band) {
        const up = s.hi.map((v, j) => `${X(j).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
        const dn = s.lo.map((v, j) => `${X(j).toFixed(1)},${Y(v).toFixed(1)}`).reverse().join(" ");
        g += `<polygon points="${up} ${dn}" fill="${s.color}" opacity=".35"/>`;
      } else {
        g += `<polyline points="${s.y.map((v, j) => `${X(j).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2.2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
      }
    });
    g += `<line class="cross" y1="${P.t}" y2="${H - P.b}" stroke="#e8edf7" stroke-opacity=".4" visibility="hidden"/>`;
    svg.innerHTML = g;
  }
  draw();
  el._redraw = draw; el.dataset.chartReady = "1";
  function show(ev) {
    const r = svg.getBoundingClientRect(), cx = ev.touches ? ev.touches[0].clientX : ev.clientX;
    const px = (cx - r.left) / r.width * W;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - P.l) / (W - P.l - P.r) * (n - 1))));
    const cr = $(".cross", svg); cr.setAttribute("x1", X(i)); cr.setAttribute("x2", X(i)); cr.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>${opt.x[i]}</b>` + opt.series.map((s, k) => hidden.has(k) ? "" : s.band
      ? `<div><span style="color:${s.color}">${s.name}</span><span>${opt.fmt(s.lo[i])} – ${opt.fmt(s.hi[i])}</span></div>`
      : `<div><span style="color:${s.color}">${s.name}</span><span>${opt.fmt(s.y[i])}</span></div>`).join("");
    tip.style.display = "block";
    const left = X(i) / W * r.width;
    tip.style.left = (left > r.width * 0.6 ? Math.max(8, left - tip.offsetWidth - 14) : left + 16) + "px";
    tip.style.top = "36px";
  }
  svg.addEventListener("mousemove", show);
  svg.addEventListener("touchstart", show, {passive: true});
  svg.addEventListener("touchmove", show, {passive: true});
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; $(".cross", svg).setAttribute("visibility", "hidden"); });
}

// ---------- Interactive bar chart (neighbours of N) ----------
function barChart(el, opt) {
  const W = Math.max(300, Math.round(el.clientWidth - 20) || 900), H = 240, P = {l: 48, r: 14, t: 12, b: 30};
  el.classList.add("chart");
  el.innerHTML = `<div class="legend"><span><i style="background:${opt.color}"></i>${opt.name}</span><span><i style="background:#95a3bf"></i>${opt.lineName}</span></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:${H}px"></svg><div class="hint">Hover or tap a bar to see its numbers</div><div class="tip"></div>`;
  const svg = $("svg", el), tip = $(".tip", el), n = opt.labels.length;
  const top = Math.max(...opt.values, opt.line) * 1.15, bw = (W - P.l - P.r) / n;
  const Y = v => P.t + (1 - v / top) * (H - P.t - P.b);
  let g = "";
  for (let k = 0; k <= 4; k++) { const v = top * k / 4; g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#26324f"/><text x="${P.l - 8}" y="${Y(v) + 4}" fill="#95a3bf" font-size="11" text-anchor="end">${v.toFixed(2)}</text>`; }
  opt.values.forEach((v, i) => {
    const x = P.l + i * bw + bw * 0.18, w = bw * 0.64;
    g += `<rect data-i="${i}" x="${x}" y="${Y(Math.max(v, 0))}" width="${w}" height="${Math.abs(Y(0) - Y(v))}" rx="6" fill="${opt.main[i] ? opt.color : opt.color + "88"}"/>` +
         `<text x="${x + w / 2}" y="${H - 10}" fill="#95a3bf" font-size="12" text-anchor="middle">${opt.labels[i]}</text>`;
  });
  g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(opt.line)}" y2="${Y(opt.line)}" stroke="#95a3bf" stroke-dasharray="5 4"/>`;
  svg.innerHTML = g;
  $$("rect", svg).forEach(rc => {
    const on = ev => { const i = +rc.dataset.i, r = svg.getBoundingClientRect();
      tip.innerHTML = `<b>N = ${opt.labels[i]}${opt.main[i] ? " (main)" : ""}</b>` + opt.details[i].map(([a, b]) => `<div><span>${a}</span><span>${b}</span></div>`).join("");
      tip.style.display = "block"; const x = (+rc.getAttribute("x")) / W * r.width; tip.style.left = Math.min(x, r.width - 190) + "px"; tip.style.top = "30px"; };
    rc.addEventListener("mousemove", on); rc.addEventListener("touchstart", on, {passive: true});
  });
  svg.addEventListener("mouseleave", () => tip.style.display = "none");
  el._redraw = () => barChart(el, opt); el.dataset.chartReady = "1";
}

// ---------- Forecast section ----------
const FC = { data: null, selected: null, timer: null };
const COLORS = { b95: "#7aa7ff", b75: "#a78bfa", mid: "#e8edf7", target: "#fbbf24", put: "#95a3bf" };

function forecastRender() {
  const d = FC.data; if (!d || !d.items.length) return;
  const item = d.items.find(i => i.asset === FC.selected) || d.items[0]; FC.selected = item.asset;
  const pct = d.pct, hz = d.horizons;
  $$(".v-amt").forEach(e => e.textContent = money(d.amount));
  $$(".v-add").forEach(e => e.textContent = d.add > 0 ? ` + ${money(d.add)} USD a month` : "");
  const bar = c => `<span class="chbar" style="width:${Math.round(c * 60)}px"></span>`;
  $("#chances").innerHTML = `<tr><th>Chance to get ${pct}% a year</th>${hz.map(y => `<th>in ${y} years</th>`).join("")}</tr>` +
    d.items.map(i => `<tr class="pick ${i.asset === item.asset ? "on" : ""}" data-asset="${i.asset}"><td><span class="tk">${i.name}</span><span class="desc">${i.label}</span></td>` +
      i.rows.map(r => `<td class="ch">${pctText(r.chance_target)}${bar(r.chance_target)}</td>`).join("") + "</tr>").join("");
  $$("#chances tr.pick").forEach(tr => tr.onclick = () => { FC.selected = tr.dataset.asset; forecastRender(); });
  $("#cards").innerHTML = item.rows.map(r => `<div class="hcard"><div class="y">${item.name} · in ${r.years} years</div>
      <div class="big">${money(r.values[50])} USD</div>
      <div class="line">Put in <b>${money(r.put_in)} USD</b></div>
      <div class="line">At your ${pct}%/yr <b>${money(r.target)} USD</b></div>
      <div class="line">Bad · Good <b>${money(r.values[5])} · ${money(r.values[95])}</b></div>
      ${d.inflation ? `<div class="line">Middle in today's money <b>≈ ${money(r.values[50] / Math.pow(1 + d.inflation.rate, r.years))} USD</b></div>` : ""}
      <div class="line">Chance to end below what you put in <b>${pctText(r.loss_chance)}</b></div>
      <div class="line">Chance to reach ${pct}%/yr <b>${pctText(r.chance_target)}</b></div>
      <div class="meter"><i style="width:${Math.round(r.chance_target * 100)}%"></i></div></div>`).join("");
  const x = Array.from({length: d.items[0].band["50"].length}, (_, i) => (i / 12).toFixed(1) + " y");
  lineChart($("#fc-chart"), { x, fmt: v => money(v) + " $",
    series: [{name: "5–95% of scenarios", color: COLORS.b95, band: true, lo: item.band["5"], hi: item.band["95"]},
             {name: "25–75% of scenarios", color: COLORS.b75, band: true, lo: item.band["25"], hi: item.band["75"]},
             {name: "Middle scenario", color: COLORS.mid, y: item.band["50"], width: 2.6},
             {name: `At your ${pct}%/yr`, color: COLORS.target, y: item.target_path},
             {name: "Money put in", color: COLORS.put, y: item.put_in_path, dash: "5 4"}]});
  $("#fc-history").textContent = `${item.name}: history ${item.history.start} to ${item.history.end} (${item.history.months} months), ` +
    `grew ${pctText(item.history.cagr, 2)} a year, worst fall from a peak ${pctText(item.history.max_drawdown, 2)}.`;
  // the summary card shows the 5-year chance of the best match
  const best = $("#sum-chance"); if (best) { const it = d.items.find(i => i.asset === best.dataset.asset);
    if (it) best.textContent = pctText(it.rows.find(r => r.years === 5).chance_target); }
  $("#fc-errors").textContent = d.errors && d.errors.length ? "Not included: " + d.errors.join("; ") : "";
  planRefresh();
}

// ---------- How much to add every month? ----------
const PLAN = { timer: null, seq: 0 };
function planRefresh() {
  const out = $("#plan-out"), d = FC.data; if (!out || !d || !FC.selected) return;
  clearTimeout(PLAN.timer);
  PLAN.timer = setTimeout(() => {
    const params = new URLSearchParams({asset: FC.selected, target: d.pct, amount: d.amount,
                                        goal: $("#goal").value, years: $("#years").value});
    const seq = ++PLAN.seq; out.classList.add("updating");
    fetch("/api/plan?" + params).then(r => r.json()).then(j => {
      if (seq !== PLAN.seq) return; out.classList.remove("updating");
      if (j.error) { out.innerHTML = `<div class="fc-err">${j.error}</div>`; return; }
      const a = j.adds, months = j.years * 12;
      const cell = (k, label) => `<div class="pcard"><small>${label}</small><b>${money(a[k])} USD</b><span>a month · you put in ${money(j.amount + a[k] * months)} USD</span></div>`;
      out.innerHTML = `<p>To have <b>${money(j.goal)} USD</b> in <b>${j.years} years</b> with <b>${j.asset}</b>, starting with ${money(j.amount)} USD:</p>
        <div class="pcards">${cell("50", "50% chance")}${cell("75", "75% chance")}${cell("90", "90% chance")}</div>
        <p class="note">If the money grew exactly ${d.pct}% every year you would need ${money(j.steady)} USD a month.
          ${j.chance_without > 0 ? `Without any top-ups the chance is ${pctText(j.chance_without)}.` : ""}
          ${j.goal_today ? `${money(j.goal)} USD in ${j.years} years is worth about ${money(j.goal_today)} USD of today's money.` : ""}</p>`;
    }).catch(() => out.classList.remove("updating"));
  }, 350);
}

// keep the CSV link in step with the amount and top-up typed at the top
function csvLinkUpdate() {
  const link = $("#csv-link"); if (!link) return;
  const url = new URL(link.href, location.href);
  url.searchParams.set("amount", $("#amt").value); url.searchParams.set("add", $("#add").value);
  link.href = url.pathname + url.search;
}

// ---------- Mix of funds ----------
const MIX_COLORS = ["#fbbf24", "#7aa7ff", "#a78bfa", "#4ade80"];
function initMix() {
  const form = $("#mix-form"); if (!form) return;
  form.addEventListener("submit", ev => {
    ev.preventDefault();
    const rows = $$(".mix-row", form).map(r => [$(".mix-fund", r).value, $(".mix-w", r).value]).filter(r => r[0] || r[1]);
    const params = new URLSearchParams({funds: rows.map(r => r[0]).join(","), weights: rows.map(r => r[1]).join(","),
                                        start: form.dataset.start, end: form.dataset.end});
    const out = $("#mix-out"); out.classList.add("updating");
    fetch("/api/mix?" + params).then(r => r.json()).then(j => {
      out.classList.remove("updating");
      if (j.error) { out.innerHTML = `<div class="error">${j.error}</div>`; return; }
      const name = Object.entries(j.weights).map(([f, w]) => `${w}% ${f}`).join(" + ");
      const pct = v => v === null || v === undefined ? "-" : pctText(v, 2);
      const row = (label, s, cls) => `<tr class="${cls || ""}"><td><b>${label}</b></td><td>${pct(s.cagr)}</td><td>${pct(s.volatility)}</td>
        <td>${s.sharpe === null ? "-" : s.sharpe.toFixed(2)}</td><td class="neg">${pct(s.max_drawdown)}</td><td>${s.fee === null ? "-" : s.fee.toFixed(2) + "%"}</td></tr>`;
      out.innerHTML = `${j.offline.length ? `<div class="offline">${j.offline.join("<br>")}</div>` : ""}
        <div class="table-wrap"><table><tr><th></th><th>Return / year</th><th>Swings</th><th>Sharpe</th><th>Worst fall</th><th>Fee / year</th></tr>
        ${row("Mix: " + name, j.mix, "mixrow")}${Object.entries(j.funds).map(([f, s]) => row(f + " alone", s)).join("")}</table></div>
        <div id="mix-chart" style="margin-top:12px"></div>
        <p class="note">${j.first} to ${j.last}, ${j.days} trading days in common. Growth of 100 USD at the end of each month. Fees are already inside the prices.</p>`;
      const names = Object.keys(j.chart.series);
      lineChart($("#mix-chart"), { x: j.chart.dates, fmt: v => money(v) + " $",
        series: names.map((n, i) => ({name: n === "Mix" ? "Mix: " + name : n, color: MIX_COLORS[i % MIX_COLORS.length],
                                      y: j.chart.series[n], width: n === "Mix" ? 2.8 : 1.6})) });
    }).catch(() => { out.classList.remove("updating"); out.innerHTML = '<div class="error">The mix could not be calculated. Please try again.</div>'; });
  });
}

function forecastRefresh() {
  const d = FC.data; if (!d) return;
  const amount = $("#amt").value, add = $("#add").value;
  const params = new URLSearchParams({target: d.pct, amount, add, assets: d.items.map(i => i.asset).join(",")});
  $("#fc-body").classList.add("updating");
  fetch("/api/forecast?" + params).then(r => r.json()).then(j => {
    $("#fc-body").classList.remove("updating");
    if (j.error) { $("#fc-errors").textContent = j.error; return; }
    FC.data = j; forecastRender();
  }).catch(() => $("#fc-body").classList.remove("updating"));
}

// ---------- Fund rows: open the full check inside the table ----------
function initDetailCharts(root) {
  $$("[data-fund-chart]", root).forEach(el => {
    const c = JSON.parse(el.dataset.fundChart);
    lineChart(el, { x: c.dates, fmt: v => v.toFixed(1),
      series: [{name: `${c.fund} (growth of 100)`, color: "#7aa7ff", y: c.fund_values}, {name: `${c.bench} (benchmark)`, color: "#a78bfa", y: c.bench_values}]});
  });
  $$("select[data-refetch]", root).forEach(sel => sel.onchange = () => loadDetail(sel.closest(".loader"), sel.value));
}

function loadDetail(box, benchmark) {
  const url = box.dataset.src + (benchmark ? "&benchmark=" + encodeURIComponent(benchmark) : "");
  box.innerHTML = '<div class="loading">Loading the full check…</div>';
  fetch(url).then(r => r.text()).then(html => { box.innerHTML = html; box.dataset.loaded = "1"; initDetailCharts(box); })
    .catch(() => box.innerHTML = '<div class="error">Could not load the full check. Please try again.</div>');
}

function initFundRows() {
  $$("tr.fund").forEach(tr => {
    const detail = document.getElementById("fund-" + tr.dataset.fund);
    const toggle = ev => { ev.preventDefault(); if (!detail) return;
      detail.hidden = !detail.hidden; tr.classList.toggle("open", !detail.hidden);
      const box = $(".loader", detail); if (!detail.hidden && box && !box.dataset.loaded) loadDetail(box); };
    tr.addEventListener("click", toggle);
  });
}

// ---------- Rule Lab: costs toggle and charts ----------
function initRuleLab() {
  $$(".cost-pill").forEach(p => p.onclick = () => {
    $$(".cost-pill").forEach(x => x.classList.toggle("on", x === p));
    $$(".ev-block").forEach(b => b.classList.toggle("on", b.dataset.cost === p.dataset.cost));
  });
  const raw = $("#rl-data"); if (!raw) return;
  const ev = JSON.parse(raw.textContent);
  lineChart($("#rl-equity"), { x: ev.dates, fmt: v => v.toFixed(2) + " $",
    series: [{name: ev.rule_name, color: "#4ade80", y: ev.rule}, {name: ev.bench_name, color: "#95a3bf", y: ev.bench}]});
  lineChart($("#rl-drawdown"), { x: ev.dates, fmt: v => (v * 100).toFixed(1) + "%", height: 220,
    series: [{name: ev.rule_name, color: "#4ade80", y: ev.dd_rule}, {name: ev.bench_name, color: "#f87171", y: ev.dd_bench}]});
  barChart($("#rl-neigh"), { labels: ev.neigh.map(r => r.n), values: ev.neigh.map(r => r.sharpe_gross || 0), main: ev.neigh.map(r => r.main),
    line: ev.neigh[0].sharpe_bench || 0, color: "#7aa7ff", name: "Sharpe of the rule (gross)", lineName: "Buy and hold",
    details: ev.neigh.map(r => [["Sharpe gross", (r.sharpe_gross ?? 0).toFixed(2)], ["Sharpe net", (r.sharpe_net ?? 0).toFixed(2)],
      ["CAGR gross", pctText(r.cagr_gross, 2)], ["Max drawdown", pctText(r.max_drawdown, 2)]]) });
}

// ---------- Tabs ----------
function initTabs() {
  $$(".tabs .pill").forEach(p => p.onclick = () => {
    const box = p.closest(".body");
    $$(".tabs .pill", box).forEach(x => x.classList.toggle("on", x === p));
    $$(".tabpane", box).forEach(x => x.classList.toggle("on", x.id === p.dataset.tab));
  });
}

// redraw charts at their real width when a section opens or the window size changes
function redrawIn(root) { $$("[data-chart-ready]", root).forEach(el => el.offsetParent !== null && el._redraw && el._redraw()); }
document.addEventListener("toggle", ev => { if (ev.target.open) redrawIn(ev.target); }, true);
let resizeTimer = null;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => redrawIn(document), 200); });

document.addEventListener("DOMContentLoaded", () => {
  const raw = $("#fc-data");
  if (raw) { FC.data = JSON.parse(raw.textContent); FC.selected = FC.data.items.length ? FC.data.items[0].asset : null; forecastRender(); }
  ["#amt", "#add"].forEach(id => { const el = $(id); if (el) el.addEventListener("input", () => {
    csvLinkUpdate(); clearTimeout(FC.timer); FC.timer = setTimeout(forecastRefresh, 450); }); });
  ["#goal", "#years"].forEach(id => { const el = $(id); if (el) el.addEventListener("input", planRefresh); });
  initFundRows(); initRuleLab(); initTabs(); initMix();
  $$("details[data-lazy]").forEach(dt => dt.addEventListener("toggle", () => {
    const box = $(".loader", dt); if (dt.open && box && !box.dataset.loaded) loadDetail(box); }));
});
