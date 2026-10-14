// Small SVG charts for the Mission page: line (crosshair + tooltip), stacked bars, grouped histogram.
// Colours come from the caller (validated palette); text always uses the page's text colours.

const NS = 'http://www.w3.org/2000/svg';
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const T = () => ({ grid: css('--border-soft') || '#222832', axis: css('--border') || '#2a313c', text: css('--text-secondary') || '#b3bdcb', muted: css('--text-muted') || '#7f8a9a', surface: css('--surface-2') || '#1b2129' });
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.appendChild(e); return e; };
const txt = (parent, x, y, s, attrs = {}) => { const t = el('text', { x, y, 'font-size': 11, fill: T().muted, ...attrs }, parent); t.textContent = s; return t; };

function niceTicks(lo, hi, n = 4) {
  if (hi - lo < 1e-12) { hi = lo + 1; }
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, out = [];
  for (let v = a; v <= b + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}
const fmtN = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

function frame(container, title, legend) {
  container.innerHTML = '';
  container.classList.add('chart-box');
  const head = document.createElement('div'); head.className = 'chart-head';
  const h = document.createElement('div'); h.className = 'chart-title'; h.textContent = title; head.appendChild(h);
  if (legend && legend.length > 1) {
    const lg = document.createElement('div'); lg.className = 'chart-legend';
    for (const s of legend) { const it = document.createElement('span'); const sw = document.createElement('i'); sw.style.background = s.color; it.append(sw, document.createTextNode(s.name)); lg.appendChild(it); }
    head.appendChild(lg);
  }
  const wrap = document.createElement('div'); wrap.className = 'chart-plot';
  const tip = document.createElement('div'); tip.className = 'chart-tip'; tip.hidden = true;
  container.append(head, wrap); wrap.appendChild(tip);
  const w = Math.max(240, wrap.clientWidth || container.clientWidth || 400), hgt = Math.max(140, wrap.clientHeight || 180);
  const svg = el('svg', { width: w, height: hgt, viewBox: `0 0 ${w} ${hgt}`, role: 'img', 'aria-label': title }, wrap);
  return { svg, tip, w, h: hgt, wrap };
}
function showTip(tip, wrap, x, rows, head) {
  tip.replaceChildren();
  if (head) { const hd = document.createElement('div'); hd.className = 'tt-head'; hd.textContent = head; tip.appendChild(hd); }
  for (const r of rows) {
    const line = document.createElement('div'); line.className = 'tt-row';
    if (r.color) { const sw = document.createElement('i'); sw.style.background = r.color; line.appendChild(sw); }
    const v = document.createElement('b'); v.textContent = r.value; const l = document.createElement('span'); l.textContent = r.label;
    line.append(v, l); tip.appendChild(line);
  }
  tip.hidden = false;
  const tw = tip.offsetWidth;
  tip.style.left = `${Math.min(wrap.clientWidth - tw - 4, Math.max(4, x + 12))}px`;
  tip.style.top = '6px';
}

// series: [{ name, color, values: [y per x] }]; refs: [{ y, label }]; events: [{ x, label }]
export function lineChart(container, { title, x, series, yLabel = '', xLabel = 'step', refs = [], events = [], yMin, yMax, unit = '' }) {
  const { svg, tip, w, h, wrap } = frame(container, title, series);
  const L = 44, R = 10, Tp = 10, B = 26;
  const ys = series.flatMap((s) => s.values).filter((v) => v != null && Number.isFinite(v)).concat(refs.map((r) => r.y));
  let lo = yMin ?? Math.min(...ys), hi = yMax ?? Math.max(...ys);
  if (!(hi > lo)) hi = lo + 1;
  const ticks = niceTicks(lo, hi); lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const x0 = x[0] ?? 0, x1 = x[x.length - 1] ?? 1;
  const X = (v) => L + ((v - x0) / Math.max(1e-9, x1 - x0)) * (w - L - R), Y = (v) => Tp + (1 - (v - lo) / (hi - lo)) * (h - Tp - B);
  const t = T();
  for (const v of ticks) { el('line', { x1: L, x2: w - R, y1: Y(v), y2: Y(v), stroke: t.grid, 'stroke-width': 1 }, svg); txt(svg, L - 6, Y(v) + 4, fmtN(v), { 'text-anchor': 'end' }); }
  txt(svg, w - R, h - 6, `${xLabel} →`, { 'text-anchor': 'end' });
  if (yLabel) txt(svg, 4, Tp + 2, yLabel, { 'dominant-baseline': 'hanging', 'font-size': 10 });
  for (const r of refs) {
    el('line', { x1: L, x2: w - R, y1: Y(r.y), y2: Y(r.y), stroke: t.muted, 'stroke-width': 1, 'stroke-dasharray': '4 3' }, svg);
    txt(svg, w - R - 2, Y(r.y) - 4, r.label, { 'text-anchor': 'end', 'font-size': 10, fill: t.text });
  }
  // event markers; labels are staggered onto rows so neighbouring events never overlap
  const rowEnd = [];
  for (const e of [...events].sort((a, b) => a.x - b.x)) {
    const xe = X(e.x), wLab = 6 * e.label.length + 8;
    el('line', { x1: xe, x2: xe, y1: Tp, y2: h - B, stroke: t.muted, 'stroke-width': 1, 'stroke-dasharray': '2 3' }, svg);
    let row = rowEnd.findIndex((end) => xe > end);
    if (row < 0) { row = rowEnd.length; rowEnd.push(0); }
    const right = xe + 3 + wLab > w - R;                     // flip to the left of the marker near the right edge
    txt(svg, right ? xe - 3 : xe + 3, Tp + 10 + 12 * row, e.label, { 'font-size': 10, fill: t.text, 'text-anchor': right ? 'end' : 'start' });
    rowEnd[row] = right ? xe : xe + 3 + wLab;
  }
  for (const s of series) {
    let d = '';
    s.values.forEach((v, i) => { if (v == null || !Number.isFinite(v)) return; d += `${d ? 'L' : 'M'}${X(x[i]).toFixed(1)},${Y(Math.max(lo, Math.min(hi, v))).toFixed(1)}`; });
    el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
  }
  // crosshair + tooltip with every series at that x
  const hair = el('line', { y1: Tp, y2: h - B, stroke: t.text, 'stroke-width': 1, visibility: 'hidden' }, svg);
  const dots = series.map((s) => el('circle', { r: 4, fill: s.color, stroke: t.surface, 'stroke-width': 2, visibility: 'hidden' }, svg));
  const hit = el('rect', { x: L, y: Tp, width: w - L - R, height: h - Tp - B, fill: 'transparent' }, svg);
  hit.addEventListener('pointermove', (ev) => {
    const r = svg.getBoundingClientRect(), px = ev.clientX - r.left;
    const xv = x0 + ((px - L) / (w - L - R)) * (x1 - x0);
    let i = 0; for (let k = 1; k < x.length; k++) if (Math.abs(x[k] - xv) < Math.abs(x[i] - xv)) i = k;
    hair.setAttribute('x1', X(x[i])); hair.setAttribute('x2', X(x[i])); hair.setAttribute('visibility', 'visible');
    series.forEach((s, k) => { const v = s.values[i]; const ok = v != null && Number.isFinite(v); dots[k].setAttribute('visibility', ok ? 'visible' : 'hidden'); if (ok) { dots[k].setAttribute('cx', X(x[i])); dots[k].setAttribute('cy', Y(Math.max(lo, Math.min(hi, v)))); } });
    showTip(tip, wrap, X(x[i]), series.map((s) => ({ color: s.color, label: s.name, value: s.values[i] == null ? '—' : `${fmtN(s.values[i])}${unit}` })), `${xLabel} ${x[i]}`);
  });
  hit.addEventListener('pointerleave', () => { hair.setAttribute('visibility', 'hidden'); dots.forEach((d) => d.setAttribute('visibility', 'hidden')); tip.hidden = true; });
}

// rows: [{ name, parts: [{ label, value (0-100), color }] }] - horizontal 100% stacked bars with labels
export function stackedBars(container, { title, rows, legend }) {
  const { svg, tip, w, h, wrap } = frame(container, title, legend);
  const L = 120, R = 10, Tp = 6, gap = 10;
  const bh = Math.min(26, (h - Tp - 18 - gap * (rows.length - 1)) / rows.length);
  const X = (v) => L + (v / 100) * (w - L - R), t = T();
  [0, 25, 50, 75, 100].forEach((v) => { el('line', { x1: X(v), x2: X(v), y1: Tp, y2: h - 16, stroke: t.grid, 'stroke-width': 1 }, svg); txt(svg, X(v), h - 3, `${v}%`, { 'text-anchor': 'middle' }); });
  rows.forEach((row, i) => {
    const y = Tp + i * (bh + gap);
    txt(svg, L - 8, y + bh / 2 + 4, row.name, { 'text-anchor': 'end', fill: t.text, 'font-size': 12 });
    let acc = 0;
    row.parts.forEach((part) => {
      if (part.value <= 0) return;
      const x = X(acc), wd = Math.max(0, X(acc + part.value) - x - 2);       // 2 px surface gap between segments
      const rect = el('rect', { x, y, width: wd, height: bh, rx: 3, fill: part.color }, svg);
      if (wd > 34) txt(svg, x + wd / 2, y + bh / 2 + 4, `${part.value.toFixed(0)}%`, { 'text-anchor': 'middle', fill: '#fff', 'font-size': 11, 'font-weight': 700 });
      rect.addEventListener('pointermove', (ev) => showTip(tip, wrap, ev.clientX - svg.getBoundingClientRect().left, [{ color: part.color, label: part.label, value: `${part.value.toFixed(1)}%` }], row.name));
      rect.addEventListener('pointerleave', () => (tip.hidden = true));
      acc += part.value;
    });
  });
}

// grouped histogram: series [{ name, color, values: [...] }] binned over [lo, hi] in `bins` bins
export function histogram(container, { title, series, lo, hi, bins = 10, xLabel = '', unit = '' }) {
  const { svg, tip, w, h, wrap } = frame(container, title, series);
  const L = 36, R = 10, Tp = 10, B = 26, t = T();
  const width = (hi - lo) / bins;
  const counts = series.map((s) => { const c = new Array(bins).fill(0); s.values.forEach((v) => { if (v == null || !Number.isFinite(v)) return; c[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / width)))]++; }); return c; });
  const maxC = Math.max(1, ...counts.flat());
  const ticks = niceTicks(0, maxC);
  const Y = (v) => Tp + (1 - v / ticks[ticks.length - 1]) * (h - Tp - B);
  for (const v of ticks) { el('line', { x1: L, x2: w - R, y1: Y(v), y2: Y(v), stroke: t.grid, 'stroke-width': 1 }, svg); txt(svg, L - 6, Y(v) + 4, String(v), { 'text-anchor': 'end' }); }
  const gw = (w - L - R) / bins, bw = Math.max(2, (gw - 6) / series.length - 2);
  for (let b = 0; b <= bins; b += Math.max(1, Math.round(bins / 5))) txt(svg, L + b * gw, h - 8, `${fmtN(lo + b * width)}`, { 'text-anchor': 'middle' });
  void xLabel;                                                // the unit is in the chart title
  series.forEach((s, k) => counts[k].forEach((c, b) => {
    if (!c) return;
    const x = L + b * gw + 3 + k * (bw + 2), y = Y(c);
    const rect = el('rect', { x, y, width: bw, height: Y(0) - y, rx: 2, fill: s.color }, svg);
    rect.addEventListener('pointermove', (ev) => showTip(tip, wrap, ev.clientX - svg.getBoundingClientRect().left,
      [{ color: s.color, label: s.name, value: `${c} mission${c === 1 ? '' : 's'}` }], `${fmtN(lo + b * width)} – ${fmtN(lo + (b + 1) * width)}${unit}`));
    rect.addEventListener('pointerleave', () => (tip.hidden = true));
  }));
}
