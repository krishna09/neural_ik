// Minimal 2-D canvas drawing for charts and diagrams inside scenes (redrawn every frame).

import { clamp } from './anim.js';

export const C = {
  text1: '#eef3fa', text2: '#aebbd0', muted: '#74849e', grid: 'rgba(120,150,190,0.16)', axis: 'rgba(160,185,220,0.45)',
  blue: '#3987e5', orange: '#d95926', aqua: '#199e70', yellow: '#c98500', accent: '#4fd6f2',
  good: '#0ca30c', warning: '#fab219', critical: '#d03b3b', surface: '#0f1b2d',
};

export class Layer {
  constructor(x, y, w, h, parent = document.getElementById('ui')) {
    this.c = document.createElement('canvas');
    this.c.width = w; this.c.height = h;
    Object.assign(this.c.style, { position: 'absolute', left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` });
    parent.appendChild(this.c);
    this.g = this.c.getContext('2d');
    this.w = w; this.h = h;
  }
  clear() { this.g.clearRect(0, 0, this.w, this.h); }
  opacity(a) { this.c.style.opacity = a.toFixed(3); this.c.style.visibility = a <= 0.001 ? 'hidden' : 'visible'; }
}

export function font(g, size, weight = 500, family = 'Inter') { g.font = `${weight} ${size}px ${family}`; }

export function text(g, s, x, y, { size = 18, weight = 500, color = C.text2, align = 'left', base = 'alphabetic', family = 'Inter' } = {}) {
  font(g, size, weight, family);
  g.fillStyle = color; g.textAlign = align; g.textBaseline = base;
  g.fillText(s, x, y);
}

export function roundRect(g, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  g.beginPath();
  g.moveTo(x + rr, y);
  g.arcTo(x + w, y, x + w, y + h, rr);
  g.arcTo(x + w, y + h, x, y + h, rr);
  g.arcTo(x, y + h, x, y, rr);
  g.arcTo(x, y, x + w, y, rr);
  g.closePath();
}

// scale helpers
export function linScale(d0, d1, r0, r1) { return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0); }
export function logScale(d0, d1, r0, r1) { const a = Math.log10(d0), b = Math.log10(d1); return (v) => r0 + ((Math.log10(Math.max(v, 1e-12)) - a) / (b - a)) * (r1 - r0); }

// line chart: series = [{pts: [[x,y],...], color, label, progress (0..1 share of points drawn), width}]
export function lineChart(g, { x, y, w, h, xd, yd, ylog = false, xticks = [], yticks = [], xlabel = '', ylabel = '',
  series = [], yfmt = (v) => String(v), xfmt = (v) => String(v), alpha = 1, endLabels = true }) {
  const sx = linScale(xd[0], xd[1], x, x + w);
  const sy = (ylog ? logScale : linScale)(yd[0], yd[1], y + h, y);
  g.save();
  g.globalAlpha = alpha;
  g.lineWidth = 1;
  for (const t of yticks) {
    const py = Math.round(sy(t)) + 0.5;
    g.strokeStyle = C.grid; g.beginPath(); g.moveTo(x, py); g.lineTo(x + w, py); g.stroke();
    text(g, yfmt(t), x - 12, py + 6, { size: 16, color: C.muted, align: 'right' });
  }
  g.strokeStyle = C.axis; g.beginPath(); g.moveTo(x, y + h + 0.5); g.lineTo(x + w, y + h + 0.5); g.stroke();
  for (const t of xticks) {
    const px = sx(t);
    text(g, xfmt(t), px, y + h + 26, { size: 16, color: C.muted, align: 'center' });
  }
  if (xlabel) text(g, xlabel, x + w, y + h + 52, { size: 16, color: C.muted, align: 'right' });
  if (ylabel) text(g, ylabel, x, y - 18, { size: 16, color: C.muted, align: 'left' });
  for (const s of series) {
    const n = s.pts.length;
    const upto = clamp(s.progress ?? 1) * (n - 1);
    const k = Math.floor(upto);
    if (n < 2 || upto <= 0) continue;
    g.strokeStyle = s.color; g.lineWidth = s.width || 3; g.lineJoin = 'round'; g.lineCap = 'round';
    g.beginPath();
    g.moveTo(sx(s.pts[0][0]), sy(s.pts[0][1]));
    for (let i = 1; i <= k; i++) g.lineTo(sx(s.pts[i][0]), sy(s.pts[i][1]));
    let ex = s.pts[k][0], ey = s.pts[k][1];
    if (k < n - 1) {
      const f = upto - k;
      ex = s.pts[k][0] + (s.pts[k + 1][0] - s.pts[k][0]) * f;
      ey = ylog ? Math.pow(10, Math.log10(s.pts[k][1]) + (Math.log10(s.pts[k + 1][1]) - Math.log10(s.pts[k][1])) * f)
        : s.pts[k][1] + (s.pts[k + 1][1] - s.pts[k][1]) * f;
      g.lineTo(sx(ex), sy(ey));
    }
    g.stroke();
    // end marker with a 2px surface ring
    g.fillStyle = C.surface; g.beginPath(); g.arc(sx(ex), sy(ey), 7, 0, Math.PI * 2); g.fill();
    g.fillStyle = s.color; g.beginPath(); g.arc(sx(ex), sy(ey), 5, 0, Math.PI * 2); g.fill();
    if (endLabels && s.label) {
      text(g, s.label(ex, ey), sx(ex) + 14, sy(ey) + (s.labelDy || 6), { size: 17, weight: 600, color: C.text1 });
    }
  }
  g.restore();
  return { sx, sy };
}

// horizontal bars: rows = [{label, value, color, text}], domain [0, max]
export function hbars(g, { x, y, w, rowH = 46, gap = 14, labelW = 250, max, rows, progress = 1, valueFmt = (v) => String(v) }) {
  const sx = linScale(0, max, 0, w - labelW - 120);
  rows.forEach((r, i) => {
    const py = y + i * (rowH + gap);
    text(g, r.label, x, py + rowH / 2 + 7, { size: 19, color: C.text2 });
    const bw = Math.max(4, sx(r.value) * clamp(progress));
    g.fillStyle = r.color;
    roundRect(g, x + labelW, py + 6, bw, rowH - 12, 4);
    g.fill();
    text(g, r.text ?? valueFmt(r.value * clamp(progress)), x + labelW + bw + 12, py + rowH / 2 + 7,
      { size: 19, weight: 600, color: C.text1 });
  });
}
