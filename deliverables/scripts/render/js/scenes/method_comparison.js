// Simulation 9 - three methods on identical targets (cold start from the home pose), side by side,
// then the measured rates over all 2,000 held-out targets.
import { Studio, renderViews, targetMarker, setPose } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, elBack, show, setHTML, badge } from '../ui.js';
import { seg, smooth, easeInOut, lerp, lerpArr, num, window_ } from '../anim.js';
import { Layer, text, C, roundRect } from '../chart2d.js';

const METHODS = [
  { key: 'network', agg: 'residual_mlp', name: 'Network alone', sub: 'residual MLP, one forward pass', color: '#199e70', light: '#45c99a' },
  { key: 'dls', agg: 'classical_dls', name: 'Classical IK', sub: 'PyBullet DLS, 100 iterations', color: '#d95926', light: '#f08a5d' },
  { key: 'hybrid', agg: 'hybrid_lm', name: 'Network + LM', sub: 'refined and PyBullet-checked', color: '#3987e5', light: '#7db4f5' },
];

export default async function (ctx) {
  const model = await robotModel(ctx);
  const cmp = await ctx.data('comparison');
  const cases = cmp.cases;
  const A = (m, k) => parseFloat(cmp.aggregate[`${m}|zero`][k]);
  const { D, c } = timeline(ctx, { ex0: 2.2, per: 6.2, agg: 21.6, note: 24.1 }, 34);
  const W0 = 584, H0 = 470, X0 = 72, Y0 = 250, GAP = 12;
  const views = METHODS.map((m, i) => {
    const st = new Studio({ floorRadius: 1.3 });
    const r = new Robot(model);
    st.zup.add(r.group);
    const tg = targetMarker();
    st.zup.add(tg);
    const x = X0 + i * (W0 + GAP);
    const card = elBack(`<div class="panel" style="padding:0;width:${W0}px;height:${H0 + 160}px;background:rgba(15,27,45,.5)">
        <div style="padding:16px 20px 0 20px;display:flex;align-items:center;gap:12px">
          <span style="width:14px;height:14px;border-radius:3px;background:${m.color}"></span>
          <span style="font-size:24px;font-weight:650">${m.name}</span><span class="muted" style="font-size:16px;margin-left:auto">${m.sub}</span></div>
        <div class="mres" style="position:absolute;left:20px;right:20px;top:${H0 + 40}px"></div></div>`,
      { left: `${x}px`, top: `${Y0 - 10}px` });
    return { m, st, r, tg, rect: [x + 1, Y0 + 40, W0 - 2, H0 - 20], card, res: card.querySelector('.mres') };
  });
  const css = document.createElement('style');
  css.textContent = `.mres .row{display:flex;justify-content:space-between;font-size:19px;color:var(--text-2);padding:3px 0}
    .mres .row .v{font-family:var(--mono);color:var(--text-1)} .mres .b{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
    .mres .badge{font-size:16px;padding:4px 10px}`;
  document.head.appendChild(css);
  const h = header(ctx, { chap: 'Level 1 · Comparison', ttl: 'Same targets, three methods',
    subt: 'Cold start from the home pose (all joints 0°) · 2,000 held-out targets' });
  const exTag = el('<div class="badge info"></div>', { right: '72px', top: '112px' });
  const L = new Layer(72, 250, 1776, 640);
  const lmNote = el(`<div class="panel" style="padding:16px 24px;width:1776px;font-size:19px;line-height:1.45;color:var(--text-2)">
      <b style="color:var(--text-1)">Fair caveat.</b> The LM solver alone (no network) also reaches 100 %, but needs random restarts for
      <span class="mono">${num(A('lm_only', 'won_restart_pct'), 0)} %</span> of targets instead of <span class="mono">${num(A('hybrid_lm', 'won_restart_pct'), 0)} %</span>, and
      <span class="mono">${num(A('lm_only', 'time_ms_mean'), 1)} ms</span> instead of <span class="mono">${num(A('hybrid_lm', 'time_ms_mean'), 1)} ms</span> per target: the network is a learned warm start.</div>`,
    { left: '72px', top: '770px' });
  const cap = captions(ctx, [
    [c.ex0, c.agg, 'Each view: the same target, the same starting pose, a different solver'],
    [c.agg, D - 0.2, 'Rates over all 2,000 held-out targets (cold start)'],
  ]);

  const resHTML = (r) => `<div class="row"><span>position error</span><span class="v">${r.pos_err_mm < 0.001 ? '< 0.001' : num(r.pos_err_mm, r.pos_err_mm < 10 ? 2 : 1)} mm</span></div>
      <div class="row"><span>orientation error</span><span class="v">${num(r.orn_err_deg, 2)}°</span></div>
      <div class="b">${badge(r.pos_err_mm <= 1 && r.orn_err_deg <= 1 ? 'ok' : 'bad', 'within 1 mm & 1°')}
      ${badge(r.self_collision ? 'bad' : 'ok', r.self_collision ? 'self-collision' : 'no self-collision')}
      ${badge(r.limit_violation ? 'bad' : 'ok', r.limit_violation ? 'outside joint limits' : 'inside limits')}</div>`;

  function drawAgg(a) {
    const g = L.g;
    L.clear();
    L.opacity(a);
    if (a <= 0) return;
    const charts = [
      { t: 'Within 1 mm & 1°, valid', k: 'success_1mm_1deg_pct', unit: '%', max: 100, d: 1 },
      { t: 'Self-collision', k: 'self_collision_pct', unit: '%', max: 50, d: 1 },
      { t: 'Joint-limit violation', k: 'limit_violation_pct', unit: '%', max: 50, d: 1 },
      { t: 'Time per target', k: 'time_ms_mean', unit: 'ms', max: 50, d: 1 },
    ];
    const cw = 420, ch = 330, gap = 32;
    charts.forEach((ct, i) => {
      const x = i * (cw + gap), y = 30;
      g.fillStyle = 'rgba(15,27,45,0.9)'; roundRect(g, x, y, cw, ch + 120, 16); g.fill();
      g.strokeStyle = 'rgba(35,52,79,1)'; g.lineWidth = 1; g.stroke();
      text(g, ct.t, x + 22, y + 38, { size: 21, weight: 650, color: C.text1 });
      const base = y + ch + 30, top = y + 90, bw = 82;
      g.strokeStyle = 'rgba(160,185,220,0.45)'; g.beginPath(); g.moveTo(x + 22, base + 0.5); g.lineTo(x + cw - 22, base + 0.5); g.stroke();
      METHODS.forEach((m, j) => {
        const v = A(m.agg, ct.k);
        const hgt = Math.max(3, ((base - top) * Math.min(v, ct.max) / ct.max) * seg(a, 0.2, 1, easeInOut));
        const bx = x + 50 + j * (bw + 34);
        g.fillStyle = m.color; roundRect(g, bx, base - hgt, bw, hgt, 4); g.fill();
        g.fillRect(bx, base - Math.min(hgt, 4), bw, Math.min(hgt, 4));
        text(g, `${num(v, v < 10 && ct.unit === 'ms' ? 2 : ct.d)}${ct.unit === '%' ? ' %' : ' ms'}`, bx + bw / 2, base - hgt - 12,
          { size: 19, weight: 650, color: C.text1, align: 'center' });
        text(g, m.name.replace('Network + LM', 'Net + LM').replace('Network alone', 'Network').replace('Classical IK', 'DLS'), bx + bw / 2, base + 30,
          { size: 16, color: C.text2, align: 'center' });
      });
    });
  }

  return {
    duration: D,
    info: { cases: cases.map((cs) => cs.k) },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      const aggA = seg(t, c.agg, c.agg + 1.0, smooth);
      const i = Math.max(0, Math.min(2, Math.floor((t - c.ex0) / c.per)));
      const cs = cases[i];
      const u = (t - c.ex0 - i * c.per) / c.per;
      setHTML(exTag, `example target ${i + 1} of 3`);
      show(exTag, Math.min(seg(t, c.ex0, c.ex0 + 0.5, smooth), 1 - aggA));
      const list = [];
      views.forEach((v) => {
        const res = cs[v.m.key];
        setPose(v.tg, cs.target_pos, cs.target_orn);
        v.tg.userData.ring.rotation.z = t;
        v.tg.scale.setScalar(Math.max(1e-3, window_(t, c.ex0 + i * c.per, c.ex0 + (i + 1) * c.per, 0.3)));
        v.r.setQ(lerpArr([0, 0, 0, 0, 0, 0], res.q, seg(u, 0.1, 0.5, easeInOut)));
        v.r.clearHighlights();
        const bad = res.self_collision || res.limit_violation;
        const flag = seg(u, 0.52, 0.6, smooth);
        if (res.self_collision) for (const l of res.collision_links) v.r.highlight(l, flag, '#d03b3b');
        else if (bad) for (let k = 1; k <= 6; k++) v.r.highlight(k, 0.5 * flag, '#fab219');
        show(v.card, 1 - aggA);
        setHTML(v.res, u > 0.52 ? resHTML(res) : '');
        const tp = cs.target_pos;
        v.st.orbit([tp[0] * 0.5, tp[1] * 0.5, 0.3 + 0.25 * tp[2]], 1.55, lerp(-55, -35, t / D) + Math.atan2(tp[1], tp[0]) * 57.3 * 0.5, 14);
        if (aggA < 0.999) list.push({ scene: v.st.scene, camera: v.st.camera, rect: v.rect });
      });
      renderViews(list);
      document.getElementById('gl').style.opacity = (1 - aggA).toFixed(3);
      drawAgg(aggA);
      show(lmNote, seg(t, c.note, c.note + 0.8, smooth));
      cap(t);
    },
  };
}
