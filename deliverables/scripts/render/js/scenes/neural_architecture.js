// Simulation 5 - the IK network: 13 inputs -> 256 -> 256 -> 128 -> 12 outputs -> atan2 -> 6 angles.
// Neuron brightness = real ReLU activations of the baseline network for one held-out target.
import * as THREE from 'three';
import { Studio, renderViews, targetMarker, setPose } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, elBack, show, setHTML } from '../ui.js';
import { seg, smooth, easeInOut, easeOut, deg, num, clamp, rng, lerp } from '../anim.js';
import { Layer, text, C, roundRect } from '../chart2d.js';

const IN_LABELS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 's1', 's2', 's3', 's4', 's5', 's6'];
const OUT_LABELS = [1, 2, 3, 4, 5, 6].map((i) => `sin q${i}`).concat([1, 2, 3, 4, 5, 6].map((i) => `cos q${i}`));

export default async function (ctx) {
  const model = await robotModel(ctx);
  const nn = await ctx.data('network');
  const { D, c } = timeline(ctx, { inputs: 1.8, h1: 4.6, h2: 6.4, h3: 8.2, out: 9.8, decode: 11.6, robot: 13.2, circle: 16.4, note: 22.4 }, 27);

  const h = header(ctx, { chap: 'Level 1 · Neural model', ttl: 'From a target pose to six joint angles',
    subt: 'Baseline network with its real activations: 13 → 256 → 256 → 128 → 12' });
  const lay = new Layer(60, 236, 1290, 720);
  const g = lay.g;
  const cols = [
    { x: 300, n: 13, idx: null, vals: nn.x, r: 9 },
    { x: 530, n: 24, idx: 0, full: 256, r: 7 },
    { x: 730, n: 24, idx: 1, full: 256, r: 7 },
    { x: 920, n: 16, idx: 2, full: 128, r: 7.5 },
    { x: 1085, n: 12, idx: null, vals: nn.raw, r: 9 },
  ];
  const top = 40, bottom = 650;
  cols.forEach((col) => {
    const sp = (bottom - top) / (col.n - 1 + (col.full ? 1.2 : 0));
    col.y = Array.from({ length: col.n }, (_, i) => top + i * sp + (col.full ? 0 : 0));
    if (col.full) {
      const a = nn.acts[col.idx];
      const mx = Math.max(...a);
      col.act = col.y.map((_, i) => a[Math.round((i * (a.length - 1)) / (col.n - 1))] / mx);
    } else {
      col.act = col.vals.map((v) => Math.min(1, Math.abs(v) / (Math.max(...col.vals.map(Math.abs)) || 1)));
    }
  });
  // sparse, deterministic edges between consecutive columns
  const R = rng(7);
  const edges = [];
  for (let l = 0; l < cols.length - 1; l++) {
    const A = cols[l], B = cols[l + 1];
    for (let j = 0; j < B.n; j++) for (let k = 0; k < 6; k++) edges.push({ l, i: Math.floor(R() * A.n), j, ph: R() });
  }
  const layerT = [c.inputs, c.h1, c.h2, c.h3, c.out];

  // decode panel + 3D inset
  const rows = nn.q_pred_deg.map((v, i) => `<div class="drow"><b>q${i + 1}</b><span class="mono">atan2(${num(nn.raw[i], 2)}, ${num(nn.raw[i + 6], 2)})</span><span class="mono v">${num(v, 1)}°</span></div>`).join('');
  const dec = el(`<div class="panel" style="padding:18px 22px;width:500px"><h3>Decode each sin / cos pair</h3>${rows}
      <div class="muted" style="font-size:15px;margin-top:8px">network output → joint angle (degrees)</div></div>`,
    { right: '72px', top: '236px' });
  const css = document.createElement('style');
  css.textContent = `.drow{display:grid;grid-template-columns:40px 1fr 92px;gap:8px;font-size:18px;padding:5px 2px;color:var(--text-2)}
    .drow b{color:var(--text-1)} .drow .v{text-align:right;color:var(--text-1)}`;
  document.head.appendChild(css);
  const st = new Studio({ floorRadius: 1.2 });
  const ghostBase = new Robot(model, { style: 'ghost', color: '#e9b440', opacity: 0.62 });
  const ghostBest = new Robot(model, { style: 'ghost', color: '#45c99a', opacity: 0.62 });
  const solidTrue = new Robot(model);
  st.zup.add(ghostBase.group, ghostBest.group);
  ghostBase.setQ(nn.baseline.q);
  ghostBest.setQ(nn.best.q);
  const tgt = targetMarker();
  st.zup.add(tgt);
  const insetRect = [1848 - 500, 552, 500, 262];
  const insetCard = elBack(`<div class="panel" style="padding:0;width:${insetRect[2]}px;height:${insetRect[3] + 72}px;background:rgba(15,27,45,.55)"></div>`,
    { left: `${insetRect[0]}px`, top: `${insetRect[1] - 6}px` });
  const insetLbl = el(`<div style="font-size:17px;line-height:1.5">
      <div><span style="color:#e9b440">■</span> baseline network: <span class="mono">${num(nn.baseline.pos_err_mm, 0)} mm · ${num(nn.baseline.orn_err_deg, 0)}°</span></div>
      <div><span style="color:#45c99a">■</span> best network (2.05 M): <span class="mono">${num(nn.best.pos_err_mm, 1)} mm · ${num(nn.best.orn_err_deg, 1)}°</span></div></div>`,
    { left: `${insetRect[0] + 18}px`, top: `${insetRect[1] + insetRect[3] + 2}px` });
  // target pose = the input's first 7 numbers
  setPose(tgt, nn.x.slice(0, 3), nn.x.slice(3, 7));

  // sin / cos explainer (unit circle)
  const circ = new Layer(330, 300, 760, 560);
  const note = el(`<div class="panel" style="padding:18px 24px;width:760px"><div style="font-size:21px;line-height:1.45">
      <b>Same inputs and outputs, bigger body:</b> the best model is a residual MLP with
      <span class="mono">2,050,316</span> parameters (baseline <span class="mono">103,820</span>).</div></div>`,
    { left: '330px', top: '880px' });
  const cap = captions(ctx, [
    [c.inputs, c.h1, 'Input: target position (3), orientation quaternion (4) and the <em>seed</em> joints (6)'],
    [c.h1, c.decode, 'Three hidden layers with ReLU; brightness shows the real activations'],
    [c.decode, c.circle, 'Output: sine and cosine of each joint, decoded with <em>atan2</em>'],
    [c.circle, c.note, 'On the circle, +179° and −179° are neighbours, so there is no jump at ±180°'],
  ]);

  function drawNet(t, dim) {
    g.clearRect(0, 0, lay.w, lay.h);
    g.save();
    g.globalAlpha = dim;
    // edges
    for (const e of edges) {
      const A = cols[e.l], B = cols[e.l + 1];
      const a = seg(t, layerT[e.l + 1] - 0.6, layerT[e.l + 1] + 0.4, smooth);
      if (a <= 0) continue;
      g.strokeStyle = `rgba(110,160,220,${0.10 * a})`;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(A.x, A.y[e.i]); g.lineTo(B.x, B.y[e.j]); g.stroke();
      // travelling pulse while this layer lights up
      const u = (t - (layerT[e.l + 1] - 1.0)) / 1.0 - e.ph * 0.35;
      if (u > 0 && u < 1 && e.ph < 0.5) {
        const px = A.x + (B.x - A.x) * u, py = A.y[e.i] + (B.y[e.j] - A.y[e.i]) * u;
        g.fillStyle = `rgba(79,214,242,${0.85 * Math.sin(Math.PI * u)})`;
        g.beginPath(); g.arc(px, py, 2.6, 0, Math.PI * 2); g.fill();
      }
    }
    // nodes
    cols.forEach((col, l) => {
      const a = seg(t, layerT[l] - 0.2, layerT[l] + 0.5, smooth);
      if (a <= 0) return;
      col.y.forEach((y, i) => {
        const b = col.act[i] * a;
        g.fillStyle = `rgba(${Math.round(lerp(26, 79, b))},${Math.round(lerp(44, 214, b))},${Math.round(lerp(71, 242, b))},${0.35 + 0.65 * a})`;
        g.beginPath(); g.arc(col.x, y, col.r, 0, Math.PI * 2); g.fill();
        g.strokeStyle = `rgba(160,200,240,${0.35 * a})`; g.lineWidth = 1; g.stroke();
      });
      if (col.full) {
        text(g, `… ${col.full} neurons`, col.x, bottom + 46, { size: 16, color: C.muted, align: 'center' });
        text(g, ['hidden 1', 'hidden 2', 'hidden 3'][col.idx], col.x, 16, { size: 16, weight: 600, color: C.text2, align: 'center' });
      }
    });
    // input labels + values
    const ia = seg(t, layerT[0], layerT[0] + 0.6, smooth);
    cols[0].y.forEach((y, i) => {
      text(g, IN_LABELS[i], 195, y + 6, { size: 17, weight: 600, color: C.text1, align: 'right', family: 'Inter' });
      text(g, num(nn.x[i], 3), 275, y + 6, { size: 16, color: C.text2, align: 'right', family: 'DejaVu Sans Mono' });
    });
    g.globalAlpha = dim * ia;
    const groups = [[0, 2, 'position'], [3, 6, 'orientation'], [7, 12, 'seed joints']];
    for (const [a0, b0, name] of groups) {
      const y0 = cols[0].y[a0] - 10, y1 = cols[0].y[b0] + 10;
      g.strokeStyle = 'rgba(79,214,242,0.6)'; g.lineWidth = 2;
      g.beginPath(); g.moveTo(130, y0); g.lineTo(122, y0); g.lineTo(122, y1); g.lineTo(130, y1); g.stroke();
      g.save(); g.translate(104, (y0 + y1) / 2); g.rotate(-Math.PI / 2);
      text(g, name, 0, 0, { size: 15, weight: 600, color: C.accent, align: 'center' });
      g.restore();
    }
    g.globalAlpha = dim;
    // output labels + values
    const oa = seg(t, layerT[4], layerT[4] + 0.6, smooth);
    g.globalAlpha = dim * oa;
    cols[4].y.forEach((y, i) => {
      text(g, OUT_LABELS[i], 1106, y + 6, { size: 16, weight: 600, color: C.text1 });
      text(g, num(nn.raw[i], 3), 1262, y + 6, { size: 15, color: C.text2, align: 'right', family: 'DejaVu Sans Mono' });
    });
    g.globalAlpha = dim;
    text(g, '13 inputs', 300, 16, { size: 16, weight: 600, color: C.text2, align: 'center' });
    text(g, '12 outputs', 1085, 16, { size: 16, weight: 600, color: C.text2, align: 'center' });
    g.restore();
  }

  function drawCircle(t, a) {
    const cg = circ.g;
    cg.clearRect(0, 0, circ.w, circ.h);
    if (a <= 0) return;
    cg.save();
    cg.globalAlpha = a;
    cg.fillStyle = 'rgba(15,27,45,0.94)'; roundRect(cg, 0, 0, circ.w, circ.h, 18); cg.fill();
    cg.strokeStyle = 'rgba(35,52,79,1)'; cg.lineWidth = 1; cg.stroke();
    const cx = 210, cy = 290, R0 = 170;
    cg.strokeStyle = 'rgba(160,190,230,0.35)'; cg.lineWidth = 1.5;
    cg.beginPath(); cg.moveTo(cx - R0 - 20, cy); cg.lineTo(cx + R0 + 20, cy); cg.moveTo(cx, cy - R0 - 20); cg.lineTo(cx, cy + R0 + 20); cg.stroke();
    cg.strokeStyle = 'rgba(79,214,242,0.55)'; cg.lineWidth = 3; cg.beginPath(); cg.arc(cx, cy, R0, 0, Math.PI * 2); cg.stroke();
    // angle sweeps from +165° through 180° to -165° (= 195°)
    const u = seg(t, c.circle + 0.8, c.note - 1.2, easeInOut);
    const thDeg = 165 + 30 * u;
    const th = (thDeg * Math.PI) / 180;
    const px = cx + R0 * Math.cos(th), py = cy - R0 * Math.sin(th);
    cg.strokeStyle = 'rgba(238,243,250,0.9)'; cg.lineWidth = 2.5; cg.beginPath(); cg.moveTo(cx, cy); cg.lineTo(px, py); cg.stroke();
    cg.setLineDash([6, 6]);
    cg.strokeStyle = 'rgba(69,201,154,0.9)'; cg.beginPath(); cg.moveTo(px, py); cg.lineTo(px, cy); cg.stroke();
    cg.strokeStyle = 'rgba(233,180,64,0.9)'; cg.beginPath(); cg.moveTo(px, py); cg.lineTo(cx, py); cg.stroke();
    cg.setLineDash([]);
    cg.fillStyle = '#4fd6f2'; cg.beginPath(); cg.arc(px, py, 9, 0, Math.PI * 2); cg.fill();
    const wrapped = thDeg > 180 ? thDeg - 360 : thDeg;
    text(cg, 'Why sine and cosine?', 430, 70, { size: 24, weight: 650, color: C.text1 });
    text(cg, 'angle written as a number', 430, 130, { size: 17, color: C.muted });
    text(cg, `θ = ${num(wrapped, 0)}°`, 430, 168, { size: 30, weight: 650, color: C.text1, family: 'DejaVu Sans Mono' });
    text(cg, 'jumps from +180° to −180°', 430, 198, { size: 17, color: '#ff9c9c' });
    text(cg, 'the same angle as (sin, cos)', 430, 262, { size: 17, color: C.muted });
    text(cg, `sin θ = ${num(Math.sin(th), 3)}`, 430, 300, { size: 24, weight: 600, color: '#45c99a', family: 'DejaVu Sans Mono' });
    text(cg, `cos θ = ${num(Math.cos(th), 3)}`, 430, 336, { size: 24, weight: 600, color: '#e9b440', family: 'DejaVu Sans Mono' });
    text(cg, 'changes smoothly', 430, 366, { size: 17, color: '#8ff08f' });
    text(cg, 'θ = atan2(sin θ, cos θ)', 430, 440, { size: 22, weight: 600, color: C.accent, family: 'DejaVu Sans Mono' });
    text(cg, '(the pair fixes the angle within one turn)', 430, 470, { size: 16, color: C.muted });
    cg.restore();
  }

  return {
    duration: D,
    info: { baseline_mm: nn.baseline.pos_err_mm, best_mm: nn.best.pos_err_mm },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      const ca = Math.min(seg(t, c.circle, c.circle + 0.6, smooth), 1 - seg(t, c.note - 0.3, c.note + 0.3, smooth));
      drawNet(t, 1 - 0.8 * ca);
      drawCircle(t, ca);
      show(dec, seg(t, c.decode, c.decode + 0.7, smooth));
      const ra = seg(t, c.robot, c.robot + 0.8, smooth);
      show(insetCard, ra);
      show(insetLbl, seg(t, c.robot + 0.6, c.robot + 1.4, smooth));
      ghostBase.setOpacity(seg(t, c.robot + 0.2, c.robot + 1.0, smooth));
      ghostBest.setOpacity(seg(t, c.robot + 1.2, c.robot + 2.0, smooth));
      tgt.visible = ra > 0.05;
      tgt.userData.ring.rotation.z = t;
      show(note, seg(t, c.note, c.note + 0.8, smooth));
      const views = [];
      if (ra > 0.01) {
        st.orbit([0.0, 0.05, 0.4], 1.55, lerp(-25, 5, t / D), 16);
        views.push({ scene: st.scene, camera: st.camera, rect: insetRect });
      }
      renderViews(views);
      document.getElementById('gl').style.opacity = ra.toFixed(3);
      cap(t);
    },
  };
}
