// Simulation 7 - training and validation, from recorded histories only:
// baseline reproduction (instrumented train_ik loop, CPU, 60 epochs) and the team's best run (120 epochs).
import { Studio, renderViews, targetMarker, setPose, project } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, elBack, show, place, setHTML } from '../ui.js';
import { seg, smooth, easeInOut, lerp, lerpArr, num, clamp } from '../anim.js';
import { Layer, lineChart, C, text } from '../chart2d.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const tr = await ctx.data('training');
  const { D, c } = timeline(ctx, { charts: 1.6, ep0: 3.0, ep1: 17.0, note: 18.6 }, 25);
  const base = tr.baseline.filter((r) => r.phase === 'ik');
  const best = tr.best;
  const nB = base.length, nBest = best.length;

  const h = header(ctx, { chap: 'Level 1 · Training', ttl: 'Training and validation',
    subt: 'Adam · learning rate 0.001 · batch 256 · 85 / 15 split · recorded runs only' });
  const L1 = new Layer(72, 250, 700, 320);
  const L2 = new Layer(72, 600, 700, 320);
  const st = new Studio({ floorRadius: 1.4 });
  const ghost = new Robot(model, { style: 'ghost', color: '#f2c45a', opacity: 0.78 });
  st.zup.add(ghost.group);
  const tgt = targetMarker();
  setPose(tgt, tr.case.target_pos, tr.case.target_orn);
  st.zup.add(tgt);
  const rect = [880, 250, 968, 500];
  const card = elBack(`<div class="panel" style="padding:0;width:${rect[2]}px;height:${rect[3] + 110}px;background:rgba(15,27,45,.45)"></div>`,
    { left: `${rect[0]}px`, top: `${rect[1] - 10}px` });
  const ep = el(`<div style="font-size:22px;line-height:1.4"><div class="t2" style="font-size:16px;letter-spacing:.12em;text-transform:uppercase">one validation target · baseline prediction per epoch</div>
      <div><span id="e" class="mono" style="font-size:30px;font-weight:650">epoch 1</span> &nbsp; <span class="t2">error</span> <span id="err" class="mono" style="font-size:30px;font-weight:650"></span></div></div>`,
    { left: `${rect[0] + 26}px`, top: `${rect[1] + rect[3] - 4}px` });
  const note = el(`<div style="font-size:16px;line-height:1.5;color:var(--muted);width:930px">
      Baseline: this reproduction of <span class="mono">train_ik.py</span> on a CPU (20,000 samples, 26 s).
      Best: team run <span class="mono">${tr.best_run.id}</span>, residual MLP, 200,000 samples, ${tr.best_run.gpu}, ${Math.round(tr.best_run.train_secs)} s.</div>`,
    { left: `${rect[0] + 26}px`, top: `${rect[1] + rect[3] + 66}px` });
  const cap = captions(ctx, [
    [c.ep0, c.ep1, 'Both losses fall together: the network is learning, not memorising'],
    [c.ep1, D - 0.2, 'More data and a deeper network: 115 mm → 23 mm mean validation error'],
  ]);

  const cq = tr.case.epochs;
  const sMSE = { train: base.map((r) => [r.epoch, r.train_fit_mse]), val: base.map((r) => [r.epoch, r.val_mse]) };
  const sPos = { base: base.map((r) => [r.epoch, r.val_pos_mean_mm]), best: best.map((r) => [r.epoch, r.val_pos_mean_mm]) };

  return {
    duration: D,
    info: { baseline_final_mm: base[nB - 1].val_pos_mean_mm, best_final_mm: best[nBest - 1].val_pos_mean_mm },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      const a = seg(t, c.charts, c.charts + 0.8, smooth);
      const u = seg(t, c.ep0, c.ep1, (x) => x);
      L1.opacity(a); L2.opacity(a);
      L1.clear(); L2.clear();
      text(L1.g, 'Baseline network · loss per epoch (sin / cos MSE)', 0, 22, { size: 19, weight: 600, color: C.text1 });
      lineChart(L1.g, { x: 70, y: 60, w: 450, h: 200, xd: [1, 60], yd: [0.02, 0.4], ylog: true, xticks: [1, 20, 40, 60],
        yticks: [0.02, 0.05, 0.1, 0.2, 0.4], yfmt: (v) => String(v), xlabel: 'epoch',
        series: [
          { pts: sMSE.train, color: '#c8d3e3', progress: u, label: (x, y) => `train ${num(y, 3)}`, labelDy: 20 },
          { pts: sMSE.val, color: C.accent, progress: u, label: (x, y) => `validation ${num(y, 3)}`, labelDy: -6 },
        ] });
      text(L2.g, 'Mean validation position error (FK-checked, mm)', 0, 22, { size: 19, weight: 600, color: C.text1 });
      lineChart(L2.g, { x: 70, y: 60, w: 450, h: 200, xd: [1, 120], yd: [15, 600], ylog: true, xticks: [1, 30, 60, 90, 120],
        yticks: [20, 50, 100, 200, 500], xlabel: 'epoch',
        series: [
          { pts: sPos.base, color: '#e9b440', progress: u, label: (x, y) => `baseline ${num(y, 0)} mm`, labelDy: -8 },
          { pts: sPos.best, color: '#45c99a', progress: u, label: (x, y) => `best ${num(y, 1)} mm`, labelDy: 18 },
        ] });
      // ghost: baseline prediction for one validation target, interpolated between epoch snapshots
      const fe = u * (cq.length - 1);
      const k0 = Math.floor(fe), k1 = Math.min(cq.length - 1, k0 + 1);
      const f = smooth(fe - k0);
      ghost.setQ(lerpArr(cq[k0].q, cq[k1].q, f));
      const ra = seg(t, c.charts + 0.4, c.charts + 1.2, smooth);
      ghost.setOpacity(ra);
      tgt.visible = ra > 0.05;
      tgt.userData.ring.rotation.z = t;
      show(card, ra);
      show(ep, ra);
      const kk = Math.round(fe);
      setHTML(ep.querySelector('#e'), `epoch ${cq[kk].epoch}`);
      setHTML(ep.querySelector('#err'), `${num(cq[kk].pos_err_mm, 0)} mm`);
      show(note, seg(t, c.note, c.note + 0.8, smooth));
      st.orbit([-0.1, 0.26, 0.34], 1.75, lerp(8, 38, t / D), 14);
      renderViews([{ scene: st.scene, camera: st.camera, rect }]);
      cap(t);
    },
  };
}
