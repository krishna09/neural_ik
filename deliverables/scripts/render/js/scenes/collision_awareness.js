// Simulation 6 - collision awareness: PyBullet contact vs the learned classifier along a real motion,
// and how the classifier enters the IK training loss.
import * as THREE from 'three';
import { Studio, render, glow, project } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, show, place, setHTML, badge } from '../ui.js';
import { seg, smooth, easeInOut, lerp, lerpArr, num, clamp } from '../anim.js';
import { Layer, lineChart, C, text } from '../chart2d.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const cd = await ctx.data('collision');
  const path = cd.path;
  const { D, c } = timeline(ctx, { panels: 1.6, move0: 3.4, move1: 12.4, loss: 14.6, note: 19.0 }, 24);
  const st = new Studio();
  const robot = new Robot(model);
  st.zup.add(robot.group);
  const sparks = [0, 1, 2, 3, 4, 5].map(() => { const g = glow(0xff4b4b, 0.08, 1); st.zup.add(g); return g; });

  const h = header(ctx, { chap: 'Level 1 · Collision awareness', ttl: 'Teaching the model what to avoid',
    subt: 'A small network predicts self-collision from the six joint angles alone.' });
  const cls = el(`<div class="panel" style="width:600px"><h3>Collision classifier C(q)</h3>
      <div style="font-size:21px;line-height:1.5;color:var(--text-2)">6 joint angles → 64 → 64 → 1 · <span class="mono" style="color:var(--text-1)">4,673</span> parameters<br>
      trained on the labelled dataset (binary cross-entropy)</div>
      <div class="kv" style="margin-top:14px"><span class="k">test accuracy (30,000 poses)</span><span class="v">98.4 %</span>
      <span class="k">mean P on free poses</span><span class="v">${num(cd.mean_p_free, 3)}</span>
      <span class="k">mean P on colliding poses</span><span class="v">${num(cd.mean_p_colliding, 3)}</span></div></div>`,
    { left: '72px', top: '250px' });
  const live = el(`<div class="panel" style="width:560px"><h3>Along this motion</h3>
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:20px;color:var(--text-2)">
        <span>PyBullet contact check</span><span id="pb"></span></div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:14px;font-size:20px;color:var(--text-2)">
        <span>classifier P(collision)</span><span class="big mono" id="pv" style="font-size:40px">0.00</span></div></div>`,
    { right: '72px', top: '250px' });
  const chartL = new Layer(1920 - 72 - 560, 470, 560, 300);
  const loss = el(`<div class="panel" style="padding:20px 26px;width:600px"><h3>IK training loss</h3><div class="eq" style="font-size:30px"><i>L</i> <span class="op">=</span> MSE(sin, cos) <span class="op">+</span> <span class="hl">0.05 · σ(<i>C</i>(<i>q̂</i>))</span></div>
      <div class="t2" style="font-size:19px;margin-top:10px;line-height:1.4">The frozen classifier scores each predicted pose <i>q̂</i>;<br>the penalty is differentiable, so it shapes training.</div>
      <div style="font-size:19px;margin-top:8px">penalty for this pose: <span class="mono" id="pen">0.000</span></div></div>`,
    { left: '72px', top: '540px' });
  const note = el(`<div>${badge('warn', 'A soft penalty, not a guarantee: every answer is re-checked in PyBullet')}</div>`, { right: '72px', top: '800px' });
  const ctag = el('<div class="label3d" style="border-color:rgba(208,59,59,.7);color:#ffb3b3"></div>');
  const cap = captions(ctx, [
    [c.move0, c.move1, 'The arm leans forward until the <em>wrist touches the base</em>'],
    [c.move1, c.loss, 'The classifier\'s probability rises as the arm enters contact'],
    [c.loss, D - 0.2, 'In training, this probability is added to the loss for every predicted pose'],
  ]);

  const pts = path.map((s) => [s.t, Math.max(s.p, 1e-3)]);
  const tContact = path[cd.first_contact].t;

  return {
    duration: D,
    info: { first_contact_t: tContact, p_at_contact: path[cd.first_contact].p },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      const u = seg(t, c.move0, c.move1, easeInOut);
      const fi = u * (path.length - 1);
      const k = Math.min(path.length - 1, Math.round(fi));
      const sNow = path[k];
      robot.setQ(lerpArr(cd.q_safe, cd.q_col, u));
      robot.clearHighlights();
      if (sNow.contact) for (const l of sNow.links) robot.highlight(l, 1, '#d03b3b');
      sparks.forEach((g, i) => { const p = sNow.contact ? sNow.points[i] : null; g.visible = !!p; if (p) g.position.set(...p); });
      place(ctag, sNow.contact ? project(sNow.points[0], st.camera)?.map((v, j) => v + (j === 0 ? 190 : 30)) : null, 1);
      setHTML(ctag, sNow.contact ? `contact: link ${sNow.links.join(' ↔ link ')}` : '');
      show(cls, seg(t, c.panels, c.panels + 0.8, smooth));
      show(live, seg(t, c.panels + 0.3, c.panels + 1.1, smooth));
      // probability interpolated between path samples
      const k0 = Math.floor(fi), k1 = Math.min(path.length - 1, k0 + 1);
      const p = lerp(path[k0].p, path[k1].p, fi - k0);
      setHTML(live.querySelector('#pv'), num(p, 2));
      live.querySelector('#pv').style.color = p >= 0.5 ? '#ff9c9c' : '#eef3fa';
      setHTML(live.querySelector('#pb'), sNow.contact ? badge('bad', `contact · links ${sNow.links.join(', ')}`) : badge('ok', 'no contact'));
      // chart
      const g = chartL.g;
      chartL.clear();
      chartL.opacity(seg(t, c.panels + 0.6, c.panels + 1.4, smooth));
      g.fillStyle = 'rgba(208,59,59,0.14)';
      const x0 = 70, w0 = 450;
      g.fillRect(x0 + w0 * tContact, 30, w0 * (1 - tContact), 200);
      text(g, 'PyBullet: contact', x0 + w0 - 6, 24, { size: 15, color: '#ff9c9c', align: 'right' });
      lineChart(g, { x: x0, y: 30, w: w0, h: 200, xd: [0, 1], yd: [0, 1], yticks: [0, 0.5, 1], xticks: [0, 0.5, 1],
        xfmt: (v) => `${Math.round(v * 100)} %`, yfmt: (v) => v.toFixed(1), xlabel: 'progress of the motion', ylabel: 'P(collision)',
        series: [{ pts, color: C.accent, progress: u, width: 3 }] });
      g.setLineDash([5, 5]); g.strokeStyle = 'rgba(250,178,25,0.7)'; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x0, 30 + 100); g.lineTo(x0 + w0, 30 + 100); g.stroke(); g.setLineDash([]);
      show(loss, seg(t, c.loss, c.loss + 0.8, smooth));
      setHTML(loss.querySelector('#pen'), `0.05 × ${num(p, 2)} = ${num(0.05 * p, 3)}`);
      show(note, seg(t, c.note, c.note + 0.8, smooth));
      st.orbit([0.1, 0.02, 0.3], 2.25, lerp(-75, -55, t / D), 13, st.camera, [40, -40]);
      render(st);
      cap(t);
    },
  };
}
