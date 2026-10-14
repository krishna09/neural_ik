// Simulation 4 - dataset generation: sample joints -> FK pose -> self-collision label -> one row.
// Rows are the first samples of the 200 000-row dataset exactly as generate_dataset.py wrote them.
import * as THREE from 'three';
import { Studio, render, points, glow, project } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, show, place, setHTML } from '../ui.js';
import { seg, smooth, easeInOut, deg, lerp, num, commas, clamp } from '../anim.js';

const FREE = '#5aa2f0', COLL = '#e04646';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const ds = await ctx.data('dataset');
  const S = ds.samples;
  const { D, c } = timeline(ctx, { slow0: 2.2, slowN: 6, slowDt: 1.15, fast1: 9.6, fast2: 17.0, cloud: 18.6, stats: 21.0 }, 26);
  const st = new Studio({ floorRadius: 1.8 });
  const robot = new Robot(model);
  st.zup.add(robot.group);
  const near = points(S.map((s) => s.pos), S.map((s) => (s.label ? COLL : FREE)), 0.05);
  st.zup.add(near);
  const cloud = points(ds.cloud_pos, ds.cloud_label.map((l) => (l ? COLL : FREE)), 0.026, 0.9);
  st.zup.add(cloud);
  const sparks = [0, 1, 2, 3].map(() => { const g = glow(0xff4b4b, 0.07, 1); st.zup.add(g); return g; });

  // sample index shown at time t (slow phase: one per slowDt; fast phase accelerates through the rest)
  const nS = S.length;
  const slowEnd = c.slow0 + c.slowN * c.slowDt;
  const idxAt = (t) => {
    if (t < c.slow0) return -1;
    if (t < slowEnd) return Math.floor((t - c.slow0) / c.slowDt);
    const u = clamp((t - slowEnd) / (c.fast2 - slowEnd));
    return Math.min(nS - 1, c.slowN + Math.floor((nS - c.slowN) * (u * u * 0.4 + 0.6 * u)));
  };

  const h = header(ctx, { chap: 'Level 1 · Dataset', ttl: 'Generating the training data',
    subt: 'No download needed: every sample comes from the simulator.' });
  const steps = ['Sample random joint angles', 'Forward kinematics → pose', 'PyBullet self-collision check', 'Save one row'];
  const strip = el(`<div class="strip">${steps.map((s, i) => `<div class="step"><span class="n">${i + 1}</span>${s}</div>`).join('<span class="arr">→</span>')}</div>`,
    { left: '72px', top: '228px' });
  const css = document.createElement('style');
  css.textContent = `.strip{display:flex;align-items:center;gap:12px;font-size:19px;color:var(--text-2)}
    .strip .step{display:flex;align-items:center;gap:10px;padding:9px 16px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border)}
    .strip .step.on{background:var(--accent-soft);border-color:rgba(79,214,242,.6);color:var(--text-1)}
    .strip .n{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;background:var(--surface-3);font-size:14px;font-weight:700;color:var(--text-1)}
    .strip .arr{color:var(--muted)}
    .tbl{font-family:var(--mono);font-size:16.5px;color:var(--text-2);border-collapse:collapse}
    .tbl td,.tbl th{padding:5px 7px;text-align:right;white-space:nowrap}
    .tbl th{font-family:var(--font);font-weight:600;font-size:14px;color:var(--muted);letter-spacing:.06em}
    .tbl tr.new td{color:var(--text-1);background:rgba(79,214,242,.10)}
    .lab{display:inline-flex;align-items:center;gap:6px;font-family:var(--font);font-weight:600}
    .lab i{display:inline-block;width:10px;height:10px;border-radius:50%}`;
  document.head.appendChild(css);
  const stepEls = [...strip.querySelectorAll('.step')];
  const table = el(`<div class="panel" style="padding:18px 20px"><h3>Dataset rows</h3><table class="tbl"><thead><tr><th>#</th><th>q1</th><th>q2</th><th>q3</th><th>q4</th><th>q5</th><th>q6</th><th>x</th><th>y</th><th>z</th><th>label</th></tr></thead><tbody></tbody></table>
      <div class="muted" style="font-size:14px;margin-top:8px;font-family:var(--font)">angles in degrees · position in metres · + quaternion (4 numbers) per row</div></div>`,
    { left: '72px', top: '320px' });
  const tbody = table.querySelector('tbody');
  const counter = el(`<div class="panel" style="padding:18px 24px;min-width:420px"><h3>Generated</h3>
      <div class="big mono" id="cnt">0</div><div class="t2" style="font-size:19px" id="cc"></div>
      <div class="legend" style="margin-top:12px"><span><span class="sw" style="background:#7db4f5"></span>collision-free</span><span><span class="sw" style="background:#d03b3b"></span>self-colliding</span></div></div>`,
    { right: '72px', top: '320px' });
  const phase = el('<div class="badge info"></div>', { right: '72px', top: '560px' });
  const stats = el(`<div class="panel" style="padding:20px 26px;width:620px"><h3>Main dataset (ds_20260928-163059)</h3>
      <div class="kv"><span class="k">samples</span><span class="v">${commas(ds.n_total)}</span>
      <span class="k">self-colliding</span><span class="v">${commas(ds.n_colliding)} (${num((100 * ds.n_colliding) / ds.n_total, 1)} %)</span>
      <span class="k">uniform samples</span><span class="v">70 %</span>
      <span class="k">near the joint limits</span><span class="v">30 %</span>
      <span class="k">IK training rows</span><span class="v">collision-free only</span></div></div>`,
    { right: '72px', top: '320px' });
  const ctag = el('<div class="label3d" style="border-color:rgba(208,59,59,.7);color:#ffb3b3">links in contact</div>');
  // the near-limit caption starts with the first near-limit sample, together with the phase badge
  const k2 = S.findIndex((s) => s.phase === 'near-limit');
  let t2 = slowEnd;
  while (k2 >= 0 && t2 < c.fast2 && idxAt(t2) < k2) t2 += 0.05;
  const cap = captions(ctx, [
    [c.slow0, slowEnd, 'Each sample: random joint angles → pose → <em>collision label</em>'],
    [slowEnd, t2, '70 % of samples: uniform inside the joint limits'],
    [t2, c.fast2, '30 % of samples sit near the joint limits, where self-collisions concentrate'],
    [c.cloud, D - 0.2, 'End-effector positions of 7,000 dataset rows: <span style="color:#7db4f5">free</span> and <span style="color:#ff8a8a">self-colliding</span>'],
  ]);
  const rowHTML = (s, isNew) => `<tr class="${isNew ? 'new' : ''}"><td>${s.i}</td>${s.q.map((v) => `<td>${num(deg(v), 0)}</td>`).join('')}
      ${s.pos.map((v) => `<td>${num(v, 3)}</td>`).join('')}<td><span class="lab" style="color:${s.label ? '#ff9c9c' : '#a9cdf7'}"><i style="background:${s.label ? '#d03b3b' : '#7db4f5'}"></i>${s.label ? 'collision' : 'free'}</span></td></tr>`;

  return {
    duration: D,
    info: { n_samples_shown: nS, agree: ds.agree },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      show(strip, Math.min(seg(t, 0.4, 1.2, smooth), 1 - seg(t, c.cloud - 0.6, c.cloud, smooth)));
      const k = idxAt(t);
      const s = k >= 0 ? S[k] : null;
      robot.setQ(s && t < c.cloud ? s.q : [0, -0.45, -0.95, 0, 1.35, 0]);
      robot.clearHighlights();
      // pipeline step highlight (slow phase only)
      let stepOn = -1;
      if (t >= c.slow0 && t < slowEnd) stepOn = Math.min(3, Math.floor((((t - c.slow0) % c.slowDt) / c.slowDt) * 4));
      stepEls.forEach((e, i) => e.classList.toggle('on', i === stepOn || (t >= slowEnd && t < c.fast2)));
      const showContact = s && s.label && t < c.cloud && (t < slowEnd ? stepOn >= 2 : true);
      if (showContact) for (const cp of s.contacts) { robot.highlight(cp.a, 1, '#d03b3b'); robot.highlight(cp.b, 1, '#d03b3b'); }
      sparks.forEach((g, i) => {
        const cp = showContact ? s.contacts[i] : null;
        g.visible = !!cp;
        if (cp) g.position.set(...cp.pa);
      });
      place(ctag, showContact && t < slowEnd ? project(s.contacts[0].pa, st.camera)?.map((v, j) => v + (j === 0 ? 120 : -40)) : null, 1);
      // points: samples up to k (the slow phase adds the point at the "save" step), then the big cloud
      let n = k + 1;
      if (t < slowEnd && stepOn < 3) n = k;
      const cloudN = Math.floor(ds.cloud_pos.length * seg(t, c.cloud, c.cloud + 3.0, (u) => u * u));
      near.geometry.setDrawRange(0, Math.max(0, n));
      cloud.geometry.setDrawRange(0, cloudN);
      // table: the last 7 rows
      show(table, Math.min(seg(t, 1.2, 2.0, smooth), 1 - seg(t, c.cloud, c.cloud + 0.6, smooth)));
      const last = Math.max(0, n);
      const from = Math.max(0, last - 7);
      setHTML(tbody, S.slice(from, last).map((r, j) => rowHTML(r, j === last - from - 1)).join(''));
      show(counter, Math.min(seg(t, 1.2, 2.0, smooth), 1 - seg(t, c.stats - 0.6, c.stats, smooth)));
      const shown = Math.max(0, n) + (cloudN > 0 ? cloudN : 0);
      const nc = S.slice(0, Math.max(0, n)).filter((r) => r.label).length + ds.cloud_label.slice(0, cloudN).filter((l) => l).length;
      setHTML(counter.querySelector('#cnt'), commas(shown));
      setHTML(counter.querySelector('#cc'), shown ? `${commas(nc)} self-colliding (${num((100 * nc) / shown, 1)} %)` : '&nbsp;');
      const ph = s && s.phase === 'near-limit' ? 'Phase 2 · near the joint limits (30 %)' : 'Phase 1 · uniform in the joint limits (70 %)';
      setHTML(phase, ph);
      show(phase, Math.min(seg(t, slowEnd, slowEnd + 0.5, smooth), 1 - seg(t, c.cloud, c.cloud + 0.5, smooth)));
      show(stats, seg(t, c.stats, c.stats + 0.8, smooth));
      const pull = seg(t, c.cloud, c.cloud + 4, easeInOut);
      st.orbit([0.05, 0, 0.3 + 0.05 * pull], lerp(2.2, 3.3, pull), lerp(-40, -15, t / D), lerp(18, 24, pull), st.camera, [lerp(170, -180, pull), lerp(40, 110, pull)]);
      render(st);
      cap(t);
    },
  };
}
