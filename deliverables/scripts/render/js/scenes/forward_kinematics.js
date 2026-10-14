// Simulation 2 - forward kinematics: known joint angles -> end-effector pose (PyBullet values).
import * as THREE from 'three';
import { Studio, render, axesMarker, project, Trail } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, show, place, setHTML } from '../ui.js';
import { seg, smooth, easeInOut, deg, lerp, num, range } from '../anim.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const fk = await ctx.data('fk');
  const lim = model.data.limits;
  const { D, c } = timeline(ctx, { j0: 2.4, step: 1.9, pose: 14.4, eq: 17.0 }, 23);
  const st = new Studio();
  const robot = new Robot(model);
  st.zup.add(robot.group);
  const ee = axesMarker(0.11, 0.0045);
  ee.matrixAutoUpdate = false;
  st.zup.add(ee);
  const trail = new Trail(0x4fd6f2, 0.0035, 0.9);
  st.zup.add(trail.mesh);

  const h = header(ctx, { chap: 'Level 1 · Forward kinematics', ttl: 'Joint angles in, gripper pose out',
    subt: 'Chaining the six joint rotations gives one exact pose.' });
  const css = document.createElement('style');
  css.textContent = `.srow{display:grid;grid-template-columns:44px 1fr 98px;align-items:center;gap:14px;font-size:21px;
    padding:7px 8px;border-radius:10px;color:var(--text-2)} .srow b{color:var(--text-1)}
    .srow.on{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
    .track{position:relative;height:8px;border-radius:4px;background:var(--surface-3)}
    .track .fill{position:absolute;top:0;height:8px;border-radius:4px;background:var(--accent);opacity:.75}
    .track .zero{position:absolute;top:-5px;width:2px;height:18px;background:var(--muted)}
    .srow .v{text-align:right;font-family:var(--mono);color:var(--text-1)}`;
  document.head.appendChild(css);
  const rows = lim.map((l, i) => `<div class="srow"><b>J${i + 1}</b><div class="track"><div class="zero"></div><div class="fill"></div></div><span class="v">0.0°</span></div>`).join('');
  const qp = el(`<div class="panel"><h3>Input · joint angles q</h3>${rows}<div class="muted" style="font-size:16px;margin-top:10px">bar = position inside the joint's limits</div></div>`,
    { left: '72px', top: '290px', width: '520px' });
  const rowEls = [...qp.querySelectorAll('.srow')];
  rowEls.forEach((r, i) => {
    const [lo, hi] = lim[i];
    r.querySelector('.zero').style.left = `${(100 * (0 - lo)) / (hi - lo)}%`;
  });
  const P = fk.ee_pos, RPY = fk.ee_rpy_deg, Q = fk.ee_orn;
  const out = el(`<div class="panel"><h3>Output · end-effector pose</h3>
      <div class="kv"><span class="k">position x</span><span class="v">${num(P[0], 3)} m</span>
      <span class="k">position y</span><span class="v">${num(P[1], 3)} m</span>
      <span class="k">position z</span><span class="v">${num(P[2], 3)} m</span>
      <span class="k">roll</span><span class="v">${num(RPY[0], 1)}°</span>
      <span class="k">pitch</span><span class="v">${num(RPY[1], 1)}°</span>
      <span class="k">yaw</span><span class="v">${num(RPY[2], 1)}°</span></div>
      <div class="t2" style="font-size:17px;margin-top:12px">quaternion x y z w: <span class="mono">${Q.map((v) => num(v, 3)).join('  ')}</span></div>
      <div class="muted" style="font-size:15px;margin-top:8px">PyBullet getLinkState(link 6) for the q on the left</div></div>`,
    { right: '72px', top: '290px', width: '470px' });
  const eq = el(`<div class="panel" style="padding:20px 28px"><div class="eq"><i>T</i>(<i>q</i>) <span class="op">=</span> <i>T</i><sub>0</sub>·<i>T</i><sub>1</sub>(<i>q</i><sub>1</sub>)·<i>T</i><sub>2</sub>(<i>q</i><sub>2</sub>) ··· <i>T</i><sub>6</sub>(<i>q</i><sub>6</sub>)</div>
      <div class="t2" style="font-size:20px;margin-top:8px">FK: <span class="mono">q → (p, R)</span> · always defined, one answer</div></div>`,
    { right: '72px', top: '700px' });
  const eeTag = el('<div class="label3d acc">end effector</div>');
  const cap = captions(ctx, [
    [0.5, c.j0, 'Start from the home pose: every joint at <em>0°</em>'],
    [c.j0, c.pose, 'Rotate the joints one by one; the <em>end effector</em> traces its path'],
    [c.pose, D - 0.2, 'Forward kinematics returns <em>one exact pose</em> for any joint vector'],
  ]);

  const qT = fk.q;
  const qAt = (t) => qT.map((v, i) => v * seg(t, c.j0 + i * c.step, c.j0 + (i + 1) * c.step, easeInOut));
  const N = 140;                                              // trail samples
  const trailPts = [];
  for (let k = 0; k <= N; k++) {
    const tt = c.j0 + ((c.pose - c.j0) * k) / N;
    robot.setQ(qAt(tt));
    const p = robot.eePos();
    trailPts.push([tt, [p.x, p.y, p.z]]);
  }

  return {
    duration: D,
    info: { fk_js_vs_pybullet_mm: (() => { robot.setQ(qT); const p = robot.eePos(); return 1000 * Math.hypot(p.x - P[0], p.y - P[1], p.z - P[2]); })() },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      show(qp, seg(t, 0.5, 1.4, smooth));
      const q = qAt(t);
      robot.setQ(q);
      robot.clearHighlights();
      let active = -1;
      for (let i = 0; i < 6; i++) if (t >= c.j0 + i * c.step && t < c.j0 + (i + 1) * c.step) active = i;
      if (active >= 0) robot.highlight(active + 1, 1);
      rowEls.forEach((r, i) => {
        const [lo, hi] = lim[i];
        const z = (0 - lo) / (hi - lo), v = (q[i] - lo) / (hi - lo);
        const f = r.querySelector('.fill');
        f.style.left = `${100 * Math.min(z, v)}%`;
        f.style.width = `${Math.max(0.6, 100 * Math.abs(v - z))}%`;
        r.classList.toggle('on', i === active);
        setHTML(r.querySelector('.v'), `${num(deg(q[i]), 1)}°`);
      });
      trail.set(trailPts.filter(([tt]) => tt <= t).map(([, p]) => p));
      const ea = seg(t, c.pose - 0.4, c.pose + 0.5, smooth);
      ee.matrix.copy(robot.frames[6]).multiply(new THREE.Matrix4().makeScale(...Array(3).fill(Math.max(1e-3, ea))));
      ee.visible = ea > 0.01;
      show(out, seg(t, c.pose, c.pose + 0.8, smooth));
      show(eq, seg(t, c.eq, c.eq + 0.8, smooth));
      place(eeTag, project(robot.eePos(), st.camera)?.map((v, k) => v + (k === 0 ? 150 : 10)), Math.min(seg(t, 1.0, 1.8, smooth), 1 - ea));
      st.orbit([0.12, 0.0, 0.34], 2.15, lerp(-58, -30, t / D), 16, st.camera, [-90, 10]);
      render(st);
      cap(t);
    },
  };
}
