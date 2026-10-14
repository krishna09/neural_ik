// Simulation 3 - inverse kinematics: one target pose, several joint solutions; the seed selects one.
import * as THREE from 'three';
import { Studio, render, targetMarker, setPose, project } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, show, place, setHTML, badge } from '../ui.js';
import { seg, smooth, easeInOut, deg, lerp, lerpArr, num, window_ } from '../anim.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const ik = await ctx.data('ik');
  const { D, c } = timeline(ctx, { target: 1.6, b0: 3.6, bStep: 1.6, seed: 11.2, move: 13.2, eq: 17.6 }, 24);
  const st = new Studio();
  const solid = new Robot(model);
  st.zup.add(solid.group);
  const ghosts = ik.branches.map(() => {
    const g = new Robot(model, { style: 'ghost', color: '#cfe3ff', opacity: 0.24 });
    st.zup.add(g.group);
    return g;
  });
  const seedGhost = new Robot(model, { style: 'ghost', color: '#8b9bb4', opacity: 0.35 });
  st.zup.add(seedGhost.group);
  const tgt = targetMarker(0x4fd6f2);
  setPose(tgt, ik.target_pos, ik.target_orn);
  st.zup.add(tgt);
  ik.branches.forEach((b, i) => ghosts[i].setQ(b.q));
  seedGhost.setQ(ik.seed);
  const sel = ik.selected;

  const h = header(ctx, { chap: 'Level 1 · Inverse kinematics', ttl: 'One target pose, several answers',
    subt: 'Inverse kinematics runs backwards: from the pose to the six joint angles.' });
  const p = ik.target_pos, r = ik.target_rpy_deg;
  const left = el(`<div class="panel"><h3>Target pose (input)</h3>
      <div class="kv"><span class="k">position</span><span class="v">${num(p[0], 3)}, ${num(p[1], 3)}, ${num(p[2], 3)} m</span>
      <span class="k">roll · pitch · yaw</span><span class="v">${num(r[0], 0)}°, ${num(r[1], 0)}°, ${num(r[2], 0)}°</span></div>
      <h3 style="margin-top:22px">Joint solutions (output)</h3>
      <div class="solrow head"><b></b><span class="mono">${['J1', 'J2', 'J3', 'J4', 'J5', 'J6'].map((j) => j.padStart(4, '\u2007')).join(' ')}</span><span class="err">FK error</span></div>
      <div id="sols"></div>
      <div class="muted" style="font-size:15px;margin-top:10px">${ik.n_branches_found} distinct solutions found by LM from 96 random starts; 4 shown.<br>Each re-checked in PyBullet.</div></div>`,
    { left: '72px', top: '280px', width: '600px' });
  const sols = left.querySelector('#sols');
  sols.innerHTML = ik.branches.map((b, i) => `<div class="solrow" data-i="${i}"><b>#${i + 1}</b>
      <span class="mono">${b.q.map((v) => num(deg(v), 0).padStart(4, ' ')).join(' ')}</span>
      <span class="mono err">${b.pos_err_mm < 0.001 ? '<0.001' : num(b.pos_err_mm, 3)} mm</span></div>`).join('');
  const css = document.createElement('style');
  css.textContent = `.solrow{display:grid;grid-template-columns:44px 1fr 110px;gap:10px;font-size:19px;padding:7px 8px;border-radius:9px;color:var(--text-2)}
    .solrow b{color:var(--text-1)} .solrow .err{text-align:right;color:var(--muted)} .solrow.head{font-size:15px;color:var(--muted);padding-bottom:0}
    .solrow.sel{background:rgba(57,135,229,.18);box-shadow:inset 3px 0 0 var(--s-blue);color:var(--text-1)}`;
  document.head.appendChild(css);
  const rowEls = [...sols.querySelectorAll('.solrow')];
  const tags = ik.branches.map((_, i) => el(`<div class="label3d">#${i + 1}</div>`));
  const seedTag = el('<div class="label3d">seed · current joints</div>');
  const eq = el(`<div class="panel" style="padding:20px 28px"><div class="eq"><i>q</i> <span class="op">=</span> IK(<i>p</i>, <i>R</i>)</div>
      <div class="t2" style="font-size:20px;margin-top:8px;max-width:520px">Several valid answers, no general closed form:<br>solved iteratively from a <span class="accent">starting guess</span></div></div>`,
    { right: '72px', top: '640px' });
  const pick = el(`<div>${badge('info', `seed → solution #${sel + 1} (joint angles in degrees)`)}</div>`, { right: '72px', top: '560px' });
  const cap = captions(ctx, [
    [c.target, c.b0, 'The task gives a <em>target pose</em> for the gripper'],
    [c.b0, c.seed, 'Several joint configurations reach exactly the same pose'],
    [c.seed, c.eq, 'A <em>seed</em> (the arm\'s current joints) picks the nearby solution'],
    [c.eq, D - 0.2, 'This is why the neural model gets the seed as an input'],
  ]);

  return {
    duration: D,
    info: { selected: sel, branches: ik.branches.map((b) => [b.pos_err_mm, b.self_collision]) },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      show(left, seg(t, 0.5, 1.4, smooth));
      const ta = seg(t, c.target, c.target + 0.7, smooth);
      tgt.visible = ta > 0.01;
      tgt.scale.setScalar(Math.max(1e-3, ta * (1 + 0.06 * Math.sin(t * 3))));
      tgt.userData.ring.rotation.z = t * 0.8;
      const fadeOthers = seg(t, c.move, c.move + 1.0, smooth);
      ghosts.forEach((g, i) => {
        const a0 = c.b0 + i * c.bStep;
        let a = seg(t, a0, a0 + 0.6, smooth);
        if (i !== sel) a *= 1 - 0.75 * fadeOthers;
        g.setOpacity(a);
        g.setColor(i === sel ? lerpColor('#cfe3ff', '#3987e5', seg(t, c.seed + 0.6, c.seed + 1.4)) : '#cfe3ff');
        if (i === sel) a *= 1 - seg(t, c.move + 2.0, c.move + 2.6, smooth);   // the solid arm takes its place
        rowEls[i].style.opacity = String(Math.max(0.15, seg(t, a0, a0 + 0.4, smooth)) * (i === sel ? 1 : 1 - 0.5 * fadeOthers));
        rowEls[i].classList.toggle('sel', i === sel && t > c.seed + 0.6);
        const xy = project(g.linkPos(3), st.camera);
        place(tags[i], xy && [xy[0], xy[1] - 30], a * (1 - fadeOthers * (i === sel ? 0 : 1)));
      });
      // solid robot: hidden until the seed appears, then moves from the seed to the selected solution
      const sa = seg(t, c.seed, c.seed + 0.8, smooth);
      seedGhost.setOpacity(sa * (1 - 0.6 * seg(t, c.move + 1.5, c.move + 2.5)));
      place(seedTag, project(seedGhost.linkPos(4), st.camera)?.map((v, k) => v + (k === 0 ? -150 : 0)), sa * (1 - seg(t, c.move + 1.5, c.move + 2.5)));
      solid.setOpacity(sa);
      solid.setQ(lerpArr(ik.seed, ik.branches[sel].q, seg(t, c.move, c.move + 2.6, easeInOut)));
      show(pick, seg(t, c.seed + 0.8, c.seed + 1.5, smooth));
      show(eq, seg(t, c.eq, c.eq + 0.8, smooth));
      st.orbit([0.05, 0.12, 0.36], 2.3, lerp(-30, -5, t / D), 17, st.camera, [80, 0]);
      render(st);
      cap(t);
    },
  };
}

function lerpColor(a, b, u) {
  return '#' + new THREE.Color(a).lerp(new THREE.Color(b), u).getHexString();
}
