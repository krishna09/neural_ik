// Simulation 8 - inference: target -> network guess -> LM refinement -> PyBullet check, three held-out targets.
import * as THREE from 'three';
import { Studio, render, targetMarker, setPose, project } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, show, place, setHTML, badge } from '../ui.js';
import { seg, smooth, easeInOut, lerp, lerpArr, num, window_ } from '../anim.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const inf = await ctx.data('inference');
  const cases = inf.cases;
  const { D, c } = timeline(ctx, { c0: 1.8, per: 9.0 }, 1.8 + 3 * 9.0 + 1.2);
  const st = new Studio();
  const robot = new Robot(model);
  const ghost = new Robot(model, { style: 'ghost', color: '#45c99a', opacity: 0.55 });
  st.zup.add(robot.group, ghost.group);
  const tgt = targetMarker();
  st.zup.add(tgt);
  const missGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const miss = new THREE.Line(missGeo, new THREE.LineDashedMaterial({ color: 0xffd47a, dashSize: 0.01, gapSize: 0.007, depthTest: false }));
  miss.renderOrder = 20;
  st.zup.add(miss);

  const h = header(ctx, { chap: 'Level 1 · Inference', ttl: 'Guess, polish, check',
    subt: 'The network guesses, the LM solver removes the error, PyBullet verifies.' });
  const inp = el(`<div class="panel" style="width:640px"><h3>Input · 13 numbers</h3><div id="iv"></div></div>`, { left: '72px', top: '270px' });
  const steps = el(`<div class="panel" style="width:600px"><h3 id="ct">Target 1 of 3</h3>
      <div class="istep" id="s1"><div class="ih"><span class="n">1</span>Network prediction<span class="mono t">${num(inf.timing.network_ms, 1)} ms</span></div><div class="ib" id="b1"></div></div>
      <div class="istep" id="s2"><div class="ih"><span class="n">2</span>LM refinement<span class="mono t">${num(inf.timing.hybrid_ms, 0)} ms in total</span></div><div class="ib" id="b2"></div></div>
      <div class="istep" id="s3"><div class="ih"><span class="n">3</span>PyBullet check</div><div class="ib" id="b3"></div></div>
      <div class="muted" style="font-size:14px;margin-top:10px">mean times over 2,000 targets, one CPU thread · acceptance: ≤ 1 mm and ≤ 1°, no self-collision, inside limits</div></div>`,
    { right: '72px', top: '250px' });
  const css = document.createElement('style');
  css.textContent = `.istep{padding:12px 4px;border-top:1px solid var(--border)} .istep:first-of-type{border-top:none}
    .ih{display:flex;align-items:center;gap:12px;font-size:21px;font-weight:600;color:var(--text-1)}
    .ih .t{margin-left:auto;font-size:17px;font-weight:400;color:var(--muted)}
    .ih .n{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--surface-3);font-size:15px}
    .ib{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px;font-size:19px;color:var(--text-2);min-height:34px}
    .istep.dim{opacity:.35} .iv{display:grid;grid-template-columns:150px 1fr;gap:6px 12px;font-size:19px}
    .iv .k{color:var(--text-2)} .iv .v{font-family:var(--mono);color:var(--text-1)}`;
  document.head.appendChild(css);
  const missTag = el('<div class="label3d" style="border-color:rgba(250,178,25,.7);color:#ffd47a"></div>');
  const cap = captions(ctx, [
    [c.c0, c.c0 + c.per, 'The network (green ghost) answers in about a millisecond, but misses by centimetres'],
    [c.c0 + c.per, c.c0 + 2 * c.per, 'Starting from that guess, the LM solver removes the remaining error'],
    [c.c0 + 2 * c.per, D - 0.2, 'Only answers that pass every PyBullet check are accepted'],
  ]);
  const ivHTML = (cs) => {
    const x = cs.x;
    return `<div class="iv"><span class="k">position (m)</span><span class="v">${x.slice(0, 3).map((v) => num(v, 3)).join('  ')}</span>
      <span class="k">quaternion</span><span class="v">${x.slice(3, 7).map((v) => num(v, 2)).join('  ')}</span>
      <span class="k">seed (rad)</span><span class="v">${x.slice(7, 13).map((v) => num(v, 2)).join('  ')}</span></div>`;
  };

  return {
    duration: D,
    info: { cases: cases.map((cs) => [cs.network.pos_err_mm, cs.hybrid.pos_err_mm]) },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      const i = Math.max(0, Math.min(2, Math.floor((t - c.c0) / c.per)));
      const cs = cases[i];
      const t0 = c.c0 + i * c.per, u = (t - t0) / c.per;           // 0..1 inside this case
      const ta = window_(t, t0, t0 + c.per, 0.35);
      setPose(tgt, cs.target_pos, cs.target_orn);
      tgt.scale.setScalar(Math.max(1e-3, seg(u, 0.0, 0.08, smooth)));
      tgt.userData.ring.rotation.z = t;
      // robot: seed -> (network guess shown as ghost) -> final answer
      const uN = seg(u, 0.14, 0.24, smooth), uR = seg(u, 0.42, 0.62, easeInOut);
      ghost.setQ(cs.network.q);
      ghost.setOpacity(uN * (1 - 0.6 * seg(u, 0.75, 0.85)) * ta);
      robot.setQ(lerpArr(cs.seed, cs.hybrid.q, uR));                // the arm itself only executes the verified answer
      robot.clearHighlights();
      const okGlow = seg(u, 0.66, 0.74, smooth) * (1 - seg(u, 0.9, 0.98, smooth));
      for (let k = 1; k <= 6; k++) robot.highlight(k, 0.55 * okGlow, '#3987e5');
      // miss line from the network's end effector to the target
      const ge = ghost.eePos();
      missGeo.setFromPoints([ge, new THREE.Vector3(...cs.target_pos)]);
      miss.computeLineDistances();
      const ma = uN * (1 - seg(u, 0.42, 0.5, smooth)) * ta;
      miss.visible = ma > 0.02;
      miss.material.opacity = ma;
      miss.material.transparent = true;
      const mid = ge.clone().add(new THREE.Vector3(...cs.target_pos)).multiplyScalar(0.5);
      setHTML(missTag, `${num(cs.network.pos_err_mm, 1)} mm`);
      place(missTag, project(mid, st.camera)?.map((v, k) => v + (k === 0 ? 70 : -24)), ma);
      // panels
      show(inp, seg(t, 1.0, 1.8, smooth));
      setHTML(inp.querySelector('#iv'), ivHTML(cs));
      show(steps, seg(t, 1.2, 2.0, smooth));
      setHTML(steps.querySelector('#ct'), `Target ${i + 1} of 3 · held-out test pose`);
      const n = cs.network, f = cs.hybrid;
      setHTML(steps.querySelector('#b1'), u > 0.2 ? `<span class="mono">${num(n.pos_err_mm, 1)} mm · ${num(n.orn_err_deg, 1)}°</span> ${badge('bad', 'outside 1 mm / 1°')}` : '');
      setHTML(steps.querySelector('#b2'), u > 0.55 ? `<span class="mono">${f.pos_err_mm < 0.001 ? '< 0.001' : num(f.pos_err_mm, 3)} mm · ${num(f.orn_err_deg, 2)}°</span>` : '');
      const checks = [badge(f.pos_err_mm <= 1 && f.orn_err_deg <= 1 ? 'ok' : 'bad', 'within 1 mm & 1°'),
        badge(f.self_collision ? 'bad' : 'ok', f.self_collision ? 'self-collision' : 'no self-collision'),
        badge(f.limit_violation ? 'bad' : 'ok', f.limit_violation ? 'limit violation' : 'inside joint limits')];
      setHTML(steps.querySelector('#b3'), u > 0.68 ? checks.join('') : '');
      ['#s1', '#s2', '#s3'].forEach((id, k) => steps.querySelector(id).classList.toggle('dim', u < [0.14, 0.42, 0.66][k]));
      st.orbit([0.22, 0.0, 0.36], 2.2, lerp(-48, -18, t / D), 15, st.camera, [170, 20]);
      render(st);
      cap(t);
    },
  };
}
