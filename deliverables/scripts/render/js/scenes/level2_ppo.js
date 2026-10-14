// Simulation 10 - Level 2 (implemented): PPO re-routing agent vs straight path vs potential field on one
// recorded episode (same seed: same obstacles, same sudden obstacle, same joint slip), then the team's
// 200-episode evaluation.
import * as THREE from 'three';
import { Studio, renderViews, targetMarker, setPose, sphere, project } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, elBack, show, setHTML, badge, place } from '../ui.js';
import { seg, smooth, easeInOut, lerp, lerpArr, num, clamp } from '../anim.js';
import { Layer, text, C, roundRect } from '../chart2d.js';

const CTRL = [
  { key: 'baseline', name: 'Straight path', sub: 'interpolate to the IK goal', color: '#199e70' },
  { key: 'apf', name: 'Potential field', sub: 'classical repulsion', color: '#d95926' },
  { key: 'policy', name: 'DRL agent (PPO)', sub: 'potential field + learned correction', color: '#3987e5' },
];

export default async function (ctx) {
  const model = await robotModel(ctx);
  const L2 = await ctx.data('level2');
  const { D, c } = timeline(ctx, { loop: 1.6, ep0: 8.6, res: 24.4, sps: 9 }, 32);
  const W0 = 584, H0 = 450, X0 = 72, Y0 = 250, GAP = 12;
  const views = CTRL.map((ct, i) => {
    const st = new Studio({ floorRadius: 1.3 });
    const r = new Robot(model);
    const goal = new Robot(model, { style: 'ghost', color: '#cfe3ff', opacity: 0.18 });
    goal.setQ(L2.goal.q);
    const tg = targetMarker();
    setPose(tg, L2.goal.ee, null);
    tg.userData.ax.visible = false;
    st.zup.add(r.group, goal.group, tg);
    const obs = [0, 1, 2].map(() => { const s = sphere(1, 0xd03b3b, 0.55, 0.25); s.visible = false; st.zup.add(s); return s; });
    const x = X0 + i * (W0 + GAP);
    const card = elBack(`<div class="panel" style="padding:0;width:${W0}px;height:${H0 + 140}px;background:rgba(15,27,45,.5)">
        <div style="padding:16px 20px 0 20px;display:flex;align-items:center;gap:12px">
          <span style="width:14px;height:14px;border-radius:3px;background:${ct.color}"></span>
          <span style="font-size:24px;font-weight:650">${ct.name}</span><span class="muted" style="font-size:16px;margin-left:auto">${ct.sub}</span></div></div>`,
      { left: `${x}px`, top: `${Y0 - 10}px` });
    const res = el('<div style="font-size:19px"></div>', { left: `${x + 20}px`, top: `${Y0 + H0 + 40}px`, width: `${W0 - 40}px` });
    const ev = el('<div></div>', { left: `${x + 20}px`, top: `${Y0 + 52}px` });
    return { ct, st, r, goal, tg, obs, card, res, ev, ep: L2.episodes[ct.key], rect: [x + 1, Y0 + 40, W0 - 2, H0] };
  });
  const h = header(ctx, { chap: 'Level 2 · DRL re-routing (implemented)', ttl: 'Steering around the unexpected',
    subt: 'Same episode for all three: an obstacle appears mid-motion and a joint slips.' });
  const loopL = new Layer(72, 250, 1776, 620);
  const resL = new Layer(72, 250, 1776, 560);
  const resNote = el(`<div style="font-size:18px;line-height:1.5;color:var(--muted);width:1776px">
      Team evaluation: agent <span class="mono">l2_20261013-185525</span>, 20 M PPO steps, 200 unseen episodes per scenario, identical seeds for every controller
      (<span class="mono">media/data/…_meta_n200.json</span>). Policy: ${num(L2.team_params, 0).replace(/\B(?=(\d{3})+(?!\d))/g, ',')} parameters, ${num(L2.team_decision_us, 1)} µs per decision on one CPU core.
      ${L2.repro_eval ? `Reproduction in this environment (CPU, 20 M steps): ${num(L2.repro_eval.scenarios.combined.policy.success_pct, 1)} % success / ${num(L2.repro_eval.scenarios.combined.policy.collision_pct, 1)} % collisions.` : ''}</div>`,
    { left: '72px', top: '830px' });
  const cap = captions(ctx, [
    [c.loop, c.ep0, 'The agent learns by trial and error in simulation, rewarded for progress and safety'],
    [c.ep0, c.res, 'Same start, goal, obstacles and joint slip for every controller'],
    [c.res, D - 0.2, '"Everything at once" scenario: 200 unseen episodes per controller'],
  ]);

  function drawLoop(t, a) {
    const g = loopL.g;
    loopL.clear(); loopL.opacity(a);
    if (a <= 0) return;
    const nodes = [
      { x: 330, y: 150, t: 'State', s: ['joint angles, goal error', 'obstacle distances, escape directions', '101 numbers'] },
      { x: 900, y: 150, t: 'Policy network', s: ['MLP 2 × 64, tanh', '11,078 parameters', 'NumPy at deployment'] },
      { x: 1470, y: 150, t: 'Action', s: ['correction for 6 joints', 'added to a potential field', '≤ 0.03 rad per step'] },
      { x: 1470, y: 470, t: 'PyBullet environment', s: ['obstacles appear and drift', 'gain errors, joint slips', 'contact = failure'] },
      { x: 900, y: 470, t: 'Reward', s: ['+ progress, + 15 on arrival', '− proximity, − 25 on contact', '− 0.3 · Level 1 collision risk'] },
      { x: 330, y: 470, t: 'PPO update', s: ['clipped objective (ε = 0.2)', 'GAE γ = 0.995, λ = 0.95', '80 parallel environments'] },
    ];
    const act = Math.floor((t - c.loop) / 0.9) % 6;
    nodes.forEach((n, i) => {
      const w = 440, hh = 170;
      const on = i === act;
      g.fillStyle = on ? 'rgba(79,214,242,0.12)' : 'rgba(15,27,45,0.92)';
      roundRect(g, n.x - w / 2, n.y - hh / 2, w, hh, 16); g.fill();
      g.strokeStyle = on ? 'rgba(79,214,242,0.8)' : 'rgba(35,52,79,1)'; g.lineWidth = on ? 2 : 1; g.stroke();
      text(g, n.t, n.x, n.y - 38, { size: 26, weight: 650, color: C.text1, align: 'center' });
      n.s.forEach((s, k) => text(g, s, n.x, n.y + 2 + k * 30, { size: 18, color: C.text2, align: 'center' }));
    });
    const arrow = (x0, y0, x1, y1) => {
      g.strokeStyle = 'rgba(160,190,230,0.7)'; g.lineWidth = 2.5;
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      const ang = Math.atan2(y1 - y0, x1 - x0);
      g.fillStyle = 'rgba(160,190,230,0.9)';
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - 14 * Math.cos(ang - 0.4), y1 - 14 * Math.sin(ang - 0.4));
      g.lineTo(x1 - 14 * Math.cos(ang + 0.4), y1 - 14 * Math.sin(ang + 0.4)); g.closePath(); g.fill();
    };
    arrow(552, 150, 678, 150); arrow(1122, 150, 1248, 150); arrow(1470, 237, 1470, 383);
    arrow(1248, 470, 1122, 470); arrow(678, 470, 552, 470); arrow(330, 383, 330, 237);
  }

  function drawRes(a) {
    const g = resL.g;
    resL.clear(); resL.opacity(a);
    if (a <= 0) return;
    const sc = L2.team_eval.combined;
    const blocks = [{ t: 'Reached the goal', k: 'success_pct' }, { t: 'Collided', k: 'collision_pct' }, { t: 'Stuck (timeout)', k: 'timeout_pct' }];
    const cw = 570, ch = 380;
    blocks.forEach((b, i) => {
      const x = i * (cw + 33), y = 20;
      g.fillStyle = 'rgba(15,27,45,0.9)'; roundRect(g, x, y, cw, ch + 120, 16); g.fill();
      g.strokeStyle = 'rgba(35,52,79,1)'; g.lineWidth = 1; g.stroke();
      text(g, b.t, x + 24, y + 40, { size: 23, weight: 650, color: C.text1 });
      text(g, '"everything at once" · 200 episodes', x + 24, y + 68, { size: 16, color: C.muted });
      const base = y + ch + 30, top = y + 110, bw = 110;
      g.strokeStyle = 'rgba(160,185,220,0.45)'; g.beginPath(); g.moveTo(x + 24, base + 0.5); g.lineTo(x + cw - 24, base + 0.5); g.stroke();
      CTRL.forEach((ct, j) => {
        const v = sc[ct.key][b.k];
        const hgt = Math.max(3, (base - top) * (v / 100) * seg(a, 0.2, 1, easeInOut));
        const bx = x + 60 + j * (bw + 50);
        g.fillStyle = ct.color; roundRect(g, bx, base - hgt, bw, hgt, 4); g.fill();
        g.fillRect(bx, base - Math.min(hgt, 4), bw, Math.min(hgt, 4));
        text(g, `${num(v, 1)} %`, bx + bw / 2, base - hgt - 12, { size: 21, weight: 650, color: C.text1, align: 'center' });
        text(g, ct.name.replace('DRL agent (PPO)', 'DRL agent'), bx + bw / 2, base + 30, { size: 16, color: C.text2, align: 'center' });
      });
    });
  }

  return {
    duration: D,
    info: { seed: L2.seed, outcomes: Object.fromEntries(Object.entries(L2.episodes).map(([k, v]) => [k, v.outcome])) },
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      const la = Math.min(seg(t, c.loop, c.loop + 0.8, smooth), 1 - seg(t, c.ep0 - 0.8, c.ep0, smooth));
      drawLoop(t, la);
      const ea = Math.min(seg(t, c.ep0 - 0.4, c.ep0 + 0.4, smooth), 1 - seg(t, c.res - 0.6, c.res, smooth));
      const list = [];
      views.forEach((v) => {
        const fr = v.ep.frames;
        const f = clamp((t - c.ep0 - 0.6) * c.sps, 0, fr.length - 1);
        const k0 = Math.floor(f), k1 = Math.min(fr.length - 1, k0 + 1), u = f - k0;
        v.r.setQ(lerpArr(fr[k0].q, fr[k1].q, u));
        const O0 = fr[k0].obstacles, O1 = fr[k1].obstacles;
        v.obs.forEach((s, j) => {
          const o = O0[j];
          s.visible = !!o;
          if (!o) return;
          const o1 = O1[j] && Math.abs(O1[j].r - o.r) < 1e-6 ? O1[j] : o;
          s.position.set(...lerpArr(o.pos, o1.pos, u));
          s.scale.setScalar(o.r);
        });
        const done = f >= fr.length - 1;
        v.r.clearHighlights();
        if (done && v.ep.outcome === 'collision') for (let k = 0; k <= 6; k++) v.r.highlight(k, 0.7, '#d03b3b');
        if (done && v.ep.outcome === 'success') for (let k = 1; k <= 6; k++) v.r.highlight(k, 0.35, '#3987e5');
        v.tg.userData.ring.rotation.z = t;
        show(v.card, ea);
        const step = fr[k0].t;
        const evs = v.ep.events.filter((e) => e.t <= step).map((e) => (e.type === 'joint_slip' ? badge('warn', `joint ${e.joint + 1} slipped`) : badge('warn', 'obstacle appeared')));
        setHTML(v.ev, evs.join(' '));
        show(v.ev, ea);
        const outcome = done ? (v.ep.outcome === 'success' ? badge('ok', `reached the goal · ${v.ep.steps} steps`) :
          v.ep.outcome === 'collision' ? badge('bad', `collision at step ${v.ep.steps}`) : badge('warn', 'stuck')) :
          `<span class="t2">step <span class="mono">${step}</span></span>`;
        setHTML(v.res, outcome);
        show(v.res, ea);
        v.st.orbit([0.0, 0.0, 0.4], 2.05, lerp(-60, -40, t / D), 18);
        if (ea > 0.001) list.push({ scene: v.st.scene, camera: v.st.camera, rect: v.rect });
      });
      renderViews(list);
      document.getElementById('gl').style.opacity = ea.toFixed(3);
      const ra = seg(t, c.res, c.res + 1.0, smooth);
      drawRes(ra);
      show(resNote, seg(t, c.res + 1.5, c.res + 2.3, smooth));
      cap(t);
    },
  };
}
