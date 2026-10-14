// Video closing: measured headline results and thanks.
import { Studio, render, targetMarker, setPose } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, stageFade } from '../common.js';
import { el, show } from '../ui.js';
import { seg, smooth, lerp, lerpArr, easeInOut } from '../anim.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const ik = await ctx.data('ik');
  const cmp = await ctx.data('comparison');
  const L2 = await ctx.data('level2');
  const A = (k, m, f) => parseFloat(cmp.aggregate[`${m}|${k}`][f]);
  const { D, c } = timeline(ctx, { chips: 1.2, thanks: 6.5 }, 13);
  const st = new Studio({ floorRadius: 1.8 });
  const robot = new Robot(model);
  st.zup.add(robot.group);
  const tgt = targetMarker();
  setPose(tgt, ik.target_pos, ik.target_orn);
  st.zup.add(tgt);
  const sc = L2.team_eval.combined.policy;
  const chips = [
    [`${A('zero', 'hybrid_lm', 'success_1mm_1deg_pct').toFixed(0)} %`, 'of 2,000 held-out targets within 1 mm and 1° (network + LM)'],
    [`${A('zero', 'hybrid_lm', 'self_collision_pct').toFixed(0)} %`, 'self-collisions and joint-limit violations in accepted answers'],
    [`${sc.success_pct.toFixed(0)} % / ${sc.collision_pct.toFixed(0)} %`, 'Level 2 agent: goals reached / collisions, hardest scenario'],
  ].map(([v, d]) => el(`<div class="panel" style="width:900px;padding:18px 26px;display:flex;align-items:center;gap:24px">
      <span class="big" style="min-width:300px;white-space:nowrap;color:var(--text-1)">${v}</span><span style="font-size:21px;color:var(--text-2);line-height:1.35">${d}</span></div>`));
  chips.forEach((e, i) => Object.assign(e.style, { left: '72px', top: `${250 + i * 132}px` }));
  const thanks = el(`<div><div class="title" style="position:relative;left:0;top:0;font-size:64px">Thank you</div>
      <div class="subtitle" style="position:relative;left:0;top:0;margin-top:12px">STP 2026 · Team 06 · Neural Kinematics for Intelligent Space Manipulators</div></div>`,
    { left: '72px', top: '700px' });
  const note = el('<div class="footer">All results: simulation (PyBullet), xArm6. Sources in deliverables/results/.</div>');
  const qA = ik.seed, qB = ik.branches[ik.selected].q;

  return {
    duration: D,
    render(t) {
      stageFade(t, D, 0.6, 1.2);
      chips.forEach((e, i) => show(e, seg(t, c.chips + i * 0.7, c.chips + i * 0.7 + 0.7, smooth)));
      show(thanks, seg(t, c.thanks, c.thanks + 0.9, smooth));
      note.style.opacity = seg(t, c.thanks, c.thanks + 1, smooth).toFixed(3);
      robot.setQ(lerpArr(qA, qB, seg(t, 0.5, 5.0, easeInOut)));
      tgt.userData.ring.rotation.z = t;
      st.orbit([0.05, 0.12, 0.4], 2.2, lerp(-10, 25, t / D), 14, st.camera, [420, 0]);
      render(st);
    },
  };
}
