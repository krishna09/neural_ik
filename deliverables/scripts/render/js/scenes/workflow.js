// Video: what was built - the implemented modules as a pipeline, with the file that implements each step.
import { timeline, header, headerFade, stageFade } from '../common.js';
import { el, show } from '../ui.js';
import { seg, smooth } from '../anim.js';

const STEPS = [
  ['robot_config.py', 'Robot registry', 'URDF, joints by name, limits'],
  ['generate_dataset.py', 'Data generation', 'pose + collision label per sample'],
  ['train_ik.py', 'Classifier + IK network', 'collision-aware loss'],
  ['refine/lm_solver.py', 'LM refinement', 'below 1 mm and 1°'],
  ['evaluate.py · ik_metrics', 'Evaluation', 'FK, limits, self-collision'],
];
const L2 = [
  ['level2/env.py', 'Obstacles & hardware faults'],
  ['level2/ppo.py', 'PPO agent, NumPy deployment'],
  ['level2/evaluate.py', 'Agent vs classical controllers'],
];

export default async function (ctx) {
  const { D, c } = timeline(ctx, { s0: 1.4, step: 1.1, l2: 8.0, reg: 10.8 }, 16);
  const h = header(ctx, { chap: 'Development', sub: 'Level 1 complete', ttl: 'What we built', subt: 'Every step is a script in the repository, reproducible from scratch on a CPU.' });
  const css = document.createElement('style');
  css.textContent = `.wf{width:318px;padding:20px 22px} .wf .f{font-family:var(--mono);font-size:16px;color:var(--accent)}
    .wf .t{font-size:25px;font-weight:650;margin-top:8px} .wf .d{font-size:18px;color:var(--text-2);margin-top:6px}
    .arrow{font-size:34px;color:var(--muted)} .l2c{width:520px;padding:18px 22px}`;
  document.head.appendChild(css);
  const cards = STEPS.map(([f, t, d], i) => el(`<div class="panel wf"><div class="f">${f}</div><div class="t">${t}</div><div class="d">${d}</div></div>`,
    { left: `${72 + i * 358}px`, top: '300px' }));
  const arrows = STEPS.slice(1).map((_, i) => el('<div class="arrow">→</div>', { left: `${72 + i * 358 + 324}px`, top: '360px' }));
  const studio = el(`<div class="panel" style="width:1776px;padding:16px 24px;font-size:20px;color:var(--text-2)">
      <b style="color:var(--text-1)">ui/ · Neural IK Studio</b> — web app that runs every step: generate, browse, train, evaluate, infer, Level 2 and missions (Level 1 + 2 combined).</div>`,
    { left: '72px', top: '520px' });
  const l2 = L2.map(([f, d], i) => el(`<div class="panel l2c"><div class="f mono" style="font-size:16px;color:#a99bff">${f}</div>
      <div style="font-size:22px;font-weight:600;margin-top:6px">${d}</div></div>`, { left: `${72 + i * 628}px`, top: '640px' }));
  const reg = el(`<div class="panel" style="width:1776px;padding:20px 26px;border-color:rgba(79,214,242,.5)">
      <div style="font-size:24px"><b>Next robot = one entry.</b> <span class="t2">Swapping in the real space manipulator means adding its URDF and joint names to</span>
      <span class="mono accent">ROBOTS</span> <span class="t2">in</span> <span class="mono accent">robot_config.py</span><span class="t2">; the pipeline does not change.</span></div></div>`,
    { left: '72px', top: '800px' });

  return {
    duration: D,
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      cards.forEach((e, i) => show(e, seg(t, c.s0 + i * c.step, c.s0 + i * c.step + 0.6, smooth)));
      arrows.forEach((e, i) => show(e, seg(t, c.s0 + (i + 1) * c.step - 0.3, c.s0 + (i + 1) * c.step + 0.3, smooth)));
      show(studio, seg(t, c.s0 + 5 * c.step, c.s0 + 5 * c.step + 0.6, smooth));
      l2.forEach((e, i) => show(e, seg(t, c.l2 + i * 0.4, c.l2 + i * 0.4 + 0.6, smooth)));
      show(reg, seg(t, c.reg, c.reg + 0.7, smooth));
    },
  };
}
