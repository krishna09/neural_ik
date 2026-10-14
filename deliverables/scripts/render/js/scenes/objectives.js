// Video: the project brief's two levels, and what this presentation covers.
import { Studio, render } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, stageFade } from '../common.js';
import { el, show, badge } from '../ui.js';
import { seg, smooth, lerp } from '../anim.js';
import { READY } from './robot_intro.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const { D, c } = timeline(ctx, { l1: 2.6, l2: 7.0, robot: 11.0 }, 15);
  const st = new Studio();
  const robot = new Robot(model);
  st.zup.add(robot.group);
  robot.setQ(READY);
  const h = header(ctx, { chap: 'Project brief', sub: 'Team 06', ttl: 'Two levels, one goal: collision-free motion',
    subt: 'Neural Kinematics and Reinforcement Learning for Collision-Free Space Manipulator Path Planning' });
  const card = (lvl, ttl, txt, status) => el(`<div class="panel" style="width:860px;padding:26px 30px">
      <div style="display:flex;align-items:center;gap:14px"><span style="font-size:17px;font-weight:700;letter-spacing:.16em;color:var(--accent)">${lvl}</span>
      <span style="margin-left:auto">${status}</span></div>
      <div style="font-size:32px;font-weight:650;margin-top:10px">${ttl}</div>
      <div style="font-size:22px;line-height:1.45;color:var(--text-2);margin-top:10px">${txt}</div></div>`);
  const c1 = card('LEVEL 1', 'Kinematics-aware AI model',
    'Use simulated manipulator joint data to compute joint angles for a target pose and prevent self-collision.',
    badge('ok', 'implemented'));
  const c2 = card('LEVEL 2', 'Lightweight DRL policy agent',
    'Re-route the manipulator in real time when unexpected obstacles or hardware position errors occur.',
    badge('ok', 'implemented in simulation'));
  Object.assign(c1.style, { left: '72px', top: '300px' });
  Object.assign(c2.style, { left: '72px', top: '560px' });
  const src = el('<div class="source">source: Project Brief - Team 6.pdf</div>');

  return {
    duration: D,
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      show(c1, seg(t, c.l1, c.l1 + 0.8, smooth));
      show(c2, seg(t, c.l2, c.l2 + 0.8, smooth));
      src.style.opacity = seg(t, 1, 2, smooth).toFixed(3);
      st.orbit([0.05, 0, 0.36], 2.3, lerp(-50, -20, t / D), 15, st.camera, [430, 40]);
      render(st);
    },
  };
}
