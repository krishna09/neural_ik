// Video opening: the xArm6 reaches a glowing target; title card.
import { Studio, render, targetMarker, setPose, Trail } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, stageFade } from '../common.js';
import { el, show } from '../ui.js';
import { seg, smooth, easeInOut, lerp, lerpArr } from '../anim.js';

export default async function (ctx) {
  const model = await robotModel(ctx);
  const ik = await ctx.data('ik');
  const { D, c } = timeline(ctx, { move0: 1.0, move1: 7.0, title: 3.2, sub: 4.4, trail: 1 }, 16);
  const st = new Studio({ floorRadius: 1.8 });
  const robot = new Robot(model);
  st.zup.add(robot.group);
  const tgt = targetMarker();
  setPose(tgt, ik.target_pos, ik.target_orn);
  st.zup.add(tgt);
  const trail = new Trail(0x4fd6f2, 0.004, 0.85);
  st.zup.add(trail.mesh);
  const qA = [0, 0, 0, 0, 0, 0], qB = ik.branches[ik.selected].q;
  const pts = [];
  for (let k = 0; k <= 120; k++) { robot.setQ(lerpArr(qA, qB, easeInOut(k / 120))); const p = robot.eePos(); pts.push([k / 120, [p.x, p.y, p.z]]); }

  const kicker = el('<div style="font-size:20px;font-weight:600;letter-spacing:.22em;text-transform:uppercase;color:var(--accent)">STP 2026 · Team 06</div>', { left: '96px', top: '300px' });
  const title = el('<div class="title" style="position:absolute;left:96px;top:340px;font-size:68px;max-width:1180px;line-height:1.06">Neural Kinematics for Intelligent<br>Space Manipulators</div>', {});
  const sub = el('<div class="subtitle" style="left:96px;top:520px;max-width:900px;font-size:28px;white-space:nowrap">Design, implementation and simulation-based evaluation<br>of a collision-aware neural inverse-kinematics framework</div>', {});
  const tag = el('<div class="footer" style="left:96px">UFactory xArm6 · PyBullet · PyTorch</div>');

  return {
    duration: D,
    render(t) {
      stageFade(t, D, 0.8, 0.8);
      const u = seg(t, c.move0, c.move1, easeInOut);
      robot.setQ(lerpArr(qA, qB, u));
      trail.set(pts.filter(([k]) => k <= u).map(([, p]) => p));
      trail.mesh.visible = c.trail > 0.5;
      tgt.scale.setScalar(Math.max(1e-3, seg(t, 0.2, 1.0, smooth) * (1 + 0.05 * Math.sin(t * 3))));
      tgt.userData.ring.rotation.z = t * 0.9;
      show(kicker, seg(t, c.title - 0.4, c.title + 0.4, smooth));
      show(title, seg(t, c.title, c.title + 0.9, smooth), 24);
      show(sub, seg(t, c.sub, c.sub + 0.9, smooth));
      tag.style.opacity = seg(t, c.sub + 0.4, c.sub + 1.2, smooth).toFixed(3);
      st.orbit([0.05, 0.12, 0.42], lerp(2.6, 2.15, seg(t, 0, D, smooth)), lerp(-80, -35, seg(t, 0, D, (x) => x)), lerp(8, 16, seg(t, 0, D, smooth)), st.camera, [380, 30]);   // keep the trail clear of the subtitle
      render(st);
    },
  };
}
