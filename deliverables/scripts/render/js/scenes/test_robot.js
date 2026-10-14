// Check scene: FK vs PyBullet, and the studio look.
import { Studio, render, targetMarker, setPose } from '../world.js';
import { loadRobot, Robot, validateFK } from '../robot.js';
import { chapter, title, footer } from '../ui.js';

export default async function (ctx) {
  const model = await loadRobot(await ctx.data('robot'));
  const check = validateFK(model);
  const st = new Studio();
  const robot = new Robot(model);
  st.zup.add(robot.group);
  const ghost = new Robot(model, { style: 'ghost', color: '#3987e5', opacity: 0.35 });
  st.zup.add(ghost.group);
  const tgt = targetMarker();
  st.zup.add(tgt);
  chapter('Renderer check', 'xArm6');
  title('FK check vs PyBullet');
  footer();
  const q1 = [0.6, -0.4, -1.2, 0.3, 1.3, 0.2];
  const q2 = [0.2, 0.1, -0.9, 0.0, 0.9, 0.0];
  return {
    duration: 2,
    info: { fk_check: check },
    render(t) {
      robot.setQ(q1);
      ghost.setQ(q2);
      const e = robot.eePos();
      setPose(tgt, [e.x, e.y, e.z], null);
      st.orbit([0, 0, 0.32], 2.0, -50 + 20 * t, 18);
      render(st);
    },
  };
}
