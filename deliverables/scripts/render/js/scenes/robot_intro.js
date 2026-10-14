// Simulation 1 - robot introduction: the xArm6, its six revolute joints and the end-effector frame.
import * as THREE from 'three';
import { Studio, render, axesMarker, project } from '../world.js';
import { Robot } from '../robot.js';
import { robotModel, timeline, header, headerFade, captions, stageFade } from '../common.js';
import { el, show, place, setHTML } from '../ui.js';
import { seg, smooth, easeInOut, deg, lerp, window_, num, range } from '../anim.js';

export const READY = [0.0, -0.45, -0.95, 0.0, 1.35, 0.0];
const NAMES = ['base rotation', 'shoulder', 'elbow', 'forearm roll', 'wrist pitch', 'flange roll'];
const AMP = [1.0, 0.38, 0.42, 1.2, 0.7, 1.6];          // swing amplitude (rad), inside every joint limit

function jointGizmo() {
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: 0x4fd6f2, transparent: true, opacity: 0.95, depthTest: false });
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, 0.26, 12), mat);
  rod.rotation.x = Math.PI / 2;                             // local Z = joint axis
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.085, 0.004, 10, 72, Math.PI * 1.6), mat);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.012, 0.03, 16), mat);
  const a = Math.PI * 1.6;
  tip.position.set(0.085 * Math.cos(a), 0.085 * Math.sin(a), 0);
  tip.rotation.z = a;                                       // cone points along the ring tangent
  g.add(rod, ring, tip);
  g.renderOrder = 10;
  g.traverse((o) => { o.renderOrder = 10; });
  g.userData.mat = mat;
  return g;
}

export default async function (ctx) {
  const model = await robotModel(ctx);
  const lim = model.data.limits;
  const { D, c } = timeline(ctx, { j0: 3.2, step: 2.75, ee: 20.0 }, 24);
  const st = new Studio();
  const robot = new Robot(model);
  st.zup.add(robot.group);
  const gizmo = jointGizmo();
  st.zup.add(gizmo);
  const ee = axesMarker(0.11, 0.0045);
  st.zup.add(ee);

  const h = header(ctx, { chap: 'Level 1 · Robot', ttl: 'Meet the xArm6', subt: 'Six revolute joints, loaded from its URDF and simulated in PyBullet.' });
  const rows = NAMES.map((n, i) => `<div class="jrow" data-i="${i}">
      <span class="jn">J${i + 1}</span><span class="jt">${n}</span>
      <span class="jr mono">${range(deg(lim[i][0]), deg(lim[i][1]))}°</span>
      <span class="jv mono">0.0°</span></div>`).join('');
  const panel = el(`<div class="panel jpanel"><h3>Joints · limits · angle</h3>${rows}</div>`,
    { left: '72px', top: '300px', width: '560px' });
  const style = document.createElement('style');
  style.textContent = `.jrow{display:grid;grid-template-columns:52px 1fr 150px 92px;align-items:center;gap:8px;
    font-size:21px;padding:9px 10px;border-radius:10px;color:var(--text-2)}
    .jrow .jn{font-weight:700;color:var(--text-1)} .jrow .jr{font-size:17px;color:var(--muted);text-align:right}
    .jrow .jv{text-align:right;color:var(--text-1)} .jrow.on{background:var(--accent-soft);color:var(--text-1);
    box-shadow:inset 3px 0 0 var(--accent)} .jrow.on .jr{color:var(--text-2)}`;
  document.head.appendChild(style);
  const rowEls = [...panel.querySelectorAll('.jrow')];
  const tag = el('<div class="label3d acc">J1</div>');
  const eeTag = el('<div class="label3d">end effector · link 6</div>');
  const cap = captions(ctx, [
    [0.6, c.j0 - 0.1, 'UFactory <em>xArm6</em>: six revolute joints, base to flange'],
    [c.ee, D - 0.3, 'Its <em>pose</em> = position (x, y, z) + orientation of the end-effector frame'],
  ]);

  return {
    duration: D,
    render(t) {
      stageFade(t, D);
      headerFade(h, t, D);
      show(panel, seg(t, 0.6, 1.6, smooth));
      const q = READY.slice();
      let active = -1, wa = 0;
      for (let i = 0; i < 6; i++) {
        const a = c.j0 + i * c.step, b = a + c.step;
        if (t >= a && t < b) {
          active = i;
          const u = (t - a) / (b - a);
          q[i] = READY[i] + AMP[i] * Math.sin(2 * Math.PI * easeInOut(u));
          wa = window_(t, a, b, 0.3);
        }
      }
      robot.setQ(q);
      robot.clearHighlights();
      rowEls.forEach((r, i) => {
        r.classList.toggle('on', i === active);
        setHTML(r.querySelector('.jv'), `${num(deg(q[i]), 1)}°`);
      });
      if (active >= 0) {
        robot.highlight(active + 1, wa);
        const o = robot.jointOrigin(active), ax = robot.jointAxis(active);
        gizmo.position.copy(o);
        gizmo.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), ax);
        gizmo.userData.mat.opacity = 0.95 * wa;
        gizmo.visible = wa > 0.01;
        setHTML(tag, `J${active + 1} · ${NAMES[active]}`);
        place(tag, project(o.clone().add(ax.clone().multiplyScalar(0.0)).add(new THREE.Vector3(0, 0, 0.0)), st.camera)
          ?.map((v, k) => v + (k === 0 ? 150 : -40)), wa);
      } else {
        gizmo.visible = false;
        place(tag, null);
      }
      const ea = seg(t, c.ee, c.ee + 0.8, smooth);
      ee.matrixAutoUpdate = false;
      ee.matrix.copy(robot.frames[6]);
      ee.visible = ea > 0.01;
      ee.scale.setScalar(ea);
      ee.matrix.multiply(new THREE.Matrix4().makeScale(Math.max(ea, 1e-3), Math.max(ea, 1e-3), Math.max(ea, 1e-3)));
      place(eeTag, project(robot.eePos(), st.camera)?.map((v, k) => v + (k === 0 ? 120 : -70)), ea);
      st.orbit([0, 0, 0.36], 2.05, lerp(-62, -28, seg(t, 0, D, (u) => u)), 15);
      render(st);
      cap(t);
    },
  };
}
