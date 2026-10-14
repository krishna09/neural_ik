// xArm6 drawn from the URDF meshes; link poses from forward kinematics of the chain exported from
// PyBullet (same maths as refine/kinematics.py, checked against PyBullet link frames in robot.json).

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { toCreasedNormals } from 'vendor/BufferGeometryUtils.js';

const rowMajor = (a) => new THREE.Matrix4().set(...a);

export async function loadRobot(data) {
  const loader = new STLLoader();
  const geoms = {};
  await Promise.all([...new Set(data.visuals.map((v) => v.mesh))].map(async (url) => {
    const g = await loader.loadAsync(url);
    geoms[url] = toCreasedNormals(g, (32 * Math.PI) / 180);
  }));
  const chain = {
    root: rowMajor(data.chain.root),
    rel: data.chain.rel.map(rowMajor),
    axes: data.chain.axes.map((a) => new THREE.Vector3(...a).normalize()),
  };
  return { data, geoms, chain, nLinks: 1 + data.dof };
}

// world matrices of links 0..dof (link 0 = base) for joint vector q (radians)
export function fk(model, q) {
  const { chain } = model;
  const T = chain.root.clone();
  const out = [T.clone()];
  const R = new THREE.Matrix4();
  for (let i = 0; i < chain.rel.length; i++) {
    T.multiply(chain.rel[i]);
    R.makeRotationAxis(chain.axes[i], q[i]);
    T.multiply(R);
    out.push(T.clone());
  }
  return out;
}

// max link-origin distance (m) between this FK and PyBullet's frames, over robot.json's check poses
export function validateFK(model) {
  let worst = 0, worstRot = 0;
  const p = new THREE.Vector3(), qa = new THREE.Quaternion(), s = new THREE.Vector3();
  for (const c of model.data.check) {
    const F = fk(model, c.q);
    for (let k = 0; k < F.length; k++) {
      const ref = c.links[String(k)];
      F[k].decompose(p, qa, s);
      worst = Math.max(worst, p.distanceTo(new THREE.Vector3(...ref[0])));
      // stable angle between rotations: 2 atan2(|v|, |w|) of qa^-1 qb (acos near 1 amplifies rounding)
      const d = qa.clone().invert().multiply(new THREE.Quaternion(...ref[1]));
      worstRot = Math.max(worstRot, 2 * Math.atan2(Math.hypot(d.x, d.y, d.z), Math.abs(d.w)));
    }
  }
  return { pos_m: worst, rot_rad: worstRot };
}

const SOLID = [
  { color: '#e3e8ef', rough: 0.36, metal: 0.05 },   // base
  { color: '#eef1f5', rough: 0.32, metal: 0.04 },   // link1
  { color: '#eef1f5', rough: 0.32, metal: 0.04 },   // link2
  { color: '#eef1f5', rough: 0.32, metal: 0.04 },   // link3
  { color: '#eef1f5', rough: 0.32, metal: 0.04 },   // link4
  { color: '#e6eaf0', rough: 0.34, metal: 0.05 },   // link5
  { color: '#3a414c', rough: 0.38, metal: 0.55 },   // link6 flange
];

export class Robot {
  // style: 'solid' (studio look) | 'ghost' (translucent single colour, front surface only)
  constructor(model, { style = 'solid', color = '#3987e5', opacity = 0.4, castShadow = true } = {}) {
    this.model = model;
    this.style = style;
    this.group = new THREE.Group();
    this.links = [];
    this.mats = [];
    for (let k = 0; k < model.nLinks; k++) {
      const g = new THREE.Group();
      g.matrixAutoUpdate = false;
      this.group.add(g);
      this.links.push(g);
      this.mats.push([]);
    }
    for (const v of model.data.visuals) {
      const holder = new THREE.Group();
      holder.position.set(...v.pos);
      holder.quaternion.set(...v.orn);
      holder.scale.set(...v.scale);
      const geo = model.geoms[v.mesh];
      if (style === 'solid') {
        const spec = SOLID[v.link] || SOLID[1];
        const m = new THREE.MeshPhysicalMaterial({ color: spec.color, roughness: spec.rough, metalness: spec.metal,
          clearcoat: 0.35, clearcoatRoughness: 0.3, envMapIntensity: 0.75, emissive: 0x000000 });
        m.userData.base = new THREE.Color(spec.color);
        const mesh = new THREE.Mesh(geo, m);
        mesh.castShadow = castShadow;
        mesh.receiveShadow = true;
        holder.add(mesh);
        this.mats[v.link].push(m);
      } else {
        const depth = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ colorWrite: false, polygonOffset: true,
          polygonOffsetFactor: 2, polygonOffsetUnits: 2 }));
        depth.renderOrder = 1;
        const m = new THREE.MeshStandardMaterial({ color, transparent: true, opacity, roughness: 0.45, metalness: 0.0,
          depthWrite: false, depthFunc: THREE.LessEqualDepth, emissive: color, emissiveIntensity: 0.25 });
        m.userData.baseOpacity = opacity;
        const mesh = new THREE.Mesh(geo, m);
        mesh.renderOrder = 2;
        holder.add(depth, mesh);
        this.mats[v.link].push(m);
        this.depthMeshes = (this.depthMeshes || []).concat([depth]);
      }
      this.links[v.link].add(holder);
    }
    this.setQ(new Array(model.data.dof).fill(0));
  }

  setQ(q) {
    this.q = Array.from(q);
    this.frames = fk(this.model, this.q);
    this.frames.forEach((M, k) => { this.links[k].matrix.copy(M); this.links[k].matrixWorldNeedsUpdate = true; });
    return this;
  }

  linkPos(k) { return new THREE.Vector3().setFromMatrixPosition(this.frames[k]); }
  eePos() { return this.linkPos(this.model.data.dof); }
  eeQuat() { const q = new THREE.Quaternion(); this.frames[this.model.data.dof].decompose(new THREE.Vector3(), q, new THREE.Vector3()); return q; }
  // origin and world axis of joint i (0-based), PyBullet coordinates
  jointOrigin(i) { return this.linkPos(i + 1); }
  jointAxis(i) {
    const R = new THREE.Matrix4().extractRotation(this.frames[i + 1]);
    return this.model.chain.axes[i].clone().applyMatrix4(R).normalize();
  }

  // tint link k towards a colour (0..1), with a soft emissive glow
  highlight(k, amount, color = '#4fd6f2') {
    const c = new THREE.Color(color);
    for (const m of this.mats[k]) {
      if (this.style === 'solid') {
        m.color.copy(m.userData.base).lerp(c, 0.55 * amount);
        m.emissive.copy(c).multiplyScalar(0.35 * amount);
      } else {
        m.emissiveIntensity = 0.25 + 0.6 * amount;
      }
    }
  }
  clearHighlights() { for (let k = 0; k < this.mats.length; k++) this.highlight(k, 0); }

  setOpacity(a) {
    this.group.visible = a > 0.002;
    if (this.style === 'solid') {
      for (const ms of this.mats) for (const m of ms) { m.transparent = a < 0.999; m.opacity = a; m.depthWrite = a >= 0.999; }
    } else {
      for (const ms of this.mats) for (const m of ms) m.opacity = m.userData.baseOpacity * a;
    }
  }
  setColor(color) {
    const c = new THREE.Color(color);
    for (const ms of this.mats) for (const m of ms) { m.color.copy(c); if (this.style !== 'solid') m.emissive.copy(c); }
  }
}
