// Three.js world: renderer, studio scene (lights, reflections, floor), cameras, helpers.
// Simulation content lives in PyBullet coordinates (Z up) inside studio.zup.

import * as THREE from 'three';
import { RoomEnvironment } from 'vendor/RoomEnvironment.js';

export { THREE };
export const W = 1920, H = 1080;

export function pb(x, y, z) { return new THREE.Vector3(x, z, -y); }          // PyBullet -> three
export function pbv(a) { return pb(a[0], a[1], a[2]); }

let _renderer = null;
export function renderer() {
  if (_renderer) return _renderer;
  const r = new THREE.WebGLRenderer({ canvas: document.getElementById('gl'), antialias: true, alpha: true, preserveDrawingBuffer: true });
  r.setPixelRatio(1);
  r.setSize(W, H, false);
  r.outputColorSpace = THREE.SRGBColorSpace;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.toneMappingExposure = 1.05;
  r.shadowMap.enabled = true;
  r.shadowMap.type = THREE.PCFSoftShadowMap;
  r.setClearColor(0x000000, 0);
  r.autoClear = false;
  _renderer = r;
  return r;
}

let _env = null;
function envMap() {
  if (_env) return _env;
  const pm = new THREE.PMREMGenerator(renderer());
  _env = pm.fromScene(new RoomEnvironment(renderer()), 0.04).texture;
  return _env;
}

function canvasTexture(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

let _floorTex = null;
function floorTexture() {
  if (_floorTex) return _floorTex;
  _floorTex = canvasTexture(2048, 2048, (g, w, h) => {
    const cx = w / 2, cy = h / 2, R = w / 2;
    const grad = g.createRadialGradient(cx, cy, 0, cx, cy, R);
    grad.addColorStop(0, 'rgba(34, 58, 96, 0.95)');
    grad.addColorStop(0.55, 'rgba(18, 34, 60, 0.75)');
    grad.addColorStop(1, 'rgba(8, 18, 34, 0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, w, h);
    g.save();
    g.beginPath(); g.arc(cx, cy, R * 0.98, 0, Math.PI * 2); g.clip();
    const step = w / 32;                                   // 0.1 m grid for a 3.2 m disk
    for (let i = 0; i <= 32; i++) {
      const a = 0.11 * (i % 5 === 0 ? 1.7 : 1);
      g.strokeStyle = `rgba(120, 170, 230, ${a})`;
      g.lineWidth = i % 5 === 0 ? 2.2 : 1.2;
      g.beginPath(); g.moveTo(i * step, 0); g.lineTo(i * step, h); g.stroke();
      g.beginPath(); g.moveTo(0, i * step); g.lineTo(w, i * step); g.stroke();
    }
    g.restore();
    // fade the grid towards the rim
    const fade = g.createRadialGradient(cx, cy, R * 0.12, cx, cy, R * 0.95);
    fade.addColorStop(0, 'rgba(0,0,0,0)');
    fade.addColorStop(1, 'rgba(0,0,0,1)');
    g.globalCompositeOperation = 'destination-out';
    g.fillStyle = fade;
    g.fillRect(0, 0, w, h);
  });
  return _floorTex;
}

let _glowTex = null;
export function glowTexture() {
  if (_glowTex) return _glowTex;
  _glowTex = canvasTexture(256, 256, (g, w) => {
    const grad = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, w, w);
  });
  return _glowTex;
}

let _dotTex = null;
export function dotTexture() {
  if (_dotTex) return _dotTex;
  _dotTex = canvasTexture(64, 64, (g, w) => {
    const grad = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.6, 'rgba(255,255,255,0.9)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.beginPath(); g.arc(w / 2, w / 2, w / 2, 0, Math.PI * 2); g.fill();
  });
  return _dotTex;
}

export class Studio {
  constructor({ floor = true, floorRadius = 1.6, shadows = true, exposure = null } = {}) {
    const s = new THREE.Scene();
    s.environment = envMap();
    this.scene = s;
    this.zup = new THREE.Group();
    this.zup.rotation.x = -Math.PI / 2;
    s.add(this.zup);

    s.add(new THREE.HemisphereLight(0xcfe3ff, 0x0b1424, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 2.3);
    key.position.set(1.8, 3.2, 2.0);
    key.castShadow = shadows;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -1.3; key.shadow.camera.right = 1.3;
    key.shadow.camera.top = 1.3; key.shadow.camera.bottom = -1.3;
    key.shadow.camera.near = 0.5; key.shadow.camera.far = 8;
    key.shadow.bias = -0.0004;
    key.shadow.radius = 4;
    s.add(key);
    const rim = new THREE.DirectionalLight(0x6fb6ff, 1.6);
    rim.position.set(-2.4, 1.6, -2.6);
    s.add(rim);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(-2.0, 1.0, 2.2);
    s.add(fill);
    this.key = key;

    if (floor) {
      const disk = new THREE.Mesh(new THREE.CircleGeometry(floorRadius, 96),
        new THREE.MeshBasicMaterial({ map: floorTexture(), transparent: true, depthWrite: false }));
      disk.rotation.x = -Math.PI / 2;
      disk.position.y = -0.0005;
      disk.renderOrder = -2;
      s.add(disk);
      const sh = new THREE.Mesh(new THREE.CircleGeometry(floorRadius, 64), new THREE.ShadowMaterial({ opacity: 0.42 }));
      sh.rotation.x = -Math.PI / 2;
      sh.receiveShadow = true;
      sh.renderOrder = -1;
      s.add(sh);
      this.floor = disk;
    }
    this.camera = new THREE.PerspectiveCamera(32, W / H, 0.01, 60);
  }

  // camera on a sphere around a PyBullet target point: azimuth from +x towards +y, elevation above the floor
  // shift = [dx, dy] moves the rendered content on screen (px) without changing the perspective
  orbit(target, dist, azDeg, elDeg, cam = this.camera, shift = null) {
    const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
    const t = pbv(target);
    cam.position.copy(t).add(pb(dist * Math.cos(el) * Math.cos(az), dist * Math.cos(el) * Math.sin(az), dist * Math.sin(el)));
    cam.lookAt(t);
    if (shift) cam.setViewOffset(W, H, -shift[0], -shift[1], W, H); else cam.clearViewOffset();
    cam.updateMatrixWorld();
    return cam;
  }
}

// render several (scene, camera, rect) views into the one canvas; rect = [x, y, w, h] from top-left
export function renderViews(views) {
  const r = renderer();
  r.setScissorTest(false);
  r.clear(true, true, true);
  for (const v of views) {
    const [x, y, w, h] = v.rect || [0, 0, W, H];
    v.camera.aspect = w / h;
    v.camera.updateProjectionMatrix();
    r.setViewport(x, H - y - h, w, h);
    r.setScissor(x, H - y - h, w, h);
    r.setScissorTest(true);
    r.render(v.scene, v.camera);
  }
  r.setScissorTest(false);
}

export function render(studio) { renderViews([{ scene: studio.scene, camera: studio.camera }]); }

// screen position (px) of a PyBullet-frame point
export function project(pPB, camera, rect = [0, 0, W, H]) {
  const v = (pPB.isVector3 ? pb(pPB.x, pPB.y, pPB.z) : pbv(pPB)).project(camera);
  if (v.z > 1) return null;
  return [rect[0] + ((v.x + 1) / 2) * rect[2], rect[1] + ((1 - v.y) / 2) * rect[3]];
}

// ------------------------------------------------------------------------------------------ helpers (Z-up)
export function axesMarker(size = 0.09, radius = 0.0045, opacity = 1) {
  const g = new THREE.Group();
  const cols = [0xff5a5a, 0x4be37a, 0x4f9dff];
  const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  dirs.forEach((d, i) => {
    const m = new THREE.MeshBasicMaterial({ color: cols[i], transparent: opacity < 1, opacity });
    const c = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, size, 12), m);
    c.position.copy(d.clone().multiplyScalar(size / 2));
    c.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
    const tip = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.4, radius * 5, 14), m);
    tip.position.copy(d.clone().multiplyScalar(size + radius * 2.5));
    tip.quaternion.copy(c.quaternion);
    g.add(c, tip);
  });
  g.add(new THREE.Mesh(new THREE.SphereGeometry(radius * 2.2, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })));
  return g;
}

export function glow(color = 0x4fd6f2, size = 0.12, opacity = 0.9) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color, transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  s.scale.set(size, size, size);
  return s;
}

export function targetMarker(color = 0x4fd6f2) {
  const g = new THREE.Group();
  const ax = axesMarker(0.1, 0.004);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.016, 24, 16),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9, roughness: 0.3 }));
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.045, 0.0025, 10, 64),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 }));
  const halo = glow(color, 0.2, 0.85);
  g.add(ax, ball, ring, halo);
  g.userData = { ax, ball, ring, halo };
  return g;
}

export function setPose(obj, pos, quatXYZW) {
  obj.position.set(pos[0], pos[1], pos[2]);
  if (quatXYZW) obj.quaternion.set(quatXYZW[0], quatXYZW[1], quatXYZW[2], quatXYZW[3]);
}

export function setOpacity(obj, a) {
  obj.traverse((o) => {
    if (o.material) {
      const ms = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of ms) {
        if (m.userData.baseOpacity === undefined) m.userData.baseOpacity = m.opacity;
        m.transparent = true;
        m.opacity = m.userData.baseOpacity * a;
      }
    }
  });
  obj.visible = a > 0.002;
}

// tube trail through a list of PyBullet points (rebuilt on demand)
export class Trail {
  constructor(color = 0x4fd6f2, radius = 0.004, opacity = 0.95) {
    this.mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity });
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.mat);
    this.key = '';
    this.radius = radius;
  }
  set(points) {
    const key = points.length + ':' + (points.length ? points[points.length - 1].join(',') : '');
    if (key === this.key) return;
    this.key = key;
    this.mesh.geometry.dispose();
    if (points.length < 2) { this.mesh.geometry = new THREE.BufferGeometry(); return; }
    const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
    this.mesh.geometry = new THREE.TubeGeometry(curve, Math.min(400, points.length * 3), this.radius, 8, false);
  }
}

export function sphere(radius, color, opacity = 1, emissive = 0.0) {
  const m = new THREE.MeshPhysicalMaterial({ color, roughness: 0.35, metalness: 0.0, transparent: opacity < 1, opacity,
    emissive: color, emissiveIntensity: emissive, clearcoat: 0.6 });
  const s = new THREE.Mesh(new THREE.SphereGeometry(radius, 40, 28), m);
  s.castShadow = true;
  return s;
}

// colors: CSS hex strings (sRGB) per point
export function points(positions, colors, size = 0.03, opacity = 1) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions.flat(), 3));
  const lin = colors.map((c) => { const k = new THREE.Color(c); return [k.r, k.g, k.b]; });
  g.setAttribute('color', new THREE.Float32BufferAttribute(lin.flat(), 3));
  const m = new THREE.PointsMaterial({ size, vertexColors: true, map: dotTexture(), transparent: true, alphaTest: 0.05,
    depthWrite: false, sizeAttenuation: true, toneMapped: false, opacity });
  return new THREE.Points(g, m);
}

export function hex(c) { return new THREE.Color(c); }
