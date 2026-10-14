// Shared three.js pieces for the Mission page: scene, robot drawn from PyBullet link frames,
// interpolation between simulation steps. (Robot poses always come from the server.)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

export { THREE };

export function createViewer(el, { camera = [0.95, 0.72, 0.95], target = [0, 0.28, 0] } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#12161c');
  const cam = new THREE.PerspectiveCamera(45, 1, 0.01, 60);
  cam.position.set(...camera);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a28, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8); sun.position.set(2.5, 4, 3); scene.add(sun);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-3, 2, -2); scene.add(fill);
  scene.add(new THREE.GridHelper(4, 40, 0x3a414c, 0x232932));
  const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);   // PyBullet is Z-up
  const v = { el, renderer, scene, camera: cam, controls, root };
  v.reset = () => { cam.position.set(...camera); controls.target.set(...target); controls.update(); };
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = h + 'px';
    cam.aspect = w / h; cam.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el); resize();
  const loop = () => { controls.update(); renderer.render(scene, cam); requestAnimationFrame(loop); };
  loop();
  return v;
}

const meshCache = new Map();
function loadMesh(url, ext) {
  if (!meshCache.has(url)) {
    const loader = ext === 'stl' ? new STLLoader() : new OBJLoader();
    meshCache.set(url, loader.loadAsync(url).then((res) => { if (ext === 'stl') res.computeVertexNormals(); return res; }));
  }
  return meshCache.get(url);
}

// style: 'solid' (URDF colours) or a colour string for a transparent ghost
export class RobotView {
  constructor(model, style = 'solid', opacity = 0.4) { this.group = new THREE.Group(); this.frames = {}; this.style = style; this.opacity = opacity; this.model = model; }
  material(v) {
    if (this.style === 'solid') { const [r, g, b] = v.rgba; return new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), metalness: 0.15, roughness: 0.55 }); }
    return new THREE.MeshStandardMaterial({ color: this.style, transparent: true, opacity: this.opacity, depthWrite: false, roughness: 0.6 });
  }
  async load() {
    await Promise.all(this.model.visuals.map(async (v) => {
      let f = this.frames[v.link];
      if (!f) { f = new THREE.Group(); this.frames[v.link] = f; this.group.add(f); }
      const holder = new THREE.Group(); holder.position.set(...v.pos); holder.quaternion.set(...v.orn); f.add(holder);
      const mat = this.material(v); let obj;
      if (v.mesh) {
        const res = await loadMesh(v.mesh, v.ext);
        if (res.isBufferGeometry) obj = new THREE.Mesh(res, mat);
        else { obj = res.clone(true); obj.traverse((c) => { if (c.isMesh) c.material = mat; }); }
        obj.scale.set(...v.dims);
      } else if (v.type === 2) obj = new THREE.Mesh(new THREE.SphereGeometry(v.dims[0], 16, 12), mat);
      else if (v.type === 3) obj = new THREE.Mesh(new THREE.BoxGeometry(...v.dims), mat);
      else if (v.type === 4 || v.type === 7) { obj = new THREE.Mesh(new THREE.CylinderGeometry(v.dims[1], v.dims[1], v.dims[0], 20), mat); obj.rotation.x = Math.PI / 2; }
      else return;
      holder.add(obj);
    }));
    return this;
  }
  setFrames(links) {
    for (const [k, [p, q]] of Object.entries(links || {})) { const f = this.frames[k]; if (f) { f.position.set(...p); f.quaternion.set(...q); } }
  }
}

// interpolate link frames between two simulation steps (position lerp, rotation slerp)
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
export function lerpLinks(A, B, t) {
  if (!B || t <= 0) return A;
  const out = {};
  for (const [k, [pa, qa]] of Object.entries(A)) {
    const [pb, qb] = B[k] || [pa, qa];
    _qa.set(...qa); _qb.set(...qb); _qa.slerp(_qb, t);
    out[k] = [[pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t], [_qa.x, _qa.y, _qa.z, _qa.w]];
  }
  return out;
}

// coordinate frame marker (target pose): sphere + RGB axes
export function frameMarker(color = 0xe34948, size = 0.1) {
  const g = new THREE.Group();
  g.add(new THREE.Mesh(new THREE.SphereGeometry(0.016, 20, 14), new THREE.MeshStandardMaterial({ color, emissive: 0x551010 })));
  for (const [x, y, z, c] of [[1, 0, 0, 0xff4040], [0, 1, 0, 0x40d040], [0, 0, 1, 0x4080ff]]) {
    g.add(new THREE.ArrowHelper(new THREE.Vector3(x, y, z), new THREE.Vector3(), size, c, 0.025, 0.014));
  }
  return g;
}

// polyline with one colour per vertex
export function colorLine(points, colors, { dashed = false } = {}) {
  const geo = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors.flatMap((c) => { const k = new THREE.Color(c); return [k.r, k.g, k.b]; }), 3));
  const mat = dashed ? new THREE.LineDashedMaterial({ vertexColors: true, dashSize: 0.02, gapSize: 0.012 }) : new THREE.LineBasicMaterial({ vertexColors: true });
  const line = new THREE.Line(geo, mat);
  if (dashed) line.computeLineDistances();
  return line;
}
