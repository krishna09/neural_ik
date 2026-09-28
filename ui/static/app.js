// Neural IK Studio - front end.
// All robot poses shown here come from PyBullet forward kinematics on the
// server (the same robot_config.load_robot the project uses); this file only
// draws them. Datasets and trained models live in the server's library
// (ui/workspace/library/) and are never overwritten.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const $ = (id) => document.getElementById(id);
const DEG = 180 / Math.PI;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = {
  s1: css('--series-1'), s2: css('--series-2'), s3: css('--series-3'),
  grid: '#2c2c29', axis: css('--text-muted'), text: css('--text-secondary'),
};
const POS_T = ['10', '25', '50', '100'];
// Tolerance -> key used in the stored accuracy dicts ("100", "12.5"); matches ik_metrics.tkey.
const tolKey = (v) => String(+v);
// The accuracy tolerance a run was trained with (older runs: 50 mm).
const runTol = (r) => tolKey((r && r.params && r.params.pos_tol_mm) || 50);
const ORN_T = ['5', '15', '30'];

const state = {
  robot: null, model: null, dof: 0,
  job: null,
  lib: { datasets: [], runs: [], project_datasets: [] },
  selRun: null, runDetail: null,
  dsId: null, dsN: 0, dsIndex: 0, playing: false,
  stats: [], liveIk: [], liveCol: [], liveErr: [], liveRun: null,
  posT: '50', ornT: '15',
  liveTol: '50', liveEs: null,        // tolerance + early-stop rule of the training job being watched
  lb: { caseIdx: 0, imgIdx: 0 },
  inf: { test: null },
};

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

let toastTimer;
function toast(msg, err = false) {
  const t = $('toast');
  t.textContent = msg; t.className = err ? 'err' : ''; t.style.display = 'block';
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.style.display = 'none'), 5000);
}

const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const pct = (v, d = 1) => (v === null || v === undefined ? '—' : `${Number(v).toFixed(d)}%`);
const fmtVec = (a, d = 3) => '[' + a.map((v) => fmt(v, d)).join(', ') + ']';
const degs = (q) => q.map((v) => (v * DEG).toFixed(1)).join('  ');
const when = (t) => (t ? new Date(t * 1000).toLocaleString(undefined, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—');
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const shortJoint = (n) => n.replace(/^.*?(joint_?\d+)$/, '$1');
function gauss() { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }

// ===================================================================
// 3D viewer
// ===================================================================

const viewer = {};

function initViewer() {
  const el = $('viewport');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#141413');
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 60);
  camera.position.set(1.5, 1.1, 1.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.35, 0);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a28, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(2.5, 4, 3);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5);
  fill.position.set(-3, 2, -2);
  scene.add(fill);
  scene.add(new THREE.GridHelper(4, 40, 0x4a4a45, 0x2a2a27));

  // PyBullet is Z-up, three.js is Y-up: everything robot-related lives in `root`.
  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;
  scene.add(root);
  root.add(new THREE.AxesHelper(0.15));

  const marker = new THREE.Group();
  marker.add(new THREE.Mesh(new THREE.SphereGeometry(0.016, 20, 14),
    new THREE.MeshStandardMaterial({ color: 0xe34948, emissive: 0x551010 })));
  for (const [x, y, z, c] of [[1, 0, 0, 0xff4040], [0, 1, 0, 0x40d040], [0, 0, 1, 0x4080ff]]) {
    marker.add(new THREE.ArrowHelper(new THREE.Vector3(x, y, z), new THREE.Vector3(), 0.1, c, 0.025, 0.014));
  }
  marker.visible = false;
  root.add(marker);

  const errGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const errLine = new THREE.Line(errGeom, new THREE.LineDashedMaterial({ color: 0xfab219, dashSize: 0.02, gapSize: 0.012 }));
  errLine.visible = false;
  root.add(errLine);

  const cloud = new THREE.Points(new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 0.009, vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false }));
  root.add(cloud);

  Object.assign(viewer, { el, renderer, scene, camera, controls, root, marker, errLine, cloud, solid: null, ghost: null, refined: null });

  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = w + 'px';
    renderer.domElement.style.height = h + 'px';
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el);
  resize();
  const loop = () => { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); };
  loop();
}

function setMarker(pos, orn) {
  if (!pos) { viewer.marker.visible = false; return; }
  viewer.marker.position.set(pos[0], pos[1], pos[2]);
  viewer.marker.quaternion.set(orn[0], orn[1], orn[2], orn[3]);
  viewer.marker.visible = true;
}

function setErrLine(a, b) {
  if (!a || !b) { viewer.errLine.visible = false; return; }
  const pos = viewer.errLine.geometry.attributes.position;
  pos.setXYZ(0, a[0], a[1], a[2]);
  pos.setXYZ(1, b[0], b[1], b[2]);
  pos.needsUpdate = true;
  viewer.errLine.computeLineDistances();
  viewer.errLine.visible = true;
}

const meshCache = new Map();
function loadMesh(url, ext) {
  if (!meshCache.has(url)) {
    const loader = ext === 'stl' ? new STLLoader() : new OBJLoader();
    meshCache.set(url, loader.loadAsync(url).then((res) => {
      if (ext === 'stl') res.computeVertexNormals();
      return res;
    }));
  }
  return meshCache.get(url);
}

class RobotView {
  constructor(model, style) {        // style: 'solid' | 'ghost' | 'refined'
    this.group = new THREE.Group();
    this.frames = {};
    this.meshes = {};
    this.style = style;
    this.model = model;
  }

  material(v) {
    if (this.style !== 'solid') {
      const c = this.style === 'ghost' ? COLORS.s2 : COLORS.s3;
      return new THREE.MeshStandardMaterial({
        color: c, emissive: this.style === 'ghost' ? 0x4a1a06 : 0x06331f, transparent: true, opacity: 0.4, depthWrite: false, roughness: 0.6,
      });
    }
    const [r, g, b] = v.rgba;
    return new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), metalness: 0.15, roughness: 0.55 });
  }

  async load() {
    await Promise.all(this.model.visuals.map(async (v) => {
      let f = this.frames[v.link];
      if (!f) { f = new THREE.Group(); this.frames[v.link] = f; this.group.add(f); }
      const holder = new THREE.Group();
      holder.position.set(...v.pos);
      holder.quaternion.set(v.orn[0], v.orn[1], v.orn[2], v.orn[3]);
      f.add(holder);
      let obj;
      const mat = this.material(v);
      if (v.mesh) {
        const res = await loadMesh(v.mesh, v.ext);
        if (res.isBufferGeometry) obj = new THREE.Mesh(res, mat);
        else { obj = res.clone(true); obj.traverse((c) => { if (c.isMesh) c.material = mat; }); }
        obj.scale.set(...v.dims);
      } else if (v.type === 2) obj = new THREE.Mesh(new THREE.SphereGeometry(v.dims[0], 16, 12), mat);
      else if (v.type === 3) obj = new THREE.Mesh(new THREE.BoxGeometry(...v.dims), mat);
      else if (v.type === 4 || v.type === 7) {
        obj = new THREE.Mesh(new THREE.CylinderGeometry(v.dims[1], v.dims[1], v.dims[0], 20), mat);
        obj.rotation.x = Math.PI / 2;
      } else return;
      holder.add(obj);
      (this.meshes[v.link] ||= []).push(mat);
    }));
    return this;
  }

  setFrames(links) {
    for (const [k, [p, q]] of Object.entries(links)) {
      const f = this.frames[k];
      if (f) { f.position.set(p[0], p[1], p[2]); f.quaternion.set(q[0], q[1], q[2], q[3]); }
    }
  }

  setCollision(links) {
    if (this.style !== 'solid') return;
    const hit = new Set((links || []).map(String));
    for (const [k, mats] of Object.entries(this.meshes)) for (const m of mats) m.emissive.set(hit.has(k) ? 0xc01010 : 0x000000);
  }
}

function showPose(fkRes, { ghost = null, refined = null, target = null, targetOrn = null } = {}) {
  if (!viewer.solid) return;
  if (fkRes) {
    viewer.solid.setFrames(fkRes.links);
    viewer.solid.setCollision(fkRes.collision_links);
  }
  const showGhost = $('show-ghost').checked;
  viewer.ghost.group.visible = !!(ghost && showGhost);
  if (ghost) viewer.ghost.setFrames(ghost.links);
  viewer.refined.group.visible = !!(refined && showGhost);
  if (refined) viewer.refined.setFrames(refined.links);
  setMarker(target, targetOrn);
  const tip = refined || ghost;
  setErrLine(tip && target ? target : null, tip && target ? tip.ee_pos : null);
}

function fitView() {
  const box = new THREE.Box3().setFromObject(viewer.solid.group);
  const size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.y, size.z, 0.4);
  const c = box.getCenter(new THREE.Vector3());
  viewer.controls.target.set(0, c.y * 0.8, 0);
  viewer.camera.position.set(r * 1.5, r * 1.1 + c.y * 0.4, r * 1.6);
  viewer.controls.update();
}

function setHud(title, lines) {
  $('hud-title').textContent = title;
  $('hud-lines').textContent = lines || '';
}

// ---------------------------------------------------------------- cloud
const cloudData = { pos: [], col: [] };
const C_FREE = new THREE.Color(COLORS.s1), C_COLL = new THREE.Color('#e66767');

function setCloud(positions, labels) {
  cloudData.pos = Array.from(positions);
  cloudData.col = [];
  for (const l of labels) { const c = l ? C_COLL : C_FREE; cloudData.col.push(c.r, c.g, c.b); }
  flushCloud();
}
function appendCloud(p, collided) {
  cloudData.pos.push(p[0], p[1], p[2]);
  const c = collided ? C_COLL : C_FREE;
  cloudData.col.push(c.r, c.g, c.b);
}
function flushCloud() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(cloudData.pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(cloudData.col, 3));
  viewer.cloud.geometry.dispose();
  viewer.cloud.geometry = g;
  viewer.cloud.visible = $('show-cloud').checked;
}

// ===================================================================
// Robot selection + jog
// ===================================================================

async function selectRobot(name) {
  if (state.robot === name && viewer.solid) return;
  state.robot = name;
  $('robot-select').value = name;
  $('loading').style.display = 'grid';
  const model = await api(`/api/robot_model?robot=${name}`);
  state.model = model;
  state.dof = model.dof;
  $('robot-desc').textContent = `${model.dof}-DOF · ${model.description}`;
  for (const k of ['solid', 'ghost', 'refined']) if (viewer[k]) viewer.root.remove(viewer[k].group);
  viewer.solid = await new RobotView(model, 'solid').load();
  viewer.ghost = await new RobotView(model, 'ghost').load();
  viewer.refined = await new RobotView(model, 'refined').load();
  viewer.ghost.group.visible = viewer.refined.group.visible = false;
  viewer.root.add(viewer.solid.group, viewer.ghost.group, viewer.refined.group);
  buildJog();
  await jogTo(model.joints.map(() => 0));
  fitView();
  $('loading').style.display = 'none';
  fillSelects();
  await loadDataset($('ds-src').value || null);
  await refreshModels();
}

// ---- network architectures: train_ik.py base model + modelset/*.py ----
const fmtParams = (n) => (n == null ? '' : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M params` : `${Math.round(n / 1e3)}k params`);
function modelLabel(info) {
  if (!info || info.id === 'base') return 'base · train_ik.py';
  return info.id.replace(/^modelset\//, '');
}

async function refreshModels() {
  const res = await api(`/api/models?robot=${state.robot}`);
  state.models = res.models;
  state.modelFolder = res.folder;
  const sel = $('tr-model');
  const prev = sel.value;
  sel.innerHTML = '';
  for (const e of res.models) {
    const o = document.createElement('option');
    o.value = e.id;
    o.disabled = !e.ok;
    o.textContent = e.ok
      ? `${e.base ? 'Base model – train_ik.py' : e.file} · ${e.cls} · ${fmtParams(e.params)}`
      : `⚠ ${e.file} – not loadable`;
    sel.appendChild(o);
  }
  if (res.models.some((e) => e.id === prev && e.ok)) sel.value = prev;
  showModelInfo();
}

function showModelInfo() {
  const e = (state.models || []).find((m) => m.id === $('tr-model').value);
  const broken = (state.models || []).filter((m) => !m.ok);
  const extra = (state.models || []).length - 1;
  $('tr-model-info').innerHTML = (e ? `${e.base ? 'The project model, trained exactly as train_ik.py defines it.' : `From <code>${esc(e.file)}</code>`}${e.description ? ' – ' + esc(e.description) : ''}. ` : '')
    + `Every model is trained by train_ik.main() (same loss, optimiser, splits). ${extra} model file${extra === 1 ? '' : 's'} in <code>${esc(state.modelFolder || 'modelset')}/</code> – add a .py there and press ↻.`
    + broken.map((b) => `<br>⚠ <code>${esc(b.file)}</code>: ${esc(b.error)}`).join('');
  $('tr-model-info').classList.toggle('warn', broken.length > 0);
}
$('tr-model').onchange = showModelInfo;
$('tr-model-rescan').onclick = () => refreshModels().then(() => toast('Rescanned the modelset folder.')).catch((e) => toast(e.message, true));

function buildJog() {
  const el = $('jog');
  el.innerHTML = '';
  for (const [i, j] of state.model.joints.entries()) {
    const row = document.createElement('div');
    row.className = 'joint';
    row.innerHTML = `<span class="name" title="${j.name}">${shortJoint(j.name)}</span>
      <input type="range" min="${j.lo}" max="${j.hi}" step="0.001" value="0" data-i="${i}">
      <span class="val" id="jog-v${i}">0.0°</span>`;
    el.appendChild(row);
  }
  el.querySelectorAll('input').forEach((s) => s.addEventListener('input', scheduleJog));
}

const jogQ = () => [...$('jog').querySelectorAll('input')].map((s) => parseFloat(s.value));

let jogPending = false;
function scheduleJog() {
  if (jogPending) return;
  jogPending = true;
  requestAnimationFrame(async () => { jogPending = false; await jogTo(jogQ(), false); });
}

async function jogTo(q, setSliders = true) {
  const sliders = [...$('jog').querySelectorAll('input')];
  if (setSliders) sliders.forEach((s, i) => (s.value = q[i]));
  q.forEach((v, i) => ($(`jog-v${i}`).textContent = (v * DEG).toFixed(1) + '°'));
  const r = await api('/api/fk', { robot: state.robot, q });
  showPose(r, { target: r.ee_pos, targetOrn: r.ee_orn });
  $('jog-readout').innerHTML = `<dt>EE position (m)</dt><dd>${fmtVec(r.ee_pos)}</dd>
    <dt>EE quaternion</dt><dd>${fmtVec(r.ee_orn)}</dd>
    <dt>Self-collision</dt><dd>${r.colliding ? '<span class="badge bad">colliding</span>' : '<span class="badge good">free</span>'}</dd>`;
  setHud('Manual joint control (forward kinematics)',
    `q (deg)  ${degs(q)}\nEE pos   ${fmtVec(r.ee_pos)}\nEE quat  ${fmtVec(r.ee_orn)}\n${r.colliding ? 'SELF-COLLISION: links ' + r.collision_links.join(', ') : 'no self-collision'}`);
  return r;
}

$('jog-zero').onclick = () => jogTo(state.model.joints.map(() => 0));
$('jog-random').onclick = () => jogTo(state.model.joints.map((j) => j.lo + Math.random() * (j.hi - j.lo)));

// ===================================================================
// Library
// ===================================================================

const dsById = (id) => state.lib.datasets.find((d) => d.id === id) || state.lib.project_datasets.find((d) => d.id === id);
const runById = (id) => state.lib.runs.find((r) => r.id === id);
const KIND = { 'ui-train': 'UI training', 'legacy-ui': 'UI (pre-library)', 'project-import': 'project models/' };

function dsLabel(d) {
  if (!d) return '—';
  if (d.id.startsWith('project:')) return `project data/ (live, read-only) · ${d.robot}`;
  const st = d.status !== 'ready' ? ` · ${d.status}` : '';
  const imported = !/^ui\b/i.test(d.source || '');
  const coll = d.n && d.n_colliding != null ? ` · ${((100 * d.n_colliding) / d.n).toFixed(0)}% coll.` : '';
  return `${when(d.created)} · n=${(d.n ?? 0).toLocaleString()} · seed ${d.seed ?? '?'}${coll}${imported ? ' · imported' : ''}${st}`;
}

function runLabel(r) {
  const t = r.metrics && r.metrics.test;
  const k = runTol(r);
  const acc = t && t.acc_pos[k] != null ? ` · test ${fmt(t.acc_pos[k], 1)}% ≤${k}mm` : '';
  return `${r.id} · ${modelLabel(r.model)}${acc}`;
}

async function refreshLibrary() {
  state.lib = await api('/api/library');
  fillSelects();
  renderLibrary();
}

function setOptions(sel, items, labelFn, emptyText) {
  const prev = sel.value;
  sel.innerHTML = '';
  if (!items.length) {
    const o = document.createElement('option'); o.value = ''; o.textContent = emptyText; sel.appendChild(o);
    return;
  }
  for (const it of items) {
    const o = document.createElement('option'); o.value = it.id; o.textContent = labelFn(it); sel.appendChild(o);
  }
  if (items.some((i) => i.id === prev)) sel.value = prev;
}

function fillSelects() {
  const mine = state.lib.datasets.filter((d) => d.robot === state.robot && d.status === 'ready');
  const proj = state.lib.project_datasets.filter((d) => d.robot === state.robot);
  setOptions($('ds-src'), [...mine, ...proj], dsLabel, 'No dataset yet – generate one');
  setOptions($('tr-ds'), mine, dsLabel, 'No library dataset – generate one');
  setOptions($('inf-ds'), [...mine, ...proj], dsLabel, 'No dataset');
  const models = state.lib.runs.filter((r) => r.robot === state.robot && r.has_models);
  setOptions($('inf-run'), models, runLabel, 'No trained model for this robot');
  setOptions($('run-select'), state.lib.runs, (r) => `${r.id} · ${KIND[r.kind] || r.kind}`, 'no runs yet');
  if (state.selRun) $('run-select').value = state.selRun;
  updateSplitUI();
}

function renderLibrary() {
  const { datasets, runs, project_datasets: proj } = state.lib;
  const dsRows = datasets.map((d) => `<tr>
      <td><span class="id">${esc(d.id)}</span></td><td>${d.robot}</td>
      <td class="num">${(d.n ?? 0).toLocaleString()}</td><td class="num">${d.seed ?? '?'}</td>
      <td class="num">${d.n ? pct((100 * (d.n_colliding ?? 0)) / d.n) : '—'}</td>
      <td>${esc(d.source)}${d.split && d.split.mode === 'custom' ? `<span class="tag" title="${esc(d.split.note || '')}">custom split</span>` : ''}${d.status !== 'ready' ? `<span class="tag warn">${d.status}</span>` : ''}</td>
      <td>${when(d.created)}</td>
      <td>${d.used_by.length ? d.used_by.map((r) => `<span class="id">${r}</span>`).join(', ') : '<span class="hint">—</span>'}</td>
      <td><button data-act="ds-browse" data-id="${d.id}">Browse</button> <button data-act="ds-train" data-id="${d.id}" ${d.status !== 'ready' ? 'disabled' : ''}>Train on</button></td></tr>`).join('');
  const projRows = proj.map((d) => `<tr><td><span class="id">${esc(d.file)}</span></td><td>${d.robot}</td><td colspan="5" class="muted">live project file, read-only – "Import from project" copies it into the library</td>
      <td></td><td><button data-act="ds-browse" data-id="${d.id}">Browse</button></td></tr>`).join('');
  const runRows = runs.map((r) => {
    const te = r.metrics && r.metrics.test, va = r.metrics && r.metrics.val;
    const k = runTol(r);                              // each run is shown at its own training tolerance
    const ev = r.evaluations.filter((e) => e.overlap && e.overlap.level === 'independent').slice(-1)[0];
    const testCell = te ? `${pct(te.acc_pos[k])} <span class="hint">(${fmt(te.pos_mean_mm, 0)} mm mean)</span>`
      : ev ? `${pct(ev.summary.acc_pos[k])} <span class="tag good">independent eval</span>` : '<span class="hint">no test split</span>';
    const es = r.early_stop_result;
    return `<tr class="${r.id === state.selRun ? 'sel' : ''}">
      <td><span class="id">${r.id}</span></td><td>${r.robot}</td><td>${esc(modelLabel(r.model))}</td><td>${KIND[r.kind] || r.kind}</td>
      <td>${r.status === 'done' ? '<span class="tag good">done</span>' : `<span class="tag ${r.status === 'running' ? 'warn' : 'bad'}">${esc(r.status)}</span>`}${es ? `<span class="tag" title="target met – stopped automatically">auto-stopped</span>` : ''}</td>
      <td><span class="id">${esc(r.dataset_id || '?')}</span></td>
      <td class="num">${r.epochs_done || '—'}${es ? `<span class="hint"> / ${es.max_epochs}</span>` : ''}</td>
      <td class="num">${k} mm</td>
      <td class="num">${va ? pct(va.acc_pos[k]) : '—'}</td>
      <td class="num">${testCell}</td>
      <td class="num">${r.evaluations.length}</td><td class="num">${r.n_inferences}</td><td>${when(r.created)}</td>
      <td><button data-act="run-details" data-id="${r.id}">Details</button>
          <button data-act="run-infer" data-id="${r.id}" ${r.has_models ? '' : 'disabled'}>Inference</button></td></tr>`;
  }).join('');
  $('library-pane').innerHTML = `<div class="pane-pad">
    <div class="toolbar"><span class="grow hint">Everything lives in <code>ui/workspace/library/</code>; each dataset and each training run has its own folder + <code>meta.json</code> and is never overwritten. Accuracy = % of predictions whose end-effector lands within the run's <b>tolerance</b> of the target (set when training; other tolerances in Run details).</span>
      <button id="import-btn" title="Copy the project's data/*.npz and models/*.pt into the library (the project files are only read)">Import from project</button></div>
    <h3 class="sec">Datasets <small>${datasets.length} saved</small></h3>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Dataset</th><th>Robot</th><th class="num">Samples</th><th class="num">Seed</th><th class="num">Colliding</th><th>Source</th><th>Created</th><th>Used by runs</th><th></th></tr></thead>
      <tbody>${dsRows}${projRows || ''}</tbody></table></div>
    <h3 class="sec">Trained models (runs) <small>${runs.length} saved</small></h3>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Run</th><th>Robot</th><th>Model</th><th>Origin</th><th>Status</th><th>Trained on</th><th class="num">Epochs</th>
      <th class="num">Tolerance</th><th class="num">Val acc</th><th class="num">Test acc</th><th class="num">Evals</th><th class="num">Inferences</th><th>Created</th><th></th></tr></thead>
      <tbody>${runRows || '<tr><td colspan="14" class="muted">no trained models yet</td></tr>'}</tbody></table></div></div>`;
  $('import-btn').onclick = async () => {
    try {
      const r = await api('/api/import_project', {});
      toast(`Imported ${r.datasets.length} dataset(s) and ${r.runs.length} model(s) from the project (copies).`);
      await refreshLibrary();
    } catch (e) { toast(e.message, true); }
  };
}

$('library-pane').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-act]');
  if (!b) return;
  const id = b.dataset.id;
  try {
    if (b.dataset.act === 'ds-browse') {
      const d = dsById(id);
      await selectRobot(d.robot);
      $('ds-src').value = id;
      await loadDataset(id);
      await showSample(0);
      openPanel('ds-src');
    } else if (b.dataset.act === 'ds-train') {
      const d = dsById(id);
      await selectRobot(d.robot);
      $('tr-ds').value = id;
      openPanel('tr-ds');
      toast('Training dataset selected – press Train in panel 3.');
    } else if (b.dataset.act === 'run-details') {
      await selectRun(id);
      switchTab('details-pane');
    } else if (b.dataset.act === 'run-infer') {
      const r = runById(id);
      await selectRobot(r.robot);
      $('inf-run').value = id;
      await selectRun(id);
      openPanel('inf-run');
      syncInferenceMode();
    }
  } catch (err) { toast(err.message, true); }
});

function openPanel(childId) {
  const det = $(childId).closest('details');
  det.open = true;
  det.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===================================================================
// Run details / gallery / inference log
// ===================================================================

async function selectRun(id) {
  if (!id) return;
  const changed = id !== state.selRun;
  state.selRun = id;
  $('run-select').value = id;
  state.runDetail = await api(`/api/run?id=${encodeURIComponent(id)}`);
  if (changed) state.posT = runTol(state.runDetail);   // show a run at the tolerance it was trained with
  renderLibrary();
  renderDetails();
  renderGallery();
  renderInfLog();
}

let detailTimer;
function refreshDetailSoon() {
  clearTimeout(detailTimer);
  detailTimer = setTimeout(() => state.selRun && selectRun(state.selRun), 400);
}

function accRow(label, s, colAcc, extra = '') {
  if (!s || !s.n) return `<tr><td>${label}${extra}</td><td colspan="10" class="muted">not available</td></tr>`;
  const pa = s.acc_pos && s.acc_pos[state.posT], oa = s.acc_orn && s.acc_orn[state.ornT];
  return `<tr><td>${label}${extra}</td><td class="num">${s.n.toLocaleString()}</td>
    <td class="num">${pct(pa)}<span class="accbar"><span style="width:${pa ?? 0}%"></span></span></td>
    <td class="num">${pct(oa)}</td>
    <td class="num">${fmt(s.pos_mean_mm, 1)}</td><td class="num">${fmt(s.pos_median_mm, 1)}</td>
    <td class="num">${fmt(s.orn_mean_deg, 1)}</td><td class="num">${fmt(s.joint_mae_deg, 1)}</td>
    <td class="num">${pct(s.self_collision_pct)}</td><td class="num">${pct(s.limit_violation_pct)}</td>
    <td class="num">${pct(colAcc)}</td></tr>`;
}

const OVERLAP_TAG = { independent: 'good', 'same-seed': 'bad', 'training-data': 'bad', unknown: 'warn' };

function overlapPreview(run, dsId) {
  const ds = dsById(dsId);
  if (!ds || !run) return null;
  if (!run.dataset_id) return { level: 'unknown', note: 'training dataset of this model is unknown' };
  if (run.dataset_id === ds.id) return { level: 'training-data', note: 'this is the dataset the model was trained on' };
  const tr = dsById(run.dataset_id);
  if (!tr) return { level: 'unknown', note: 'training dataset not in the library' };
  if (tr.sha256 && tr.sha256 === ds.sha256) return { level: 'training-data', note: 'identical file to the training dataset' };
  if (tr.seed == null || ds.seed == null) return { level: 'unknown', note: 'generator seed unknown' };
  if (tr.seed === ds.seed) return { level: 'same-seed', note: `same generator seed (${ds.seed}) as the training data – rows largely coincide` };
  return { level: 'independent', note: `independent (seed ${ds.seed} vs training seed ${tr.seed})` };
}

function esRuleText(es, tol) {
  const where = es.split === 'val' ? 'validation' : 'training';
  const what = es.metric === 'mean_error'
    ? `the mean position error on the ${where} split is ≤ ${tol} mm`
    : `≥ ${+es.target_pct}% of ${where} samples land within ${tol} mm`;
  return `${what} for ${es.patience} epoch${es.patience === 1 ? '' : 's'} in a row`;
}

function esSummary(m) {
  const es = m.params && m.params.early_stop;
  if (!es) return '';
  const tol = runTol(m), r = m.early_stop_result;
  const ik = (m.epochs || []).filter((e) => e.phase === 'ik');
  return r
    ? `<div class="hint"><span class="tag good">auto-stopped</span> at epoch <b>${r.stopped_at_epoch}</b> of max ${r.max_epochs}: ${esRuleText(es, tol)}.</div>`
    : `<div class="hint"><span class="tag warn">target not reached</span> Automatic stop was on (${esRuleText(es, tol)}) – ${m.status === 'running' ? 'still training' : `ran all ${ik.length} epochs without meeting it`}.</div>`;
}

function renderDetails() {
  const m = state.runDetail;
  const pane = $('details-pane');
  if (!m) { pane.innerHTML = '<div class="empty">Select a run in the Library.</div>'; return; }
  const cm = m.collision_metrics || {};
  const met = m.metrics || {};
  const sp = m.split || {};
  const inf = (m.inference && m.inference.summary) || { n: 0 };
  const evRows = (m.evaluations || []).map((e) => accRow(
    `Evaluation on <span class="id">${esc(e.dataset_id)}</span>`, e.summary, e.collision && e.collision.accuracy,
    ` <span class="tag ${OVERLAP_TAG[e.overlap?.level] || 'warn'}" title="${esc(e.overlap?.note)}">${esc(e.overlap?.level || '?')}</span>`)).join('');
  const dsOpts = state.lib.datasets.filter((d) => d.robot === m.robot && d.status === 'ready')
    .map((d) => `<option value="${d.id}">${esc(dsLabel(d))}</option>`).join('');
  const hasSplit = !!met.test;
  pane.innerHTML = `<div class="pane-pad">
    <div class="toolbar">
      <h3 class="sec"><span class="id">${esc(m.id)}</span><small>${m.robot} · model ${esc(modelLabel(m.model))}${m.model && m.model.params ? ` (${fmtParams(m.model.params)})` : ''} · ${KIND[m.kind] || m.kind} · ${esc(m.status)} · ${when(m.created)}${m.device ? ' · ' + esc(m.gpu || m.device) : ''}</small></h3>
      <span class="grow"></span>
      <label class="hint">Position tolerance <select id="thr-pos" class="inline-sel">${[...new Set([...POS_T, runTol(m), state.posT])].sort((a, b) => a - b)
        .map((t) => `<option value="${t}" ${t === state.posT ? 'selected' : ''}>${t}${t === runTol(m) ? ' (training)' : ''}</option>`).join('')}</select> mm</label>
      <label class="hint">Orientation <select id="thr-orn" class="inline-sel">${ORN_T.map((t) => `<option ${t === state.ornT ? 'selected' : ''}>${t}</option>`).join('')}</select> °</label>
    </div>
    ${m.notes ? `<div class="notes">${esc(m.notes)}</div>` : ''}
    ${esSummary(m)}
    <div class="hint">Trained on <span class="id">${esc(m.dataset_id || 'unknown')}</span>${sp.n_dataset ? ` · ${sp.n_dataset.toLocaleString()} samples → test ${sp.n_test.toLocaleString()} held out (${sp.n_test_free.toLocaleString()} collision-free) · train_ik got ${sp.n_given_to_train_ik.toLocaleString()} → IK train ${(sp.n_ik_train ?? 0).toLocaleString()} / validation ${(sp.n_ik_val ?? 0).toLocaleString()}` : ''}
      ${m.inference_speed ? ` · network inference ${fmt(m.inference_speed.single_ms, 3)} ms per call on ${esc(m.inference_speed.device)}` : ''}</div>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Where</th><th class="num">n</th><th class="num">Pos. acc ≤${state.posT} mm</th><th class="num">Orn. acc ≤${state.ornT}°</th>
      <th class="num">Mean pos err (mm)</th><th class="num">Median (mm)</th><th class="num">Mean orn err (°)</th><th class="num">Joint MAE (°)</th>
      <th class="num">Pred. self-collision</th><th class="num">Joint-limit viol.</th><th class="num">Collision-net acc</th></tr></thead><tbody>
      <tr class="group"><td colspan="11">Training (train_ik's own splits)</td></tr>
      ${accRow('Train split', met.train, cm.train && cm.train.accuracy)}
      ${accRow('Validation split', met.val, cm.val && cm.val.accuracy)}
      <tr class="group"><td colspan="11">Testing</td></tr>
      ${hasSplit ? accRow('Held-out test split', met.test, cm.test && cm.test.accuracy, ' <span class="tag good">never seen in training</span>') : '<tr><td colspan="11" class="muted">No held-out test split for this run – use "Evaluate on dataset" below with a dataset generated from a different seed.</td></tr>'}
      ${evRows}
      <tr class="group"><td colspan="11">Inference (every call from panel 4, logged)</td></tr>
      ${inf.n ? accRow(`Inference log – network output <span class="hint">(${fmt(inf.network.latency_ms_mean, 3)} ms avg)</span>`, inf.network, null) : '<tr><td colspan="11" class="muted">no inferences yet</td></tr>'}
      ${inf.refined ? accRow('Inference log – after DLS refinement', inf.refined, null) : ''}
    </tbody></table></div>
    <div class="details-grid">
      <div class="box"><h4><span>Loss per epoch (MSE)</span><span class="legend"><span><i style="background:${COLORS.s1}"></i>train</span><span><i style="background:${COLORS.s2}"></i>validation</span></span></h4><canvas id="d-loss"></canvas></div>
      <div class="box"><h4><span>IK accuracy per epoch (≤${state.posT} mm, %)</span><span class="legend"><span><i style="background:${COLORS.s1}"></i>train</span><span><i style="background:${COLORS.s2}"></i>validation</span></span></h4><canvas id="d-acc"></canvas></div>
      <div class="box"><h4><span>Collision-net accuracy per epoch (%)</span><span class="legend"><span><i style="background:${COLORS.s1}"></i>train</span><span><i style="background:${COLORS.s2}"></i>validation</span></span></h4><canvas id="d-col"></canvas></div>
    </div>
    <div class="box"><h4><span>Evaluate this model on a dataset (batch inference → test accuracy)</span></h4>
      <div class="toolbar">${m.has_models ? `<select id="ev-ds" style="width:auto;max-width:420px">${dsOpts}</select>
        <label class="hint">max samples <input id="ev-max" type="number" min="0" value="0" style="width:90px" title="0 = all"></label>
        <button id="ev-btn" class="primary">Evaluate</button><span id="ev-overlap" class="hint grow"></span>` : '<span class="hint">This run has no saved model.</span>'}</div></div>
  </div>`;

  $('thr-pos').onchange = (e) => { state.posT = e.target.value; rerenderAccuracy(); };
  $('thr-orn').onchange = (e) => { state.ornT = e.target.value; rerenderAccuracy(); };
  const ik = (m.epochs || []).filter((e) => e.phase === 'ik');
  const col = (m.epochs || []).filter((e) => e.phase === 'collision');
  const epFmt = (x, long) => (long ? `epoch ${x}` : `ep ${x}`);
  new LineChart($('d-loss'), { series: [{ name: 'train', color: COLORS.s1 }, { name: 'validation', color: COLORS.s2 }], yFmt: (v) => v.toFixed(3), xLabel: epFmt })
    .set(ik.map((e) => e.epoch), [ik.map((e) => e.train_loss ?? null), ik.map((e) => e.val_mse ?? null)]);
  const es = m.params && m.params.early_stop;
  const ref = es && es.metric === 'pct_within' && state.posT === runTol(m) ? { value: +es.target_pct, label: `stop target ${+es.target_pct}%` } : null;
  new LineChart($('d-acc'), { series: [{ name: 'train', color: COLORS.s1 }, { name: 'validation', color: COLORS.s2 }], yMin: 0, yFmt: (v) => v.toFixed(1), xLabel: epFmt, ref })
    .set(ik.map((e) => e.epoch), [ik.map((e) => e.acc?.train?.acc_pos?.[state.posT] ?? null), ik.map((e) => e.acc?.val?.acc_pos?.[state.posT] ?? null)]);
  new LineChart($('d-col'), { series: [{ name: 'train', color: COLORS.s1 }, { name: 'validation', color: COLORS.s2 }], yFmt: (v) => v.toFixed(1), xLabel: epFmt })
    .set(col.map((e) => e.epoch), [col.map((e) => e.train_acc ?? null), col.map((e) => e.val_acc ?? null)]);

  if (m.has_models) {
    const upd = () => {
      const o = overlapPreview(m, $('ev-ds').value);
      $('ev-overlap').innerHTML = o ? `<span class="tag ${OVERLAP_TAG[o.level]}">${o.level}</span> ${esc(o.note)}` : '';
    };
    $('ev-ds').onchange = upd;
    const indep = [...$('ev-ds').options].find((o) => overlapPreview(m, o.value)?.level === 'independent');
    if (indep) $('ev-ds').value = indep.value;
    upd();
    $('ev-btn').onclick = async () => {
      try {
        await api('/api/evaluate', { run_id: m.id, dataset_id: $('ev-ds').value, max_samples: +$('ev-max').value });
        toast('Evaluation started – results appear here when done.');
      } catch (e) { toast(e.message, true); }
    };
  }
}

function rerenderAccuracy() {
  renderDetails();
  renderInfLog();
}

function renderInfLog() {
  const m = state.runDetail;
  const pane = $('inflog-pane');
  if (!m) { pane.innerHTML = '<div class="empty">Select a run.</div>'; return; }
  const recs = (m.inference && m.inference.recent) || [];
  const s = m.inference && m.inference.summary;
  if (!recs.length) {
    pane.innerHTML = `<div class="empty">No inferences logged for ${esc(m.id)} yet – use panel 4 (Inference).</div>`;
    return;
  }
  const rows = recs.map((r) => `<tr><td>${new Date(r.t * 1000).toLocaleTimeString()}</td>
    <td>${esc(r.source)}${r.dataset_index != null ? ` #${r.dataset_index}` : ''}</td><td>${esc(r.seed_mode || '')}</td>
    <td class="num">${fmt(r.pos_err_mm, 1)}</td><td class="num">${fmt(r.orn_err_deg, 1)}</td><td class="num">${fmt(r.joint_err_deg, 1)}</td>
    <td class="num">${fmt(r.latency_ms, 3)}</td>
    <td class="num">${r.refined ? `${fmt(r.refined.pos_err_mm, 1)} mm / ${fmt(r.refined.orn_err_deg, 2)}° (${r.refined.iters} it)` : '—'}</td>
    <td>${r.pred_self_collision ? '<span class="tag bad">yes</span>' : '<span class="tag good">no</span>'}</td>
    <td class="num">${r.collision_net_prob != null ? fmt(r.collision_net_prob, 2) : '—'}</td></tr>`).join('');
  pane.innerHTML = `<div class="pane-pad">
    <div class="hint">${s.n} inference${s.n === 1 ? '' : 's'} logged for <span class="id">${esc(m.id)}</span> · network output: ${pct(s.network.acc_pos[state.posT])} within ${state.posT} mm, mean ${fmt(s.network.pos_mean_mm, 1)} mm / ${fmt(s.network.orn_mean_deg, 1)}°${s.refined ? ` · after refinement: ${pct(s.refined.acc_pos[state.posT])} within ${state.posT} mm` : ''}</div>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Time</th><th>Target</th><th>Seed</th><th class="num">Pos err (mm)</th><th class="num">Orn err (°)</th><th class="num">Joint err (°)</th>
      <th class="num">Latency (ms)</th><th class="num">Refined</th><th>Self-collision</th><th class="num">Collision-net P</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

// ---------------------------------------------------------------- gallery
function manifest() { return state.runDetail && state.runDetail.manifest_data; }

function renderGallery() {
  const g = $('gallery');
  const man = manifest();
  if (!man || !man.cases || !man.cases.length) {
    g.innerHTML = `<div class="empty">${state.runDetail ? 'This run has no rendered training cases.' : 'Select a run.'}</div>`;
    return;
  }
  g.innerHTML = '';
  man.cases.forEach((c, idx) => {
    const imgs = [...c.images].sort((a, b) => a.epoch - b.epoch);
    const last = imgs[imgs.length - 1];
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = last
      ? `<img src="/files/${last.file}" loading="lazy" alt="case ${c.case}">
         <div class="meta"><b>Case ${c.case} · dataset sample #${c.dataset_index ?? '?'}</b>
         <span>epoch ${last.epoch}: pos ${last.pos_err_mm.toFixed(1)} mm · orn ${last.orn_err_deg.toFixed(1)}° · joints ${last.joint_err_deg.toFixed(1)}°</span>
         <span>${imgs.length} image${imgs.length === 1 ? '' : 's'} (epochs ${imgs.map((i) => i.epoch).join(', ')})</span></div>`
      : `<div class="meta"><b>Case ${c.case} · dataset sample #${c.dataset_index ?? '?'}</b><span>waiting for first render (end of epoch 1)…</span></div>`;
    card.onclick = () => last && openLightbox(idx);
    g.appendChild(card);
  });
}

function openLightbox(caseIdx, keepEpoch = false) {
  const man = manifest();
  if (!man) return;
  caseIdx = (caseIdx + man.cases.length) % man.cases.length;
  const c = man.cases[caseIdx];
  const imgs = [...c.images].sort((a, b) => a.epoch - b.epoch);
  if (!imgs.length) return;
  const same = state.lb.caseIdx === caseIdx;
  state.lb.caseIdx = caseIdx;
  if (!keepEpoch || !same) state.lb.imgIdx = imgs.length - 1;
  state.lb.imgIdx = Math.min(state.lb.imgIdx, imgs.length - 1);
  $('lb-slider').max = imgs.length - 1;
  $('lb-table').innerHTML = `<tr><th>epoch</th><th>EE position error (mm)</th><th>orientation error (°)</th><th>mean joint error (°)</th></tr>` +
    imgs.map((im, k) => `<tr data-k="${k}"><td>${im.epoch}</td><td>${im.pos_err_mm.toFixed(1)}</td><td>${im.orn_err_deg.toFixed(1)}</td><td>${im.joint_err_deg.toFixed(1)}</td></tr>`).join('');
  $('lb-table').querySelectorAll('tr[data-k]').forEach((tr) => (tr.onclick = () => showLbImage(+tr.dataset.k)));
  $('lightbox').classList.add('open');
  showLbImage(state.lb.imgIdx);
}

function lbCurrent() {
  const man = manifest();
  const c = man.cases[state.lb.caseIdx];
  const imgs = [...c.images].sort((a, b) => a.epoch - b.epoch);
  return { man, c, im: imgs[state.lb.imgIdx] };
}

function showLbImage(k) {
  state.lb.imgIdx = k;
  const { man, c, im } = lbCurrent();
  $('lb-slider').value = k;
  $('lb-img').src = `/files/${im.file}`;
  $('lb-open').href = `/files/${im.file}`;
  $('lb-epoch').textContent = `${im.epoch} / ${man.epochs ?? '?'}`;
  $('lb-title').textContent = `${man.robot} · case ${c.case} of ${man.cases.length} · dataset sample #${c.dataset_index ?? '?'} · run ${man.run_id}`;
  $('lb-table').querySelectorAll('tr[data-k]').forEach((tr) => tr.classList.toggle('sel', +tr.dataset.k === k));
}

$('lb-slider').oninput = (e) => showLbImage(+e.target.value);
$('lb-close').onclick = () => $('lightbox').classList.remove('open');
$('lightbox').onclick = (e) => { if (e.target.id === 'lightbox') $('lightbox').classList.remove('open'); };
$('lb-prev').onclick = () => openLightbox(state.lb.caseIdx - 1);
$('lb-next').onclick = () => openLightbox(state.lb.caseIdx + 1);
$('lb-3d').onclick = async () => {
  const { man, c, im } = lbCurrent();
  if (man.robot !== state.robot) await selectRobot(man.robot);
  $('follow').checked = false;
  const [t, pr] = await Promise.all([api('/api/fk', { robot: man.robot, q: c.q_true }), api('/api/fk', { robot: man.robot, q: im.q_pred })]);
  showPose(t, { ghost: pr, target: c.target_pos, targetOrn: c.target_orn });
  renderJointTable($('ds-detail'), c.q_true, im.q_pred);
  setHud(`Training case ${c.case} · dataset sample #${c.dataset_index} · network at epoch ${im.epoch}`,
    `solid = dataset ground truth, ghost = network prediction\npos error ${im.pos_err_mm.toFixed(1)} mm · orn error ${im.orn_err_deg.toFixed(1)}° · joint error ${im.joint_err_deg.toFixed(1)}°\n("Follow live job" was switched off so this view stays put)`);
  $('lightbox').classList.remove('open');
};
document.addEventListener('keydown', (e) => {
  if (!$('lightbox').classList.contains('open')) return;
  if (e.key === 'Escape') $('lightbox').classList.remove('open');
  if (e.key === 'ArrowRight') showLbImage(Math.min(+$('lb-slider').max, state.lb.imgIdx + 1));
  if (e.key === 'ArrowLeft') showLbImage(Math.max(0, state.lb.imgIdx - 1));
});

// ===================================================================
// Dataset browser
// ===================================================================

async function loadDataset(id) {
  state.dsId = id;
  const d = id && dsById(id);
  if (!d || (d.status && d.status !== 'ready')) {
    state.dsN = 0;
    $('ds-info').textContent = 'No dataset for this robot yet.';
    $('ds-slider').max = 0;
    setCloud([], []);
    return;
  }
  const cloud = await api(`/api/dataset/cloud?ds=${encodeURIComponent(id)}`);
  state.dsN = cloud.n;
  const coll = cloud.labels.reduce((a, b) => a + b, 0);
  $('ds-info').innerHTML = id.startsWith('project:')
    ? `${cloud.n.toLocaleString()} samples · project file <code>${esc(d.file)}</code> (read-only)`
    : `${d.n.toLocaleString()} samples · seed ${d.seed ?? '?'} · ${(d.n - (d.n_colliding ?? 0)).toLocaleString()} successful / ${(d.n_colliding ?? 0).toLocaleString()} collision (${pct((100 * (d.n_colliding ?? 0)) / d.n)})${d.split && d.split.mode === 'custom' ? ' · custom split' : ''}
       · used by ${d.used_by && d.used_by.length ? d.used_by.length + ' run(s)' : 'no runs'}<br><code>${esc(d.id)}</code>`;
  void coll;
  $('ds-slider').max = cloud.n - 1;
  $('ds-index').max = cloud.n - 1;
  setCloud(cloud.positions, cloud.labels);
}

async function showSample(i) {
  if (!state.dsId || !state.dsN) return;
  i = Math.max(0, Math.min(state.dsN - 1, i));
  state.dsIndex = i;
  $('ds-slider').value = i;
  $('ds-index').value = i;
  const r = await api(`/api/dataset/sample?ds=${encodeURIComponent(state.dsId)}&i=${i}`);
  showPose(r, { target: r.pos, targetOrn: r.orn });
  renderJointTable($('ds-detail'), r.q, null);
  const mem = (r.membership || []).map((m) => `<dt>Run <span class="id">${esc(m.run_id)}</span></dt><dd>${esc(m.split)}</dd>`).join('');
  $('ds-detail').insertAdjacentHTML('beforeend', `<dl class="kv" style="margin-top:8px">
    <dt>Label</dt><dd>${r.label ? '<span class="badge bad">self-colliding</span>' : '<span class="badge good">collision-free</span>'}</dd>
    <dt>EE position (m)</dt><dd>${fmtVec(r.pos)}</dd>
    <dt>EE quaternion (xyzw)</dt><dd>${fmtVec(r.orn)}</dd>
    <dt>FK re-check</dt><dd>${r.colliding === !!r.label ? 'matches label' : 'differs from label'}</dd>${mem}</dl>`);
  setHud(`Dataset sample #${i.toLocaleString()} / ${(state.dsN - 1).toLocaleString()}`,
    `q (deg)   ${degs(r.q)}\nEE pos    ${fmtVec(r.pos)}   ← model input\nEE quat   ${fmtVec(r.orn)}   ← model input\nlabel     ${r.label ? 'SELF-COLLIDING (excluded from IK training)' : 'collision-free (used for IK training)'}${(r.membership || []).map((m) => `\n${m.run_id}: ${m.split}`).join('')}`);
}

function renderJointTable(el, q, qPred) {
  const rows = state.model.joints.map((j, i) => {
    const pos = (v) => Math.max(0, Math.min(100, (100 * (v - j.lo)) / (j.hi - j.lo))).toFixed(1);
    return `<div class="joint"><span class="name">${shortJoint(j.name)}</span>
      <div class="limitbar" title="limits ${(j.lo * DEG).toFixed(0)}° … ${(j.hi * DEG).toFixed(0)}°">
        <div class="mark" style="left:${pos(q[i])}%"></div>
        ${qPred ? `<div class="mark pred" style="left:${pos(qPred[i])}%"></div>` : ''}
      </div><span class="val">${(q[i] * DEG).toFixed(1)}°</span></div>`;
  });
  el.innerHTML = `<div class="joints">${rows.join('')}</div>`;
}

$('ds-src').onchange = (e) => loadDataset(e.target.value);
$('ds-slider').oninput = (e) => showSample(+e.target.value);
$('ds-index').onchange = (e) => showSample(+e.target.value);
$('ds-prev').onclick = () => showSample(state.dsIndex - 1);
$('ds-next').onclick = () => showSample(state.dsIndex + 1);
$('ds-play').onclick = async () => {
  state.playing = !state.playing;
  $('ds-play').textContent = state.playing ? '❚❚ Pause' : '▶ Play';
  while (state.playing) {
    const t0 = performance.now();
    if (state.dsIndex >= state.dsN - 1) state.dsIndex = -1;
    await showSample(state.dsIndex + 1);
    const wait = 1000 / +$('ds-fps').value - (performance.now() - t0);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
};

// ===================================================================
// Jobs: generate / train / stop
// ===================================================================

// ---- movement split (successful vs collision %) ----
function naturalRate() {
  // share of colliding samples in this robot's most recent natural-split dataset
  const d = state.lib.datasets
    .filter((x) => x.robot === state.robot && x.status === 'ready' && x.n && (!x.split || x.split.mode === 'natural'))
    .sort((a, b) => (b.created || 0) - (a.created || 0))[0];
  return d ? (100 * d.n_colliding) / d.n : null;
}

function updateSplitUI(src) {
  const clamp = (v) => Math.max(0, Math.min(100, Math.round(Number.isFinite(v) ? v : 0)));
  const natural = $('gen-natural').checked;
  let free = clamp(+$('gen-free-pct').value), col = clamp(+$('gen-col-pct').value);
  if (src === 'free') col = 100 - free; else col = 100 - (free = 100 - col);
  $('gen-free-pct').value = free; $('gen-col-pct').value = col;
  $('gen-free-pct').disabled = $('gen-col-pct').disabled = natural;
  $('gen-free-pct').closest('.split-box').classList.toggle('natural', natural);
  const n = Math.max(0, +$('gen-n').value || 0);
  const nat = naturalRate();
  const showCol = natural ? (nat ?? 15.5) : col;
  $('gen-split-free').style.width = `${100 - showCol}%`;
  $('gen-split-col').style.width = `${showCol}%`;
  const info = $('gen-split-info');
  info.classList.remove('warn');
  if (natural) {
    info.textContent = `Keeps whatever generate() produces – ${nat != null ? `your last natural ${state.robot} dataset had ${nat.toFixed(1)}% collision movements` : 'typically ≈15–16% collision movements for xArm6 / KUKA'}. The file is exactly generate()'s output.`;
    return;
  }
  const nCol = Math.round((n * col) / 100), nFree = n - nCol;
  const rate = (nat ?? 15.5) / 100;
  const pool = Math.ceil(Math.max(n, Math.max(nCol / rate, nFree / (1 - rate)) * 1.15));
  const warns = [];
  if (nFree < 200) warns.push('too few successful movements to train the IK network (it trains only on collision-free samples)');
  else if (free < 30) warns.push(`the IK network trains only on the ${nFree.toLocaleString()} successful movements`);
  if (col === 0) warns.push('the collision classifier will see only one class');
  if (pool > 5 * n) warns.push(`needs a pool of ≈${pool.toLocaleString()} samples – takes longer`);
  info.innerHTML = `<b>${nFree.toLocaleString()}</b> successful + <b>${nCol.toLocaleString()}</b> collision movements, drawn from a pool of ≈${pool.toLocaleString()} generated by generate().`
    + (warns.length ? `<br>⚠ ${warns.join('; ')}.` : '');
  if (warns.length) info.classList.add('warn');
}
$('gen-natural').onchange = () => updateSplitUI();
$('gen-free-pct').oninput = () => updateSplitUI('free');
$('gen-col-pct').oninput = () => updateSplitUI('col');
$('gen-n').oninput = () => updateSplitUI();

$('gen-btn').onclick = async () => {
  try {
    const r = await api('/api/generate', {
      robot: state.robot, n: +$('gen-n').value, seed: +$('gen-seed').value, delay_ms: +$('gen-delay').value,
      collision_pct: $('gen-natural').checked ? null : +$('gen-col-pct').value,
    });
    state.pendingDs = r.ds_id;
  } catch (e) { toast(e.message, true); }
};

// ---- accuracy tolerance + automatic stop ----
function esSettings() {
  if (!$('es-on').checked) return null;
  return { metric: $('es-metric').value, target_pct: +$('es-pct').value, split: $('es-split').value, patience: Math.max(1, +$('es-patience').value || 1) };
}

function updateEsUI() {
  const on = $('es-on').checked;
  $('es-fields').querySelectorAll('input, select').forEach((el) => (el.disabled = !on));
  $('es-pct-wrap').hidden = $('es-metric').value === 'mean_error';
  const tol = tolKey(+$('tr-tol').value || 0), max = +$('tr-epochs').value || 0, es = esSettings();
  $('es-info').innerHTML = es
    ? `Training stops as soon as ${esRuleText(es, tol)} – otherwise it runs the full ${max} epochs. Checked after every epoch on 2,000 fixed rows of that split.${es.split === 'train' ? ' The validation split is the stricter check (data the model does not learn from).' : ''}`
    : `Runs all ${max} epochs. Accuracy is still reported at ${tol} mm everywhere.`;
}
['es-on', 'es-metric', 'es-split'].forEach((id) => ($(id).onchange = updateEsUI));
['es-pct', 'es-patience', 'tr-tol', 'tr-epochs'].forEach((id) => ($(id).oninput = updateEsUI));

$('tr-btn').onclick = async () => {
  if (!$('tr-ds').value) { toast('Generate a dataset first (panel 1).', true); return; }
  if (!(+$('tr-tol').value > 0)) { toast('Set a position tolerance in mm (> 0).', true); return; }
  try {
    const r = await api('/api/train', {
      robot: state.robot, dataset_id: $('tr-ds').value, test_frac: +$('tr-test').value / 100,
      epochs: +$('tr-epochs').value, collision_epochs: +$('tr-cepochs').value,
      cases: +$('tr-cases').value, render_every: +$('tr-every').value, sample_hz: +$('tr-hz').value,
      pos_tol_mm: +$('tr-tol').value, early_stop: esSettings(), model_id: $('tr-model').value || 'base',
    });
    state.liveRun = r.run_id;
  } catch (e) { toast(e.message, true); }
};

$('stop-btn').onclick = () => api('/api/stop', {}).catch((e) => toast(e.message, true));

function applyJob(job) {
  state.job = job;
  const running = job && job.status === 'running';
  $('job-dot').className = 'dot ' + (running ? 'run' : job && job.status === 'failed' ? 'bad' : job ? 'on' : '');
  let txt = 'idle';
  if (job) {
    const pr = job.progress || {};
    if (running && job.kind === 'generate') txt = `generating ${job.params.robot} · ${pr.i ?? 0}/${pr.n ?? job.params.n}`;
    else if (running && job.kind === 'train') txt = `training ${job.params.robot} · ${pr.phase ?? ''} ${pr.epoch ?? ''}/${pr.epochs ?? ''}`;
    else if (running) txt = `evaluating ${job.params.run_id} · ${pr.i ?? 0}/${pr.n ?? '?'}`;
    else txt = `${job.kind} ${job.status}`;
  }
  $('job-text').textContent = txt;
  for (const id of ['gen-btn', 'tr-btn']) $(id).disabled = running;
  const ev = $('ev-btn'); if (ev) ev.disabled = running;
  $('stop-btn').disabled = !running;
  $('robot-select').disabled = running && job.kind !== 'evaluate';
}

// ===================================================================
// Inference (panel 4)
// ===================================================================

function syncInferenceMode() {
  const mode = $('inf-target').value;
  $('inf-t-test').hidden = mode !== 'test';
  $('inf-t-dataset').hidden = mode !== 'dataset';
  $('inf-t-custom').hidden = mode !== 'custom';
  const known = mode !== 'custom';
  for (const o of $('inf-seed').options) if (o.value === 'noisy' || o.value === 'truth') o.disabled = !known;
  if (!known && ['noisy', 'truth'].includes($('inf-seed').value)) $('inf-seed').value = 'jog';
  if (known && $('inf-seed').value === 'jog' && mode === 'test') $('inf-seed').value = 'noisy';
}
$('inf-target').onchange = syncInferenceMode;
$('inf-run').onchange = () => { state.inf.test = null; $('inf-test-info').textContent = ''; if ($('inf-run').value) selectRun($('inf-run').value); };

async function pickTestSample() {
  const run = $('inf-run').value;
  const t = await api(`/api/test_sample?run=${encodeURIComponent(run)}`);
  state.inf.test = t;
  $('inf-test-info').textContent = `dataset sample #${t.index} (held out)`;
  return t;
}
$('inf-random-test').onclick = () => pickTestSample().catch((e) => toast(e.message, true));

$('inf-from-ee').onclick = async () => {
  const r = await api('/api/fk', { robot: state.robot, q: jogQ() });
  const e = new THREE.Euler().setFromQuaternion(new THREE.Quaternion(...r.ee_orn), 'ZYX');
  // PyBullet's getQuaternionFromEuler uses roll-pitch-yaw about fixed X, Y, Z == intrinsic ZYX
  $('inf-x').value = r.ee_pos[0].toFixed(3); $('inf-y').value = r.ee_pos[1].toFixed(3); $('inf-z').value = r.ee_pos[2].toFixed(3);
  $('inf-r').value = (e.x * DEG).toFixed(1); $('inf-p').value = (e.y * DEG).toFixed(1); $('inf-yaw').value = (e.z * DEG).toFixed(1);
};

async function runInference({ quiet = false } = {}) {
  const runId = $('inf-run').value;
  const run = runById(runId);
  if (!run) throw new Error('No trained model selected.');
  if (run.robot !== state.robot) await selectRobot(run.robot);
  const mode = $('inf-target').value;
  const req = { run_id: runId, source: mode, refine_iters: +$('inf-refine').value, seed_mode: $('inf-seed').value };
  let qTrue = null;
  if (mode === 'test' || mode === 'dataset') {
    let ds, i;
    if (mode === 'test') { const t = state.inf.test || await pickTestSample(); ds = t.dataset_id; i = t.index; req.source = 'held-out test'; }
    else { ds = $('inf-ds').value; i = +$('inf-ds-i').value; req.source = 'dataset'; }
    const s = await api(`/api/dataset/sample?ds=${encodeURIComponent(ds)}&i=${i}`);
    Object.assign(req, { target_pos: s.pos, target_orn: s.orn, q_true: s.q, dataset_id: ds, dataset_index: s.index });
    qTrue = s.q;
  } else if (mode === 'jog') {
    const q = jogQ();
    const r = await api('/api/fk', { robot: state.robot, q });
    Object.assign(req, { target_pos: r.ee_pos, target_orn: r.ee_orn, q_true: q, source: 'joint sliders' });
    qTrue = q;
  } else {
    req.target_pos = [+$('inf-x').value, +$('inf-y').value, +$('inf-z').value];
    req.target_rpy_deg = [+$('inf-r').value, +$('inf-p').value, +$('inf-yaw').value];
    req.source = 'custom pose';
  }
  const sm = $('inf-seed').value;
  if ((sm === 'noisy' || sm === 'truth') && !qTrue) throw new Error('That seed needs a target with known true joints.');
  req.seed_q = sm === 'noisy' ? qTrue.map((v) => v + 0.3 * gauss()) : sm === 'truth' ? qTrue.slice() : sm === 'jog' ? jogQ() : qTrue ? qTrue.map(() => 0) : state.model.joints.map(() => 0);
  req.seed_mode = { noisy: 'truth + N(0,0.3)', truth: 'truth', jog: 'joint sliders', zero: 'zero pose' }[sm];

  const res = await api('/api/infer', req);
  const rec = res.record;
  $('follow').checked = false;
  const base = await api('/api/fk', { robot: state.robot, q: qTrue || req.seed_q });
  showPose(base, { ghost: res.fk_pred, refined: res.fk_refined || null, target: rec.target_pos, targetOrn: rec.target_orn });
  renderJointTable($('ds-detail'), qTrue || req.seed_q, rec.q_pred);
  $('inf-kv').innerHTML = `<dt>Network latency</dt><dd>${fmt(rec.latency_ms, 3)} ms (CPU)</dd>
    <dt>EE position error</dt><dd>${fmt(rec.pos_err_mm, 1)} mm ${rec.pos_err_mm <= +runTol(run) ? `<span class="tag good">within ${runTol(run)} mm</span>` : `<span class="tag bad">outside ${runTol(run)} mm</span>`}</dd>
    <dt>EE orientation error</dt><dd>${fmt(rec.orn_err_deg, 1)}°</dd>
    <dt>Mean joint error vs truth</dt><dd>${rec.joint_err_deg != null ? fmt(rec.joint_err_deg, 1) + '°' : '— (no truth)'}</dd>
    <dt>Prediction self-collides</dt><dd>${rec.pred_self_collision ? '<span class="badge bad">yes</span>' : '<span class="badge good">no</span>'} · collision-net P ${fmt(rec.collision_net_prob, 2)}</dd>
    <dt>Joint-limit violation</dt><dd>${rec.limit_violation ? '<span class="badge bad">yes</span>' : '<span class="badge good">no</span>'}</dd>
    ${rec.refined ? `<dt>After ${rec.refined.iters} DLS iterations</dt><dd>${fmt(rec.refined.pos_err_mm, 1)} mm · ${fmt(rec.refined.orn_err_deg, 2)}° · ${fmt(rec.refined.ms, 2)} ms</dd>` : ''}`;
  setHud(`Inference · ${runId} · target: ${req.source}${req.dataset_index != null ? ' #' + req.dataset_index : ''}`,
    `target pos   ${fmtVec(rec.target_pos)}\ntarget quat  ${fmtVec(rec.target_orn)}\nseed q (deg) ${degs(rec.seed_q)}   (${req.seed_mode})\n${qTrue ? 'truth q (deg) ' + degs(qTrue) + '\n' : ''}model q (deg) ${degs(rec.q_pred)}\n→ ${fmt(rec.pos_err_mm, 1)} mm · ${fmt(rec.orn_err_deg, 1)}° · ${fmt(rec.latency_ms, 3)} ms${rec.refined ? `\nrefined (${rec.refined.iters} DLS it, aqua ghost) → ${fmt(rec.refined.pos_err_mm, 1)} mm · ${fmt(rec.refined.orn_err_deg, 2)}°` : ''}\n${qTrue ? 'solid = true solution' : 'solid = seed pose'} · orange ghost = network`);
  if (!quiet && state.selRun === runId) refreshDetailSoon();
  return rec;
}

$('inf-btn').onclick = () => runInference().catch((e) => toast(e.message, true));
$('inf-batch').onclick = async () => {
  const b = $('inf-batch');
  b.disabled = true;
  const prevMode = $('inf-target').value;
  const tol = runTol(runById($('inf-run').value));
  try {
    $('inf-target').value = 'test'; syncInferenceMode();
    let within = 0, sum = 0;
    for (let k = 0; k < 20; k++) {
      await pickTestSample();
      const rec = await runInference({ quiet: true });
      sum += rec.pos_err_mm; within += rec.pos_err_mm <= +tol ? 1 : 0;
      b.textContent = `${k + 1}/20…`;
    }
    toast(`20 held-out test inferences: ${within}/20 within ${tol} mm, mean ${(sum / 20).toFixed(1)} mm (logged)`);
    refreshDetailSoon();
  } catch (e) { toast(e.message, true); }
  finally { b.disabled = false; b.textContent = '×20 test samples'; $('inf-target').value = prevMode; syncInferenceMode(); }
};

// ===================================================================
// Live events
// ===================================================================

const follow = () => $('follow').checked;
let genFlushT = 0;

async function onEvent(ev) {
  switch (ev.type) {
    case 'hello': return onHello(ev);
    case 'stats': return onStats(ev);
    case 'log':
      if (!/finished with exit code/.test(ev.line)) state.lastLog = ev.line;
      return appendLog(ev.line);
    case 'job': {
      const was = state.job && state.job.status;
      applyJob(ev.job);
      if (was === 'running' && ev.job.status === 'failed') toast(`${ev.job.kind} failed: ${state.lastLog || 'see the Log tab'}`, true);
      return;
    }
    case 'library_changed': {
      await refreshLibrary();
      if (state.pendingDs && dsById(state.pendingDs)?.status === 'ready') {
        const id = state.pendingDs; state.pendingDs = null;
        $('ds-src').value = id; $('tr-ds').value = id;
        await loadDataset(id);
      }
      if (state.selRun) refreshDetailSoon();
      return;
    }
    case 'gen_start':
      if (ev.robot !== state.robot) await selectRobot(ev.robot);
      state.pendingDs = ev.ds_id;
      cloudData.pos = []; cloudData.col = []; flushCloud();
      $('gen-bar').style.width = '0%';
      if (ev.attempt > 0) toast(`Pool was too small for the requested split – regenerating with ${ev.pool.toLocaleString()} samples.`);
      return;
    case 'gen_sample': return onGenSample(ev);
    case 'gen_rebalance':
      $('gen-label').textContent = `drawing ${ev.n_free.toLocaleString()} successful + ${ev.n_colliding.toLocaleString()} collision movements from the pool…`;
      return;
    case 'gen_done': {
      $('gen-bar').style.width = '100%';
      const cp = ((100 * ev.n_colliding) / ev.n).toFixed(1);
      $('gen-label').textContent = `saved: ${ev.n.toLocaleString()} samples (${(100 - cp).toFixed(1)}% successful / ${cp}% collision) in ${ev.secs.toFixed(1)} s`;
      $('gen-pct').textContent = '100%';
      $('gen-kv').innerHTML = `<dt>Successful (collision-free)</dt><dd>${(ev.n - ev.n_colliding).toLocaleString()}</dd>
        <dt>Collision movements</dt><dd>${ev.n_colliding.toLocaleString()}</dd>
        ${ev.split && ev.split.mode === 'custom' ? `<dt>Pool generated</dt><dd>${ev.split.pool_n.toLocaleString()} (${ev.split.pool_colliding.toLocaleString()} colliding)</dd>` : '<dt>Split</dt><dd>natural (generate() output as-is)</dd>'}
        <dt>Library entry</dt><dd>${esc(ev.ds_id)}</dd>`;
      return;
    }
    case 'train_start':
      if (ev.robot !== state.robot) await selectRobot(ev.robot);
      state.liveIk = []; state.liveCol = []; state.liveErr = []; state.liveRun = ev.run_id;
      state.liveTol = tolKey(ev.pos_tol_mm || 50); state.liveEs = ev.early_stop || null;
      drawTrainCharts();
      $('tr-device').textContent = ev.device === 'cuda' ? `on ${ev.gpu}` : 'on CPU';
      $('tr-col-bar').style.width = '0%'; $('tr-ik-bar').style.width = '0%';
      $('tr-kv').innerHTML = `<dt>Run</dt><dd>${esc(ev.run_id)}</dd><dt>Dataset</dt><dd>${ev.n_samples.toLocaleString()} samples</dd>
        <dt>Held-out test</dt><dd>${ev.n_test.toLocaleString()}</dd><dt>Given to train_ik</dt><dd>${ev.n_given.toLocaleString()} (${ev.n_free.toLocaleString()} collision-free for IK)</dd>
        <dt>Model</dt><dd>${esc(modelLabel(ev.model))} · ${fmtParams(ev.model && ev.model.params)}</dd>
        <dt>Tolerance</dt><dd>${state.liveTol} mm</dd><dt>Automatic stop</dt><dd>${ev.early_stop ? esc(esRuleText(ev.early_stop, state.liveTol)) : 'off'}</dd>`;
      await refreshLibrary();
      await selectRun(ev.run_id);
      return;
    case 'phase':
      if (ev.phase === 'evaluating') $('tr-ik-label').textContent = 'final evaluation on train / validation / test…';
      return;
    case 'train_sample': return onTrainSample(ev);
    case 'epoch_end': return onEpochEnd(ev);
    case 'early_stop':
      $('tr-ik-bar').style.width = '100%';
      $('tr-ik-label').textContent = `stopped automatically after epoch ${ev.stopped_at_epoch}/${ev.max_epochs} – target met`;
      toast(`Target met: ${esRuleText(ev.rule, tolKey(ev.pos_tol_mm))}. Training stopped at epoch ${ev.stopped_at_epoch} of ${ev.max_epochs} – saving model and evaluating train / validation / test…`);
      return;
    case 'cases':
      if (state.selRun === ev.run_id && state.runDetail) {
        state.runDetail.manifest_data = { run_id: ev.run_id, robot: ev.robot, epochs: state.runDetail.params?.epochs, cases: ev.cases.map((c) => ({ ...c, images: [] })) };
        renderGallery();
      }
      return;
    case 'render': {
      const man = state.selRun === ev.run_id && manifest();
      const c = man && man.cases.find((x) => x.case === ev.case);
      if (c && !c.images.some((im) => im.epoch === ev.epoch)) { c.images.push(ev); renderGallery(); }
      return;
    }
    case 'final_metrics':
      if (state.selRun === ev.run_id) refreshDetailSoon();
      return;
    case 'train_done':
      toast(`Training finished in ${ev.secs}s – saved as run ${ev.run_id}. See Run details for train / validation / test accuracy.`);
      await refreshLibrary();
      if (state.selRun === ev.run_id) { await selectRun(ev.run_id); switchTab('details-pane'); }
      return;
    case 'eval_done':
      {
        const k = runTol(runById(ev.run_id));
        toast(`Evaluation done: ${pct(ev.summary.acc_pos[k])} within ${k} mm on ${ev.summary.n.toLocaleString()} samples (${ev.overlap?.level}).`);
      }
      await refreshLibrary();
      if (state.selRun === ev.run_id) await selectRun(ev.run_id);
      return;
    default:
  }
}

function onHello(ev) {
  applyJob(ev.job);
  state.stats = ev.stats || [];
  if (state.stats.length) onStats(state.stats[state.stats.length - 1], true);
  drawStatCharts();
  $('log').textContent = '';
  (ev.log || []).forEach(appendLog);
  state.liveIk = (ev.epochs || []).filter((e) => e.phase === 'ik');
  state.liveCol = (ev.epochs || []).filter((e) => e.phase === 'collision');
  const last = state.liveIk[state.liveIk.length - 1];
  state.liveTol = tolKey((last && last.pos_tol_mm) || (ev.job && ev.job.params && ev.job.params.pos_tol_mm) || 50);
  state.liveEs = (ev.job && ev.job.kind === 'train' && ev.job.params && ev.job.params.early_stop) || null;
  drawTrainCharts();
}

function onGenSample(ev) {
  if (ev.robot !== state.robot) return;
  const p = (100 * ev.i) / ev.n;
  $('gen-bar').style.width = p + '%';
  $('gen-label').textContent = ev.target_n && ev.target_n !== ev.n
    ? `pool sample ${ev.i.toLocaleString()} / ${ev.n.toLocaleString()} · ${ev.phase}`
    : `sample ${ev.i.toLocaleString()} / ${ev.n.toLocaleString()} · ${ev.phase}`;
  $('gen-pct').textContent = p.toFixed(0) + '%';
  $('gen-kv').innerHTML = `<dt>Colliding so far</dt><dd>${ev.n_colliding.toLocaleString()} (${((100 * ev.n_colliding) / ev.i).toFixed(1)}%)</dd>
    <dt>Sampling phase</dt><dd>${ev.phase === 'uniform' ? 'uniform in joint limits' : 'near joint limits (15% edges)'}</dd>`;
  appendCloud(ev.pos, ev.collided);
  const now = performance.now();
  if (now - genFlushT > 150 || ev.i === ev.n) { flushCloud(); genFlushT = now; }
  if (!follow()) return;
  showPose(ev.fk, { target: ev.pos, targetOrn: ev.orn });
  setHud(`Generating dataset · sample ${ev.i.toLocaleString()} / ${ev.n.toLocaleString()}`,
    `1. random q (deg)  ${degs(ev.q)}\n2. FK → EE pos     ${fmtVec(ev.pos)}\n        EE quat    ${fmtVec(ev.orn)}\n3. collision check → ${ev.collided ? 'SELF-COLLIDING (label 1)' : 'free (label 0)'}`);
}

function onTrainSample(ev) {
  if (ev.robot !== state.robot) return;
  const bar = ev.phase === 'ik' ? 'tr-ik' : 'tr-col';
  const frac = ((ev.epoch - 1) + ev.batch / ev.nb) / ev.epochs;
  $(`${bar}-bar`).style.width = (100 * Math.min(1, frac)) + '%';
  $(`${bar}-label`).textContent = `epoch ${ev.epoch}/${ev.epochs} · batch ${ev.batch}/${ev.nb}`;
  if (ev.phase === 'ik' && ev.pos_err_mm !== undefined) {
    state.liveErr.push({ y: ev.pos_err_mm, epoch: ev.epoch });
    if (state.liveErr.length > 600) state.liveErr.shift();
    charts.err.set(state.liveErr.map((p) => p.epoch), [state.liveErr.map((p) => p.y)]);
  }
  if (!follow()) return;
  if (ev.phase === 'collision') {
    showPose(ev.fk_true, {});
    setMarker(ev.fk_true.ee_pos, ev.fk_true.ee_orn);
    setHud(`Training collision classifier · epoch ${ev.epoch}/${ev.epochs} · batch ${ev.batch}/${ev.nb}`,
      `input q (deg)       ${degs(ev.q_true)}\ntrue label          ${ev.label ? 'SELF-COLLIDING' : 'free'}\npredicted P(coll.)  ${fmt(ev.prob, 3)}  → ${ev.prob > 0.5 ? 'colliding' : 'free'}  ${(ev.prob > 0.5) === !!ev.label ? '✓' : '✗'}`);
    renderJointTable($('ds-detail'), ev.q_true, null);
    return;
  }
  showPose(ev.fk_true, { ghost: ev.fk_pred, target: ev.target_pos, targetOrn: ev.target_orn });
  renderJointTable($('ds-detail'), ev.q_true, ev.q_pred);
  setHud(`Training IK network · epoch ${ev.epoch}/${ev.epochs} · batch ${ev.batch}/${ev.nb} · dataset sample #${ev.dataset_index ?? '?'} (train split)`,
    `input  target pos   ${fmtVec(ev.target_pos)}\n       target quat  ${fmtVec(ev.target_orn)}\n       seed q (deg) ${degs(ev.seed)}\ntruth  q (deg)      ${degs(ev.q_true)}\nmodel  q (deg)      ${degs(ev.q_pred)}\nFK of prediction → pos error ${fmt(ev.pos_err_mm, 1)} mm · orn error ${fmt(ev.orn_err_deg, 1)}° · sample MSE ${fmt(ev.sample_mse, 4)}`);
  $('tr-kv').innerHTML = `<dt>Run</dt><dd>${esc(ev.run_id)}</dd><dt>Current batch loss</dt><dd>${fmt(ev.batch_loss, 4)}</dd>
    <dt>Live sample pos / orn error</dt><dd>${fmt(ev.pos_err_mm, 1)} mm / ${fmt(ev.orn_err_deg, 1)}°</dd>
    <dt>Prediction self-collides</dt><dd>${ev.fk_pred.colliding ? '<span class="badge bad">yes</span>' : '<span class="badge good">no</span>'}</dd>`;
}

function onEpochEnd(ev) {
  if (ev.phase === 'ik') {
    state.liveIk.push(ev);
    $('tr-ik-bar').style.width = (100 * ev.epoch / ev.epochs) + '%';
    const a = ev.acc, T = tolKey(ev.pos_tol_mm || state.liveTol), s = ev.early_stop;
    $('tr-ik-label').textContent = `epoch ${ev.epoch}/${ev.epochs}`
      + (a ? ` · ≤${T} mm: train ${pct(a.train.acc_pos[T])}, val ${pct(a.val.acc_pos[T])}` : '')
      + (s ? ` · target met ${s.streak}/${s.patience}` : '');
  } else {
    state.liveCol.push(ev);
    $('tr-col-bar').style.width = (100 * ev.epoch / ev.epochs) + '%';
    $('tr-col-label').textContent = `epoch ${ev.epoch}/${ev.epochs} · val acc ${pct(ev.val_acc)}`;
  }
  drawTrainCharts();
}

function appendLog(line) {
  const el = $('log');
  el.textContent += line + '\n';
  if (el.textContent.length > 120000) el.textContent = el.textContent.slice(-100000);
  const pane = $('log-pane');
  pane.scrollTop = pane.scrollHeight;
}

// ===================================================================
// Charts (single y-axis each; hover crosshair + tooltip)
// ===================================================================

class LineChart {
  // ref: optional dashed reference line {value, label} (e.g. the early-stop target)
  constructor(canvas, { series, yMin, yMax, yFmt = (v) => v.toFixed(1), xLabel = (x) => x, ref = null }) {
    Object.assign(this, { canvas, series, yMin, yMax, yFmt, xLabel, ref });
    this.xs = []; this.ys = [];
    this.hover = null;
    canvas.addEventListener('mousemove', (e) => this.onMove(e));
    canvas.addEventListener('mouseleave', () => { this.hover = null; $('tooltip').style.display = 'none'; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  set(xs, ys) { this.xs = xs; this.ys = ys; this.draw(); return this; }

  geom() { return { w: this.canvas.clientWidth, h: this.canvas.clientHeight, l: 38, r: 8, t: 8, b: 16 }; }

  range() {
    let lo = this.yMin, hi = this.yMax;
    const all = this.ys.flat().filter((v) => v !== null && v !== undefined && Number.isFinite(v));
    if (this.ref && Number.isFinite(this.ref.value) && all.length) all.push(this.ref.value);
    if (lo === undefined) lo = all.length ? Math.min(...all) : 0;
    if (hi === undefined) hi = all.length ? Math.max(...all) : 1;
    if (hi - lo < 1e-9) hi = lo + 1;
    if (this.yMin === undefined) lo = Math.max(0, lo - (hi - lo) * 0.08);
    if (this.yMax === undefined) hi += (hi - lo) * 0.08;
    return [lo, hi];
  }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const { w, h, l, r, t, b } = this.geom();
    if (!w) return;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext('2d');
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const [lo, hi] = this.range();
    const n = this.xs.length;
    const X = (i) => l + (n <= 1 ? 0 : (i / (n - 1)) * (w - l - r));
    const Y = (v) => t + (1 - (v - lo) / (hi - lo)) * (h - t - b);

    g.font = '10px Segoe UI, sans-serif';
    g.fillStyle = COLORS.axis; g.strokeStyle = COLORS.grid; g.lineWidth = 1;
    for (let k = 0; k <= 3; k++) {
      const v = lo + (k / 3) * (hi - lo), y = Math.round(Y(v)) + 0.5;
      g.beginPath(); g.moveTo(l, y); g.lineTo(w - r, y); g.stroke();
      g.textAlign = 'right'; g.textBaseline = 'middle';
      g.fillText(this.yFmt(v), l - 5, y);
    }
    const hasData = this.ys.some((arr) => arr.some((v) => Number.isFinite(v)));
    if (hasData && this.ref && Number.isFinite(this.ref.value) && this.ref.value >= lo && this.ref.value <= hi) {
      const y = Math.round(Y(this.ref.value)) + 0.5;
      g.save();
      g.strokeStyle = css('--warning'); g.setLineDash([5, 4]); g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(l, y); g.lineTo(w - r, y); g.stroke();
      g.restore();
      g.fillStyle = COLORS.text; g.textAlign = 'left'; g.textBaseline = 'bottom';
      g.fillText(this.ref.label, l + 4, y - 2);
      g.fillStyle = COLORS.axis;
    }
    if (n && hasData) {
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      g.fillText(String(this.xLabel(this.xs[0])), l, h - 3);
      g.textAlign = 'right';
      g.fillText(String(this.xLabel(this.xs[n - 1])), w - r, h - 3);
    } else {
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText('no data', (l + w - r) / 2, (t + h - b) / 2);
    }
    this.ys.forEach((arr, si) => {
      g.strokeStyle = this.series[si].color; g.lineWidth = 2; g.lineJoin = 'round';
      g.beginPath();
      let started = false;
      arr.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        if (!started) { g.moveTo(X(i), Y(v)); started = true; } else g.lineTo(X(i), Y(v));
      });
      g.stroke();
      const last = arr.length - 1;
      if (last >= 0 && Number.isFinite(arr[last])) {
        g.fillStyle = this.series[si].color;
        g.beginPath(); g.arc(X(last), Y(arr[last]), 3, 0, 7); g.fill();
      }
    });
    if (this.hover !== null && n) {
      const i = this.hover, x = Math.round(X(i)) + 0.5;
      g.strokeStyle = COLORS.axis; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, t); g.lineTo(x, h - b); g.stroke();
      this.ys.forEach((arr, si) => {
        if (!Number.isFinite(arr[i])) return;
        g.fillStyle = this.series[si].color; g.strokeStyle = css('--surface-1'); g.lineWidth = 2;
        g.beginPath(); g.arc(X(i), Y(arr[i]), 4, 0, 7); g.fill(); g.stroke();
      });
    }
  }

  onMove(e) {
    const n = this.xs.length;
    if (!n) return;
    const rect = this.canvas.getBoundingClientRect();
    const { w, l, r } = this.geom();
    this.hover = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - rect.left - l) / (w - l - r)) * (n - 1))));
    this.draw();
    const tt = $('tooltip');
    const i = this.hover;
    tt.innerHTML = `<div class="muted">${this.xLabel(this.xs[i], true)}</div>` + this.series.map((s, si) =>
      `<div class="tt-row"><i style="background:${s.color}"></i>${s.name}: <b>${Number.isFinite(this.ys[si][i]) ? this.yFmt(this.ys[si][i], true) : '—'}</b></div>`).join('');
    tt.style.display = 'block';
    tt.style.left = Math.min(window.innerWidth - tt.offsetWidth - 8, e.clientX + 12) + 'px';
    tt.style.top = (e.clientY + 12) + 'px';
  }
}

const ago = (t, long) => {
  const s = Math.round(Date.now() / 1000 - t);
  return long ? `${s}s ago` : (s <= 1 ? 'now' : `-${s}s`);
};
const charts = {};

function initCharts() {
  const epFmt = (x, long) => (long ? `epoch ${x}` : `ep ${x}`);
  const two = [{ name: 'train', color: COLORS.s1 }, { name: 'validation', color: COLORS.s2 }];
  charts.cpu = new LineChart($('c-cpu'), { series: [{ name: 'CPU', color: COLORS.s1 }], yMin: 0, yMax: 100, yFmt: (v) => v.toFixed(0), xLabel: ago });
  charts.gpu = new LineChart($('c-gpu'), { series: [{ name: 'GPU', color: COLORS.s3 }], yMin: 0, yMax: 100, yFmt: (v) => v.toFixed(0), xLabel: ago });
  charts.vram = new LineChart($('c-vram'), { series: [{ name: 'VRAM', color: COLORS.s1 }], yMin: 0, yFmt: (v, long) => v.toFixed(long ? 2 : 1), xLabel: ago });
  charts.loss = new LineChart($('c-loss'), { series: two, yFmt: (v) => v.toFixed(3), xLabel: epFmt });
  charts.ikacc = new LineChart($('c-ikacc'), { series: two, yMin: 0, yFmt: (v) => v.toFixed(1), xLabel: epFmt });
  charts.acc = new LineChart($('c-acc'), { series: two, yFmt: (v) => v.toFixed(1), xLabel: epFmt });
  charts.err = new LineChart($('c-err'), { series: [{ name: 'pos error', color: COLORS.s2 }], yFmt: (v) => v.toFixed(0), xLabel: (x, long) => (long ? `sample during epoch ${x}` : `ep ${x}`) });
}

function drawStatCharts() {
  const s = state.stats;
  const ts = s.map((x) => x.t);
  charts.cpu.set(ts, [s.map((x) => x.cpu)]);
  charts.gpu.set(ts, [s.map((x) => (x.gpus && x.gpus[0] ? x.gpus[0].util : null))]);
  const g0 = s.length && s[s.length - 1].gpus && s[s.length - 1].gpus[0];
  charts.vram.yMax = g0 ? g0.mem_total / 1024 : undefined;
  charts.vram.set(ts, [s.map((x) => (x.gpus && x.gpus[0] ? x.gpus[0].mem_used / 1024 : null))]);
}

function drawTrainCharts() {
  const ik = state.liveIk, col = state.liveCol, T = state.liveTol, es = state.liveEs;
  document.querySelectorAll('.thr-label').forEach((el) => (el.textContent = `${T} mm`));
  charts.ikacc.ref = es && es.metric === 'pct_within' ? { value: +es.target_pct, label: `stop target ${+es.target_pct}%` } : null;
  charts.err.ref = { value: +T, label: `tolerance ${T} mm` };
  charts.loss.set(ik.map((e) => e.epoch), [ik.map((e) => e.train_loss), ik.map((e) => e.val_mse)]);
  charts.ikacc.set(ik.map((e) => e.epoch), [ik.map((e) => e.acc?.train?.acc_pos?.[T] ?? null), ik.map((e) => e.acc?.val?.acc_pos?.[T] ?? null)]);
  charts.acc.set(col.map((e) => e.epoch), [col.map((e) => e.train_acc ?? null), col.map((e) => e.val_acc ?? null)]);
  charts.err.set(state.liveErr.map((p) => p.epoch), [state.liveErr.map((p) => p.y)]);
}

function onStats(s, skipPush) {
  if (!skipPush) {
    state.stats.push(s);
    if (state.stats.length > 300) state.stats.shift();
  }
  $('t-cpu').innerHTML = `${s.cpu.toFixed(0)}<small>%</small>`;
  $('t-cpu-sub').textContent = `${s.cpu_count} logical cores`;
  $('t-ram').innerHTML = `${s.ram_used_gb.toFixed(1)}<small> / ${s.ram_total_gb.toFixed(0)} GB</small>`;
  $('t-ram-sub').textContent = `${s.ram_pct.toFixed(0)}% used`;
  const g = s.gpus && s.gpus[0];
  if (g) {
    $('gpu-name').textContent = g.name;
    $('t-gpu').innerHTML = `${fmt(g.util, 0)}<small>%</small>`;
    $('t-gpu-sub').textContent = `SM clock ${fmt(g.clock_sm, 0)} MHz`;
    $('t-vram').innerHTML = `${(g.mem_used / 1024).toFixed(2)}<small> / ${(g.mem_total / 1024).toFixed(0)} GB</small>`;
    $('t-vram-sub').textContent = `mem controller ${fmt(g.mem_util, 0)}%`;
    $('t-temp').innerHTML = `${fmt(g.temp, 0)}<small> °C</small>`;
    $('t-power').textContent = `${fmt(g.power, 0)} / ${fmt(g.power_limit, 0)} W`;
  } else {
    $('gpu-name').textContent = 'no NVIDIA GPU detected';
  }
  if (s.job_cpu !== undefined) {
    $('t-job').innerHTML = `${s.job_cpu.toFixed(0)}<small>% CPU</small>`;
    $('t-job-sub').textContent = `${s.job_rss_gb.toFixed(2)} GB RAM · ${s.job_threads} threads`;
  } else {
    $('t-job').innerHTML = '<small>no job running</small>';
    $('t-job-sub').textContent = '';
  }
  const cores = $('cores');
  if (cores.children.length !== s.per_cpu.length) cores.innerHTML = s.per_cpu.map(() => '<div></div>').join('');
  s.per_cpu.forEach((v, i) => {
    const d = cores.children[i];
    d.style.height = Math.max(4, v) + '%';
    d.title = `core ${i}: ${v.toFixed(0)}%`;
  });
  if (!skipPush) drawStatCharts();
}

// ===================================================================
// Misc wiring + boot
// ===================================================================

function switchTab(id) {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x.dataset.tab === id));
  document.querySelectorAll('.tabpane').forEach((p) => p.classList.toggle('active', p.id === id));
}
document.querySelectorAll('.tab').forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
$('run-select').onchange = (e) => selectRun(e.target.value).catch((err) => toast(err.message, true));
$('expand-btn').onclick = () => {
  const big = document.documentElement.style.getPropertyValue('--bottom-h') === '62vh';
  document.documentElement.style.setProperty('--bottom-h', big ? '330px' : '62vh');
};
$('show-cloud').onchange = (e) => (viewer.cloud.visible = e.target.checked);
$('show-ghost').onchange = (e) => { if (!e.target.checked) viewer.ghost.group.visible = viewer.refined.group.visible = false; };
$('fit-btn').onclick = () => fitView();
$('robot-select').onchange = (e) => selectRobot(e.target.value).catch((err) => toast(err.message, true));

function connectEvents() {
  const es = new EventSource('/events');
  es.onopen = () => { $('conn-dot').className = 'dot on'; $('conn-text').textContent = 'connected'; };
  es.onerror = () => { $('conn-dot').className = 'dot bad'; $('conn-text').textContent = 'reconnecting'; };
  let chain = Promise.resolve();
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    chain = chain.then(() => onEvent(ev)).catch((err) => console.error(err));
  };
}

async function boot() {
  initViewer();
  initCharts();
  const robots = await api('/api/robots');
  for (const r of robots) {
    const o = document.createElement('option');
    o.value = r.name; o.textContent = `${r.name} (${r.dof}-DOF)`;
    $('robot-select').appendChild(o);
  }
  state.lib = await api('/api/library');
  await selectRobot('xarm6');
  renderLibrary();
  syncInferenceMode();
  updateEsUI();
  const first = state.lib.runs.find((r) => r.robot === state.robot && r.has_models) || state.lib.runs[0];
  if (first) await selectRun(first.id);
  connectEvents();
}

boot().catch((e) => { console.error(e); toast('Failed to start: ' + e.message, true); });
