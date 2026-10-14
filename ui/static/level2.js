// Neural IK Studio – Level 2 page: train / test / evaluate the DRL re-routing agent.
// Robot poses come from the server's PyBullet episodes (level2/); this file only draws them.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const pct = (v) => (v == null ? '—' : `${Number(v).toFixed(1)}%`);
const when = (t) => (t ? new Date(t * 1000).toLocaleString(undefined, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—');
const C = { agent: '#3987e5', base: '#d95926', good: '#2bb673', bad: '#e5534b', warn: '#fab219', grid: '#2a313c', text: '#b3bdcb', muted: '#7f8a9a' };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
let toastTimer;
function toast(msg, err = false) {
  const t = $('toast'); t.textContent = msg; t.className = err ? 'err' : ''; t.style.display = 'block';
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.style.display = 'none'), 5000);
}

const state = { policies: [], job: null, curves: {}, ep: null, frame: 0, playing: false, models: {} };

// ===================================================================== steps
const panels = [...document.querySelectorAll('#left > details.panel')];
function showStep(id) {
  for (const p of panels) p.open = p.id === id;
  document.querySelectorAll('#stepper button').forEach((b) => b.classList.toggle('active', b.dataset.step === id));
  try { localStorage.setItem('nik.l2step', id); } catch { /* ignore */ }
}
document.querySelectorAll('#stepper button').forEach((b) => (b.onclick = () => showStep(b.dataset.step)));
panels.forEach((p) => p.querySelector(':scope > summary').addEventListener('click', (e) => e.preventDefault()));
let savedStep = null; try { savedStep = localStorage.getItem('nik.l2step'); } catch { /* ignore */ }
showStep(panels.some((p) => p.id === savedStep) ? savedStep : 'p-env');

document.querySelectorAll('.tab').forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
function switchTab(id) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === id));
  document.querySelectorAll('.tabpane').forEach((p) => p.classList.toggle('active', p.id === id));
  if (id === 'curve-pane') renderCurves();
}

// ===================================================================== 3D viewer
const viewer = {};
function initViewer() {
  const el = $('viewport');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#12161c');
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 60);
  camera.position.set(0.95, 0.72, 0.95);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.28, 0);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a28, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8); sun.position.set(2.5, 4, 3); scene.add(sun);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-3, 2, -2); scene.add(fill);
  scene.add(new THREE.GridHelper(4, 40, 0x3a414c, 0x232932));
  const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);   // PyBullet Z-up
  const obstacles = new THREE.Group(); root.add(obstacles);
  const trails = new THREE.Group(); root.add(trails);
  const goalMark = new THREE.Mesh(new THREE.SphereGeometry(0.018, 20, 14), new THREE.MeshStandardMaterial({ color: 0x2bb673, emissive: 0x0b4026 }));
  goalMark.visible = false; root.add(goalMark);
  Object.assign(viewer, { el, renderer, scene, camera, controls, root, obstacles, trails, goalMark });
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = h + 'px';
    camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el); resize();
  const loop = () => { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); };
  loop();
}
const meshCache = new Map();
function loadMesh(url, ext) {
  if (!meshCache.has(url)) {
    const loader = ext === 'stl' ? new STLLoader() : new OBJLoader();
    meshCache.set(url, loader.loadAsync(url).then((res) => { if (ext === 'stl') res.computeVertexNormals(); return res; }));
  }
  return meshCache.get(url);
}
class RobotView {
  constructor(model, style) { this.group = new THREE.Group(); this.frames = {}; this.style = style; this.model = model; }
  material(v) {
    if (this.style === 'solid') { const [r, g, b] = v.rgba; return new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), metalness: 0.15, roughness: 0.55 }); }
    const col = this.style === 'base' ? C.base : C.good;
    return new THREE.MeshStandardMaterial({ color: col, transparent: true, opacity: this.style === 'base' ? 0.45 : 0.22, depthWrite: false, roughness: 0.6 });
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
async function loadRobot(name) {
  if (state.robot === name) return;
  $('loading').style.display = 'grid';
  const model = state.models[name] || (state.models[name] = await api(`/api/robot_model?robot=${name}`));
  for (const k of ['agent', 'base', 'goal']) if (viewer[k]) viewer.root.remove(viewer[k].group);
  viewer.agent = await new RobotView(model, 'solid').load();
  viewer.base = await new RobotView(model, 'base').load();
  viewer.goal = await new RobotView(model, 'goal').load();
  viewer.root.add(viewer.agent.group, viewer.base.group, viewer.goal.group);
  viewer.base.group.visible = viewer.goal.group.visible = false;
  const zero = await api('/api/fk', { robot: name, q: model.joints.map(() => 0) });
  viewer.agent.setFrames(zero.links);
  state.robot = name;
  $('loading').style.display = 'none';
}

// ===================================================================== policies
async function refreshPolicies() {
  state.policies = await api('/api/l2/policies');
  const usable = state.policies.filter((p) => p.has_policy);
  for (const id of ['l2-policy', 'l2-policy-eval']) {
    const sel = $(id), prev = sel.value;
    sel.innerHTML = usable.length ? usable.map((p) => `<option value="${esc(p.id)}">${esc(p.id)} · ${polScore(p)}</option>`).join('')
      : '<option value="">No trained agent yet – train one in step 1</option>';
    if (usable.some((p) => p.id === prev)) sel.value = prev;
  }
  renderPolicies();
}
function polScore(p) {
  const s = p.evaluation && p.evaluation.scenarios && p.evaluation.scenarios.combined;
  if (s) return `everything at once: ${fmt(s.policy.success_pct, 0)}% success (straight ${fmt(s.baseline.success_pct, 0)}%)`;
  if (p.last) return `training: ${fmt(p.last.success_pct, 0)}% success`;
  return p.status || '';
}
function renderPolicies() {
  const rows = state.policies.map((p) => {
    const ev = p.evaluation && p.evaluation.scenarios, cb = ev && ev.combined;
    const st = p.status === 'done' ? '<span class="tag good">done</span>' : `<span class="tag ${['running', 'evaluating'].includes(p.status) ? 'warn' : 'bad'}">${esc(p.status)}</span>`;
    return `<tr>
      <td class="acts"><button data-act="play" data-id="${esc(p.id)}" ${p.has_policy ? '' : 'disabled'}>Test</button>
        <button data-act="results" data-id="${esc(p.id)}" ${ev ? '' : 'disabled'}>Results</button>
        <button data-act="curves" data-id="${esc(p.id)}">Curves</button>
        <button class="del" data-act="delete" data-id="${esc(p.id)}">🗑 Delete</button></td>
      <td><span class="id">${esc(p.id)}</span></td><td>${esc(p.robot)}</td><td>${st}</td>
      <td class="num">${p.steps_done ? (p.steps_done / 1e6).toFixed(2) + ' M' : '—'}</td>
      <td class="num">${p.train_secs ? fmt(p.train_secs / 60, 1) + ' min' : '—'}</td>
      <td>${esc(p.device || (p.params && p.params.device) || '')}</td>
      <td class="num">${cb ? pct(cb.policy.success_pct) : '—'}</td><td class="num">${cb ? `${pct(cb.baseline.success_pct)} / ${cb.apf ? pct(cb.apf.success_pct) : '—'}` : '—'}</td>
      <td class="num">${cb ? pct(cb.policy.collision_pct) : '—'}</td><td class="num">${cb ? `${pct(cb.baseline.collision_pct)} / ${cb.apf ? pct(cb.apf.collision_pct) : '—'}` : '—'}</td>
      <td class="num">${p.evaluation ? fmt(p.evaluation.decision_us, 1) + ' µs' : '—'}</td><td>${when(p.created)}</td></tr>`;
  }).join('');
  $('pol-pane').innerHTML = `<div class="pane-pad">
    <div class="toolbar"><span class="grow hint">Trained agents live in <code>ui/workspace/level2/policies/</code>, separate from the Level 1 library.
      “Combined” = obstacles on the path + a sudden obstacle + a joint slip in the same episode.</span></div>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Actions</th><th>Agent</th><th>Robot</th><th>Status</th><th class="num">Steps</th><th class="num">Train time</th><th>Device</th>
      <th class="num">Success (agent)</th><th class="num">Success (straight / pot. field)</th><th class="num">Collisions (agent)</th><th class="num">Collisions (straight / pot. field)</th><th class="num">Decision time</th><th>Created</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="13" class="muted">No agent yet – press “Train agent”.</td></tr>'}</tbody></table></div></div>`;
}
$('pol-pane').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-act]'); if (!b) return;
  const id = b.dataset.id;
  try {
    if (b.dataset.act === 'play') { $('l2-policy').value = id; showStep('p-play'); await runScenario(); }
    else if (b.dataset.act === 'results') { await showResults(id); switchTab('res-pane'); }
    else if (b.dataset.act === 'curves') { state.curveId = id; await loadCurves(id); switchTab('curve-pane'); }
    else if (b.dataset.act === 'delete') await deletePolicy(id);
  } catch (err) { toast(err.message, true); }
});
async function deletePolicy(id) {
  const ok = await confirmDialog('Delete agent?', `<p>Permanently delete the trained agent <span class="id">${esc(id)}</span>?</p>
    <p>Its policy weights, training history and evaluation results are removed.</p>`);
  if (!ok) return;
  await api('/api/delete', { kind: 'policy', id });
  toast(`Deleted agent ${id}.`);
  await refreshPolicies();
}
function confirmDialog(title, html) {
  const dlg = $('confirm');
  $('confirm-title').textContent = title; $('confirm-body').innerHTML = html;
  return new Promise((resolve) => {
    const done = (ok) => { dlg.close(); resolve(ok); };
    $('confirm-yes').onclick = () => done(true); $('confirm-no').onclick = () => done(false);
    dlg.oncancel = (e) => { e.preventDefault(); done(false); };
    dlg.showModal(); $('confirm-no').focus();
  });
}

// ===================================================================== training
$('l2-train-btn').onclick = async () => {
  try {
    const r = await api('/api/l2/train', { robot: $('l2-robot').value, steps: +$('l2-steps').value, envs: +$('l2-envs').value,
      seed: +$('l2-seed').value, eval_n: +$('l2-evaln').value });
    state.trainingId = r.id; state.curves[r.id] = []; state.curveId = r.id;
    toast(`Training started: ${r.id}`);
    switchTab('curve-pane');
    await refreshPolicies();
  } catch (e) { toast(e.message, true); }
};
$('l2-eval-stop').onclick = async () => { try { await api('/api/stop', {}); toast('Evaluation stopped.'); } catch (e) { toast(e.message, true); } };
// header ■ Stop: stops whatever job is running (asks first)
$('job-stop').onclick = async () => {
  if (!confirm('Stop the running job? Training keeps what it has learned so far; an evaluation is discarded.')) return;
  try { await api('/api/stop', {}); toast('Stopping…'); } catch (e) { toast(e.message, true); }
};
$('l2-stop-btn').onclick = async () => { try { await api('/api/stop', {}); toast('Stopping – the best policy so far is kept.'); } catch (e) { toast(e.message, true); } };

function onUpdate(ev) {
  (state.curves[ev.policy_id] ||= []).push(ev);
  state.trainingId = ev.policy_id;
  const f = ev.update / ev.updates;
  $('l2-prog-bar').style.width = `${100 * f}%`;
  $('l2-prog-label').textContent = `update ${ev.update}/${ev.updates} · ${(ev.steps / 1e6).toFixed(2)} M steps · ${ev.fps.toLocaleString()} steps/s`;
  $('l2-prog-pct').textContent = `${(100 * f).toFixed(0)}%`;
  $('l2-train-kv').innerHTML = `<dt>Success (last update)</dt><dd>${pct(ev.success_pct)}</dd><dt>Collisions</dt><dd>${pct(ev.collision_pct)}</dd>
    <dt>Timeouts</dt><dd>${pct(ev.timeout_pct)}</dd><dt>Episodes this update</dt><dd>${ev.episodes}</dd>
    <dt>PPO updates on</dt><dd>${esc(ev.device)}</dd><dt>Simulation</dt><dd>${ev.envs} envs · ${ev.workers} CPU workers</dd>`;
  if (state.curveId === ev.policy_id && $('curve-pane').classList.contains('active')) renderCurves();
}

// ===================================================================== curves
async function loadCurves(id) {
  const d = await api(`/api/l2/policy?id=${encodeURIComponent(id)}`);
  if (d.history && d.history.length) state.curves[id] = d.history;
  renderCurves();
}
function renderCurves() {
  const id = state.curveId || state.trainingId;
  const h = (id && state.curves[id]) || [];
  if (!h.length) { $('curve-pane').innerHTML = '<div class="empty">Start training or pick an agent’s “Curves” to see training progress.</div>'; return; }
  $('curve-pane').innerHTML = `<div class="pane-pad" style="padding-bottom:0"><h3 class="sec">Training curves <small>${esc(id)} · smoothed over 5 updates</small></h3>
    <div class="hint">Shaded = warm-up: the first updates only count short (mostly successful) episodes, because an episode can last up to 300 steps but an update collects 128 per environment. Return and success then drop to their true level – that drop is not the agent getting worse. Judge progress after the warm-up, or with Evaluate.</div></div>
    <div class="l2-charts">
      <div class="box"><h4><span>Success rate (%)</span></h4><canvas id="cv-succ"></canvas></div>
      <div class="box"><h4><span>Collision rate (%)</span></h4><canvas id="cv-coll"></canvas></div>
      <div class="box"><h4><span>Episode return</span></h4><canvas id="cv-ret"></canvas></div></div>`;
  const xs = h.map((r) => r.steps / 1e6);
  const smooth = (k) => h.map((_, i) => { const w = h.slice(Math.max(0, i - 4), i + 1).map((r) => r[k]).filter((v) => v != null); return w.length ? w.reduce((a, b) => a + b, 0) / w.length : null; });
  const warm = xs[Math.min(2, xs.length - 1)];          // episodes last up to 300 steps = ~3 updates of 128 steps
  lineChart($('cv-succ'), xs, [{ y: smooth('success_pct'), color: C.good }], [0, 100], warm);
  lineChart($('cv-coll'), xs, [{ y: smooth('collision_pct'), color: C.bad }], [0, 100], warm);
  lineChart($('cv-ret'), xs, [{ y: smooth('ep_return'), color: C.agent }], null, warm);
}
function lineChart(cv, xs, series, yr, warmX) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, hgt = cv.clientHeight;
  cv.width = w * dpr; cv.height = hgt * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const L = 40, R = 8, T = 8, B = 20;
  const all = series.flatMap((s) => s.y).filter((v) => v != null);
  let lo = yr ? yr[0] : Math.min(...all), hi = yr ? yr[1] : Math.max(...all);
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const x0 = xs[0], x1 = xs[xs.length - 1] || 1;
  const X = (x) => L + ((x - x0) / Math.max(1e-9, x1 - x0)) * (w - L - R), Y = (y) => T + (1 - (y - lo) / (hi - lo)) * (hgt - T - B);
  if (warmX != null && warmX > x0) {                      // warm-up region (see the note above the charts)
    g.fillStyle = 'rgba(127, 138, 154, 0.12)'; g.fillRect(L, T, X(warmX) - L, hgt - T - B);
    g.fillStyle = C.muted; g.font = '10px Segoe UI, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText('warm-up', L + 4, T + 2);
  }
  g.font = '11px Segoe UI, sans-serif'; g.fillStyle = C.muted; g.strokeStyle = C.grid; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
    g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1), L - 6, y);
  }
  g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText(`${x1.toFixed(2)} M steps`, w - R, hgt);
  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = 2; g.beginPath(); let first = true;
    s.y.forEach((v, i) => { if (v == null) return; const px = X(xs[i]), py = Y(v); if (first) { g.moveTo(px, py); first = false; } else g.lineTo(px, py); });
    g.stroke();
  }
}

// ===================================================================== results
async function showResults(id) {
  const d = await api(`/api/l2/policy?id=${encodeURIComponent(id)}`);
  const ev = d.evaluation;
  if (!ev) { $('res-pane').innerHTML = '<div class="empty">This agent has not been evaluated yet.</div>'; return; }
  const names = { combined: 'Everything at once', sudden: 'Sudden obstacle', static: 'Obstacles from the start', moving: 'Drifting obstacles', slip: 'Joint slip (hardware error)', mixed: 'Random mix' };
  // agent cell is green when it beats both baselines, red when it is worse than both
  const cell = (r, key, higherBetter) => {
    const a = r.policy[key], others = ['baseline', 'apf'].filter((c) => r[c]).map((c) => r[c][key]);
    const beats = (x) => (higherBetter ? a > x + 0.5 : a < x - 0.5), loses = (x) => (higherBetter ? a < x - 0.5 : a > x + 0.5);
    const cls = others.every(beats) ? 'better' : others.every(loses) ? 'worse' : '';
    return `<td class="num ${cls}">${pct(a)}</td>` + ['baseline', 'apf'].map((c) => `<td class="num">${r[c] ? pct(r[c][key]) : '—'}</td>`).join('');
  };
  const rows = Object.entries(ev.scenarios).map(([k, r]) => `<tr><td>${esc(names[k] || k)}</td>
      ${cell(r, 'success_pct', true)}${cell(r, 'collision_pct', false)}${cell(r, 'timeout_pct', false)}
      <td class="num">${fmt(r.policy.min_clearance_mm, 0)} mm</td></tr>`).join('');
  const cb = ev.scenarios.combined || Object.values(ev.scenarios)[0];
  $('res-pane').innerHTML = `<div class="pane-pad">
    <div class="res-head"><h3 class="sec">Agent vs. baseline <small>${esc(id)} · ${ev.robot} · ${ev.n_per_scenario} episodes per scenario, identical seeds</small></h3></div>
    <div class="kpi-row">
      <div class="tile"><div class="label">Success, everything at once</div><div class="value">${pct(cb.policy.success_pct)}</div><div class="sub">straight ${pct(cb.baseline.success_pct)} · pot. field ${cb.apf ? pct(cb.apf.success_pct) : '—'}</div></div>
      <div class="tile"><div class="label">Collisions, everything at once</div><div class="value">${pct(cb.policy.collision_pct)}</div><div class="sub">straight ${pct(cb.baseline.collision_pct)} · pot. field ${cb.apf ? pct(cb.apf.collision_pct) : '—'}</div></div>
      <div class="tile"><div class="label">Policy size</div><div class="value">${(ev.policy_params / 1000).toFixed(1)} k</div><div class="sub">parameters (2 × 64 MLP)</div></div>
      <div class="tile"><div class="label">Decision time</div><div class="value">${fmt(ev.decision_us, 1)} µs</div><div class="sub">NumPy, 1 CPU core, no GPU</div></div>
    </div>
    <div class="tbl-wrap"><table class="tbl"><thead>
      <tr><th></th><th colspan="3" class="num">Success ↑</th><th colspan="3" class="num">Collisions ↓</th><th colspan="3" class="num">Stuck / timeout ↓</th><th></th></tr>
      <tr><th>Scenario</th><th class="num">DRL agent</th><th class="num">straight</th><th class="num">pot. field</th>
      <th class="num">DRL agent</th><th class="num">straight</th><th class="num">pot. field</th>
      <th class="num">DRL agent</th><th class="num">straight</th><th class="num">pot. field</th><th class="num">Agent min. clearance</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <div class="hint">Straight = Level 1 IK goal reached along a straight joint-space path, no obstacle awareness. Pot. field = straight path + repulsion along
      the Jacobian escape direction (classical method; gets stuck in local minima). Green = agent beats both baselines, red = worse than both.</div></div>`;
}
$('l2-eval-btn').onclick = async () => {
  const id = $('l2-policy-eval').value; if (!id) return toast('Train an agent first.', true);
  try { await api('/api/l2/evaluate', { id, n: +$('l2-eval-n').value }); state.evalId = id; toast('Evaluation started – see the Log tab.'); switchTab('log-pane'); }
  catch (e) { toast(e.message, true); }
};

// ===================================================================== scenario playback
$('l2-run').onclick = () => runScenario().catch((e) => toast(e.message, true));
$('l2-next').onclick = () => { $('l2-seed-play').value = +$('l2-seed-play').value + 1; runScenario().catch((e) => toast(e.message, true)); };
async function runScenario() {
  const id = $('l2-policy').value;
  if (!id) return toast('Train an agent first (step 1).', true);
  $('l2-run').disabled = true;
  try {
    const ep = await api('/api/l2/rollout', { id, scenario: $('l2-scenario').value, seed: +$('l2-seed-play').value });
    ep.mode = 'agent';
    await showEpisode(ep);
  } finally { $('l2-run').disabled = false; }
}
async function showEpisode(ep) {
  await loadRobot(ep.robot);
  if (ep.mode === 'preview') { ep.policy = ep.apf; }            // no agent yet: the potential field is the solid robot
  state.ep = ep;
  viewer.goal.setFrames(ep.goal_links);
  viewer.goalMark.position.set(...ep.goal_ee);
  buildTrails();
  const n = Math.max(...['policy', 'baseline', 'apf'].filter((k) => ep[k]).map((k) => ep[k].frames.length)) - 1;
  $('l2-slider').max = n; setFrame(0);
  renderVerdict();
  play(true);
}
// step 1: preview the environment with the classical controllers
async function previewEnv() {
  $('l2-env-show').disabled = true;
  try {
    const ep = await api('/api/l2/rollout', { robot: $('l2-env-robot').value, scenario: $('l2-env-scenario').value, seed: +$('l2-env-seed').value });
    ep.mode = 'preview';
    if ($('l2-show').value === 'apf') $('l2-show').value = 'baseline';
    await showEpisode(ep);
  } catch (e) { toast(e.message, true); } finally { $('l2-env-show').disabled = false; }
}
$('l2-env-show').onclick = previewEnv;
$('l2-env-next').onclick = () => { $('l2-env-seed').value = +$('l2-env-seed').value + 1; previewEnv(); };
// live view during training: one episode of the current agent every few updates
async function onLive(ev) {
  if (!$('l2-follow').checked || (state.playing && state.ep && state.ep.mode !== 'live')) return;   // never interrupt a scenario you are watching
  const e = ev.episode;
  await showEpisode({ mode: 'live', robot: ev.robot, scenario: 'mixed (training)', seed: `update ${ev.update}/${ev.updates}`,
    policy: e, baseline: null, apf: null, goal_links: e.goal_links, goal_ee: e.goal_ee });
}
function renderVerdict() {
  const ep = state.ep;
  const row = (sw, name, e) => `<div class="vrow"><span><i class="${sw}"></i><b>${name}</b></span>
      <span><span class="outcome ${e.outcome}">${e.outcome === 'collision' ? `collision (${e.collision})` : e.outcome}</span>
      · ${e.steps} steps · min. clearance ${fmt(1000 * e.min_clearance_m, 0)} mm</span></div>`;
  const first = ep.mode === 'preview' ? 'Potential field' : ep.mode === 'live' ? 'Agent (live, training)' : 'DRL agent';
  $('l2-verdict').innerHTML = `<div class="vcard">${row('swatch-agent', first, ep.policy)}${ep.baseline ? row('swatch-base', 'Straight path', ep.baseline) : ''}${ep.apf && ep.mode !== 'preview' ? row('swatch-base', 'Potential field', ep.apf) : ''}</div>`;
  const evs = ep.policy.events.map((e) => `<span class="ev"><i style="background:${e.type === 'joint_slip' ? C.warn : C.bad}"></i>step ${e.t}: ${e.type === 'joint_slip' ? `joint ${e.joint + 1} slipped` : 'obstacle appeared'}</span>`);
  $('l2-events').innerHTML = evs.length ? `Events (agent run): ${evs.join('')}` : 'No mid-motion events in this scenario.';
}
function eeOf(links) { const keys = Object.keys(links).map(Number).filter((k) => k >= 0); return links[String(Math.max(...keys))][0]; }
function buildTrails() {
  viewer.trails.clear();
  const ep = state.ep;
  for (const [k, col] of [['policy', C.agent], ['baseline', C.base], ['apf', C.base]]) {
    if (!ep[k]) continue;
    const pts = ep[k].frames.map((f) => new THREE.Vector3(...eeOf(f.links)));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: col }));
    line.userData.kind = k; viewer.trails.add(line);
  }
  applyVisibility();
}
const cmpKey = () => ($('l2-show').value === 'apf' ? 'apf' : 'baseline');
function applyVisibility() {
  const show = $('l2-show').value;
  if (viewer.agent) viewer.agent.group.visible = true;
  if (viewer.base) viewer.base.group.visible = !!state.ep && show !== 'policy';
  if (viewer.goal) viewer.goal.group.visible = !!state.ep && $('l2-goal').checked;
  viewer.goalMark.visible = !!state.ep && $('l2-goal').checked;
  viewer.trails.visible = $('l2-trails').checked;
  const ep = state.ep;
  const second = ep && ep.mode === 'preview' ? 'baseline' : cmpKey();
  viewer.trails.children.forEach((l) => (l.visible = (l.userData.kind === 'policy' && !(ep && ep.mode === 'preview')) || (ep && ep.mode === 'preview' && l.userData.kind === 'apf') || (show !== 'policy' && l.userData.kind === second)));
  if (viewer.base) viewer.base.group.visible = !!ep && show !== 'policy' && !!(ep.mode === 'preview' ? ep.baseline : ep[cmpKey()]);
}
['l2-show', 'l2-goal', 'l2-trails'].forEach((id) => ($(id).onchange = () => { applyVisibility(); if (state.ep) setFrame(state.frame); }));
const obsMat = new THREE.MeshStandardMaterial({ color: 0xff4d4d, transparent: true, opacity: 0.8, roughness: 0.35, emissive: 0x3a0505 });
const obsNewMat = new THREE.MeshStandardMaterial({ color: 0xfab219, transparent: true, opacity: 0.7, emissive: 0x553300 });
// ---- smooth playback: frames are interpolated between simulation steps (position lerp, rotation slerp)
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
function lerpLinks(A, B, t) {
  if (!B || t <= 0) return A;
  const out = {};
  for (const [k, [pa, qa]] of Object.entries(A)) {
    const [pb, qb] = B[k] || [pa, qa];
    _qa.set(...qa); _qb.set(...qb); _qa.slerp(_qb, t);
    out[k] = [[pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t], [_qa.x, _qa.y, _qa.z, _qa.w]];
  }
  return out;
}
const framePair = (run, f) => {
  const n = run.frames.length - 1, i = Math.min(Math.floor(f), n), j = Math.min(i + 1, n);
  return [run.frames[i], run.frames[j], Math.min(1, f - i)];
};
const unitSphere = new THREE.SphereGeometry(1, 28, 18);
function setFrame(f) {
  const ep = state.ep; if (!ep) return;
  state.frame = f;
  const i = Math.round(f);
  $('l2-slider').value = Math.floor(f);
  const cmp = (ep.mode === 'preview' ? ep.baseline : ep[cmpKey()] || ep.baseline) || ep.policy;
  const [fa, fa2, ta] = framePair(ep.policy, f);
  const [fb, fb2, tb] = framePair(cmp, f);
  viewer.agent.setFrames(lerpLinks(fa.links, fa2.links, ta));
  viewer.base.setFrames(lerpLinks(fb.links, fb2.links, tb));
  // obstacles of the agent run; meshes are reused and glide between steps; a just-appeared one is highlighted
  const evs = ep.policy.events || [];
  const fresh = evs.some((e) => e.type === 'obstacle_appeared' && i >= e.t && i < e.t + 12);
  const obs = fa.obstacles, obs2 = fa2.obstacles;
  while (viewer.obstacles.children.length < obs.length) viewer.obstacles.add(new THREE.Mesh(unitSphere, obsMat));
  viewer.obstacles.children.forEach((m, k) => {
    const o = obs[k];
    m.visible = !!o; if (!o) return;
    const o2 = obs2[k] && obs2.length === obs.length ? obs2[k] : o;
    m.position.set(...o.pos.map((v, d) => v + (o2.pos[d] - v) * ta));
    m.scale.setScalar(o.r);
    m.material = fresh && k === obs.length - 1 ? obsNewMat : obsMat;
  });
  const endA = i >= ep.policy.frames.length - 1, endB = i >= cmp.frames.length - 1;
  const cmpName = ep.mode === 'preview' || cmpKey() !== 'apf' ? 'straight ' : 'pot.field';
  const firstName = ep.mode === 'preview' ? 'pot.field' : 'agent    ';
  const clr = (fr) => (fr.clearance == null ? '—' : fmt(1000 * fr.clearance, 0) + ' mm');
  $('l2-step-label').textContent = `step ${Math.min(i, +$('l2-slider').max)} / ${$('l2-slider').max}`;
  $('hud-title').textContent = ep.mode === 'live' ? `● Live training · ${ep.seed} · ${ep.robot}`
    : `${ep.mode === 'preview' ? 'Environment preview' : 'Scenario'} “${ep.scenario}” · seed ${ep.seed} · ${ep.robot}`;
  $('hud-lines').textContent = [
    `${firstName} ${endA ? ep.policy.outcome.toUpperCase() : 'moving'}  clearance ${clr(fa)}`,
    cmp !== ep.policy ? `${cmpName} ${endB ? cmp.outcome.toUpperCase() : 'moving'}  clearance ${clr(fb)}` : '',
    eventNow(i),
  ].filter(Boolean).join('\n');
}
function eventNow(i) {
  const e = (state.ep.policy.events || []).find((x) => i >= x.t && i < x.t + 15);
  return e ? (e.type === 'joint_slip' ? `⚠ step ${e.t}: joint ${e.joint + 1} slipped (hardware position error)` : `⚠ step ${e.t}: obstacle appeared on the path`) : '';
}
$('l2-slider').oninput = () => { play(false); setFrame(+$('l2-slider').value); };
// animation loop: STEPS_PER_SEC simulation steps per second at 1×, drawn every display frame
const STEPS_PER_SEC = 15;
let rafId = 0, lastTs = 0, holdUntil = 0;
function play(on) {
  state.playing = on; $('l2-play').textContent = on ? '❚❚ Pause' : '▶ Play';
  cancelAnimationFrame(rafId);
  if (!on) return;
  if (state.frame >= +$('l2-slider').max) setFrame(0);
  lastTs = 0; holdUntil = 0;
  const tick = (ts) => {
    if (!state.playing) return;
    const dt = lastTs ? Math.min(0.1, (ts - lastTs) / 1000) : 0; lastTs = ts;
    const max = +$('l2-slider').max;
    if (state.frame >= max) {
      if (state.ep && state.ep.mode === 'live') {            // live view: hold the end pose, then loop until a newer episode arrives
        if (!holdUntil) holdUntil = ts + 1200;
        if (ts >= holdUntil) { holdUntil = 0; setFrame(0); }
      } else return play(false);
    } else setFrame(Math.min(max, state.frame + dt * STEPS_PER_SEC * +$('l2-speed').value));
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}
$('l2-play').onclick = () => play(!state.playing);
$('l2-speed').onchange = () => { if (state.playing) play(true); };
$('fit-btn').onclick = () => { viewer.camera.position.set(0.95, 0.72, 0.95); viewer.controls.target.set(0, 0.28, 0); viewer.controls.update(); };

// ===================================================================== live events
function applyJob(job) {
  state.job = job;
  const running = job && job.status === 'running';
  const l2 = job && String(job.kind || '').startsWith('l2');
  $('job-dot').className = 'dot ' + (running ? 'run' : job && job.status === 'failed' ? 'bad' : job ? 'on' : '');
  $('job-text').textContent = job ? `${job.kind} ${job.status}` : 'idle';
  $('l2-train-btn').disabled = !!running;
  $('l2-eval-btn').disabled = !!running;
  $('l2-stop-btn').disabled = !(running && job.kind === 'l2train');
  $('l2-eval-stop').disabled = !(running && job.kind === 'l2eval');
  $('job-stop').hidden = !running;
}
function log(line) { const el = $('log'); el.textContent += line + '\n'; if (el.textContent.length > 60000) el.textContent = el.textContent.slice(-40000); el.parentElement.scrollTop = el.parentElement.scrollHeight; }
function connectEvents() {
  const es = new EventSource('/events');
  es.onopen = () => { $('conn-dot').className = 'dot on'; $('conn-text').textContent = 'connected'; };
  es.onerror = () => { $('conn-dot').className = 'dot bad'; $('conn-text').textContent = 'reconnecting'; };
  es.onmessage = async (m) => {
    const ev = JSON.parse(m.data);
    switch (ev.type) {
      case 'hello': applyJob(ev.job); (ev.log || []).slice(-80).forEach(log); return;
      case 'job': {
        const was = state.job && state.job.status;
        applyJob(ev.job);
        if (was === 'running' && ev.job.status !== 'running' && String(ev.job.kind).startsWith('l2')) {
          toast(`${ev.job.kind === 'l2train' ? 'Training' : 'Evaluation'} ${ev.job.status}.`, ev.job.status === 'failed');
          await refreshPolicies();
          const id = ev.job.params && ev.job.params.policy_id;
          if (id && ev.job.status === 'done') { await showResults(id); switchTab('res-pane'); }
        }
        return;
      }
      case 'log': return log(ev.line);
      case 'library_changed': return refreshPolicies();
      case 'l2_update': return onUpdate(ev);
      case 'l2_live': return onLive(ev);
      case 'l2_eval_progress': {
        const target = state.job && state.job.kind === 'l2eval' ? 'l2-eval' : 'l2-prog';
        $(`${target}-label`).textContent = `evaluating: scenario ${ev.i}/${ev.n} done (${ev.scenario})`;
        $(`${target}-bar`).style.width = `${(100 * ev.i) / ev.n}%`; $(`${target}-pct`).textContent = `${Math.round((100 * ev.i) / ev.n)}%`;
        return;
      }
      default:
    }
  };
}

window.addEventListener('resize', () => { if ($('curve-pane').classList.contains('active')) renderCurves(); });

// ===================================================================== start
initViewer();
await loadRobot('xarm6');
await refreshPolicies();
connectEvents();
