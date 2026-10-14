// Mission page: Level 1 (IK) + trajectory estimate + Level 2 (DRL re-routing) in one run.
// All poses come from the server (/api/mission); this file only draws them.

import { THREE, createViewer, RobotView, lerpLinks, frameMarker, colorLine } from './viewer3d.js';
import { lineChart, stackedBars, histogram } from './charts.js';

// fixed identity colours (validated palette, slots 1-3: all-pairs CVD ΔE 9.4, normal ΔE 20.9 on the dark surface)
const COL = { policy: '#3987e5', apf: '#d95926', baseline: '#199e70' };
// status colours (reserved; always shown with a label)
const STATUS = { success: '#0ca30c', collision: '#d03b3b', timeout: '#fab219' };
const JOINT_COLS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9'];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const pct = (v) => (v == null ? '—' : `${Number(v).toFixed(1)}%`);
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
}
let toastTimer;
function toast(msg, err = false) {
  const t = $('toast'); t.textContent = msg; t.className = err ? 'err' : ''; t.style.display = 'block';
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.style.display = 'none'), 6000);
}
function switchTab(id) {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x.dataset.tab === id));
  document.querySelectorAll('.tabpane').forEach((p) => p.classList.toggle('active', p.id === id));
  if (id === 'plot-pane') renderPlots();                // charts need a visible container to size themselves
  if (id === 'saved-pane') refreshSaved();
}
document.querySelectorAll('.tab').forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
window.addEventListener('resize', () => { if ($('plot-pane').classList.contains('active')) renderPlots(); });

const NAMES = { policy: 'DRL agent (Level 2)', baseline: 'Straight path', apf: 'Potential field' };
const state = { m: null, f: 0, playing: false, models: {}, robot: null, agents: [], runs: [] };

// ------------------------------------------------------------------ 3D
const v = createViewer($('viewport'));
const target = frameMarker(); target.visible = false; v.root.add(target);
const obstacles = new THREE.Group(); v.root.add(obstacles);
const lines = new THREE.Group(); v.root.add(lines);
const unitSphere = new THREE.SphereGeometry(1, 28, 18);
const obsMat = new THREE.MeshStandardMaterial({ color: 0xff4d4d, transparent: true, opacity: 0.8, roughness: 0.35, emissive: 0x3a0505 });
const obsNewMat = new THREE.MeshStandardMaterial({ color: 0xfab219, transparent: true, opacity: 0.8, emissive: 0x553300 });
$('fit-btn').onclick = () => v.reset();

async function loadRobot(name) {
  if (state.robot === name) return;
  $('loading').style.display = 'grid';
  const model = state.models[name] || (state.models[name] = await api(`/api/robot_model?robot=${name}`));
  for (const k of ['agent', 'cmpA', 'cmpB', 'goal']) if (v[k]) v.root.remove(v[k].group);
  v.agent = await new RobotView(model, 'solid').load();
  v.cmpA = await new RobotView(model, COL.apf, 0.45).load();        // potential field
  v.cmpB = await new RobotView(model, COL.baseline, 0.45).load();   // straight path
  v.goal = await new RobotView(model, '#2bb673', 0.22).load();
  v.root.add(v.agent.group, v.cmpA.group, v.cmpB.group, v.goal.group);
  v.cmpA.group.visible = v.cmpB.group.visible = v.goal.group.visible = false;
  const zero = await api('/api/fk', { robot: name, q: model.joints.map(() => 0) });
  v.agent.setFrames(zero.links);
  state.robot = name;
  $('loading').style.display = 'none';
}

// ------------------------------------------------------------------ model lists
async function loadModels() {
  const [agents, lib] = await Promise.all([api('/api/l2/policies'), api('/api/library')]);
  state.agents = agents.filter((a) => a.has_policy);
  state.runs = lib.runs.filter((r) => r.has_models);
  const sel = $('m-agent');
  sel.innerHTML = state.agents.length ? state.agents.map((a) => {
    const cb = a.evaluation && a.evaluation.scenarios && a.evaluation.scenarios.combined;
    return `<option value="${esc(a.id)}">${esc(a.id)}${cb ? ` · ${fmt(cb.policy.success_pct, 0)}% success` : ''}</option>`;
  }).join('') : '<option value="">No Level 2 agent – train one first</option>';
  // prefer the best evaluated agent
  const best = [...state.agents].sort((a, b) => score(b) - score(a))[0];
  if (best) sel.value = best.id;
  fillL1();
}
const score = (a) => { const cb = a.evaluation && a.evaluation.scenarios && a.evaluation.scenarios.combined; return cb ? cb.policy.success_pct - cb.policy.collision_pct : -1; };
function fillL1() {
  const ag = state.agents.find((a) => a.id === $('m-agent').value);
  const robot = ag ? ag.robot : 'xarm6';
  const runs = state.runs.filter((r) => r.robot === robot);
  const acc = (r) => { const te = r.metrics && r.metrics.test; return te ? te.acc_pos[String(r.tol_mm || 30)] ?? te.pos_mean_mm : null; };
  $('m-l1').innerHTML = runs.length ? runs.map((r) => {
    const te = r.metrics && r.metrics.test;
    return `<option value="${esc(r.id)}">${esc(r.id)}${te ? ` · net ${fmt(te.pos_mean_mm, 0)} mm mean` : ''}</option>`;
  }).join('') : `<option value="">No Level 1 run for ${robot}</option>`;
  // prefer the most accurate network (smallest test error)
  const best = runs.filter((r) => r.metrics && r.metrics.test).sort((a, b) => a.metrics.test.pos_mean_mm - b.metrics.test.pos_mean_mm)[0];
  if (best) $('m-l1').value = best.id;
  void acc;
  loadRobot(robot);
}
$('m-agent').onchange = fillL1;
$('m-target').onchange = () => { $('m-custom').hidden = $('m-target').value !== 'custom'; };

// ------------------------------------------------------------------ pipeline status
function stage(k, cls, html) {
  const li = document.querySelector(`#pipeline li[data-k="${k}"]`);
  li.className = cls || ''; li.querySelector('.d').innerHTML = html;
}
function resetPipeline() { ['target', 'ik', 'plan', 'move', 'final'].forEach((k) => stage(k, '', '—')); }

// ------------------------------------------------------------------ run
$('m-run').onclick = () => runMission().catch((e) => { toast(e.message, true); stage('ik', 'bad', esc(e.message)); });
$('m-next').onclick = () => { $('m-seed').value = +$('m-seed').value + 1; $('m-run').onclick(); };
async function runMission() {
  const agent = $('m-agent').value, l1 = $('m-l1').value;
  if (!agent || !l1) throw new Error('Pick a Level 2 agent and a Level 1 run.');
  const body = { l2_policy: agent, l1_run: l1, scenario: $('m-scenario').value, seed: +$('m-seed').value };
  if ($('m-target').value === 'custom') {
    body.target = { mode: 'custom', pos: [+$('m-x').value, +$('m-y').value, +$('m-z').value], rpy_deg: [+$('m-r').value, +$('m-p').value, +$('m-yaw').value] };
  }
  play(false); resetPipeline();
  stage('target', 'run', 'choosing the target…'); stage('ik', 'run', 'solving with the Level 1 network + LM…');
  $('m-run').disabled = true;
  try {
    const m = await api('/api/mission', body);
    await loadRobot(m.robot);
    state.m = m; state.batch = null;
    showMission();
    toast(`Saved as ${m.id}`);
  } finally { $('m-run').disabled = false; }
}

function showMission() {
  const m = state.m, ik = m.ik, t = m.target;
  const rpy = quatToRpy(t.orn);
  stage('target', 'ok', `${t.mode === 'custom' ? 'custom' : 'random reachable'} · p = (${t.pos.map((x) => fmt(x, 3)).join(', ')}) m · rpy = (${rpy.map((x) => fmt(x, 0)).join(', ')})°`);
  target.position.set(...t.pos); target.quaternion.set(...t.orn); target.visible = true;
  stage('ik', ik.success ? 'ok' : 'bad', `network ${fmt(ik.net_pos_err_mm, 1)} mm / ${fmt(ik.net_orn_err_deg, 1)}° → LM <b>${fmt(ik.pos_err_mm, 3)} mm / ${fmt(ik.orn_err_deg, 2)}°</b>
    · ${fmt(ik.ms, 0)} ms · self-collision ${ik.self_collision ? '<b>yes</b>' : 'no'} · limits ${ik.limit_violation ? 'violated' : 'ok'}`);
  lines.clear(); obstacles.clear();
  if (!ik.success) {
    stage('plan', '', 'skipped'); stage('move', '', 'skipped'); stage('final', 'bad', esc(m.verdict));
    v.agent.setFrames(m.start_links); v.goal.group.visible = v.cmpA.group.visible = v.cmpB.group.visible = false;
    $('hud-title').textContent = `Mission seed ${m.seed} · ${m.robot}`; $('hud-lines').textContent = m.verdict;
    renderReport(); return;
  }
  const e = m.estimate;
  stage('plan', e.obstacle_conflicts ? 'warn' : 'ok', `straight path: ${fmt(e.ee_path_m, 2)} m for the gripper, ${fmt(e.joint_path_rad, 2)} rad in joint space ·
    <b>${e.obstacle_conflicts} of ${e.waypoints} waypoints collide</b> with known obstacles, ${e.near_misses} near misses, ${e.self_collisions} self-collisions
    → ${e.obstacle_conflicts ? 're-routing needed' : 'path is clear (so far)'}`);
  const a = m.episodes.policy, evs = a.events.map((x) => (x.type === 'joint_slip' ? `joint ${x.joint + 1} slipped at step ${x.t}` : `obstacle appeared at step ${x.t}`));
  stage('move', a.outcome === 'success' ? 'ok' : 'bad', `agent: <b>${a.outcome === 'collision' ? `collision (${a.collision})` : a.outcome}</b> after ${a.steps} steps ·
    min. clearance ${fmt(1000 * a.min_clearance_m, 0)} mm${evs.length ? ` · ${evs.join(', ')}` : ''}${a.final_approach ? ` · final approach ${a.final_approach.steps} steps` : ''}`);
  const okFinal = a.outcome === 'success' && a.final_pos_err_mm < 1 && a.final_orn_err_deg < 1;
  stage('final', okFinal ? 'ok' : a.outcome === 'success' ? 'warn' : 'bad',
    `gripper vs. target: <b>${fmt(a.final_pos_err_mm, 3)} mm / ${fmt(a.final_orn_err_deg, 3)}°</b> · ${esc(m.verdict)}`);
  // estimated path, coloured per waypoint
  const pts = m.planned.map((w) => w.p), cols = m.planned.map((w) => (w.clear <= 0 || w.self ? '#ff4d4d' : w.clear < 0.05 ? '#fab219' : '#c9b8ff'));
  const plan = colorLine(pts, cols, { dashed: true }); plan.userData.kind = 'plan'; lines.add(plan);
  for (const k of ['policy', 'baseline', 'apf']) {
    const ep = m.episodes[k]; if (!ep || !ep.frames) continue;
    const ee = ep.frames.map((f) => eeOf(f.links));
    const ln = colorLine(ee, ee.map(() => COL[k])); ln.userData.kind = k; lines.add(ln);
  }
  v.goal.setFrames(m.goal_links);
  $('m-slider').max = Math.max(...['policy', 'baseline', 'apf'].map((k) => m.episodes[k].frames.length)) - 1;
  applyVisibility(); setFrame(0); renderReport(); play(true);
  if ($('plot-pane').classList.contains('active')) renderPlots();
}
function eeOf(links) { const ks = Object.keys(links).map(Number).filter((k) => k >= 0); return links[String(Math.max(...ks))][0]; }
function quatToRpy([x, y, z, w]) {
  const e = new THREE.Euler().setFromQuaternion(new THREE.Quaternion(x, y, z, w), 'XYZ');
  return [e.x, e.y, e.z].map((r) => (r * 180) / Math.PI);
}
function applyVisibility() {
  const cmp = $('m-cmp').value, m = state.m;
  lines.children.forEach((l) => {
    l.visible = l.userData.kind === 'plan' ? $('m-show-plan').checked
      : $('m-show-trail').checked && (l.userData.kind === 'policy' || l.userData.kind === cmp);
  });
  v.goal.group.visible = !!(m && m.goal_links) && $('m-show-goal').checked;
  v.cmpA.group.visible = !!(m && m.episodes) && cmp === 'apf';
  v.cmpB.group.visible = !!(m && m.episodes) && cmp === 'baseline';
  v.cmp = cmp === 'apf' ? v.cmpA : v.cmpB;
}
['m-show-plan', 'm-show-trail', 'm-show-goal', 'm-cmp'].forEach((id) => ($(id).onchange = () => { applyVisibility(); if (state.m) setFrame(state.f); }));

// ------------------------------------------------------------------ smooth playback
function pair(run, f) { const n = run.frames.length - 1, i = Math.min(Math.floor(f), n), j = Math.min(i + 1, n); return [run.frames[i], run.frames[j], Math.min(1, f - i)]; }
function setFrame(f) {
  const m = state.m; if (!m || !m.episodes) return;
  state.f = f; $('m-slider').value = Math.floor(f);
  const a = m.episodes.policy, c = m.episodes[$('m-cmp').value] || a;
  const [fa, fa2, ta] = pair(a, f), [fc, fc2, tc] = pair(c, f);
  v.agent.setFrames(lerpLinks(fa.links, fa2.links, ta));
  (v.cmp || v.cmpB).setFrames(lerpLinks(fc.links, fc2.links, tc));
  const i = Math.round(f);
  const fresh = a.events.some((e) => e.type === 'obstacle_appeared' && i >= e.t && i < e.t + 12);
  while (obstacles.children.length < fa.obstacles.length) obstacles.add(new THREE.Mesh(unitSphere, obsMat));
  obstacles.children.forEach((mesh, k) => {
    const o = fa.obstacles[k]; mesh.visible = !!o; if (!o) return;
    const o2 = (fa2.obstacles.length === fa.obstacles.length && fa2.obstacles[k]) || o;
    mesh.position.set(...o.pos.map((x, d) => x + (o2.pos[d] - x) * ta)); mesh.scale.setScalar(o.r);
    mesh.material = fresh && k === fa.obstacles.length - 1 ? obsNewMat : obsMat;
  });
  const endA = i >= a.frames.length - 1;
  const phase = fa.phase === 'final approach' ? 'final approach (servo onto the IK goal)' : endA ? a.outcome.toUpperCase() : 'moving';
  const ev = a.events.find((e) => i >= e.t && i < e.t + 15);
  $('m-step').textContent = `step ${Math.min(i, +$('m-slider').max)} / ${$('m-slider').max}`;
  $('hud-title').textContent = `Mission seed ${m.seed} · ${m.scenario} · ${m.robot}`;
  $('hud-lines').textContent = [
    `agent     ${phase}  clearance ${fa.clearance == null ? '—' : fmt(1000 * fa.clearance, 0) + ' mm'}`,
    c !== a ? `${$('m-cmp').value === 'apf' ? 'pot.field' : 'straight '} ${i >= c.frames.length - 1 ? c.outcome.toUpperCase() : 'moving'}` : '',
    ev ? (ev.type === 'joint_slip' ? `⚠ step ${ev.t}: joint ${ev.joint + 1} slipped (hardware position error)` : `⚠ step ${ev.t}: obstacle appeared on the path`) : '',
  ].filter(Boolean).join('\n');
}
let raf = 0, last = 0;
function play(on) {
  state.playing = on; $('m-play').textContent = on ? '❚❚ Pause' : '▶ Play';
  cancelAnimationFrame(raf); if (!on) return;
  if (state.f >= +$('m-slider').max) setFrame(0);
  last = 0;
  const tick = (ts) => {
    if (!state.playing) return;
    const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0; last = ts;
    const max = +$('m-slider').max;
    if (state.f >= max) return play(false);
    setFrame(Math.min(max, state.f + dt * 15 * +$('m-speed').value));
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}
$('m-play').onclick = () => play(!state.playing);
$('m-slider').oninput = () => { play(false); setFrame(+$('m-slider').value); };

// ------------------------------------------------------------------ report + statistics
function renderReport() {
  const m = state.m, ik = m.ik;
  const tile = (label, value, sub) => `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  let html = `<div class="pane-pad"><h3 class="sec">Mission report <small>${esc(m.robot)} · scenario ${esc(m.scenario)} · seed ${m.seed} · IK run ${esc(ik.run)}</small></h3>
    <div class="rep-grid">
      ${tile('IK accuracy (Level 1)', `${fmt(ik.pos_err_mm, 3)} mm`, `${fmt(ik.orn_err_deg, 2)}° · ${fmt(ik.ms, 0)} ms · network alone ${fmt(ik.net_pos_err_mm, 0)} mm`)}
      ${m.estimate ? tile('Estimated path', `${m.estimate.obstacle_conflicts} / ${m.estimate.waypoints}`, `waypoints colliding · ${m.estimate.near_misses} near · ${m.estimate.self_collisions} self-collision`) : ''}
      ${m.episodes ? tile('Agent outcome', esc(m.episodes.policy.outcome), `${m.episodes.policy.steps} steps · min. clearance ${fmt(1000 * m.episodes.policy.min_clearance_m, 0)} mm`) : ''}
      ${m.episodes ? tile('Final gripper error', `${fmt(m.episodes.policy.final_pos_err_mm, 3)} mm`, `${fmt(m.episodes.policy.final_orn_err_deg, 3)}° vs. the target pose`) : ''}
    </div>`;
  if (m.episodes) {
    html += `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Controller (same mission)</th><th>Outcome</th><th class="num">Steps</th>
      <th class="num">Min. clearance</th><th class="num">Final position error</th><th class="num">Final orientation error</th><th>Events</th></tr></thead><tbody>
      ${['policy', 'apf', 'baseline'].map((k) => { const e = m.episodes[k]; return `<tr><td>${NAMES[k]}</td>
        <td><span class="tag ${e.outcome === 'success' ? 'good' : e.outcome === 'collision' ? 'bad' : 'warn'}">${e.outcome}${e.collision ? ` (${e.collision})` : ''}</span></td>
        <td class="num">${e.steps}</td><td class="num">${fmt(1000 * e.min_clearance_m, 0)} mm</td>
        <td class="num">${fmt(e.final_pos_err_mm, 3)} mm</td><td class="num">${fmt(e.final_orn_err_deg, 3)}°</td>
        <td>${e.events.map((x) => (x.type === 'joint_slip' ? `slip j${x.joint + 1}@${x.t}` : `obstacle@${x.t}`)).join(', ') || '—'}</td></tr>`; }).join('')}
      </tbody></table></div>
      <div class="hint">All three controllers get the identical mission (start, IK goal, obstacles, errors). Successful runs end with a short collision-checked
        final approach onto the IK goal, so the final error is the IK accuracy.</div>`;
  }
  $('rep-pane').innerHTML = html + '</div>';
}
$('m-batch').onclick = async () => {
  const agent = $('m-agent').value, l1 = $('m-l1').value;
  if (!agent || !l1) return toast('Pick a Level 2 agent and a Level 1 run.', true);
  $('m-batch').disabled = true; $('m-batch').textContent = 'Running…';
  switchTab('stat-pane');
  $('stat-pane').innerHTML = '<div class="empty">Running missions… (about 0.25 s each)</div>';
  try {
    const r = await api('/api/mission/batch', { l2_policy: agent, l1_run: l1, scenario: $('m-scenario').value, n: +$('m-n').value, seed: 1000 });
    state.batch = r;
    renderStats(r);
    toast(`Statistics saved as ${r.id}`);
  } catch (e) { toast(e.message, true); $('stat-pane').innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
  finally { $('m-batch').disabled = false; $('m-batch').textContent = 'Run statistics'; }
};
function renderStats(r) {
  const C = r.controllers;
  $('stat-pane').innerHTML = `<div class="pane-pad"><h3 class="sec">Statistics <small>${r.n} missions · scenario ${esc(r.scenario)} · ${fmt(r.secs, 0)} s${r.id ? ` · saved as ${esc(r.id)}` : ''}</small></h3>
    <div class="rep-grid">
      <div class="tile"><div class="label">IK solved (Level 1)</div><div class="value">${pct(r.ik_success_pct)}</div><div class="sub">median ${fmt(r.ik_pos_err_mm, 3)} mm · ${fmt(r.ik_ms, 0)} ms</div></div>
      <div class="tile"><div class="label">Estimated path blocked</div><div class="value">${pct(r.predicted_conflict_pct)}</div><div class="sub">missions needing re-routing</div></div>
      <div class="tile"><div class="label">Agent reached target</div><div class="value">${pct(C.policy.success_pct)}</div><div class="sub">straight ${pct(C.baseline.success_pct)} · pot. field ${pct(C.apf.success_pct)}</div></div>
      <div class="tile"><div class="label">Agent collisions</div><div class="value">${pct(C.policy.collision_pct)}</div><div class="sub">self ${pct(C.policy.self_collision_pct)} · straight ${pct(C.baseline.collision_pct)}</div></div>
    </div>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Controller</th><th class="num">Reached ↑</th><th class="num">Collisions ↓</th><th class="num">of which self</th>
      <th class="num">Stuck ↓</th><th class="num">Final error (median, reached)</th><th class="num">Steps</th></tr></thead><tbody>
      ${['policy', 'apf', 'baseline'].map((k) => `<tr><td>${NAMES[k]}</td><td class="num">${pct(C[k].success_pct)}</td><td class="num">${pct(C[k].collision_pct)}</td>
        <td class="num">${pct(C[k].self_collision_pct)}</td><td class="num">${pct(C[k].timeout_pct)}</td>
        <td class="num">${fmt(C[k].final_pos_err_mm, 3)} mm / ${fmt(C[k].final_orn_err_deg, 2)}°</td><td class="num">${fmt(C[k].steps, 0)}</td></tr>`).join('')}
    </tbody></table></div>
    <div class="hint">Percentages of controllers are over the missions where IK succeeded. Same seeds for all controllers. Charts in the <b>Plots</b> tab.</div></div>`;
}

// ------------------------------------------------------------------ plots
function renderPlots() {
  const pane = $('plot-pane');
  if (state.batch && state.batch.rows) return renderBatchPlots(state.batch);
  const m = state.m;
  if (!m || !m.episodes || !m.episodes.policy.frames) { pane.innerHTML = '<div class="empty">Run or open a mission to see its plots.</div>'; return; }
  pane.innerHTML = `<div class="pane-pad" style="padding-bottom:0"><h3 class="sec">Plots <small>mission ${esc(m.id || '')} · seed ${m.seed} · ${esc(m.scenario)} · hover a chart for values</small></h3></div>
    <div class="plots"><div id="pl-clear"></div><div id="pl-goal"></div><div id="pl-joints"></div><div id="pl-risk"></div></div>`;
  const ks = ['policy', 'apf', 'baseline'];
  const n = Math.max(...ks.map((k) => m.episodes[k].frames.length));
  const x = Array.from({ length: n }, (_, i) => i);
  const col = (k, f) => x.map((i) => { const fr = m.episodes[k].frames[i]; return fr ? f(fr, i) : null; });
  const goal = m.goal_ee;
  const dist = (fr) => { const p = eeOf(fr.links); return 1000 * Math.hypot(p[0] - goal[0], p[1] - goal[1], p[2] - goal[2]); };
  const a = m.episodes.policy;
  const events = a.events.map((e) => ({ x: e.t, label: e.type === 'joint_slip' ? `slip j${e.joint + 1}` : 'obstacle' }));
  const fa = a.frames.findIndex((f) => f.phase === 'final approach');
  if (fa > 0) events.push({ x: fa, label: 'final approach' });
  const series = (f) => ks.map((k) => ({ name: NAMES[k], color: COL[k], values: col(k, f) }));
  lineChart($('pl-clear'), { title: 'Clearance to the nearest obstacle (mm)', x, series: series((fr) => (fr.clearance == null ? null : 1000 * fr.clearance)),
    refs: [{ y: 0, label: 'contact' }, { y: 80, label: 'safety zone 80 mm' }], events, yMin: -20, unit: ' mm' });
  lineChart($('pl-goal'), { title: 'Gripper distance to the IK goal (mm)', x, series: series(dist), events, yMin: 0, unit: ' mm' });
  const dof = a.frames[0].q.length;
  lineChart($('pl-joints'), { title: 'DRL agent – joint angles (°)', x: a.frames.map((_, i) => i),
    series: Array.from({ length: dof }, (_, j) => ({ name: `j${j + 1}`, color: JOINT_COLS[j], values: a.frames.map((f) => (f.q[j] * 180) / Math.PI) })), events, unit: '°' });
  lineChart($('pl-risk'), { title: 'DRL agent – self-collision risk from the Level 1 classifier (%)', x: a.frames.map((_, i) => i),
    series: [{ name: 'risk', color: COL.policy, values: a.frames.map((f) => (f.risk == null ? null : 100 * f.risk)) }], events, yMin: 0, yMax: 100, unit: '%' });
}
function renderBatchPlots(r) {
  const pane = $('plot-pane');
  pane.innerHTML = `<div class="pane-pad" style="padding-bottom:0"><h3 class="sec">Plots <small>statistics ${esc(r.id || '')} · ${r.n} missions · ${esc(r.scenario)}</small></h3></div>
    <div class="plots three"><div id="pb-out"></div><div id="pb-clear"></div><div id="pb-ik"></div></div>`;
  const ks = ['policy', 'apf', 'baseline'], C = r.controllers;
  stackedBars($('pb-out'), {
    title: 'Outcome per controller (% of missions)',
    legend: [{ name: 'reached', color: STATUS.success }, { name: 'collision', color: STATUS.collision }, { name: 'stuck', color: STATUS.timeout }],
    rows: ks.map((k) => ({ name: NAMES[k].replace(' (Level 2)', ''), parts: [
      { label: 'reached', value: C[k].success_pct, color: STATUS.success },
      { label: 'collision', value: C[k].collision_pct, color: STATUS.collision },
      { label: 'stuck', value: C[k].timeout_pct, color: STATUS.timeout }] })),
  });
  const ok = r.rows.filter((x) => x.ik_success && x.episodes.policy);
  histogram($('pb-clear'), { title: 'Minimum clearance per mission (mm)', lo: -40, hi: 200, bins: 12, xLabel: 'mm', unit: ' mm',
    series: ks.map((k) => ({ name: NAMES[k].replace(' (Level 2)', ''), color: COL[k], values: ok.map((x) => Math.min(199, 1000 * x.episodes[k].min_clearance_m)) })) });
  histogram($('pb-ik'), { title: 'Level 1 IK position error per mission (log₁₀ mm)', lo: -4, hi: 1, bins: 10, xLabel: 'log₁₀ mm',
    series: [{ name: 'IK error', color: COL.policy, values: r.rows.map((x) => Math.log10(Math.max(1e-4, x.ik_pos_err_mm))) }] });
}

// ------------------------------------------------------------------ saved runs
async function refreshSaved() {
  const list = await api('/api/missions');
  const rows = list.map((m) => {
    const res = m.kind === 'single'
      ? `${esc(m.verdict || '')}${m.episodes && m.episodes.policy ? ` · ${m.episodes.policy.steps} steps` : ''}`
      : `agent ${pct(m.controllers.policy.success_pct)} reached / ${pct(m.controllers.policy.collision_pct)} collisions (straight ${pct(m.controllers.baseline.success_pct)})`;
    return `<tr><td class="acts"><button data-act="open" data-id="${esc(m.id)}">Open</button> <button class="del" data-act="del" data-id="${esc(m.id)}">🗑 Delete</button></td>
      <td><span class="id">${esc(m.id)}</span></td><td>${m.kind === 'single' ? 'mission' : 'statistics'}</td><td>${esc(m.robot)}</td><td>${esc(m.scenario)}</td>
      <td class="num">${m.kind === 'single' ? `seed ${m.seed}` : `${m.n} missions`}</td><td>${res}</td>
      <td><span class="id">${esc(m.l1_run || '')}</span><br><span class="id">${esc(m.l2_policy || '')}</span></td>
      <td>${new Date(m.created * 1000).toLocaleString(undefined, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}</td></tr>`;
  }).join('');
  $('saved-pane').innerHTML = `<div class="pane-pad"><div class="toolbar"><span class="grow hint">Every mission and statistics run is saved in <code>ui/workspace/missions/&lt;id&gt;/</code>
    (<code>meta.json</code> summary + <code>mission.json</code> full result, including every frame for replay and plots).</span></div>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Actions</th><th>Run</th><th>Type</th><th>Robot</th><th>Scenario</th><th class="num">Size</th><th>Result</th><th>Models (Level 1 / Level 2)</th><th>Created</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="9" class="muted">Nothing saved yet – run a mission.</td></tr>'}</tbody></table></div></div>`;
}
$('saved-pane').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-act]'); if (!b) return;
  const id = b.dataset.id;
  try {
    if (b.dataset.act === 'open') {
      const m = await api(`/api/missions/get?id=${encodeURIComponent(id)}`);
      if (m.rows) { state.batch = m; renderStats(m); switchTab('stat-pane'); toast(`Opened statistics ${id} – charts in Plots.`); return; }
      state.batch = null;
      await loadRobot(m.robot); state.m = m; showMission(); switchTab('rep-pane');
    } else if (b.dataset.act === 'del') {
      if (!(await confirmDialog('Delete saved run?', `<p>Permanently delete <span class="id">${esc(id)}</span>?</p><p>Its report, replay frames and plots data are removed. Models are not affected.</p>`))) return;
      await api('/api/delete', { kind: 'mission', id });
      toast(`Deleted ${id}.`);
      refreshSaved();
    }
  } catch (err) { toast(err.message, true); }
});
function confirmDialog(title, html) {
  const dlg = $('confirm');
  $('confirm-title').textContent = title; $('confirm-body').innerHTML = html;
  return new Promise((resolve) => {
    const done = (ok) => { dlg.close(); resolve(ok); };
    $('confirm-yes').onclick = () => done(true); $('confirm-no').onclick = () => done(false);
    dlg.oncancel = (ev) => { ev.preventDefault(); done(false); };
    dlg.showModal(); $('confirm-no').focus();
  });
}

await loadModels();
