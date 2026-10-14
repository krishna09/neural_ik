// Builds presentation/Neural_IK_STP_Team06_Level1.pptx (17 slides, 16:9) with pptxgenjs.
// Every number comes from $NIK_WORK/deck/data.json (written by prepare_deck_assets.py from the result files).
// Animations and transitions are added afterwards by postprocess_deck.py.
//
//   NODE_PATH=<dir with pptxgenjs> node deliverables/scripts/deck/build_deck.js
'use strict';
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..', '..', '..');
const DELIV = path.join(ROOT, 'deliverables');
const WORK = process.env.NIK_WORK || '/tmp/nik_work';
const ASSETS = path.join(WORK, 'deck');
const OUT = path.join(DELIV, 'presentation', 'Neural_IK_STP_Team06_Level1.pptx');
const D = JSON.parse(fs.readFileSync(path.join(ASSETS, 'data.json'), 'utf8'));

// ------------------------------------------------------------------------------------------ design tokens
const BG = '0A1527', SURF = '0F1B2D', SURF2 = '14233A', SURF3 = '1A2C47', BORDER = '23344F';
const T1 = 'EEF3FA', T2 = 'AEBBD0', MUTED = '7F8FA8', ACCENT = '4FD6F2';
const BLUE = '3987E5', ORANGE = 'D95926', AQUA = '199E70', YELLOW = 'C98500';
const GOOD = '0CA30C', WARN = 'FAB219', CRIT = 'D03B3B';
const W = 13.333, H = 7.5, M = 0.6, CW = W - 2 * M;
const THEME = { name: 'Neural IK Navy', headFontFace: 'Calibri', bodyFontFace: 'Calibri',
  colors: { dk1: '0A1527', lt1: 'FFFFFF', dk2: '0F1B2D', lt2: 'EEF3FA', accent1: BLUE, accent2: ORANGE, accent3: AQUA,
    accent4: YELLOW, accent5: ACCENT, accent6: CRIT, hlink: ACCENT, folHlink: T2 } };

// decimal rounding, half away from zero (2.55 -> "2.6"): the same rule as the renderer and the charts
function fixed(v, d = 1) {
  const a = Math.abs(v), s = String(a);
  if (s.includes('e')) return v.toFixed(d);
  const r = Number(`${Math.round(Number(`${s}e${d}`))}e-${d}`);
  return (v < 0 ? -r : r).toFixed(d);
}
const fmt = (v, d = 1) => (fixed(Math.abs(v), d) === fixed(0, d) ? fixed(0, d) : fixed(v, d)).replace('-', '−');
const r1 = (v) => Number(fixed(v, 1));                 // chart values, pre-rounded so labels match the text
const pct = (v, d = 1) => `${fmt(v, d)} %`;
const commas = (n) => Math.round(n).toLocaleString('en-US');
const E = (m, mode) => D.eval[m][mode];

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'STP 2026 Team 06';
pres.company = 'STP 2026 Team 06';
pres.subject = 'Neural Kinematics and Reinforcement Learning for Collision-Free Space Manipulator Path Planning';
pres.title = 'Neural Kinematics for Intelligent Space Manipulators';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

// ------------------------------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: 'TITLE', background: { color: BG },
  objects: [],
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: BG },
  objects: [
    { ellipse: { x: M, y: 0.465, w: 0.11, h: 0.11, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 } } },
    { placeholder: { options: { name: 'chapter', type: 'body', x: M + 0.2, y: 0.34, w: 9.5, h: 0.36, fontSize: 12, bold: true,
      color: T2, charSpacing: 3, margin: 0, valign: 'middle', align: 'left' }, text: '' } },
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.72, w: CW, h: 0.8, fontSize: 32, bold: true,
      color: T1, margin: 0, valign: 'middle', align: 'left' }, text: '' } },
    { text: { text: 'STP 2026 · Team 06 · Neural kinematics for space manipulators · simulation results (PyBullet, xArm6)',
      options: { x: M, y: 7.05, w: 10, h: 0.28, fontSize: 9, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.05, w: 0.6, h: 0.28, fontSize: 9, color: MUTED, align: 'right' },
});

// ------------------------------------------------------------------------------------------ helpers
let S;                                  // current slide
const ANIM = [];                        // per slide: object names to reveal in order
function slide(section, chapter, title, notes) {
  if (section) pres.addSection({ title: section });
  S = pres.addSlide({ masterName: 'CONTENT', sectionTitle: section || undefined });
  S.addText(chapter.toUpperCase(), { placeholder: 'chapter' });
  S.addText(title, { placeholder: 'title' });
  S.addNotes(notes);
  ANIM.push([]);
  return S;
}
function anim(name) { ANIM[ANIM.length - 1].push(name); return name; }
function text(t, o) { S.addText(t, { isTextBox: true, margin: 0, fontFace: 'Calibri', color: T2, fontSize: 14, valign: 'top', ...o }); }
function box(x, y, w, h, o = {}) {
  S.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: o.r ?? 0.1, fill: { color: o.fill || SURF },
    line: { color: o.line || BORDER, width: o.lw ?? 1 }, objectName: o.name });
}
// card: rounded box + optional kicker, heading and body, grouped under one animation name
// "• a\n• b" -> real bulleted paragraphs (hanging indent on wrapped lines)
const bullets = (s) => (s.startsWith('• ') ? s.split('\n').map((l, i, a) =>
  ({ text: l.replace(/^• /, ''), options: { bullet: { indent: 12 }, breakLine: i < a.length - 1 } })) : s);
function card(x, y, w, h, { kicker, head, body, name, kColor = ACCENT, fill, line, headSize = 17, bodySize = 13.5 } = {}) {
  const n = name ? anim(name) : undefined;
  box(x, y, w, h, { fill, line, name: n && `${n}#box` });
  let yy = y + 0.2;
  if (kicker) { text(kicker, { x: x + 0.22, y: yy, w: w - 0.44, h: 0.25, fontSize: 10.5, bold: true, color: kColor, charSpacing: 2, objectName: n && `${n}#k` }); yy += 0.32; }
  if (head) { text(head, { x: x + 0.22, y: yy, w: w - 0.44, h: 0.42, fontSize: headSize, bold: true, color: T1, objectName: n && `${n}#h` }); yy += headSize > 16 ? 0.46 : headSize > 14 ? 0.4 : 0.32; }
  if (body) text(bullets(body), { x: x + 0.22, y: yy, w: w - 0.44, h: Math.max(0.3, y + h - yy - 0.12), fontSize: bodySize, color: T2,
    paraSpaceAfter: 4, objectName: n && `${n}#b` });
}
function stat(x, y, w, big, label, { color = T1, size = 30, name } = {}) {
  const n = name ? anim(name) : undefined;
  text(big, { x, y, w, h: 0.55, fontSize: size, bold: true, color, objectName: n && `${n}#v` });
  text(label, { x, y: y + 0.56, w, h: 0.5, fontSize: 12, color: T2, objectName: n && `${n}#l` });
}
function badge(x, y, w, label, kind = 'ok', name) {
  const c = { ok: [GOOD, '8FF08F', '✓ '], warn: [WARN, 'FFD47A', '! '], bad: [CRIT, 'FF9C9C', '✕ '], info: [ACCENT, 'BFF2FC', ''] }[kind];
  S.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, rectRadius: 0.16, fill: { color: SURF2 }, line: { color: c[0], width: 1 }, objectName: name && `${name}#bb` });
  text(c[2] + label, { x, y, w, h: 0.32, fontSize: 11, bold: true, color: c[1], align: 'center', valign: 'middle', objectName: name && `${name}#bt` });
}
const SCENE_OF = { level2_ppo_rerouting: 'level2_ppo' };
function posterPath(name) {
  const p = path.join(ASSETS, 'posters', `${name}.png`);
  if (fs.existsSync(p)) return p;
  const d = path.join(WORK, 'stills', SCENE_OF[name] || name);           // layout preview before the clip exists
  const f = fs.existsSync(d) ? fs.readdirSync(d).filter((x) => x.endsWith('.png')).sort().pop() : null;
  if (!f) throw new Error(`no poster for ${name}`);
  console.warn(`  preview: ${name} uses still ${f}`);
  return path.join(d, f);
}
function poster(name) { return 'data:image/png;base64,' + fs.readFileSync(posterPath(name)).toString('base64'); }
function video(name, x, y, w, caption) {
  const h = w * 9 / 16;
  const n = anim(`video_${name}`);
  box(x - 0.03, y - 0.03, w + 0.06, h + 0.06, { r: 0.04, fill: BG, line: BORDER, name: `${n}#frame` });
  const clip = path.join(DELIV, 'simulations', `${name}.mp4`);
  if (fs.existsSync(clip)) S.addMedia({ type: 'video', path: clip, cover: poster(name), x, y, w, h, objectName: `${n}#media` });
  else S.addImage({ path: posterPath(name), x, y, w, h, objectName: `${n}#media` });
  if (caption) text(caption, { x, y: y + h + 0.1, w, h: 0.3, fontSize: 10.5, color: MUTED, objectName: `${n}#cap` });
  return h;
}
const chartText = { catAxisLabelColor: T2, catAxisLabelFontSize: 11, catAxisLabelFontFace: '+mn-lt', valAxisLabelColor: MUTED,
  valAxisLabelFontSize: 10, valAxisLabelFontFace: '+mn-lt', legendColor: T2, legendFontSize: 11, legendFontFace: '+mn-lt',
  titleColor: T1, titleFontSize: 13, titleFontFace: '+mn-lt', dataLabelColor: T1, dataLabelFontSize: 10, dataLabelFontFace: '+mn-lt' };
const chartFrame = () => ({ chartArea: { fill: { color: SURF }, border: { color: BORDER, pt: 1 }, roundedCorners: false },
  plotArea: { fill: { color: SURF } }, valGridLine: { color: '1E2E47', size: 0.75 }, catGridLine: { style: 'none' } });

// ================================================================================================== slides
const sc = D.level2.team.scenarios.combined;
const repro = D.level2.repro;
const hz = (m) => E(m, 'zero'), hw = (m) => E(m, 'warm');

// 1 · title -------------------------------------------------------------------------------------------
pres.addSection({ title: 'Introduction' });
S = pres.addSlide({ masterName: 'TITLE', sectionTitle: 'Introduction' });
ANIM.push([]);
S.addImage({ path: path.join(ASSETS, 'hero.png'), x: 0, y: 0, w: W, h: H, objectName: 'hero' });
text('STP 2026 · TEAM 06', { x: 0.75, y: 1.95, w: 6, h: 0.35, fontSize: 14, bold: true, color: ACCENT, charSpacing: 4, objectName: anim('t_kicker') });
text('Neural Kinematics for Intelligent Space Manipulators', { x: 0.75, y: 2.35, w: 6.4, h: 2.05, fontSize: 38, bold: true, color: T1, valign: 'top', lineSpacingMultiple: 0.95, objectName: anim('t_title') });
text('Design, implementation and simulation-based evaluation of a collision-aware neural inverse-kinematics framework',
  { x: 0.75, y: 4.55, w: 5.9, h: 0.9, fontSize: 16, color: T2, objectName: anim('t_sub') });
const tg = anim('t_tags');
[['Level 1 · complete', 1.75], ['Level 2 · implemented in simulation', 2.95], ['UFactory xArm6 · PyBullet', 2.2]].reduce((x, [lbl, w]) => {
  S.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.65, w, h: 0.38, rectRadius: 0.19, fill: { color: SURF2 }, line: { color: BORDER, width: 1 }, objectName: `${tg}#${lbl}#b` });
  text(lbl, { x, y: 5.65, w, h: 0.38, fontSize: 11.5, color: T1, align: 'center', valign: 'middle', objectName: `${tg}#${lbl}#t` });
  return x + w + 0.15;
}, 0.75);
text('Neural Kinematics and Reinforcement Learning for Collision-Free Space Manipulator Path Planning',
  { x: 0.75, y: 6.85, w: 9, h: 0.3, fontSize: 10, color: MUTED });
S.addNotes(`Opening (about 45 s).
Good morning. We are STP Team 06. Our project is Neural Kinematics and Reinforcement Learning for Collision-Free Space Manipulator Path Planning.
In this talk we show what we actually built and measured: a complete Level 1 pipeline that turns a target pose into collision-free joint angles, and a Level 2 reinforcement-learning agent that re-routes the arm around unexpected obstacles. Everything runs in PyBullet simulation on a UFactory xArm6, a six-joint arm standing in for the space manipulator.
All numbers in this deck come from result files in deliverables/results (re-measured for this presentation) or from the team's recorded runs, which are cited where they appear.`);

// 2 · problem ------------------------------------------------------------------------------------------
slide(null, 'Introduction · Problem', 'Reach any pose: precisely, legally, safely and fast', `Problem statement (about 1 min).
Inverse kinematics asks: given where the gripper must be, and how it must be oriented, which six joint angles get it there?
For a space manipulator the answer must also be legal (inside every joint limit) and safe (the arm must never hit itself), and it has to come fast enough for real-time use.
The project brief splits the work into two levels. Level 1: a kinematics-aware AI model that computes joint angles and prevents self-collision. Level 2: a lightweight deep-reinforcement-learning agent that re-routes the arm in real time when obstacles or hardware position errors appear.
Both levels are implemented in simulation. The mentors approved using a standard arm, because the real manipulator's model is not available yet.
Source: Project Brief - Team 6.pdf; Level 1 Implementation Plan PDF.`);
text('Given a target pose for the gripper (position + orientation), find the six joint angles that reach it.',
  { x: M, y: 1.8, w: 5.7, h: 0.75, fontSize: 17, color: T1, objectName: anim('p_q') });
[['Precise', 'within 1 mm and 1° of the target'], ['Legal', 'inside every joint limit'], ['Safe', 'no self-collision, ever'],
  ['Fast', 'milliseconds, for real-time control']].forEach(([k, v], i) => {
  const n = anim(`p_c${i}`);
  const y = 2.8 + i * 0.86;
  S.addShape(pres.shapes.OVAL, { x: M, y: y + 0.06, w: 0.46, h: 0.46, fill: { color: SURF2 }, line: { color: ACCENT, width: 1.25 }, objectName: `${n}#o` });
  text(String(i + 1), { x: M, y: y + 0.06, w: 0.46, h: 0.46, fontSize: 14, bold: true, color: ACCENT, align: 'center', valign: 'middle', objectName: `${n}#n` });
  text([{ text: k + '  ', options: { bold: true, color: T1 } }, { text: v, options: { color: T2 } }],
    { x: M + 0.65, y: y + 0.08, w: 5.0, h: 0.42, fontSize: 15, valign: 'middle', objectName: `${n}#t` });
});
card(6.75, 1.8, CW - 6.15, 2.3, { kicker: 'LEVEL 1', head: 'Kinematics-aware AI model', name: 'p_l1',
  body: 'Use simulated manipulator joint data to compute joint angles for a target pose while preventing self-collision.' });
badge(6.97, 3.62, 1.55, 'Implemented', 'ok', 'p_l1b');
card(6.75, 4.3, CW - 6.15, 2.3, { kicker: 'LEVEL 2', head: 'Lightweight DRL policy agent', name: 'p_l2',
  body: 'Re-route the manipulator in real time when unexpected obstacles or hardware position errors occur.' });
badge(6.97, 6.12, 2.6, 'Implemented in simulation', 'ok', 'p_l2b');
text('Robot used: UFactory xArm6 (6 revolute joints). The space manipulator drops in as one new entry in the robot registry.',
  { x: M, y: 6.45, w: 5.8, h: 0.5, fontSize: 11.5, color: MUTED });

// 3 · FK / IK -------------------------------------------------------------------------------------------
slide(null, 'Fundamentals', 'Forward is easy; inverse is the challenge', `Fundamentals (about 1 min).
Forward kinematics, left: chain the six joint rotations and you get exactly one pose. PyBullet computes this with getLinkState; our renderer uses the same chain and matches PyBullet to 0.005 mm.
Inverse kinematics goes the other way. There is no general closed-form answer, and for the target in the clip our solver found ${D.ik_branches} different joint configurations that all reach it exactly; four are shown.
That is why classical IK solvers iterate from a starting guess, and why our network also takes a seed: the arm's current joints decide which solution we want.
Clip: simulations/inverse_kinematics.mp4 (Simulation 3). Code: robot_config.py, refine/lm_solver.py (random_candidates).`);
card(M, 1.8, 5.5, 1.95, { kicker: 'FORWARD KINEMATICS · q → (p, R)', head: 'One exact answer', name: 'f_fk',
  body: 'T(q) = T₀ · T₁(q₁) · T₂(q₂) ··· T₆(q₆). Always defined; PyBullet returns the end-effector pose with getLinkState.' });
card(M, 3.95, 5.5, 2.45, { kicker: 'INVERSE KINEMATICS · (p, R) → q', head: 'Several answers, no closed form', name: 'f_ik', kColor: WARN,
  body: `${D.ik_branches} joint solutions found for the target in the clip; solved iteratively from a starting guess; classical solvers ignore joint limits and self-collision unless told.` });
video('inverse_kinematics', 6.45, 1.8, CW - 5.85, 'Simulation 3 · one target pose, four of its joint solutions; the seed selects one');
text('This is why our network takes a seed: the arm\'s current joints pick one solution.', { x: M, y: 6.55, w: 5.6, h: 0.4, fontSize: 13, color: T1, objectName: anim('f_note') });

// 4 · approach -------------------------------------------------------------------------------------------
slide('Level 1 · Method', 'Level 1 · Approach', 'From plan to a working, reproducible pipeline', `Development approach (about 1 min).
We followed the Level 1 implementation plan step by step, and each step is one script in the repository.
1 robot setup with joints resolved by name; 2 dataset generation; 3 the collision classifier; 4 the IK network with the collision-aware loss; 5 the benchmark against PyBullet's classical solver; 6 Levenberg-Marquardt refinement, added after the benchmark showed that a network alone stays tens of millimetres off.
Beyond the plan: a deeper residual network, a web Studio that runs every step, and Level 2.
Reproducibility: the README recipe was re-run unchanged for this presentation; classical DLS reproduced exactly (5.07 mm, 9 % self-collision) and the neural baseline within noise (123 mm vs 128 mm in the README).`);
const steps = [['Robot setup', 'robot_config.py', 'joints by name, self-collision on'], ['Dataset', 'generate_dataset.py', 'pose + collision label per sample'],
  ['Collision classifier', 'train_ik.py', `6 → 64 → 64 → 1, ${fmt(D.classifier.accuracy, 1)} % accurate`], ['Neural IK', 'train_ik.py · modelset/', 'collision-aware loss'],
  ['Benchmark', 'evaluate.py', 'vs classical DLS and hybrid'], ['Refinement', 'refine/lm_solver.py', 'below 1 mm and 1°']];
const sw = (CW - 5 * 0.2) / 6;
steps.forEach(([t, f, d], i) => {
  const x = M + i * (sw + 0.2), n = anim(`a_s${i}`);
  box(x, 1.85, sw, 2.75, { name: `${n}#box` });
  S.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 2.05, w: 0.5, h: 0.5, fill: { color: SURF3 }, line: { color: ACCENT, width: 1.25 }, objectName: `${n}#o` });
  text(String(i + 1), { x: x + 0.2, y: 2.05, w: 0.5, h: 0.5, fontSize: 15, bold: true, color: ACCENT, align: 'center', valign: 'middle', objectName: `${n}#n` });
  text(t, { x: x + 0.2, y: 2.68, w: sw - 0.32, h: 0.56, fontSize: 14.5, bold: true, color: T1, valign: 'bottom', objectName: `${n}#t` });
  text(f, { x: x + 0.2, y: 3.28, w: sw - 0.32, h: 0.28, fontSize: 10.5, color: ACCENT, objectName: `${n}#f` });
  text(d, { x: x + 0.2, y: 3.56, w: sw - 0.32, h: 0.5, fontSize: 11.5, color: T2, objectName: `${n}#d` });
  badge(x + 0.2, 4.12, 0.95, 'done', 'ok', `${n}#ok`);
});
text('Beyond the original plan', { x: M, y: 4.95, w: 6, h: 0.32, fontSize: 13, bold: true, color: T1, objectName: anim('a_beyond') });
[['2.05 M-parameter residual network', 3.0], ['LM refinement with collision-free retry', 3.3], ['Neural IK Studio (web app)', 2.55], ['Level 2 agent + Level 1 + 2 missions', 3.05]]
  .reduce((x, [lbl, w], i) => {
    const n = anim(`a_x${i}`);
    S.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.38, w, h: 0.44, rectRadius: 0.22, fill: { color: SURF2 }, line: { color: ACCENT, width: 1 }, objectName: `${n}#b` });
    text(lbl, { x, y: 5.38, w, h: 0.44, fontSize: 12, color: T1, align: 'center', valign: 'middle', objectName: `${n}#t` });
    return x + w + 0.08;
  }, M);
text('Plan: "Level 1 Implementation Plan — Neural IK for 6/7-DOF Manipulators" (PDF in the repository).', { x: M, y: 6.3, w: 9, h: 0.3, fontSize: 11, color: MUTED });

// 5 · technology --------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Tools', 'Technology and simulation environment', `Technology (about 45 s).
PyBullet is the simulator: forward kinematics, Jacobians and the contact checks that label self-collision. It runs headless in DIRECT mode.
PyTorch trains both networks and also runs our batched Levenberg-Marquardt solver. NumPy and Matplotlib handle data and figures; the web Studio uses Three.js to show the robot from PyBullet's link frames.
The right column says where each result was computed. The team's best Level 1 network trained in 719 s on an NVIDIA RTX A6000; the Level 2 agent trained for 20 million steps in 134 minutes with GPU updates. For this presentation, everything was re-run on a 4-core CPU with no GPU: Python 3.13, PyTorch 2.14.1, PyBullet 3.2.7.`);
const tech = [['PyBullet 3.2.7', 'Physics engine: FK, Jacobians, self-contact checks (headless)'], ['PyTorch 2.14', 'Networks, training, batched LM solver'],
  ['NumPy · Matplotlib', 'Data handling and figures'], ['Python 3.13', 'Every pipeline step is a script'],
  ['Three.js Studio', 'Web app: generate, train, evaluate, infer, Level 2'], ['Docker', 'Offline image for the team PCs']];
tech.forEach(([t, d], i) => {
  const x = M + (i % 3) * 2.42, y = 1.85 + Math.floor(i / 3) * 2.2;
  card(x, y, 2.27, 2.0, { head: t, body: d, name: `t_${i}`, headSize: 16, bodySize: 12.5 });
});
card(8.05, 1.85, CW - 7.45, 4.2, { kicker: 'WHERE THE RESULTS CAME FROM', head: 'Hardware', name: 't_hw', headSize: 16, bodySize: 12.5,
  body: `Team, Level 1 best run: ${D.best_run.gpu}, ${commas(D.best_run.train_secs)} s training.\n\nTeam, Level 2 agent: ${commas(D.level2.steps / 1e6)} M PPO steps, ${Math.round(D.level2.train_secs / 60)} min (GPU updates, 80 simulated arms on CPU workers).\n\nThis presentation: everything re-measured on a 4-core CPU, no GPU.` });
text('Simulation only: PyBullet moves the arm kinematically (no dynamics); obstacles in Level 2 are spheres.', { x: M, y: 6.3, w: 11, h: 0.3, fontSize: 11, color: MUTED });

// 6 · robot -------------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Robot', 'Building the xArm6 digital robot', `The robot (about 1 min).
The xArm6 comes from its URDF: meshes, six revolute joints and their limits, shown in the table.
Two details mattered. First, the URDF starts with a fixed world_joint at index 0, so assuming joints 0 to 5 silently shifts every index by one. robot_config.py resolves joints by name instead, which is also what lets a new robot drop in as one registry entry.
Second, PyBullet skips self-collision checks unless the URDF is loaded with URDF_USE_SELF_COLLISION; EXCLUDE_PARENT stops neighbouring links from always counting as touching.
The pose we control is the frame of link 6: a position and a quaternion.
Clip: simulations/robot_intro.mp4 (Simulation 1).`);
video('robot_intro', M, 1.85, 6.9, 'Simulation 1 · each joint highlighted and swept inside its limits; end-effector frame at the end');
const roles = ['base rotation', 'shoulder', 'elbow', 'forearm roll', 'wrist pitch', 'flange roll'];
const rows = [[{ text: 'Joint', options: { bold: true, color: T1 } }, { text: 'Role', options: { bold: true, color: T1 } }, { text: 'Limits (°)', options: { bold: true, color: T1, align: 'right' } }]]
  .concat(roles.map((r, i) => [{ text: `J${i + 1}` }, { text: r }, { text: `${D.limits_deg[i][0]} … ${D.limits_deg[i][1]}`.replace(/-/g, '−'), options: { align: 'right' } }]));
S.addTable(rows, { x: 7.85, y: 1.85, w: CW - 7.25, colW: [0.8, 2.2, 1.88], fontSize: 13, color: T2, fontFace: 'Calibri', rowH: 0.36,
  fill: { color: SURF }, border: { type: 'solid', pt: 0.75, color: BORDER }, margin: [0, 0.1, 0, 0.1], valign: 'middle', objectName: anim('r_table') });
[['Joints resolved by name:', ' the URDF\'s fixed world_joint at index 0 would shift every index'],
  ['Self-collision enabled:', ' URDF_USE_SELF_COLLISION + EXCLUDE_PARENT'], ['End effector:', ' link 6 frame, pose = position + quaternion']].forEach(([a, b], i) => {
  text([{ text: a, options: { bold: true, color: T1 } }, { text: b }], { x: 7.85, y: 4.55 + i * 0.6, w: CW - 7.25, h: 0.55, fontSize: 12.5, objectName: anim(`r_f${i}`) });
});

// 7 · dataset ------------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Data', `${commas(D.dataset.n)} labelled samples, generated in simulation`, `Dataset (about 1 min).
No external dataset is needed: generate_dataset.py samples random joint angles, 70 % uniformly inside the limits and 30 % close to the limits where self-collisions concentrate.
For each sample PyBullet computes the end-effector pose and checks every link pair for contact; the result is one row: six angles, a position, a quaternion and a collision label.
The main dataset has ${commas(D.dataset.n)} rows, of which ${commas(D.dataset.n_colliding)} (${fmt(100 * D.dataset.n_colliding / D.dataset.n, 1)} %) are self-colliding; the clip ends with 7,000 of their end-effector positions: the red cluster is near the base, where the wrist can hit the body.
Re-checking the first 120 rows in PyBullet gives the same labels (100 % agreement).
The IK network is trained only on collision-free rows; the colliding rows train the classifier.`);
stat(M, 1.95, 4.2, commas(D.dataset.n), 'samples (main dataset, generator seed 0)', { name: 'd_n' });
stat(M, 3.05, 4.2, pct(100 * D.dataset.n_colliding / D.dataset.n), `self-colliding (${commas(D.dataset.n_colliding)} rows)`, { name: 'd_c', color: 'FF9C9C' });
stat(M, 4.15, 4.2, '70 / 30', 'uniform / near-limit sampling', { name: 'd_s' });
stat(M, 5.25, 4.2, `${Math.round(D.dataset.secs)} s`, 'to generate all 200,000 (team run)', { name: 'd_t' });
video('dataset_generation', 5.15, 1.85, CW - 4.55, 'Simulation 4 · real dataset rows: sample → FK pose → contact check → row; then 7,000 end-effector positions');

// 8 · representation ------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Representation', 'What the network sees and what it predicts', `Representation (about 1 min).
Input: 13 numbers. The target position (3), its orientation as a quaternion (4), and a seed of 6 joint angles. In training, the seed is the true answer plus Gaussian noise of 0.3 radians, standing in for the arm's current joints.
Output: 12 numbers, the sine and cosine of each joint angle. On a circle, plus and minus 180 degrees are neighbours, so the training targets have no artificial jump; atan2 turns each pair back into an angle.
Because atan2 returns -180 to +180 degrees while joint 3 runs from -225 to +11, each angle is then shifted by 360 degrees into the joint range when possible: same pose, legal value.
Code: train_ik.load_dataset, IKNet.decode_angles, refine/lm_solver.wrap_to_limits.`);
const PITCH = 0.56;
const grp = (x, y, labels, color, title, name) => {
  const n = anim(name);
  text(title, { x, y, w: Math.max(1.6, PITCH * labels.length + 0.3), h: 0.28, fontSize: 11, bold: true, color, objectName: `${n}#t` });
  labels.forEach((l, i) => {
    S.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + i * PITCH, y: y + 0.32, w: PITCH - 0.06, h: 0.5, rectRadius: 0.08, fill: { color: SURF2 }, line: { color, width: 1.25 }, objectName: `${n}#b${i}` });
    text(l, { x: x + i * PITCH, y: y + 0.32, w: PITCH - 0.06, h: 0.5, fontSize: 11.5, color: T1, align: 'center', valign: 'middle', objectName: `${n}#l${i}` });
  });
};
text('Input X · 13 numbers', { x: M, y: 1.85, w: 6, h: 0.35, fontSize: 16, bold: true, color: T1, objectName: anim('x_h') });
grp(M, 2.25, ['x', 'y', 'z'], ACCENT, 'position', 'x_p');
grp(M + 1.9, 2.25, ['qx', 'qy', 'qz', 'qw'], ACCENT, 'orientation (quaternion)', 'x_o');
grp(M + 4.36, 2.25, ['s1', 's2', 's3', 's4', 's5', 's6'], WARN, 'seed: current joint angles', 'x_s');
text('Output y · 12 numbers', { x: M, y: 3.45, w: 6, h: 0.35, fontSize: 16, bold: true, color: T1, objectName: anim('y_h') });
grp(M, 3.85, ['sin1', 'sin2', 'sin3', 'sin4', 'sin5', 'sin6'], AQUA, 'sine of each joint angle', 'y_s');
grp(M + 3.66, 3.85, ['cos1', 'cos2', 'cos3', 'cos4', 'cos5', 'cos6'], AQUA, 'cosine of each joint angle', 'y_c');
card(M, 5.15, 7.7, 1.45, { head: 'Decode: q = atan2(sin, cos), then shift by ±360° into the joint limits', name: 'x_dec', headSize: 15, bodySize: 12.5,
  body: 'Seed in training: true joints + N(0, 0.3 rad). IK rows: collision-free only. Same pose, legal joint value (xArm6 joint 3 spans −225…+11°).' });
S.addImage({ path: path.join(ASSETS, 'circle.png'), x: 8.75, y: 1.85, w: 3.98, h: 3.98 * 560 / 760, objectName: anim('x_circle') });
text('From Simulation 5: as the angle crosses 180°, the written number jumps but (sin, cos) moves smoothly.', { x: 8.75, y: 4.88, w: 3.98, h: 0.6, fontSize: 11, color: MUTED });

// 9 · architecture --------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Model', 'Neural IK architecture', `Architecture (about 1 min).
The clip shows the baseline network from the plan, with its real activations for one held-out target: 13 inputs, three hidden layers of 256, 256 and 128 ReLU units, and 12 outputs.
The best network keeps exactly the same inputs, outputs and training loop, but has a deeper residual body: 2,050,316 parameters instead of 103,820.
On the target in the clip the baseline lands 92 mm away; the best network 16.7 mm. Over the whole held-out test split the best network averages ${fmt(D.full_test.pos_mean_mm, 2)} mm, re-measured here and identical to the recorded run.
Code: train_ik.IKNet (baseline), modelset/custom/model_set.py (best). Clip: simulations/neural_architecture.mp4 (Simulation 5).`);
video('neural_architecture', M, 1.85, 7.2, 'Simulation 5 · real activations of the baseline network, decoding, and why sine/cosine');
card(8.1, 1.85, CW - 7.5, 2.1, { kicker: 'BASELINE · train_ik.IKNet', head: '13 → 256 → 256 → 128 → 12', name: 'm_base', kColor: 'E9B440',
  body: `ReLU · ${commas(D.params.baseline_mlp)} parameters · the plan's architecture` });
card(8.1, 4.15, CW - 7.5, 2.45, { kicker: 'BEST · modelset/custom/model_set.py', head: 'Residual MLP, 512 wide', name: 'm_best', kColor: '45C99A',
  body: `256 → 512 → 3 residual blocks (LayerNorm, SiLU / GELU / LeakyReLU, dropout 0.05) → 384 → 256 → 128 → 12 · ${commas(D.params.residual_mlp)} parameters · same inputs, outputs and loop` });

// 10 · collision ----------------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Safety', 'Collision-aware learning', `Collision awareness (about 1 min).
A small classifier, 6 inputs, two hidden layers of 64, one output, learns to predict self-collision from the joint angles alone. On ${commas(D.classifier.n)} held-out poses it is ${fmt(D.classifier.accuracy, 1)} % accurate, with ${fmt(D.classifier.precision, 1)} % precision and ${fmt(D.classifier.recall, 1)} % recall (re-measured here).
In the clip the arm leans forward until its wrist touches the base: PyBullet reports contact at 80 % of the motion, exactly where the classifier's probability crosses 0.5.
During IK training the classifier is frozen and its probability for the predicted angles is added to the loss with weight 0.05. Because it is differentiable, it pushes the network away from risky poses during training, not only afterwards.
It is a soft penalty, not a guarantee: every final answer is still checked in PyBullet. Code: train_ik.train_ik_net.`);
card(M, 1.85, 4.7, 2.0, { kicker: 'TRAINING LOSS', head: 'L = MSE(sin, cos) + 0.05 · σ(C(q̂))', name: 'c_loss', headSize: 18,
  body: 'C = frozen collision classifier, q̂ = predicted angles. The penalty is differentiable, so it shapes training.' });
stat(M, 4.1, 2.3, pct(D.classifier.accuracy), `accuracy · ${commas(D.classifier.n)} held-out poses`, { name: 'c_acc' });
stat(M + 2.45, 4.1, 2.25, pct(D.classifier.precision), 'precision', { name: 'c_prec' });
stat(M, 5.25, 2.3, pct(D.classifier.recall), 'recall', { name: 'c_rec' });
stat(M + 2.45, 5.25, 2.25, commas(D.params.collision_net), 'parameters (6 → 64 → 64 → 1)', { name: 'c_par' });
video('collision_awareness', 5.6, 1.85, CW - 5.0, 'Simulation 6 · PyBullet contact vs classifier probability along a real motion');
badge(5.6, 6.35, 4.6, 'Soft penalty: answers are still re-checked in PyBullet', 'warn', 'c_warn');

// 11 · training -----------------------------------------------------------------------------------------------------
slide('Level 1 · Results', 'Level 1 · Training', 'Training and validation', `Training (about 1 min).
Both networks are trained by the same train_ik loop: Adam, learning rate 0.001, batch 256, a random 85/15 train/validation split, 15 classifier epochs then the IK network.
The chart shows the mean validation position error after every epoch, checked with forward kinematics. The baseline (20,000 samples, 60 epochs, our CPU reproduction in 26 seconds) levels off near ${Math.round(D.baseline_repro.final.val_pos_mean_mm)} mm. The residual network (200,000 samples, 120 epochs, the team's run on an RTX A6000 in ${commas(D.best_run.train_secs)} s) reaches ${fmt(D.curves.best_val_mm[D.curves.best_val_mm.length - 1], 1)} mm.
Training and validation errors fall together for most of training. Over the last epochs the residual network's validation error stops improving (best ${fmt(Math.min(...D.curves.best_val_mm), 1)} mm at epoch ${D.curves.best_val_mm.indexOf(Math.min(...D.curves.best_val_mm)) + 1}, ${fmt(D.curves.best_val_mm[D.curves.best_val_mm.length - 1], 1)} mm at the end) while its training loss keeps falling: more epochs would not help, more data or regularisation might. Both histories are recorded, not reconstructed.
Sources: results/training_history.csv; ui/workspace/library/runs/${D.best_run.id}/meta.json.`);
const ep = Array.from({ length: 120 }, (_, i) => String(i + 1));
S.addChart(pres.charts.LINE, [
  { name: 'Baseline MLP · 20 k samples', labels: ep, values: D.curves.baseline_val_mm.concat(Array(60).fill(null)) },
  { name: 'Residual MLP · 200 k samples', labels: ep, values: D.curves.best_val_mm },
], { x: M, y: 1.85, w: 7.7, h: 4.75, ...chartFrame(), ...chartText, chartColors: ['E9B440', '45C99A'], lineSize: 2.25, lineDataSymbol: 'none',
  valAxisLogScaleBase: 10, valAxisMinVal: 10, valAxisMaxVal: 1000, valAxisTitle: 'mm (log scale)', showValAxisTitle: true, valAxisTitleColor: MUTED,
  valAxisTitleFontSize: 10, catAxisLabelFrequency: '20', catAxisTitle: 'epoch', showCatAxisTitle: true, catAxisTitleColor: MUTED, catAxisTitleFontSize: 10,
  showLegend: true, legendPos: 't', showTitle: true, title: 'Mean validation position error per epoch (FK-checked)', objectName: anim('tr_chart') });
stat(8.65, 1.95, 4.1, `${Math.round(D.baseline_repro.final.val_pos_mean_mm)} → ${fmt(D.curves.best_val_mm[D.curves.best_val_mm.length - 1], 1)} mm`, 'mean validation error, baseline → residual network', { name: 'tr_s1', size: 28 });
stat(8.65, 3.1, 4.1, '10× data · 20× parameters', '200 k vs 20 k samples; 2.05 M vs 0.10 M parameters', { name: 'tr_s2', size: 22 });
card(8.65, 4.3, CW - 8.05, 2.3, { kicker: 'SET-UP', name: 'tr_setup', bodySize: 12.5,
  body: 'Adam · lr 0.001 · batch 256 · 85 / 15 split\nClassifier: 15 epochs, BCE loss\nIK: 60 epochs (baseline), 120 (residual)\nBaseline 26 s on a CPU · residual 719 s on an RTX A6000' });

// 12 · inference workflow ------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Inference', 'Inference: guess, polish, check', `Inference workflow (about 1 min).
One target is solved in five steps. The network makes a guess in about a millisecond on one CPU thread (${fmt(D.timing.network_ms, 2)} ms measured), typically about two centimetres off.
The guess is shifted into the joint limits, then refined with Levenberg-Marquardt: the step solves (JᵀJ + λI) Δq = Jᵀe with λ = |e|² + 10⁻⁶, from the network output and from the seed, for up to 30 iterations; targets still outside 1 mm and 1 degree get 16 random restarts.
PyBullet then checks the result independently: the pose error, the joint limits and self-collision. If the solution collides, another IK branch is searched from fresh random starts and the first one PyBullet confirms is kept.
Measured on 2,000 held-out targets: 100 % within 1 mm and 1 degree, ${fmt(hw('hybrid_lm').time_ms_mean, 1)} ms per target from a warm start on one CPU thread.
Code: ui/server.py infer(), ui/runners/ik_metrics.refine_ik, refine/lm_solver.py.`);
const flow = [['Target pose + seed', '13 numbers'], ['Neural network', `≈ ${fmt(D.timing.network_ms, 0)} ms · cm-level guess`], ['Wrap into limits', 'shift by ±360°, same pose'],
  ['LM refinement', 'Δq = (JᵀJ + λI)⁻¹ Jᵀe'], ['PyBullet check', 'pose · limits · self-collision']];
const fw = 2.12, fg = (CW - 5 * fw) / 4;
flow.forEach(([t, d], i) => {
  const x = M + i * (fw + fg), n = `i_f${i}`;
  card(x, 2.0, fw, 1.55, { head: t, body: d, headSize: 15, bodySize: 12.5, fill: i === 3 ? SURF3 : SURF, line: i === 3 ? BLUE : BORDER, name: n });
  if (i < 4) S.addShape(pres.shapes.LINE, { x: x + fw + 0.04, y: 2.78, w: fg - 0.08, h: 0, line: { color: ACCENT, width: 1.75, endArrowType: 'triangle' }, objectName: `${n}#arrow` });
});
badge(10.95, 3.75, 1.78, 'accept', 'ok', 'i_acc');
text('collision → search another IK branch from random starts', { x: 7.25, y: 3.75, w: 3.6, h: 0.32, fontSize: 11, color: 'FFD47A', align: 'right', valign: 'middle', objectName: anim('i_retry') });
stat(M, 4.55, 3.8, '100 %', 'of 2,000 held-out targets within 1 mm and 1°, any starting pose', { name: 'i_s1', color: '7DB4F5' });
stat(M + 4.1, 4.55, 3.8, `${fmt(hw('hybrid_lm').time_ms_mean, 1)} ms`, 'per target from a warm start (one CPU thread)', { name: 'i_s2' });
stat(M + 8.2, 4.55, 3.9, `${fmt(parseFloat(String(E('hybrid_lm', 'warm').notes).match(/batched ([\d.]+)/)[1]), 2)} ms`, 'per target when 2,000 are solved together (CPU)', { name: 'i_s3' });
text('Numbers: results/evaluation_metrics.csv (hybrid_lm rows).', { x: M, y: 6.3, w: 8, h: 0.3, fontSize: 11, color: MUTED });

// 13 · demonstration --------------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Demonstration', 'The robot reaching held-out targets', `Demonstration (about 1 min).
Three held-out test poses. For each, the green ghost is the network's guess, which misses by ${D.inference.map((c) => fmt(c.net_mm, 0)).join(', ')} millimetres; the solid arm then executes the refined answer, which PyBullet confirms below 0.001 mm, without self-collision and inside the joint limits.
The input panel shows the actual 13 numbers given to the network for each target.
Clip: simulations/inference_demo.mp4 (Simulation 8). Data: export_sim_data.py inference.`);
video('inference_demo', M, 1.85, 8.3, 'Simulation 8 · network guess (green ghost), LM refinement, PyBullet acceptance checks');
D.inference.forEach((c, i) => card(9.2, 1.85 + i * 1.45, CW - 8.6, 1.3, { kicker: `TARGET ${i + 1}`, name: `dm_${i}`, headSize: 15,
  head: `${fmt(c.net_mm, 1)} mm → < 0.001 mm`, body: `network alone → after LM refinement (${fmt(c.net_deg, 1)}° → ${fmt(c.lm_deg, 2)}°)` }));
badge(9.2, 6.25, CW - 8.6, 'no self-collision · inside joint limits', 'ok', 'dm_ok');

// 14 · evaluation --------------------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Evaluation', 'Testing on 2,000 unseen targets', `Evaluation method and accuracy (about 1 min).
Targets: 2,000 poses from the held-out test split of the best run, restricted to dataset rows neither network saw in training.
Every method receives the same target and the same starting joints. Three starting conditions: warm (true joints plus 0.3-radian noise, as in training), the home pose, and a random pose.
Every answer is re-checked in PyBullet: forward-kinematics pose error, joint limits, self-collision. An answer is valid only if it passes all three.
Left: how close each method gets, warm start. The network alone reaches 50 mm on ${fmt(hw('residual_mlp').within_50mm_pct, 0)} % of targets but never 1 mm; classical DLS reaches 1 mm on ${fmt(hw('classical_dls').within_1mm_pct, 0)} %; network + LM on 100 %.
Right: valid answers for every starting condition. LM alone, without the network, also reaches 100 %: the precision comes from the solver; the network's value is a better starting point.
Source: results/evaluation_metrics.csv, scripts/pipeline/evaluate_methods.py.`);
S.addChart(pres.charts.BAR, [
  { name: 'Network alone', labels: ['≤ 1 mm', '≤ 10 mm', '≤ 50 mm'], values: ['within_1mm_pct', 'within_10mm_pct', 'within_50mm_pct'].map((k) => hw('residual_mlp')[k]).map(r1) },
  { name: 'Classical IK (DLS)', labels: ['≤ 1 mm', '≤ 10 mm', '≤ 50 mm'], values: ['within_1mm_pct', 'within_10mm_pct', 'within_50mm_pct'].map((k) => hw('classical_dls')[k]).map(r1) },
  { name: 'Network + LM', labels: ['≤ 1 mm', '≤ 10 mm', '≤ 50 mm'], values: ['within_1mm_pct', 'within_10mm_pct', 'within_50mm_pct'].map((k) => hw('hybrid_lm')[k]).map(r1) },
], { x: M, y: 1.85, w: 6.9, h: 4.7, barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 55, ...chartFrame(), ...chartText,
  chartColors: [AQUA, ORANGE, BLUE], valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 25, showValue: true, dataLabelPosition: 'outEnd',
  dataLabelFormatCode: '0', showLegend: true, legendPos: 't', showTitle: true, title: 'Targets reached within … (% of 2,000, warm start)',
  objectName: anim('ev_chart') });
const evRows = [['Starting pose', 'Network', 'DLS', 'Net + LM', 'LM only']].map((r) => r.map((t, j) => ({ text: t, options: { bold: true, color: T1, align: j ? 'center' : 'left' } })))
  .concat([['warm (as trained)', 'warm'], ['home (all 0°)', 'zero'], ['random', 'random']].map(([lbl, mode]) => [{ text: lbl }]
    .concat(['residual_mlp', 'classical_dls', 'hybrid_lm', 'lm_only'].map((m) => ({ text: pct(E(m, mode).success_1mm_1deg_pct, 1), options: { align: 'center', color: E(m, mode).success_1mm_1deg_pct >= 99.95 ? '8FF08F' : T2, bold: E(m, mode).success_1mm_1deg_pct >= 99.95 } })))));
text('Valid answers within 1 mm and 1° (no self-collision, inside limits)', { x: 7.85, y: 1.85, w: CW - 7.25, h: 0.55, fontSize: 13, bold: true, color: T1, objectName: anim('ev_tt') });
S.addTable(evRows, { x: 7.85, y: 2.45, w: CW - 7.25, colW: [1.6, 0.8, 0.8, 0.83, 0.85], fontSize: 12.5, color: T2, fontFace: 'Calibri', rowH: 0.42,
  fill: { color: SURF }, border: { type: 'solid', pt: 0.75, color: BORDER }, margin: [0, 0.08, 0, 0.08], valign: 'middle', objectName: anim('ev_table') });
card(7.85, 4.35, CW - 7.25, 2.2, { kicker: 'HOW TO READ IT', name: 'ev_read', bodySize: 12.5,
  body: 'Warm seeds sit near the true answer, so they model "the arm is already close". Home and random starts are the honest cold-start test. The network alone never reaches 1 mm: precision comes from the LM solver.' });

// 15 · comparison ----------------------------------------------------------------------------------------------------------
slide(null, 'Level 1 · Comparison', 'Neural vs classical vs hybrid: the trade-off', `Comparison (about 1.5 min).
Cold start: all three methods start from the home pose, on the same 2,000 targets.
The network alone is fast, about one millisecond, but it is not precise, and from a cold start ${pct(hz('residual_mlp').self_collision_pct)} of its answers collide.
PyBullet's classical damped-least-squares solver is the fastest, and from the home pose it gets within 5 cm of ${pct(hz('classical_dls').within_50mm_pct, 0)} of targets, but it knows nothing about joint limits or self-collision: ${pct(hz('classical_dls').self_collision_pct)} of its answers collide and ${pct(hz('classical_dls').limit_violation_pct)} break a joint limit; only ${pct(hz('classical_dls').success_1mm_1deg_pct)} are valid within 1 mm and 1 degree.
Network plus LM refinement with PyBullet checks: 100 % valid, no collisions, no limit violations, at ${fmt(hz('hybrid_lm').time_ms_mean, 1)} ms per target on one CPU thread.
Honest caveat: the LM solver alone also reaches 100 %. The network makes it about a quarter faster (${fmt(hz('lm_only').time_ms_mean, 1)} → ${fmt(hz('hybrid_lm').time_ms_mean, 1)} ms) and cuts random restarts from ${fmt(hz('lm_only').won_restart_pct, 0)} % to ${fmt(hz('hybrid_lm').won_restart_pct, 0)} % of targets.
From a warm start it is the other way round: LM alone is faster (${fmt(hw('lm_only').time_ms_mean, 1)} vs ${fmt(hw('hybrid_lm').time_ms_mean, 1)} ms) because the seed is already close. The network pays off when the arm starts far from the answer.
Clip: simulations/method_comparison.mp4 (Simulation 9). Source: results/evaluation_metrics.csv (seed_mode = zero).`);
video('method_comparison', M, 1.85, 6.5, 'Simulation 9 · same target, same home pose, three solvers; then rates over 2,000 targets');
S.addChart(pres.charts.BAR, [
  { name: 'Network alone', labels: ['valid ≤ 1 mm & 1°', 'self-collision', 'limit violation'], values: [hz('residual_mlp').success_1mm_1deg_pct, hz('residual_mlp').self_collision_pct, hz('residual_mlp').limit_violation_pct].map(r1) },
  { name: 'Classical IK (DLS)', labels: ['valid ≤ 1 mm & 1°', 'self-collision', 'limit violation'], values: [hz('classical_dls').success_1mm_1deg_pct, hz('classical_dls').self_collision_pct, hz('classical_dls').limit_violation_pct].map(r1) },
  { name: 'Network + LM', labels: ['valid ≤ 1 mm & 1°', 'self-collision', 'limit violation'], values: [hz('hybrid_lm').success_1mm_1deg_pct, hz('hybrid_lm').self_collision_pct, hz('hybrid_lm').limit_violation_pct].map(r1) },
], { x: 7.45, y: 1.85, w: CW - 6.85, h: 3.75, barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 50, ...chartFrame(), ...chartText,
  chartColors: [AQUA, ORANGE, BLUE], valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 25, showValue: true, dataLabelPosition: 'outEnd',
  dataLabelFormatCode: '0.0', showLegend: true, legendPos: 't', showTitle: true, title: 'Cold start, % of 2,000 targets', objectName: anim('cm_chart') });
[[`Network alone · ≈ ${fmt(hz('residual_mlp').time_ms_mean, 0)} ms`, 'fast, but centimetres off'], [`Classical DLS · ${fmt(hz('classical_dls').time_ms_mean, 2)} ms`, 'collision- and limit-blind'],
  [`Network + LM · ${fmt(hz('hybrid_lm').time_ms_mean, 1)} ms`, `100 % valid · LM alone: ${fmt(hz('lm_only').time_ms_mean, 1)} ms`]].forEach(([h, b], i) =>
  card(M + i * ((CW + 0.2) / 3), 6.02, (CW - 0.4) / 3, 0.88, { head: h, body: b, name: `cm_t${i}`, headSize: 13.5, bodySize: 12 }));

// 16 · level 2 ---------------------------------------------------------------------------------------------------------------
slide('Level 2', 'Level 2 · DRL re-routing (implemented)', 'Level 2: steering around the unexpected', `Level 2 (about 1.5 min).
Level 2 is implemented and trained in simulation. A PPO agent outputs a correction for each of the six joints, added to a classical potential-field controller (residual reinforcement learning), so it only has to learn when and how to deviate.
It observes 101 numbers: joint state, goal error, end-effector and link positions, and for each obstacle its distance, velocity and a kinematics-aware escape direction. Its reward favours progress, penalises getting close to obstacles, collisions, timeouts and the Level 1 classifier's self-collision risk.
The policy has 11,078 parameters and needs ${fmt(D.level2.team.decision_us, 1)} microseconds per decision on one CPU core, in plain NumPy.
In the hardest scenario, everything at once, over 200 unseen episodes with identical seeds: the agent reached ${fmt(sc.policy.success_pct, 1)} % of goals with ${fmt(sc.policy.collision_pct, 1)} % collisions; the potential field ${fmt(sc.apf.success_pct, 1)} % / ${fmt(sc.apf.collision_pct, 1)} %, and the straight path ${fmt(sc.baseline.success_pct, 1)} % / ${fmt(sc.baseline.collision_pct, 1)} %. The agent still gets stuck in ${fmt(sc.policy.timeout_pct, 1)} % of episodes.
${repro ? `Reproduction for this presentation (20 M steps on a CPU, same code and settings): ${fmt(repro.scenarios.combined.policy.success_pct, 1)} % reached, ${fmt(repro.scenarios.combined.policy.collision_pct, 1)} % collisions on the same scenario.` : ''}
Clip: simulations/level2_ppo_rerouting.mp4 (Simulation 10). Source: media/data/l2_20261013-185525_xarm6_meta_n200.json; level2/*.py.`);
video('level2_ppo_rerouting', M, 1.85, 6.5, 'Simulation 10 · one recorded episode: straight path, potential field and the PPO agent');
S.addChart(pres.charts.BAR, [
  { name: 'Straight path', labels: ['reached goal', 'collided', 'stuck'], values: [sc.baseline.success_pct, sc.baseline.collision_pct, sc.baseline.timeout_pct].map(r1) },
  { name: 'Potential field', labels: ['reached goal', 'collided', 'stuck'], values: [sc.apf.success_pct, sc.apf.collision_pct, sc.apf.timeout_pct].map(r1) },
  { name: 'DRL agent (PPO)', labels: ['reached goal', 'collided', 'stuck'], values: [sc.policy.success_pct, sc.policy.collision_pct, sc.policy.timeout_pct].map(r1) },
], { x: 7.45, y: 1.85, w: CW - 6.85, h: 3.75, barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 50, ...chartFrame(), ...chartText,
  chartColors: [AQUA, ORANGE, BLUE], valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 25, showValue: true, dataLabelPosition: 'outEnd',
  dataLabelFormatCode: '0.0', showLegend: true, legendPos: 't', showTitle: true, title: '"Everything at once", 200 unseen episodes (%)', objectName: anim('l2_chart') });
[['11,078 parameters', `${fmt(D.level2.team.decision_us, 1)} µs per decision, 1 CPU core`], ['Residual RL', 'potential field + 2 × learned correction'],
  [repro ? `Reproduced: ${fmt(repro.scenarios.combined.policy.success_pct, 1)} % / ${fmt(repro.scenarios.combined.policy.collision_pct, 1)} %` : `${commas(D.level2.steps / 1e6)} M PPO steps`, repro ? 'retrained here, 20 M steps, CPU' : `${Math.round(D.level2.train_secs / 60)} min training`]]
  .forEach(([h, b], i) => card(M + i * ((CW + 0.2) / 3), 6.02, (CW - 0.4) / 3, 0.88, { head: h, body: b, name: `l2_t${i}`, headSize: 13.5, bodySize: 12 }));

// 17 · conclusion ----------------------------------------------------------------------------------------------------------------
slide('Conclusion', 'Conclusion', 'What we achieved, and what is still open', `Conclusion (about 1.5 min).
Achievements: a complete and reproducible Level 1 pipeline; every one of 2,000 held-out targets solved within 1 mm and 1 degree with no self-collision and no limit violation; a collision classifier that is ${fmt(D.classifier.accuracy, 1)} % accurate and is used both in training and in the Level 2 reward; and a Level 2 agent that cuts collisions from ${fmt(sc.baseline.collision_pct, 0)} % (straight path) to ${fmt(sc.policy.collision_pct, 0)} %.
Limitations: everything is simulation with kinematic motion; the xArm6 stands in for the space manipulator and there is no satellite body yet; self-collision is checked per configuration, not along a continuous path; warm-start results assume the arm starts near the answer; the network alone is not precise, the precision comes from the solver; the Level 2 agent still gets stuck in ${fmt(sc.policy.timeout_pct, 0)} % of the hardest episodes.
Next steps: the space manipulator's URDF with the satellite body as collision geometry; continuous path checking; an IK-branch fallback when the agent is stuck; dynamics and a free-floating base.
Thank the audience and open questions.`);
card(M, 1.85, 3.9, 3.85, { kicker: 'ACHIEVED', head: 'Level 1 complete, Level 2 working', name: 'k_ach', kColor: '8FF08F', bodySize: 13,
  body: `• 100 % of 2,000 held-out targets within 1 mm and 1°, 0 % self-collision (network + LM)\n• Classifier ${fmt(D.classifier.accuracy, 1)} % accurate, used in training and Level 2\n• Network alone: ${fmt(hw('residual_mlp').pos_mean_mm, 0)} mm mean error; the plan's network: ${fmt(hw('baseline_mlp').pos_mean_mm, 0)} mm on the same targets\n• Level 2: collisions ${fmt(sc.baseline.collision_pct, 0)} % → ${fmt(sc.policy.collision_pct, 0)} %, goals ${fmt(sc.baseline.success_pct, 0)} % → ${fmt(sc.policy.success_pct, 0)} %` });
card(M + 4.1, 1.85, 3.95, 3.85, { kicker: 'LIMITATIONS', head: 'What the numbers do not show', name: 'k_lim', kColor: 'FFD47A', bodySize: 13,
  body: `• Simulation only, kinematic motion\n• xArm6, not yet the space arm; no satellite body\n• Collisions checked per pose, not along the path\n• Warm seeds assume the arm starts near the answer\n• Precision comes from the solver, not the network\n• Level 2 agent stuck in ${fmt(sc.policy.timeout_pct, 0)} % of hardest episodes` });
card(M + 8.25, 1.85, CW - 8.25, 3.85, { kicker: 'NEXT', head: 'Towards the space manipulator', name: 'k_next', bodySize: 13,
  body: '• Space-arm URDF + satellite body as collision geometry\n• Continuous path collision checking\n• IK-branch fallback when the agent is stuck\n• Dynamics, free-floating base, hardware-in-the-loop' });
S.addShape(pres.shapes.LINE, { x: M, y: 6.0, w: CW, h: 0, line: { color: '22324A', width: 1 } });
text('Thank you · questions', { x: M, y: 6.12, w: 6, h: 0.5, fontSize: 22, bold: true, color: T1, objectName: anim('k_end') });
text('Code, data and every number in this deck: github.com/krishna09/neural_ik · deliverables/', { x: M + 5.2, y: 6.2, w: CW - 5.2, h: 0.36, fontSize: 12, color: MUTED, align: 'right', objectName: 'k_end#repo' });

// ------------------------------------------------------------------------------------------ write
fs.mkdirSync(path.dirname(OUT), { recursive: true });
pres.writeFile({ fileName: OUT }).then(async (f) => {
  fs.writeFileSync(path.join(ASSETS, 'anim.json'), JSON.stringify(ANIM, null, 1));
  const at = process.env.APPLY_THEME;
  if (at) { const { applyTheme } = require(at); await applyTheme(OUT, THEME); }
  console.log('wrote', f);
});
