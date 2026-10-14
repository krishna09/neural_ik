// Page chrome only (no app logic): one workflow step visible at a time, the
// System Monitor drawer, and remembering both per browser. app.js still opens
// a step via details.open = true (openPanel), which this file follows.

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// ---------- workflow steps ----------
const panels = [...document.querySelectorAll('#left > details.panel')];
const stepBtns = [...document.querySelectorAll('#stepper button')];

function showStep(id) {
  for (const p of panels) if (p.id !== id && p.open) p.open = false;
  const p = document.getElementById(id);
  if (p && !p.open) p.open = true;
  for (const b of stepBtns) b.classList.toggle('active', b.dataset.step === id);
  store.set('nik.step', id);
  document.getElementById('left').scrollTop = 0;
}

for (const p of panels) {
  // the summary is a heading now; switching happens through the stepper
  p.querySelector(':scope > summary').addEventListener('click', (e) => e.preventDefault());
  p.addEventListener('toggle', () => { if (p.open) showStep(p.id); });
}
for (const b of stepBtns) b.addEventListener('click', () => showStep(b.dataset.step));

const saved = store.get('nik.step');
showStep(panels.some((p) => p.id === saved) ? saved : 'p-gen');

// ---------- System Monitor drawer ----------
const layout = document.querySelector('.layout');
const monBtn = document.getElementById('monitor-toggle');

function setMonitor(on, remember = true) {
  layout.classList.toggle('monitor-open', on);
  monBtn.setAttribute('aria-pressed', String(on));
  if (remember) store.set('nik.monitor', on ? '1' : '0');
  window.dispatchEvent(new Event('resize'));   // let the 3D view refit
}
monBtn.addEventListener('click', () => setMonitor(!layout.classList.contains('monitor-open')));
setMonitor(store.get('nik.monitor') === '1', false);

// open the drawer automatically when a job starts (job dot pulses while running)
const jobDot = document.getElementById('job-dot');
let wasRunning = jobDot.classList.contains('run');
new MutationObserver(() => {
  const running = jobDot.classList.contains('run');
  if (running && !wasRunning) setMonitor(true, false);
  wasRunning = running;
}).observe(jobDot, { attributes: true, attributeFilter: ['class'] });
