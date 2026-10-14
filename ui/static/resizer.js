// Drag the boundary between the 3D view and the bottom tabs to resize them (both levels).
// Double-click the handle to reset. The height is remembered per browser.

const center = document.getElementById('center');
const bottom = document.getElementById('bottom');
const root = document.documentElement;
const KEY = 'nik.bottomH';
const DEFAULT = '330px';

const handle = document.createElement('div');
handle.className = 'resize-handle';
handle.title = 'Drag to resize · double-click to reset';
handle.setAttribute('role', 'separator');
handle.setAttribute('aria-orientation', 'horizontal');
bottom.prepend(handle);

const setH = (px) => {
  const max = center.clientHeight - 120;                 // keep at least 120 px of 3D view
  const h = Math.round(Math.max(110, Math.min(max, px)));
  root.style.setProperty('--bottom-h', `${h}px`);
  return h;
};
try { const saved = localStorage.getItem(KEY); if (saved) setH(+saved); } catch { /* storage unavailable */ }

handle.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  handle.setPointerCapture(e.pointerId);
  document.body.classList.add('resizing');
  const move = (ev) => setH(center.getBoundingClientRect().bottom - ev.clientY);
  const up = () => {
    handle.removeEventListener('pointermove', move);
    handle.removeEventListener('pointerup', up);
    document.body.classList.remove('resizing');
    try { localStorage.setItem(KEY, String(bottom.getBoundingClientRect().height)); } catch { /* ignore */ }
    window.dispatchEvent(new Event('resize'));           // let charts redraw at the new size
  };
  handle.addEventListener('pointermove', move);
  handle.addEventListener('pointerup', up);
});
handle.addEventListener('dblclick', () => {
  root.style.setProperty('--bottom-h', DEFAULT);
  try { localStorage.removeItem(KEY); } catch { /* ignore */ }
  window.dispatchEvent(new Event('resize'));
});
