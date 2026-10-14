// Overlay (HTML) helpers. Elements are created once in setup and only restyled per frame.

const root = () => document.getElementById('ui');

// elBack: same, but behind the 3D canvas (cards that frame a 3D view)
export function elBack(html, style = {}) { return el(html, style, document.getElementById('uiback')); }

export function el(html, style = {}, parent = null) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  const node = t.content.firstElementChild;
  Object.assign(node.style, style);
  (parent || root()).appendChild(node);
  return node;
}

// opacity + small upward slide, driven by a 0..1 amount
export function show(node, a, dy = 18) {
  node.style.opacity = a.toFixed(3);
  node.style.visibility = a <= 0.001 ? 'hidden' : 'visible';
  const base = node.dataset.baseTransform || '';
  node.style.transform = `${base} translateY(${((1 - a) * dy).toFixed(2)}px)`;
}

export function setBase(node, transform) {
  node.dataset.baseTransform = transform;
  node.style.transform = transform;
}

export function chapter(text, sub = '') {
  return el(`<div class="chapter"><span class="dot"></span><span>${text}</span>${sub ? `<span class="sep">/</span><span>${sub}</span>` : ''}</div>`);
}

export function title(text, top = 96, extra = {}) {
  return el(`<div class="title">${text}</div>`, { top: `${top}px`, ...extra });
}

export function subtitle(text, top = 170, extra = {}) {
  return el(`<div class="subtitle">${text}</div>`, { top: `${top}px`, ...extra });
}

export function footer(text = 'STP 2026 · Team 06 · xArm6 · PyBullet simulation') {
  return el(`<div class="footer">${text}</div>`);
}

export function source(text) {
  return el(`<div class="source">${text}</div>`);
}

export function caption(html) {
  const c = el(`<div class="caption">${html}</div>`);
  setBase(c, 'translateX(-50%)');
  return c;
}

export function panel(html, style) {
  return el(`<div class="panel">${html}</div>`, style);
}

export function badge(kind, text) {
  const ic = { ok: '✓', bad: '✕', warn: '!', info: '' }[kind] || '';
  return `<span class="badge ${kind}">${ic ? `<span class="ic">${ic}</span>` : ''}${text}</span>`;
}

// place a DOM label at a projected 3D position (px)
export function place(node, xy, a = 1) {
  if (!xy) { node.style.visibility = 'hidden'; return; }
  node.style.left = `${xy[0].toFixed(1)}px`;
  node.style.top = `${xy[1].toFixed(1)}px`;
  node.style.opacity = a.toFixed(3);
  node.style.visibility = a <= 0.001 ? 'hidden' : 'visible';
}

export function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

export function setHTML(node, html) {
  if (node.innerHTML !== html) node.innerHTML = html;
}
