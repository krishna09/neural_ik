// Scene runner. URL: stage.html?scene=<name>[&variant=clip|video][&timing=<url>]
// Exposes window.__render(t) (synchronous, deterministic) and window.__duration for the frame driver.

const P = new URLSearchParams(location.search);
const name = P.get('scene');
const variant = P.get('variant') || 'clip';
const timingUrl = P.get('timing');

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

try {
  const timing = timingUrl ? await json(timingUrl) : null;
  const ctx = { variant, timing, data: (n) => json(`/work/sim/${n}.json`), json };
  const { default: build } = await import(`./scenes/${name}.js`);
  const scene = await build(ctx);
  await document.fonts.ready;
  scene.render(0);
  window.__duration = scene.duration;
  window.__info = scene.info || {};
  window.__render = (t) => { scene.render(t); return true; };
  window.__ready = true;
} catch (e) {
  window.__error = String(e && e.stack ? e.stack : e);
  console.error(e);
}
