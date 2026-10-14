// Shared scene scaffolding: robot model cache, timeline (clip defaults or narration timing), header,
// captions (clip variant only - the video carries narration subtitles instead).

import { loadRobot } from './robot.js';
import { chapter, title, subtitle, footer, source, caption, show, setHTML } from './ui.js';
import { window_, seg, smooth } from './anim.js';

let _model = null;
export async function robotModel(ctx) {
  if (!_model) _model = await loadRobot(await ctx.data('robot'));
  return _model;
}

// default cue times (s) for a clip of length D0; a timing file may give {duration, cues:{...}}:
// missing cues are scaled with the duration so the scene keeps its rhythm.
export function timeline(ctx, defaults, D0) {
  const T = ctx.timing || {};
  const D = T.duration || D0;
  const s = D / D0;
  const c = {};
  for (const [k, v] of Object.entries(defaults)) c[k] = v * s;
  if (T.cues) for (const [k, v] of Object.entries(T.cues)) if (k in c) c[k] = v;
  return { D, c, s };
}

export function header(ctx, { chap, sub = '', ttl = '', subt = '', src = '', subtTop = 168 }) {
  const h = {};
  h.chapter = chapter(chap, sub);
  if (ttl) h.title = title(ttl);
  if (subt) h.subtitle = subtitle(subt, subtTop);
  h.footer = footer();
  if (src) h.source = source(src);
  return h;
}

// fade the header in at the start and (optionally) out at the end
export function headerFade(h, t, D, { fadeOut = true } = {}) {
  const a = Math.min(seg(t, 0.15, 0.9, smooth), fadeOut ? 1 - seg(t, D - 0.5, D, smooth) : 1);
  for (const k of ['chapter', 'title', 'subtitle']) if (h[k]) show(h[k], a, 12);
  for (const k of ['footer', 'source']) if (h[k]) h[k].style.opacity = a.toFixed(3);
}

// captions: [[t0, t1, html], ...] -> one caption element, clip variant only
export function captions(ctx, list) {
  if (ctx.variant !== 'clip' || !list.length) return () => {};
  const node = caption('');
  return (t) => {
    const cur = list.find(([a, b]) => t >= a && t < b);
    if (!cur) { show(node, 0); return; }
    setHTML(node, cur[2]);
    show(node, window_(t, cur[0], cur[1], 0.35), 10);
  };
}

// overall fade to/from black for clip boundaries
export function stageFade(t, D, fin = 0.5, fout = 0.6) {
  const a = Math.min(seg(t, 0, fin, smooth), 1 - seg(t, D - fout, D, smooth));
  document.getElementById('stage').style.opacity = a.toFixed(3);
}
