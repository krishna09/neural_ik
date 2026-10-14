// Deterministic animation helpers: every frame is a pure function of the time t (seconds).

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, u) => a + (b - a) * u;
export const lerpArr = (A, B, u) => A.map((a, i) => a + (B[i] - a) * u);
export const smooth = (u) => u * u * (3 - 2 * u);
export const easeInOut = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
export const easeOut = (u) => 1 - Math.pow(1 - u, 3);
export const easeIn = (u) => u * u * u;
export const easeOutBack = (u) => { const c1 = 1.4, c3 = c1 + 1; return 1 + c3 * Math.pow(u - 1, 3) + c1 * Math.pow(u - 1, 2); };

// progress of t through [a, b], clamped, optional easing
export const seg = (t, a, b, ease = easeInOut) => ease(clamp((t - a) / Math.max(1e-6, b - a)));
// fade in at [a, a+d] and out at [b-d, b]
export const window_ = (t, a, b, d = 0.4) => Math.min(seg(t, a, a + d, smooth), 1 - seg(t, b - d, b, smooth));

// shortest-path angle interpolation is NOT used for joints: robots move in joint space, the
// interpolation must stay inside the joint limits, so plain linear interpolation of joint values.
export const qLerp = lerpArr;

// piecewise keyframes [{t, q}] -> q(t), eased per segment
export function keyframes(frames, t, ease = easeInOut) {
  if (t <= frames[0].t) return frames[0].q;
  for (let i = 0; i < frames.length - 1; i++) {
    const a = frames[i], b = frames[i + 1];
    if (t <= b.t) return lerpArr(a.q, b.q, ease(clamp((t - a.t) / Math.max(1e-6, b.t - a.t))));
  }
  return frames[frames.length - 1].q;
}

// timeline: named cues with defaults that a narration timing file may override
export function cues(defaults, timing) {
  const c = { ...defaults };
  if (timing && timing.cues) for (const [k, v] of Object.entries(timing.cues)) if (k in c) c[k] = v;
  return c;
}

// decimal rounding, half away from zero (2.55 -> "2.6"), the same rule as the deck, the charts and spreadsheets;
// plain toFixed works on the binary value and would print 2.55 as "2.5"
export function fixed(v, d = 1) {
  const a = Math.abs(v), s = String(a);
  if (s.includes('e')) return v.toFixed(d);
  const r = Number(`${Math.round(Number(`${s}e${d}`))}e-${d}`);
  return (v < 0 ? -r : r).toFixed(d);
}
export const fmt = (v, d = 1) => (Number.isFinite(v) ? fixed(v, d) : '–');
export const fmtSigned = (v, d = 1) => (v >= 0 ? '+' : '−') + fixed(Math.abs(v), d);
export const deg = (r) => (r * 180) / Math.PI;
export const rad = (d) => (d * Math.PI) / 180;
export const commas = (n) => Math.round(n).toLocaleString('en-US');

// seeded PRNG (mulberry32) for any visual randomness, so renders are repeatable
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// number with a true minus sign (U+2212) and fixed decimals
export const num = (v, d = 1) => (Number.isFinite(v) ? (v < 0 && fixed(Math.abs(v), d) !== fixed(0, d) ? '−' : '') + fixed(Math.abs(v), d) : '–');
export const range = (lo, hi) => `${num(lo, 0)}…${num(hi, 0)}`;
