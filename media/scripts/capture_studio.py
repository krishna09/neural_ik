"""Record Studio footage frame by frame with headless Edge (brief section 6, automated recording).

Run with the WINDOWS python (Edge runs on Windows; the Studio must be running in WSL: `nik-server`):

    .venv\\Scripts\\python.exe media\\scripts\\capture_studio.py              # every shot
    .venv\\Scripts\\python.exe media\\scripts\\capture_studio.py --preview    # first frame of each shot only
    .venv\\Scripts\\python.exe media\\scripts\\capture_studio.py mission_run  # one shot

Frames land in media/assets/frames/<shot>/f_00000.jpg; build_video.py encodes them. Playback is stepped through
the page's own slider (#m-slider, with step='any' so frames can be fractional) and the camera is orbited with
small mouse drags, so the footage is smooth at 30 fps however slow the software renderer is.

Seeds were picked with an offline scan (ui/server.py mission(..., save=False)): the agent succeeds, the straight
path collides, an obstacle appears and a joint slips. Running a mission in the page saves it to
ui/workspace/missions/ like any Studio run; the Infer shot appends one line to the run's inference log.
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import Browser  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = os.path.join(ROOT, "media", "assets", "frames")
URL = "http://127.0.0.1:8765"
FPS = 30
L1_RUN, L2_POLICY = "20260928-174703_xarm6", "l2_20261013-185525_xarm6"
MAIN_SEED, REPLAY_SEED = 15, 39          # dynamic scenario; see the module docstring
VIEW = [1140, 470]                       # a point inside the 3D viewport for mouse orbit / zoom (updated by fullview)


def iife(body):
    return f"(async () => {{ {body} }})()"


class Studio:
    def __init__(self, b):
        self.b, self.seed = b, None

    def fullview(self, on):
        """Hide the left panel so the 3D view gets the whole width (the video adds its own captions)."""
        css = ".layout, .layout.monitor-open{grid-template-columns:minmax(0,1fr)!important} #left, #right{display:none!important} #center{grid-column:1 / -1!important}" if on else ""
        self.b.js(iife("let st = document.getElementById('cap-style'); if (!st) { st = document.createElement('style');"
                       f"st.id = 'cap-style'; document.head.appendChild(st); }} st.textContent = '{css}';"
                       "window.dispatchEvent(new Event('resize')); return true;"))
        VIEW[:] = [960, 470] if on else [1140, 470]
        time.sleep(0.5)

    def tidy(self, bottom_px=110):
        """Small bottom panel (more 3D view), no toast."""
        self.b.js(iife(f"document.documentElement.style.setProperty('--bottom-h', '{bottom_px}px');"
                       "window.dispatchEvent(new Event('resize'));"
                       "const t = document.getElementById('toast'); if (t) t.style.visibility = 'hidden'; return true;"))

    def open_mission(self):
        self.b.goto(f"{URL}/static/mission.html")
        self.b.wait("document.querySelectorAll('#m-agent option').length > 0 && "
                    "document.getElementById('loading').style.display === 'none'", 120)
        self.b.js(iife(f"const $ = (i) => document.getElementById(i); $('m-agent').value = '{L2_POLICY}'; $('m-agent').onchange();"
                       f"return true;"))
        self.b.wait("document.getElementById('loading').style.display === 'none'", 120)
        self.b.js(iife(f"document.getElementById('m-l1').value = '{L1_RUN}'; return true;"))

    def run_mission(self, seed, cmp="baseline"):
        b = self.b
        if self.seed == seed:                             # already loaded: only switch the comparison
            b.js(iife(f"const c = document.getElementById('m-cmp'); c.value = '{cmp}'; c.onchange(); return true;"))
            return int(b.js("+document.getElementById('m-slider').max"))
        self.seed = seed
        b.js(iife(f"const $ = (i) => document.getElementById(i); $('m-scenario').value = 'dynamic'; $('m-target').value = 'random';"
                  f"$('m-seed').value = {seed}; $('m-cmp').value = '{cmp}'; $('m-slider').max = 0; $('m-run').click(); return true;"))
        b.wait("+document.getElementById('m-slider').max > 0 && !document.getElementById('m-run').disabled", 180)
        b.js(iife("const $ = (i) => document.getElementById(i);"
                  "if ($('m-play').textContent.includes('Pause')) $('m-play').click();"
                  "$('m-slider').step = 'any'; return true;"))
        return int(b.js("+document.getElementById('m-slider').max"))

    def frame(self, f):
        self.b.js(iife(f"const s = document.getElementById('m-slider'); s.value = {f:.4f};"
                       "s.dispatchEvent(new Event('input')); return true;"))

    def checks(self, plan=None, trail=None, goal=None):
        for cid, on in (("m-show-plan", plan), ("m-show-trail", trail), ("m-show-goal", goal)):
            if on is not None:
                self.b.js(iife(f"const c = document.getElementById('{cid}'); if (c.checked !== {str(on).lower()}) "
                               f"{{ c.checked = {str(on).lower()}; c.onchange(); }} return true;"))

    def camera(self, zoom_clicks=4, yaw_px=0, pitch_px=0):
        b = self.b
        b.js(iife("document.getElementById('fit-btn').click(); return true;"))
        time.sleep(0.6)
        for _ in range(zoom_clicks):
            b.wheel(*VIEW, -120)
            time.sleep(0.15)
        if yaw_px or pitch_px:
            b.drag(*VIEW, yaw_px, pitch_px, steps=6)
        time.sleep(1.5)                                   # let OrbitControls' damping settle


class Recorder:
    def __init__(self, b, name, preview):
        self.b, self.preview, self.n = b, preview, 0
        self.dir = os.path.join(FRAMES, name)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)

    def grab(self, orbit_px=0.0):
        if orbit_px:                                      # slow camera orbit between frames
            self.b.drag(*VIEW, orbit_px, 0, steps=1)
            time.sleep(0.05)
        self.b.shot(os.path.join(self.dir, f"f_{self.n:05d}.jpg"), quality=93)
        self.n += 1


def shot_infer(b, preview):
    """Level 1 -> 4 Infer: held-out test sample, LM solver; orange ghost = network, solid = final answer."""
    b.goto(f"{URL}/")
    b.wait("document.querySelectorAll('#inf-run option').length > 0", 120)
    b.js(iife("const $ = (i) => document.getElementById(i);"
              "document.querySelector('#stepper button[data-step=\"p-infer\"]').click();"
              f"$('inf-run').value = '{L1_RUN}'; $('inf-run').dispatchEvent(new Event('change'));"
              "$('inf-target').value = 'test'; $('inf-target').dispatchEvent(new Event('change'));"
              "$('inf-seed').value = 'noisy'; $('inf-ref-method').value = 'lm';"
              "const c = $('show-cloud'); if (c.checked) c.click(); return true;"))
    time.sleep(2)
    b.js(iife("document.getElementById('inf-random-test').click(); return true;"))
    time.sleep(2)
    b.js(iife("document.getElementById('inf-btn').click(); return true;"))
    b.wait("document.getElementById('inf-final').textContent.trim().length > 0", 120)
    b.js(iife("const st = document.createElement('style'); st.textContent = '.layout.monitor-open{grid-template-columns:360px "
              "minmax(0,1fr)!important} #right{display:none!important} #job-stop{display:none!important}'; "
              "document.head.appendChild(st); window.dispatchEvent(new Event('resize')); "
              "document.getElementById('job-text').textContent = 'idle'; return true;"))
    Studio(b).tidy(150)
    time.sleep(1.0)
    for _ in range(1):
        b.wheel(1140, 420, -120)
        time.sleep(0.15)
    time.sleep(1.5)
    rec = Recorder(b, "infer", preview)
    rec.grab()


def shot_mission(s, preview, which):
    b = s.b
    if s.seed is None:
        s.open_mission()
    seed = REPLAY_SEED if which == "mission_replay" else MAIN_SEED
    nmax = s.run_mission(seed, cmp="none" if which == "mission_replay" else "baseline")
    s.tidy()
    s.fullview(which != "mission_plan")                   # scene 4 keeps the pipeline card ("15 of 40 waypoints collide")
    rec = Recorder(b, which, preview)
    orbit_until = None
    if which == "mission_intro":                          # scene 2: target + arm at start, nothing else
        s.checks(plan=False, trail=False, goal=False)
        s.frame(0)
        s.camera(zoom_clicks=5, yaw_px=-120, pitch_px=-20)
        n, orbit = 8 * FPS, 0.9
    elif which == "mission_plan":                         # scene 4: dashed straight-path estimate, red where it hits
        s.checks(plan=True, trail=False, goal=True)
        s.frame(0)
        s.camera(zoom_clicks=5, yaw_px=-40, pitch_px=-20)
        n, orbit = 10 * FPS, 0.9
    elif which == "mission_run":                          # scene 5: agent (solid) vs. straight path (aqua) on the same mission
        s.checks(plan=False, trail=True, goal=True)
        s.camera(zoom_clicks=5, yaw_px=60, pitch_px=-20)
        n, orbit = 16 * FPS, 0.6
        orbit_until = int(6.5 * FPS)                      # then hold: further round, the arm reaches into the camera
    else:                                                 # scene 8: a second mission, agent only, to the target
        s.checks(plan=False, trail=True, goal=True)
        s.camera(zoom_clicks=0, yaw_px=-100, pitch_px=-20)
        n, orbit = 5 * FPS, 1.0
    orbit_until = orbit_until or n
    moving = which in ("mission_run", "mission_replay")
    lead, tail = (0.6, 1.2) if moving else (0, 0)         # seconds held at the start / end of the playback
    prev = 0
    for i in ([0, n // 2, n - 1] if preview else range(n)):
        t = i / FPS
        if moving:
            k = min(1.0, max(0.0, (t - lead) / (n / FPS - lead - tail)))
            s.frame(k * nmax)
        steps = sum(1 for j in range(prev + 1 if i else 0, i + 1) if j < orbit_until)
        rec.grab(orbit * steps)                           # preview: the orbit of the skipped frames at once
        prev = i
        if i % 30 == 0:
            print(f"  {which}: frame {i}/{n}", flush=True)
    print(f"{which}: seed {seed}, {rec.n} frames, playback max {nmax}", flush=True)


def shot_browse(b, preview):
    """Level 1 -> 2 Browse: the 200 k dataset as a point cloud (blue = free, red = colliding), for the deck."""
    b.goto(f"{URL}/")
    b.wait("document.querySelectorAll('#ds-src option').length > 0", 120)
    b.js(iife("document.querySelector('#stepper button[data-step=\"p-browse\"]').click();"
              f"const s = document.getElementById('ds-src'); s.value = '{BROWSE_DS}'; s.dispatchEvent(new Event('change'));"
              "const c = document.getElementById('show-cloud'); if (!c.checked) c.click(); return true;"))
    b.wait("document.getElementById('loading').style.display === 'none'", 120)
    time.sleep(3)
    for _ in range(12):                                   # step through a few samples so the arm shows one
        b.js(iife("document.getElementById('ds-next').click(); return true;"))
        time.sleep(0.3)
    b.js(iife("const st = document.createElement('style'); st.textContent = '#right{display:none!important} "
              ".layout.monitor-open{grid-template-columns:360px minmax(0,1fr)!important} #job-stop{display:none!important}';"
              "document.head.appendChild(st); window.dispatchEvent(new Event('resize')); return true;"))
    Studio(b).tidy(150)
    time.sleep(2)
    Recorder(b, "browse", preview).grab()


def shot_robots(b, preview):
    """One clean 3D view per supported arm (no panels, no overlays), for the deck's robot strip."""
    b.goto(f"{URL}/")
    b.wait("document.querySelectorAll('#robot-select option').length > 0", 120)
    b.js(iife("const st = document.createElement('style'); st.textContent = '.layout, .layout.monitor-open"
              "{grid-template-columns:minmax(0,1fr)!important} #left, #right, .overlay, #view-tools{display:none!important} "
              "#center{grid-column:1 / -1!important}'; document.head.appendChild(st);"
              "const c = document.getElementById('show-cloud'); if (c && c.checked) c.click(); return true;"))
    Studio(b).tidy(110)
    for name in ROBOTS:
        loaded = f"document.getElementById('robot-desc').textContent.includes('{ROBOT_WORD[name]}') && " \
                 "document.getElementById('loading').style.display === 'none'"
        for _ in range(5):                                # the page may switch back to the robot of the selected run
            b.js(iife(f"const s = document.getElementById('robot-select'); s.value = '{name}'; s.onchange({{target: s}}); return true;"))
            b.wait(loaded, 120)
            time.sleep(3)
            if b.js(loaded):
                break
        b.js(iife("const f = document.getElementById('fit-btn'); if (f) f.click(); return true;"))
        time.sleep(2)
        Recorder(b, f"robot_{name}", preview).grab()


BROWSE_DS = "ds_20260928-163059_xarm6_n200000_s0"
ROBOTS = ["xarm6", "kuka_iiwa", "panda"]
ROBOT_WORD = {"xarm6": "xArm6", "kuka_iiwa": "KUKA", "panda": "Panda"}   # appears in #robot-desc once loaded
SHOTS = ["infer", "mission_intro", "mission_plan", "mission_run", "mission_replay"]
STILLS = ["browse", "robots"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="*", help=", ".join(SHOTS + STILLS) + " (default: the video shots)")
    ap.add_argument("--preview", action="store_true", help="first frame of each shot only")
    ap.add_argument("--port", type=int, default=9333, help="DevTools port (use another one to run two captures at once)")
    args = ap.parse_args()
    b = Browser(port=args.port)
    studio = Studio(b)
    try:
        for name in args.shots or SHOTS:
            t0 = time.time()
            if name in ("infer", "browse", "robots"):
                {"infer": shot_infer, "browse": shot_browse, "robots": shot_robots}[name](b, args.preview)
                studio.seed = None                        # navigated away from the mission page
            else:
                shot_mission(studio, args.preview, name)
            print(f"{name} done in {time.time() - t0:.0f} s", flush=True)
    finally:
        b.close()


if __name__ == "__main__":
    main()
