#!/usr/bin/env python
"""Render a renderer scene (deliverables/scripts/render/js/scenes/<name>.js) frame by frame.

Headless Chromium (Playwright, WebGL through SwiftShader) opens stage.html; for every frame the page
draws time t = i / fps with window.__render(t), and the screenshot is piped into ffmpeg. Because every
frame is a pure function of t, the output is smooth and repeatable however slow the software
renderer is.

    python render_scene.py test_robot --stills 0,1          # PNG stills -> $NIK_WORK/stills/<scene>/
    python render_scene.py robot_intro --out clip.mp4         # whole scene -> H.264
    python render_scene.py robot_intro --out a.mp4 --timing /work/timing/robot_intro.json --variant video
"""
import argparse
import functools
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, WORK  # noqa: E402

CHROME = os.environ.get("NIK_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--enable-webgl",
        "--disable-gpu-vsync", "--hide-scrollbars", "--force-color-profile=srgb"]


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the project root; /work/... is mapped to $NIK_WORK (generated data)."""
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".mjs": "text/javascript", ".json": "application/json", ".stl": "application/octet-stream"}

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/work/"):
            return str(WORK / clean[len("/work/"):])
        return super().translate_path(path)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(REPO)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def open_scene(pw, port, scene, variant, timing):
    browser = pw.chromium.launch(executable_path=CHROME, args=ARGS)
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    logs = []
    page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
    page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    url = f"http://127.0.0.1:{port}/deliverables/scripts/render/stage.html?scene={scene}&variant={variant}"
    if timing:
        url += f"&timing={timing}"
    page.goto(url)
    try:
        page.wait_for_function("window.__ready === true || window.__error !== undefined", timeout=180_000)
    except Exception:
        print("\n".join(logs[-30:]))
        raise
    err = page.evaluate("window.__error")
    if err:
        print("\n".join(logs[-30:]))
        raise RuntimeError(err)
    return browser, page, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--out", help="output .mp4")
    ap.add_argument("--stills", help="comma-separated times (s) to save as PNG")
    ap.add_argument("--stills-dir", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--variant", default="clip")
    ap.add_argument("--timing", default=None, help="URL of a timing JSON (e.g. /work/timing/x.json)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright
    srv, port = serve()
    t_start = time.time()
    with sync_playwright() as pw:
        browser, page, logs = open_scene(pw, port, a.scene, a.variant, a.timing)
        dur = float(page.evaluate("window.__duration"))
        info = page.evaluate("window.__info")
        print(json.dumps({"scene": a.scene, "duration": dur, "info": info}))
        if a.stills:
            d = Path(a.stills_dir or WORK / "stills" / a.scene)
            d.mkdir(parents=True, exist_ok=True)
            for t in [float(x) for x in a.stills.split(",")]:
                page.evaluate(f"window.__render({t})")
                page.screenshot(path=str(d / f"{a.scene}_{t:06.2f}.png"))
                print(d / f"{a.scene}_{t:06.2f}.png")
        if a.out:
            end = dur if a.end is None else min(dur, a.end)
            n = int(round((end - a.start) * a.fps))
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps),
                                   "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
            t0 = time.time()
            for i in range(n):
                page.evaluate(f"window.__render({a.start + i / a.fps})")
                ff.stdin.write(page.screenshot(type="jpeg", quality=94))
                if i % 150 == 0:
                    print(f"  frame {i}/{n}  {(time.time() - t0) / max(1, i):.3f} s/frame", flush=True)
            ff.stdin.close()
            ff.wait()
            print(f"wrote {a.out} ({n} frames, {time.time() - t_start:.0f} s)")
        errs = [l for l in logs if "error" in l.lower()]
        if errs:
            print("\n".join(errs[-10:]))
        browser.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
