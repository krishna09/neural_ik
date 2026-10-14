"""Tiny Chrome DevTools Protocol driver for headless Microsoft Edge (run with the Windows .venv python).

Edge runs on Windows, the Studio in WSL; Windows reaches it at http://127.0.0.1:8765 through WSL's localhost
forwarding. Only websocket-client is needed (suppress_origin=True, or Edge refuses the connection).
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websocket

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


class Browser:
    def __init__(self, port=9333, w=1920, h=1080):
        self.port = port
        profile = os.path.join(tempfile.gettempdir(), f"nik-capture-{port}")
        self.proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={port}", "--use-angle=swiftshader",
                                      f"--window-size={w},{h}", f"--user-data-dir={profile}", "--hide-scrollbars",
                                      "--mute-audio", "--no-first-run", "--force-device-scale-factor=1", "about:blank"])
        tabs = None
        for _ in range(150):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except OSError:
                time.sleep(0.2)
        page = next(t for t in tabs if t["type"] == "page")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], suppress_origin=True, timeout=300)
        self.mid = 0
        self.send("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)

    def send(self, method, **params):
        self.mid += 1
        mid = self.mid
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error in {expr[:80]!r}: {r['exceptionDetails'].get('exception', {}).get('description')}")
        return r["result"].get("value")

    def wait(self, expr, timeout=60, every=0.25):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.js(expr):
                return
            time.sleep(every)
        raise TimeoutError(f"waited {timeout} s for {expr!r}")

    def goto(self, url):
        self.send("Page.enable")
        self.send("Page.navigate", url=url)
        self.wait("document.readyState === 'complete'")

    def shot(self, path, quality=None):
        kw = {"format": "jpeg", "quality": quality} if quality else {"format": "png"}
        data = self.send("Page.captureScreenshot", **kw)["data"]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    def drag(self, x0, y0, dx, dy, steps=1, button="left"):
        """Mouse drag in page pixels (orbits the three.js view)."""
        self.send("Input.dispatchMouseEvent", type="mousePressed", x=x0, y=y0, button=button, clickCount=1)
        for i in range(1, steps + 1):
            self.send("Input.dispatchMouseEvent", type="mouseMoved", x=x0 + dx * i / steps, y=y0 + dy * i / steps, button=button)
        self.send("Input.dispatchMouseEvent", type="mouseReleased", x=x0 + dx, y=y0 + dy, button=button, clickCount=1)

    def wheel(self, x, y, dy):
        self.send("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y, deltaX=0, deltaY=dy)

    def close(self):
        try:
            self.send("Browser.close")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:          # Edge sometimes lingers after Browser.close
            self.proc.kill()
