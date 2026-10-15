#!/bin/bash
# Build documentation/STP_Spacearm_explained.pdf from spacearm_explained.src.html:
# KaTeX renders every $$...$$ and \(...\) formula to HTML (no network), then headless Chromium prints A4.
#   NODE_PATH=<dir with katex> bash deliverables/scripts/docs/build_spacearm_explained.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../documentation/STP_Spacearm_explained.pdf"
KATEX_DIR="$(node -e "console.log(require('path').dirname(require.resolve('katex/package.json')))")"
node - "$HERE/spacearm_explained.src.html" "$HERE/spacearm_explained.html" "$KATEX_DIR" <<'JS'
const fs = require('fs'), katex = require('katex');
const [src, dst, kdir] = process.argv.slice(2);
let html = fs.readFileSync(src, 'utf8').replace('KATEX_CSS', 'file://' + kdir + '/dist/katex.min.css');
const tex = (s) => s.replace(/\\\*/g, '*').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
const r = (s, display) => katex.renderToString(tex(s), { displayMode: display, throwOnError: true, strict: 'ignore' });
html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => r(m, true));
html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => r(m, false));
fs.writeFileSync(dst, html);
console.log('math rendered');
JS
python - "$HERE/spacearm_explained.html" "$OUT" <<'PY'
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
with sync_playwright() as p:
    import os, glob
    exe = os.environ.get("CHROME") or next(iter(sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))), None)
    b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
    pg = b.new_page()
    pg.goto(src.as_uri(), wait_until="networkidle")
    pg.evaluate("document.fonts.ready")
    pg.pdf(path=str(out), format="A4", print_background=True, display_header_footer=True,
           header_template="<span></span>",
           footer_template='<div style="font-size:7.5pt;color:#7a879a;width:100%;padding:0 16mm;display:flex;justify-content:space-between;font-family:Inter,sans-serif">'
                           '<span>STP_Spacearm explained with simple maths · STP 2026 Team 06</span>'
                           '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
           margin={"top": "17mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
    b.close()
print("wrote", out)
PY
rm -f "$HERE/spacearm_explained.html"
