import sys, time
sys.path.insert(0, r"H:\STP-2026\Projects\vs_code_claude\neural_ik\media\scripts")
from cdp import Browser
b = Browser()
try:
    b.goto("http://127.0.0.1:8765/static/mission.html")
    t0=time.time()
    b.wait("document.querySelectorAll('#m-agent option').length>0 && document.getElementById('loading').style.display==='none'", 90)
    print('loaded in', time.time()-t0)
    b.js("document.getElementById('m-scenario').value='dynamic'; document.getElementById('m-seed').value=15; document.getElementById('m-cmp').value='baseline'; document.getElementById('m-cmp').onchange(); true")
    b.js("document.getElementById('m-run').click(); true")
    b.wait("+document.getElementById('m-slider').max>0", 120)
    b.js("document.getElementById('m-play').textContent.includes('Pause') && document.getElementById('m-play').click(); true")
    s=b.js("const s=document.getElementById('m-slider'); s.step='any'; s.value=0; s.dispatchEvent(new Event('input')); s.max")
    print('max', s)
    time.sleep(1)
    t0=time.time(); b.shot(r"H:\STP-2026\Projects\vs_code_claude\neural_ik\media\assets\probe0.png"); print('shot', time.time()-t0)
    b.js("const s=document.getElementById('m-slider'); s.value=11.5; s.dispatchEvent(new Event('input')); true")
    time.sleep(0.3)
    b.shot(r"H:\STP-2026\Projects\vs_code_claude\neural_ik\media\assets\probe1.png")
    print(b.js("document.getElementById('hud-lines').textContent"))
finally:
    b.close()
