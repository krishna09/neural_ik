#!/usr/bin/env python
"""Add PowerPoint animations and transitions that pptxgenjs cannot write, and tidy chart data.

* Entrance animations: for every slide, the object groups listed in $NIK_WORK/deck/anim.json (written by
  build_deck.js) fade in one after another ("after previous"), starting automatically when the slide
  appears - the slide builds itself without extra clicks. Shapes named "<group>#..." move together.
* Transitions: a medium fade on every slide.
* Charts: pptxgenjs writes missing values as empty points; they are removed so lines end cleanly.

    python deliverables/scripts/deck/postprocess_deck.py deliverables/presentation/Neural_IK_STP_Team06_Level1.pptx
"""
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
STEP_MS, FADE_MS = 260, 420           # start-to-start gap between groups, fade duration


def shapes_by_group(xml, groups):
    """group name -> [(spid, kind)] in document order; kind = sp | pic | graphicFrame."""
    root = etree.fromstring(xml)
    found = {g: [] for g in groups}
    for el in root.iter("{%s}sp" % NS["p"], "{%s}pic" % NS["p"], "{%s}graphicFrame" % NS["p"]):
        c = el.find(".//p:cNvPr", NS)
        if c is None:
            continue
        name = c.get("name", "")
        for g in groups:
            if name == g or name.startswith(g + "#"):
                found[g].append((c.get("id"), etree.QName(el).localname, el.find("p:txBody", NS) is not None))
                break
    return found


def effect(ids, spid, node_type, grp):
    a, b, c = next(ids), next(ids), next(ids)
    g = ' grpId="0"' if grp else ""
    return (f'<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold"{g} nodeType="{node_type}">'
            f'<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            f'<p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
            f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>'
            f'</p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>'
            f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{c}" dur="{FADE_MS}"/>'
            f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:animEffect></p:childTnLst></p:cTn></p:par>')


def timing_xml(steps):
    """steps: [[(spid, kind, has_text), ...], ...] -> <p:timing> (auto-start, each step after the previous)."""
    counter = iter(range(4, 100000))
    pars, bld, t = [], [], 0
    for k, shapes in enumerate(steps):
        effs = []
        for j, (spid, kind, has_text) in enumerate(shapes):
            grp = kind in ("sp", "graphicFrame")
            effs.append(effect(counter, spid, "afterEffect" if j == 0 else "withEffect", grp))
            if kind == "sp":
                bld.append(f'<p:bldP spid="{spid}" grpId="0"{" animBg=\"1\"" if has_text else ""}/>')
            elif kind == "graphicFrame":
                bld.append(f'<p:bldGraphic spid="{spid}" grpId="0"><p:bldAsOne/></p:bldGraphic>')
        pars.append(f'<p:par><p:cTn id="{next(counter)}" fill="hold"><p:stCondLst><p:cond delay="{t}"/></p:stCondLst>'
                    f'<p:childTnLst>{"".join(effs)}</p:childTnLst></p:cTn></p:par>')
        t += STEP_MS if k else 0
    # ids 1-3 are the root, main sequence and click group
    body = "".join(pars)
    return ('<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
            '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
            '<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
            '<p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst>'
            f'<p:childTnLst>{body}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>'
            '<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
            '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
            '</p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>'
            f'<p:bldLst>{"".join(bld)}</p:bldLst></p:timing>') if pars else ""


def main(pptx):
    pptx = Path(pptx)
    anim = json.load(open(Path(os.environ.get("NIK_WORK", "/tmp/nik_work")) / "deck" / "anim.json"))
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(pptx) as z:
        z.extractall(tmp)
        order = z.namelist()
    report = []
    slides = sorted((tmp / "ppt" / "slides").glob("slide*.xml"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    for i, sp in enumerate(slides):
        xml = sp.read_bytes()
        groups = anim[i] if i < len(anim) else []
        found = shapes_by_group(xml, groups)
        missing = [g for g in groups if not found[g]]
        if missing:
            sys.exit(f"{sp.name}: animation groups without shapes: {missing}")
        steps = [found[g] for g in groups]
        s = xml.decode("utf-8")
        if "<p:timing" in s or "<p:transition" in s:
            sys.exit(f"{sp.name} already has timing/transition")
        extra = '<p:transition spd="med"><p:fade/></p:transition>' + timing_xml(steps)
        s, n = re.subn(r"(</p:clrMapOvr>)", r"\1" + extra.replace("\\", "\\\\"), s, count=1)
        if n != 1:
            sys.exit(f"{sp.name}: no clrMapOvr to anchor the transition")
        sp.write_text(s, encoding="utf-8")
        report.append(f"{sp.name}: {len(groups)} build steps, {sum(len(v) for v in steps)} shapes")
    for ch in (tmp / "ppt" / "charts").glob("chart*.xml"):
        s = ch.read_text(encoding="utf-8")
        s2 = re.sub(r'<c:pt idx="\d+"><c:v></c:v></c:pt>', "", s)
        if s2 != s:
            ch.write_text(s2, encoding="utf-8")
            report.append(f"{ch.name}: removed empty points")
    out = pptx.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in order:                       # keep the original part order ([Content_Types].xml first)
            src = tmp / name
            if src.is_file():
                z.write(src, name, compress_type=zipfile.ZIP_STORED if name.endswith((".mp4", ".png", ".jpg", ".jpeg")) else zipfile.ZIP_DEFLATED)
    shutil.move(out, pptx)
    shutil.rmtree(tmp)
    print("\n".join(report))


if __name__ == "__main__":
    main(sys.argv[1])
