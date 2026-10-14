# Production brief – marketing video + presentation deck

**Project:** Neural Kinematics and Reinforcement Learning for Collision-Free Space Manipulator Path Planning (STP 2026 · Team 06)
**Deliverables:** (1) a 90-second marketing video with animations and plots, (2) an optional 3-minute technical cut,
(3) a 14-slide PowerPoint deck.

This file is written to be handed to **Claude Code** (run it inside WSL, in this project folder). Section 9 has
ready-to-paste prompts. Every number you may show is in **section 2 – use nothing else**.

---

## 1. One-line story

> *Tell the arm where to go – it finds the joint angles in milliseconds, then steers itself there safely
> while obstacles appear and its own joints slip.*

Three beats, in this order, everywhere (video, deck, captions):

1. **Solve** – Level 1: neural IK + LM solver → exact, self-collision-free joint angles.
2. **Steer** – Level 2: a tiny DRL agent re-routes in real time around moving / sudden obstacles and hardware errors.
3. **Together** – Mission: target pose → IK → path check → safe movement → 0.002 mm final error.

---

## 2. Fact sheet (the only numbers allowed)

All results are **simulation (PyBullet)**, **xArm6 (6-DOF)** unless stated, measured on **unseen** data.

### Level 1 – neural IK
| Fact | Value | Source |
|---|---|---|
| Best network alone, mean position error | **23.1 mm** (median 20.3 mm), orientation 5.8° | run `20260928-174703_xarm6`, 25,090 test poses |
| Network alone within 50 mm / 100 mm | 95.5 % / 99.8 % | same |
| Network size | 2.05 M parameters (`modelset/custom/model_set.py`) | same |
| Original plan baseline (network alone) | 128 mm → improved **5.5×** to 23.1 mm | Level 1 plan PDF vs. run above |
| Network + LM refinement | **0.0003 mm / 0.009° mean, 100 % within 1 mm & 1°**, 0 % self-collision, 0 % joint-limit violation | runs `20261005-155310/155138/153122`, 2,613 test poses each, warm and cold (zero-pose) start |
| Collision classifier accuracy | **97.2 %** (precision 90.8 %, recall 87.1 %) | run `20261005-155310`, 3,000 test poses |
| Robots supported | xArm6 (6-DOF), KUKA LBR iiwa (7-DOF), Franka Panda (7-DOF) | `robot_config.py` |
| Training hardware | NVIDIA RTX 3060 Laptop GPU (CUDA, WSL2) | |

### Level 2 – DRL re-routing agent
Best agent `l2_20261013-185525_xarm6`: PPO, **20 M steps, 134 min**, GPU updates + 80 simulated arms on 10 CPU workers.
200 unseen episodes per scenario, **identical seeds** for every controller.

| Scenario | DRL agent success / collisions | Potential field | Straight path |
|---|---|---|---|
| Everything at once | **84.0 % / 2.0 %** | 71.0 % / 2.5 % | 50.0 % / 50.0 % |
| Obstacles from the start | **89.5 % / 1.5 %** | 79.0 % / 1.5 % | 55.0 % / 45.0 % |
| Random mix | **88.5 % / 2.0 %** | 77.5 % / 2.5 % | 57.0 % / 43.0 % |
| Sudden obstacle | 94.0 % / 1.5 % | 91.0 % / 1.0 % | 80.5 % / 19.5 % |
| Drifting obstacles | 94.5 % / 2.0 % | 95.5 % / 2.5 % | 50.0 % / 50.0 % |
| Joint slip | 98.5 % / 1.5 % | 98.5 % / 1.5 % | 98.5 % / 1.5 % |

| Fact | Value |
|---|---|
| Policy size | **11,078 parameters** (2 × 64 MLP) |
| Decision time | **17.5 µs** per decision, NumPy, 1 CPU core, **no GPU needed** (≈ 57,000 decisions / s) |
| Stuck episodes, everything at once | 14.0 % (potential field 26.5 %) |
| Previous agent (5 M steps) for the “we improved it” beat | 78.0 % / 2.5 % on everything at once |
| Design iterations (everything at once, success / collisions) | 30/70 → 37/55 → 67/3.5 → 78/2.5 → **84/2.0** |

### Mission – Level 1 + Level 2 together (100 missions per scenario)
| Scenario | IK solved | Straight path blocked | DRL agent reached / coll. / stuck | Potential field | Straight path |
|---|---|---|---|---|---|
| Dynamic (drifting + sudden obstacles + joint slip) | **100 %** | 53 % | **95 / 4 / 1** | 95 / 5 / 0 | 58 / 42 / 0 |
| Static + sudden + joint slip | **100 %** | 53 % | **83 / 3 / 14** | 76 / 3 / 21 | 60 / 40 / 0 |

Median IK error 0.005 mm · **0 % self-collisions** · median final gripper error **0.002 mm** · ~0.25 s per mission.
Saved runs: `ui/workspace/missions/mb_20261013-211225_xarm6_dynamic_n100`, `mb_20261013-211258_xarm6_combined_n100`.

### Claims guardrails
- ✅ Say “in simulation”, “on the xArm6”, “cuts collisions from ~50 % to ~2 %”, “sub-millimetre IK”, “runs without a GPU”.
- ❌ Never say “zero collisions” for Level 2 in general (it is 1.5–2 %), “real robot”, “tested on a satellite”,
  “beats every method” (it ties the potential field on drifting obstacles), or any number not in this section.
- KUKA / Panda: “supported”, not “benchmarked”. Obstacles are spheres; the arm is moved kinematically.

---

## 3. Look & feel

Use the Studio's own design so screen recordings and generated graphics match.

| Token | Hex | Use |
|---|---|---|
| Background | `#0e1116` | video / slide background |
| Surface | `#151a21`, `#1b2129` | cards, chart backgrounds |
| Border | `#2a313c` | card outlines, grid lines |
| Text primary / secondary / muted | `#eef2f7` / `#b3bdcb` / `#7f8a9a` | all text (never colour text with series colours) |
| Accent (Level 1) | `#4b8ff0` | Level 1 chapter, highlights |
| Level 2 | `#7c5cff` | Level 2 chapter |
| Mission | `#1f9d63` | Mission chapter |
| Logo | conic gradient `#4b8ff0 → #7c5cff → #199e70` in a rounded square | title cards, corner bug |

**Chart colours (validated for colour-blind safety on the dark background – keep these identities everywhere):**
DRL agent `#3987e5` (blue) · Potential field `#d95926` (orange) · Straight path `#199e70` (aqua).
Status (always with a text label): reached `#0ca30c` · collision `#d03b3b` · stuck `#fab219`.

Type: **Segoe UI** (or Inter) – titles 600–700 weight; numbers in tabular figures. Mono for code: Cascadia Mono.
Motion: 300–500 ms ease-out, numbers count up, charts draw left-to-right, no spinning 3D text.

---

## 4. Video A – 90-second marketing cut (1920×1080, 30 fps)

| # | Time | Visual | On-screen text | Voice-over |
|---|---|---|---|---|
| 1 | 0–6 s | Black → logo builds; xArm6 silhouette fades in | **Neural IK Studio** · Team 06 | “A robot arm in space can't afford a single collision.” |
| 2 | 6–14 s | Studio, Mission page: red target frame appears, arm at start | “Where must every joint go?” | “Tell it where the gripper must go…” |
| 3 | 14–24 s | Level 1 Infer: orange ghost (network) snaps to solid answer; counter 23 mm → **0.0003 mm** | **Level 1 · Solve** · 100 % within 1 mm & 1° | “…a neural network guesses the joint angles, a solver makes them exact – in milliseconds, and never colliding with itself.” |
| 4 | 24–34 s | Mission: dashed estimated path turns **red** where it hits obstacles | “53 % of straight paths would crash” | “But the straight path to that answer often runs straight into an obstacle.” |
| 5 | 34–50 s | Split screen, same mission: straight path (aqua) **collides**; DRL agent (blue) bends around; obstacle **appears** (amber flash); joint slip warning | **Level 2 · Steer** | “So a tiny reinforcement-learning agent steers the arm – reacting to obstacles that appear or move, and to joints that slip.” |
| 6 | 50–62 s | Animated bar chart: collisions straight **50 %** → agent **2 %**; success 50 → 84 % | “Collisions 50 % → 2 %” · “Success 50 % → 84 %” | “Collisions drop from one in two… to two in a hundred.” |
| 7 | 62–72 s | Counter: **11,078 parameters · 17.5 µs** · “no GPU” chip icon | “Runs on one CPU core” | “And it's light: eleven thousand parameters, seventeen microseconds per decision – no GPU on board.” |
| 8 | 72–84 s | Full mission replay to the target; final card **0.002 mm** | **Mission · Solve + Steer** · 95 % reached · 0 % self-collision | “Solve, then steer: 95 percent of dynamic missions reach the target, to two thousandths of a millimetre.” |
| 9 | 84–90 s | Logo + three robots (xArm6, KUKA, Panda) | STP 2026 · Team 06 · Neural Kinematics & RL for collision-free space manipulators | “Neural kinematics and reinforcement learning – for collision-free space manipulators.” |

Music: calm electronic, 90–100 BPM, duck under voice by −12 dB. Captions burned in (accessibility).

## 5. Video B – 3-minute technical cut (optional)

Same chapters, plus: problem statement (brief PDF) · data generation (Studio Browse playing samples, blue/red point cloud) ·
training curves (Level 1 loss/accuracy; Level 2 success/collision with the shaded warm-up) · the LM step equation
Δθ = (JᵀJ + λI)⁻¹Jᵀe · the residual-RL idea u = u_potential-field + 2·a · design-iteration chart (30 → 37 → 67 → 78 → 84 %) ·
honest limits slide (14 % stuck, simulation only) · next step (fallback to another IK branch when stuck).

---

## 6. How to capture each asset

Start the Studio first: `wsl` → `nik-server` → open `http://127.0.0.1:8765`. **Do not start or stop training while recording.**

| Asset | Where in the Studio | Settings that tell the story |
|---|---|---|
| Target + IK snap | **Level 1 → 4 Infer** | Model `20260928-174703_xarm6`, Target “Held-out test sample”, Refinement “LM solver” → Run inference (orange ghost = network, solid = final) |
| Data generation | **Level 1 → 2 Browse** | Dataset `ds_20260928-163059_xarm6_n200000_s0`, ▶ Play at 25/s (blue = free, red = colliding) |
| Environment preview (baselines failing) | **Level 2 → 1 Environment** | Scenario “Everything”, Next ▶ until straight path = collision and potential field = timeout |
| Agent vs. baseline | **Level 2 → 3 Test scenario** | Agent `l2_20261013-185525_xarm6`, Show “Agent vs. straight path”, try seeds 1–20 and keep one where agent = success, straight = collision |
| Full mission | **Mission · L1 + L2** | Dynamic scenario, Target random, Show “Straight path”; press Next ▶ until the pipeline shows *12+ waypoints collide* and the agent succeeds |
| Mission plots | **Mission → Plots** | clearance (with 0 / 80 mm lines), distance to goal, joint angles, risk |
| Statistics charts | **Mission → Saved runs → Open** `mb_20261013-211225…` → Plots | outcome bars, clearance histogram, IK error histogram |
| Training curves | **Level 2 → Agents → Curves** on the best agent | success / collision / return |

**Manual recording (simplest):** Windows **Win + Alt + R** (Xbox Game Bar) or OBS, browser in full-screen (F11),
hide the cursor when idle. Drag the bottom panel (handle above the tabs) to give the 3D view more room; press Reset view.

**Automated recording (Claude Code can do it):** drive headless Microsoft Edge over the DevTools protocol
(`--headless=new --remote-debugging-port=9333 --use-angle=swiftshader --window-size=1920,1080`), open the page,
click the buttons above with `Runtime.evaluate`, then either `Page.startScreencast` (JPEG frames) or a loop of
`Page.captureScreenshot` while stepping the playback slider (`#l2-slider` / `#m-slider`) one step per frame – this
gives perfectly smooth 30 fps footage independent of machine speed. The session's working example of this driver
pattern used Python + `websocket-client` from the Windows `.venv` (`suppress_origin=True` is required).

**Data for animated charts (read these files – never type numbers by hand):**

| Chart | File | Fields |
|---|---|---|
| Level 2 curves | `ui/workspace/level2/policies/l2_20261013-185525_xarm6/history.json` | per update: `steps`, `success_pct`, `collision_pct`, `ep_return` |
| Level 2 results table | `…/l2_20261013-185525_xarm6/meta.json` → `evaluation.scenarios` | `policy` / `apf` / `baseline` → `success_pct`, `collision_pct`, `timeout_pct` |
| Mission statistics | `ui/workspace/missions/mb_…/mission.json` | `controllers`, `rows[]` |
| Level 1 accuracy | `ui/workspace/library/runs/20260928-174703_xarm6/meta.json` → `metrics.test` | `pos_mean_mm`, `acc_pos` |
| Level 1 LM result | `ui/workspace/library/runs/20261005-155310_xarm6/meta.json` → `metrics.test_refined`, `test_cold_refined` | |
| Original plan comparison | `figures/comparison.png`, `figures/training_curves.png` | ready-made images |

---

## 7. Build pipeline (one-time setup, then reproducible)

```bash
wsl                                   # Ubuntu, root
apt-get update && apt-get install -y ffmpeg fonts-inter
nik                                   # project + GPU venv
pip install python-pptx imageio[ffmpeg] edge-tts      # edge-tts = free neural voice-over (optional)
```

Folder layout (`media/out/` and `media/assets/` are large and already in `.gitignore`):

```
media/
  PRODUCTION_BRIEF.md     this file
  scripts/                capture_*.py, anim_*.py, build_video.py, build_deck.py
  assets/                 captured clips (*.mp4), screenshots (*.png), voice (*.mp3)
  out/                    neural_ik_90s.mp4, neural_ik_technical.mp4, neural_ik_deck.pptx
```

Render recipe:
1. **Animated charts / counters / title cards:** matplotlib `FuncAnimation` → `.mp4` (`writer=FFMpegWriter(fps=30)`),
   dark theme from section 3, 1920×1080 (`figsize=(19.2, 10.8), dpi=100`).
2. **Studio footage:** section 6 (frames → `ffmpeg -framerate 30 -i f_%05d.png -c:v libx264 -pix_fmt yuv420p clip.mp4`).
3. **Voice-over:** `edge-tts --voice en-IN-NeerjaNeural --text "…" --write-media vo_03.mp3` (or record your own).
4. **Assemble:** ffmpeg concat with 0.4 s `xfade` transitions, overlay captions (`subtitles=captions.srt`), mix music
   at −18 dB under voice, export `libx264 -crf 18 -preset slow`, AAC 192 k.
5. **Check:** watch the whole video; verify every number against section 2.

---

## 8. Deck – 14 slides (16:9, dark theme from section 3)

Build with python-pptx (or Claude Code's pptx skill). Each slide: one message as the title, ≤ 25 words of body,
one visual, speaker notes with the detail. Charts as native PowerPoint charts (editable) with the colours in section 3.

| # | Title (the message) | Visual | Speaker notes |
|---|---|---|---|
| 1 | Neural Kinematics & RL for Collision-Free Space Manipulators | logo, 3-robot strip, Team 06 · STP 2026 | Who we are, one-line story |
| 2 | The problem: reach a pose without ever colliding | brief PDF quote, Level 1 / Level 2 cards | Problem statement verbatim |
| 3 | Our answer: Solve, then Steer | 5-step pipeline graphic (target → IK → path check → DRL → final check) | How the pieces fit |
| 4 | Three arms, one pipeline | xArm6 / KUKA / Panda table: DOF, reach, joints | Robots are config entries; space arm drops in |
| 5 | Data from simulation, not from a download | Browse screenshot (blue/red cloud), sampling formula | 200 k samples, near-limit sampling, collision labels |
| 6 | Level 1: a network guesses, a solver makes it exact | bar chart: 128 mm → 23.1 mm → 0.0003 mm (log scale, labelled) | sin/cos encoding, seed, LM step equation |
| 7 | Never colliding with itself | collision classifier 97.2 %, loss = MSE + 0.05·σ(C) | classifier in training loss and in Level 2 reward |
| 8 | Level 2: a tiny agent that steers in real time | animation still: agent bends around obstacle | residual RL on potential field, Jacobian escape direction |
| 9 | What can go wrong – and does | scenario icons: sudden obstacle, drifting, joint slip, gain error | randomised every episode |
| 10 | Collisions 50 % → 2 % | grouped bars (success / collisions) agent vs potential field vs straight, “everything at once” + 2 more scenarios | identical seeds, 200 episodes |
| 11 | Small enough to fly | big numbers: 11,078 params · 17.5 µs · no GPU | GPU for training only |
| 12 | Mission: solve + steer, end to end | mission screenshot + stats: 100 % IK, 95 % reached, 0 % self-collision, 0.002 mm | 100 missions, dynamic scenario |
| 13 | How we got here | line: 30 → 37 → 67 → 78 → 84 % success across 5 iterations | each design change and why |
| 14 | Honest limits & next steps | two columns: limits (14 % stuck, simulation, spheres, xArm6 trained) / next (IK-branch fallback, KUKA agent, real space-arm URDF, dynamics) | Q&A |

Deck rules: one chart per slide max; label the bars directly; never a dual axis; agent always blue, potential field
orange, straight aqua; status colours only for reached / collision / stuck with labels; slide numbers bottom-right;
footer “STP 2026 · Team 06 · simulation results”.

---

## 9. Prompts to paste into Claude Code

**Setup**
> Read `media/PRODUCTION_BRIEF.md`. Install what section 7 lists, create the `media/` folder layout and add
> `media/out/` to `.gitignore`. Do not start, stop or restart any Studio job; if the Studio server needs restarting,
> check `GET /api/state` first and wait while a job runs.

**Charts & title animations**
> Using only numbers read from the files in section 6 (never typed by hand), write `media/scripts/anim_charts.py`
> that renders these 1920×1080, 30 fps clips into `media/assets/`: the collisions bar race (scene 6), the
> 11,078-params / 17.5 µs counter (scene 7), the IK error drop 128 → 23.1 → 0.0003 mm on a log axis (scene 3), and
> the Level 2 training curves with the shaded warm-up. Use the colours and type of section 3. Show me a frame of each.

**Studio footage**
> Write `media/scripts/capture_studio.py` that drives headless Edge as described in section 6 and records, frame by
> frame at 30 fps, the shots in the section 6 table (Level 1 inference snap, Level 2 agent vs straight path, a full
> mission). Pick seeds where the agent succeeds and the straight path collides; print the seeds you used.

**Assemble the video**
> Write `media/scripts/build_video.py` that follows the section 4 storyboard: voice-over with edge-tts, burned-in
> captions, 0.4 s cross-fades, and exports `media/out/neural_ik_90s.mp4`. Then watch-check: extract one frame per
> scene, show them to me, and verify every on-screen number against section 2.

**Deck**
> Build `media/out/neural_ik_deck.pptx` from section 8 with python-pptx: dark theme from section 3, native editable
> charts filled from the section 6 data files, speaker notes on every slide, captured screenshots from
> `media/assets/`. Render each slide to PNG (PowerPoint on Windows or LibreOffice) and show me the slides so we can
> fix layout issues before finishing.

---

*Prepared from the project's own measured results (October 2026). If you retrain or re-evaluate, update section 2
first – everything else reads from it.*
