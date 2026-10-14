# Reproducibility guide

Everything in `deliverables/` is produced by the scripts in `deliverables/scripts/`, from the project's own
code and recorded runs. No number in the deck, the video or the result files is typed by hand: the
scripts read them from the result files, and the narration script refuses to synthesise speech if a
spoken number does not match the results (`narration.py: check_numbers()`).

The project itself is never modified: the scripts read `data/`, `models/`, `ui/workspace/` and
`media/data/` and write only under `deliverables/` and a scratch directory (`$NIK_WORK`).

## 1. What was re-run and what is cited

| Result | Status | Where |
|---|---|---|
| README baseline recipe (20,000 samples, `train_ik.py`, `evaluate.py`) | **Re-run unchanged** on a CPU | `results/baseline_repro/`, `results/training_history.csv` |
| Best network (2.05 M-parameter residual MLP), whole test split | **Re-evaluated** from the committed checkpoint; identical to the recorded numbers (23.06 mm, classifier 98.37 %) | `results/eval_summary.json` |
| Seven methods × three starting poses on 2,000 held-out targets | **Measured here** | `results/evaluation_metrics.csv` |
| Best-run training history and GPU training time (RTX A6000, 718.8 s) | **Cited** from the run's `meta.json` | `ui/workspace/library/runs/20260928-174703_xarm6/` |
| Team Level 2 agent, 200 episodes per scenario | **Cited** (the agent file is not in git, only its evaluation) | `media/data/l2_20261013-185525_xarm6_meta_n200.json` |
| Level 2 reproduction (same code and settings, 20 M steps, CPU) | **Trained and evaluated here** | `results/level2_repro_eval.json`, `results/level2_evaluation.csv` |

## 2. Environment used

| Item | Version |
|---|---|
| Machine | 4-core Intel Xeon @ 2.80 GHz, 15 GB RAM, **no GPU** |
| OS | Linux (glibc 2.39) |
| Python | 3.13.16 (virtual environment made with `uv`) |
| PyTorch / NumPy / SciPy | 2.14.1 / 2.5.3 / 1.18.1 |
| PyBullet | 3.2.7, built from source for Python 3.13 |
| Matplotlib / Pillow | 3.11.2 / 12.3.0 |
| Playwright + Chromium | 1.64.0 + Chromium 1194 (WebGL through SwiftShader) |
| Three.js (renderer) | r160: the Studio's copy in `ui/static/vendor/three`, plus two MIT add-ons in `scripts/render/vendor/` |
| Speech | kokoro-onnx 0.6.1, onnxruntime 1.31.0, Kokoro-82M v1.0 (Apache-2.0), voice `af_heart` |
| Video | FFmpeg 6.1.1 (libx264, AAC) |
| Deck | Node 22, pptxgenjs 3.12.0, python-pptx 1.0.2, lxml 6.1.3 |
| Deck preview (QA only) | LibreOffice 24.2, poppler 24.02 |

### Setup

```bash
# main environment
uv venv --python 3.13 venv && source venv/bin/activate
uv pip install torch numpy scipy matplotlib pillow python-pptx lxml playwright
uv pip install pybullet==3.2.7          # builds from source on 3.13 (a C++ compiler is needed)
python -m playwright install chromium   # skip if a Chromium is already available

# speech environment (kept separate: onnxruntime and torch do not need to share one env)
uv venv --python 3.13 ttsvenv
uv pip install --python ttsvenv kokoro-onnx soundfile numpy
mkdir -p $NIK_WORK/tts    # put kokoro-v1.0.onnx and voices-v1.0.bin here
                          # (kokoro-onnx GitHub release "model-files-v1.0"), or point NIK_TTS at them

# deck
(cd deliverables/scripts/deck && npm install pptxgenjs@3.12.0)

export NIK_WORK=/path/to/scratch        # large intermediates (frames, audio, datasets); default /tmp/nik_work
```

## 3. Steps

Run from the repository root with the main environment active. Times are for the 4-core CPU above.

| # | Command | Output | Time |
|---|---|---|---|
| 1 | `python deliverables/scripts/pipeline/run_level1_baseline.py` | `results/baseline_repro/`, `results/training_history.csv` | ~1 min |
| 2 | `python deliverables/scripts/pipeline/evaluate_methods.py` | `results/evaluation_metrics.csv`, `eval_summary.json`, `experiment_configuration.json`, `$NIK_WORK/eval_per_target.npz` | ~1 min |
| 3 | `python deliverables/scripts/pipeline/make_charts.py` | `results/accuracy_comparison.png`, `results/training_validation.png` | seconds |
| 4 | `python deliverables/scripts/pipeline/train_level2_repro.py --steps 20000000 --workers 3` | `$NIK_WORK/level2_repro/policy.pt`, `train_log.jsonl` | ~2 h (115 min of active training here) |
| 5 | `python deliverables/scripts/pipeline/evaluate_level2_repro.py` | `results/level2_repro_eval.json`, `results/level2_evaluation.csv` | ~1.5 min |
| 6 | `python deliverables/scripts/pipeline/export_sim_data.py` | `$NIK_WORK/sim/*.json` (every pose, error and event shown in the clips) | a few minutes |
| 7 | `python deliverables/scripts/render/render_clips.py` | `simulations/*.mp4` (10 clips) | ~10–15 min per clip |
| 8 | `ttsvenv/bin/python deliverables/scripts/video/narration.py` | `$NIK_WORK/voice/*.wav`, `timing.json` | a few minutes |
| 9 | `python deliverables/scripts/video/build_video.py` | `video/Neural_IK_STP_Team06_5min.mp4`, `subtitles.srt`, `scene_timeline.csv`, `voiceover_script.md` | ~1.5 h |
| 10 | `bash deliverables/scripts/deck/make_deck.sh` (set `PPTX_SKILL` only if you have the validator) | `presentation/Neural_IK_STP_Team06_Level1.pptx`, `speaker_notes.md`, preview JPGs in `$NIK_WORK/deck/qa/` | ~2 min |

Nothing needs a GPU. Steps 4, 7 and 9 dominate the run time; 7 and 9 because WebGL is rendered in
software here (with a GPU, Chromium uses it and they are several times faster).

## 4. Seeds and determinism

| What | Seed / rule |
|---|---|
| README baseline dataset | `generate_dataset.py` seed 0 (20,000 samples) |
| Baseline training (instrumented copy) | `torch.manual_seed(0)`; the unchanged run uses the script as is (no seed) |
| Best run (recorded) | dataset seed 0 (200,000 samples), test split seed 2024 |
| 2,000-target comparison subset | seed 2026, drawn from test rows with dataset index ≥ 14,000 |
| Warm seeds | the run's own test seeds: true joints + N(0, 0.3 rad) |
| LM random restarts and collision-free retry | fixed generator seeds inside `refine/lm_solver.py` |
| Level 2 training (reproduction) | `level2.ppo.train(seed=1)`, 80 environments, 3 workers |
| Level 2 evaluation | seeds 50,000–50,199 for every controller and scenario (`level2/evaluate.py`) |
| Simulation clips | every frame is a pure function of time (`window.__render(t)`), so re-rendering gives the same frames |
| Clip episodes | chosen by fixed rules in `export_sim_data.py` (for example the first Level 2 seed ≥ 50,000 where the straight path collides and the agent arrives) |

What will differ on another machine:

- **Timings** (`time_ms_*` columns, the 17.5 µs Level 2 decision time) depend on the CPU and its load.
- **Training runs** are not bit-identical across hardware and thread counts (floating-point order,
  multiprocessing). The baseline network re-trained here reaches 123 mm against 128 mm in the README;
  the Level 2 reproduction lands within 3 percentage points of the team's agent in every scenario
  ("everything at once": 81.0 % vs 84.0 % reached, 2.0 % collided for both). The straight-path and
  potential-field baselines, which involve no training, reproduce the team's evaluation exactly, which
  confirms that the seeds and environment are identical. Both agents are reported side by side; the
  cited numbers are never replaced by re-run ones.
- **PyBullet DLS** depends on its settings (100 iterations, residual threshold 1e-5, no limits or
  collision information, as in `evaluate.py`); other settings give other numbers.

## 5. Checks that ran on the outputs

- FK in the renderer (JavaScript) matches PyBullet within 4.8 µm and 1.05e-5 rad on test poses.
- `narration.py` checks 19 spoken claims against the result files before generating speech.
- The deck is generated from `data.json` (built from the result files) and passes the OOXML validator;
  every slide was rendered to an image and inspected.
- Every clip and the final video were checked frame by frame at representative times; the video's
  duration, resolution, frame rate and audio levels are reported in `deliverables/README.md`.
