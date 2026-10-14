# STP Team 06 — presentation, video and simulation deliverables

**Neural Kinematics for Intelligent Space Manipulators** — design, implementation and simulation-based
evaluation of a collision-aware neural inverse-kinematics framework (Level 1), with the Level 2
deep-reinforcement-learning re-routing agent.

Everything in this folder was generated from the project's own code and recorded runs by the scripts
in `scripts/`. All numbers come from result files; none were typed in by hand. Robot: UFactory xArm6
in PyBullet simulation.

## Contents

| Path | What it is |
|---|---|
| `presentation/Neural_IK_STP_Team06_Level1.pptx` | 17-slide 16:9 deck, fully editable (theme, layouts, sections). Fade-in builds and slide transitions; 8 of the simulation clips embedded on their slides (each with a poster frame, so the slide still works if a video cannot play); native, editable charts; speaker notes on every slide. |
| `presentation/speaker_notes.md` | The same speaker notes as a document, with sources and code references. About 18–20 minutes. |
| `video/Neural_IK_STP_Team06_5min.mp4` | The 5-minute video: 5 min 01 s, 1920×1080, 30 fps, H.264 + AAC stereo, 47.9 MB. English narration (synthetic voice), English subtitles burned in, quiet ambient music under the voice (−20.8 LUFS integrated, peaks −1 dB). |
| `video/subtitles.srt` | English subtitles (also burned into the video). |
| `video/voiceover_script.md` | The narration script, scene by scene, with timings. |
| `video/scene_timeline.csv` | Start and end time of every scene and narration sentence. |
| `simulations/*.mp4` | Ten 1920×1080, 30 fps simulation clips with on-screen captions (list below). |
| `results/` | Measured results: `evaluation_metrics.csv`, `training_history.csv`, `experiment_configuration.json`, `eval_summary.json`, `level2_evaluation.csv`, `level2_repro_eval.json`, the two charts, and the README-baseline reproduction in `baseline_repro/`. |
| `documentation/implementation_audit.md` | What the code actually does, mapped to source files (written before the deck and video). |
| `documentation/reproducibility_guide.md` | Environment, commands, seeds and run times to regenerate everything. |
| `documentation/limitations_and_future_scope.md` | The conditions behind every result, and what comes next. |
| `scripts/` | The generators: `pipeline/` (measurements), `render/` (simulation renderer), `video/`, `deck/`. |

### Simulation clips

| File | Shows |
|---|---|
| `robot_intro.mp4` | The xArm6 from its URDF; each joint swept through its limits; the end-effector frame. |
| `forward_kinematics.mp4` | Joint angles in, one exact pose out; frames chained link by link. |
| `inverse_kinematics.mp4` | One target pose, four of its six exact joint solutions; the seed picks one. |
| `dataset_generation.mp4` | Real dataset rows: random joints → FK pose → contact check; 7,000 end-effector positions. |
| `neural_architecture.mp4` | The baseline network with its real activations for one held-out target; sine/cosine decoding. |
| `collision_awareness.mp4` | PyBullet contact vs the classifier's probability along a real motion; the training loss. |
| `training_validation.mp4` | Recorded training curves of both networks. |
| `inference_demo.mp4` | Three held-out targets: network guess (ghost) → LM refinement → PyBullet check. |
| `method_comparison.mp4` | Same target and home pose for network, classical DLS and network + LM; rates over 2,000 targets. |
| `level2_ppo_rerouting.mp4` | One unseen Level 2 episode (an obstacle appears, a joint slips) for the straight path, the potential field and the PPO agent. |

## Key results

| Result | Value | Metric | Source |
|---|---|---|---|
| Network + LM refinement | **100 %** of 2,000 held-out targets valid | success rate: within 1 mm **and** 1°, no self-collision, inside joint limits | measured, `evaluation_metrics.csv` |
| Network alone (warm start) | 22.9 mm / 5.8° mean | pose error (FK of the predicted joints vs the target) | measured |
| Plan's baseline network (warm start) | 113.6 mm / 30.6° mean | pose error, same targets | measured (retrained with the README recipe) |
| Classical DLS from the home pose | 2.6 % valid; 22.8 % self-collision; 39.3 % limit violations | success rate and violation rates | measured |
| Collision classifier | 98.37 % accuracy (95.9 % precision, 94.1 % recall) | classification accuracy on 30,000 held-out poses | measured, `eval_summary.json` |
| Level 2 agent, "everything at once" | 84.0 % reached, 2.0 % collided, 14.0 % stuck (straight path: 50.0 % collided) | episode outcome rates, 200 unseen episodes | team's recorded evaluation |
| Level 2 reproduction (same code, 20 M steps, CPU) | 81.0 % reached, 2.0 % collided, 17.0 % stuck; within 3 points of the team's agent in every scenario | same | trained and measured here, `level2_evaluation.csv` |

**How to read the metrics.** *Pose error* is measured in millimetres and degrees after running the
predicted joints through PyBullet's forward kinematics. *Joint-space error* (difference between
predicted and true angles) is only used as the training loss (MSE on sine/cosine) and is not reported
as accuracy, because several joint solutions reach the same pose. *Classification accuracy* belongs
to the collision classifier only. *Success rate* counts answers that meet the position, orientation,
self-collision and joint-limit conditions together. Level 2 rates are fractions of episodes.

## Changes from the brief

| Brief | Delivered | Why |
|---|---|---|
| `proposed_ppo.mp4`, PPO as a proposal only "unless working PPO code is found" | `level2_ppo_rerouting.mp4`, Level 2 presented as **implemented in simulation** | Working PPO code is in `level2/`, with a recorded 200-episode evaluation. It was also retrained and re-evaluated here. |
| "Architecture 256 → 256 → 128" as the model | Shown as the **baseline**; the best network is a 2.05 M-parameter residual MLP | That is what the best recorded run uses (`implementation_audit.md`). |
| Accuracy figures from the earlier notes (128 mm / 5 mm / 45 mm) | Shown as the plan baseline; final results re-measured | The project now refines with Levenberg–Marquardt (100 % within 1 mm and 1°). |
| Recorded voice-over | Synthetic narration (Kokoro-82M, Apache-2.0) and music synthesised by the build script | No recording setup or licensed music was available; both are generated and reproducible. |

Nothing requested was left out. No real-hardware footage exists: all motion is simulation.

## Regenerate

See `documentation/reproducibility_guide.md`. In short, with `NIK_WORK` pointing at a scratch directory:

```bash
python deliverables/scripts/pipeline/evaluate_methods.py      # results
python deliverables/scripts/pipeline/export_sim_data.py       # data for the clips
python deliverables/scripts/render/render_clips.py            # simulations/
python deliverables/scripts/video/build_video.py              # video/ (after narration.py)
bash   deliverables/scripts/deck/make_deck.sh                  # presentation/
```
