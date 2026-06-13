---
title: "12 — AdaptShot Studio Guide"
nav_order: 12
---

AdaptShot Studio is the optional offline GUI for AdaptShot. It lives in [src/adaptshot/studio/app.py](src/adaptshot/studio/app.py) and wraps the same learner surface used by the Python API.

Studio is a convenience layer, not the primary interface. If you are writing code, start with `FewShotLearner` and use Studio only when you want a browser workflow.

!!! note
    Studio is optional and CPU-first. Install it with `pip install "adaptshot[gui]"`.

## 1. What Studio Supports Today

The UI provides eight tabs:

- Setup & Configuration
- Dataset & Class Management
- Train & Predict
- Human-in-the-Loop Correction
- Calibration & ACT Tuning
- Buffer & Memory Management
- Export & Deployment
- Logs & Diagnostics

The implementation only calls backend methods that already exist in the codebase:

- `FewShotLearner.load_support_images()`
- `FewShotLearner.predict()`
- `FewShotLearner.correct()`
- `FewShotLearner.save()`
- `FewShotLearner.load()`
- `CalibrationEngine.current_temperature` and `CalibrationEngine.current_ece`
- `ACTEngine.get_all_thresholds()` and `ACTEngine.reset_class()`
- `UPUGFPruner.compute_scores()` and `FewShotLearner._apply_buffer_management()` for buffer enforcement

Studio now exposes the native checkpoint bundle, TorchScript, and ONNX export paths that the current code supports. When a feature does not exist yet, it shows a literal TODO marker such as `[TODO: Implement in src/adaptshot/core/learner.py]` instead of inventing behavior.

## 2. Install The GUI Extra

```bash
pip install "adaptshot[gui]"
```

The `gui` extra adds Gradio, Pandas, and the lightweight ONNX exporter dependencies used by the browser interface and deployment bundles.

## 3. Launch Studio

Studio can be started in two ways:

```bash
adaptshot-studio
```

or:

```bash
python -m adaptshot.studio.app
```

The launcher binds to `http://127.0.0.1:7860` by default and does not enable sharing.

## 4. Configuration Workflow

The first tab validates a CPU-only configuration with these controls:

- backbone selection
- eco mode toggle
- max buffer size slider
- calibration method dropdown
- seed input

The validation callback returns:

- a JSON preview of the active config
- a heuristic RAM/latency estimate
- a human-readable validation message

These numbers are estimates for display only. They are not benchmark claims.

## 5. Support Set Workflow

Studio loads support images into `FewShotLearner.load_support_images()`. It supports three label strategies:

- `folder`
- `stem`
- `manual`

The dataset tab also shows a support table, a preview gallery, and a class distribution summary.

## 6. Inference And Corrections

The inference tab runs `FewShotLearner.predict()` on one image or a batch of uploaded images. The table includes:

- prediction
- raw confidence
- calibrated confidence
- uncertainty flag
- ACT action
- latency per image

The correction tab sends a selected prediction through `FewShotLearner.correct()` with a human confidence weight.

## 7. Calibration, ACT, And Buffering

The calibration tab refreshes the learner's ACT thresholds and refits temperature when enough observations exist. The buffer tab displays a snapshot of the current replay buffer and can trigger the existing buffer management path.

The export tab supports native checkpoint bundles, TorchScript, and ONNX. It also lets you load a saved project bundle back into Studio so you can continue where you left off.

## 8. Logs And Diagnostics

The diagnostics tab shows:

- Python and Torch versions
- CPU count
- current RSS memory
- offline-only status
- the in-app log buffer

The log export button writes `adaptshot_studio.log` locally.

## 9. Practical Limits

- Uploads are capped at 100 MB per session in the UI.
- State is stored in `~/.adaptshot/studio_session.json` as a lightweight snapshot.
- The browser session resets on refresh; the learner object itself stays in memory only.

## 10. Verification Checklist

- [ ] I can launch Studio with `adaptshot-studio`.
- [ ] I know the GUI extra name is `gui`.
- [ ] I know which learner methods Studio calls.
- [ ] I understand which features are TODOs because the backend does not yet expose them.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
