---
title: "11 — UI Pilot Dashboard"
nav_order: 11
---

This chapter covers the optional Gradio interface in [src/adaptshot/ui/app.py](src/adaptshot/ui/app.py). It is a pilot dashboard for loading support images, predicting on a query image, and sending a correction back into the learner.

!!! note
    This UI is optional. The `ui` extra is listed in [pyproject.toml](pyproject.toml), and the dashboard depends on `gradio`.

## 1. What The UI Does

The UI wraps `FewShotLearner` in a browser-based dashboard with three sections:

- load support images
- predict on a query image
- submit a human correction

The code that builds it is `build_ui()` in [src/adaptshot/ui/app.py](src/adaptshot/ui/app.py).

Analogy: the UI is a front desk for the library. It does not change the core rules; it gives people a friendly place to use them.

## 2. Install The UI Extra

To use the dashboard, install the optional UI dependency:

```bash
pip install adaptshot[ui]
# Expected output: pip installs gradio and the library's core dependencies
```

The `ui` extra is declared in [pyproject.toml](pyproject.toml) as `gradio>=3.50.0`.

## 3. Run The Dashboard

The module includes a main block that launches the app:

```python
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", share=True)
```

That code lives in [src/adaptshot/ui/app.py](src/adaptshot/ui/app.py).

### Start It From Python

```bash
python -m src.adaptshot.ui.app
# Expected output: a Gradio app starts and prints a local URL or launch message
```

!!! warning
    The source file launches with `share=True`. That is part of the current implementation. If you need local-only behavior, change the launch call in your own copy of the app.

## 4. Walk Through The UI Components

The dashboard has three columns/sections created with `gr.Blocks`:

| Section | Component | Source behavior |
|---|---|---|
| Load Support Set | `gr.Files`, `gr.Button`, `gr.Textbox` | Upload support images, index them, and show status |
| Inference | `gr.Image`, `gr.Button`, `gr.Label`, `gr.Number`, `gr.Textbox` | Predict on a query image and display confidence |
| Human Feedback | `gr.Textbox`, `gr.Slider`, `gr.Button`, `gr.Textbox` | Submit a correction and show routing status |

See the component wiring in [src/adaptshot/ui/app.py](src/adaptshot/ui/app.py).

## 5. What Each Callback Returns

### `load_support_files(files)`

This method:

1. checks that files were uploaded
2. extracts file paths and labels from the upload locations
3. calls `learner.load_support_images(paths, labels)`
4. returns a status string

If no files are uploaded, it returns `❌ No files uploaded.`

### `predict_image(image)`

This method returns a 5-tuple:

1. prediction label as a string
2. raw confidence as a float
3. calibrated confidence as a float
4. ACT action as a string
5. a feedback prompt string

If the learner is not initialized, it returns an error tuple that starts with `Error` and includes `Learner not initialized. Upload support images first.`

### `submit_correction(true_label, confidence_weight)`

This method sends the last predicted image back into `learner.correct()` and returns a status string such as:

- `✅ Correction routed! Fine-tuned: ...`
- `❌ No prediction made yet to correct.`

## 6. How The UI Fits The Human-In-The-Loop Loop

The UI is built around the same learner methods used in the tutorials:

- `load_support_images()` from [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py)
- `predict()` from [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py)
- `correct()` from [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py)

That means the dashboard is not a separate system. It is a browser layer on top of the same source-backed pipeline.

Analogy: the UI is a window through which a person talks to the same machine, not a different machine.

## 7. What To Expect In Practice

| User action | What happens |
|---|---|
| Upload support images | The learner indexes them and prepares support embeddings |
| Upload a query image and press Predict | The learner returns prediction, raw confidence, calibrated confidence, and ACT action |
| Enter a true label and submit correction | The learner routes the correction and may fine-tune later if enough corrections accumulate |

## 8. Practical Notes

- The UI uses `AdaptShotConfig(device="cpu", seed=42)` in the source file.
- The app creates a temporary support directory with `tempfile.mkdtemp(prefix="adaptshot_support_")`.
- Labels are derived from the parent folder name in the current implementation.

## 9. Verification Checklist

- [ ] I know the UI lives in `src/adaptshot/ui/app.py`.
- [ ] I know how to install the `ui` extra.
- [ ] I can describe the three sections of the dashboard.
- [ ] I know what each callback returns.
- [ ] I can explain how the UI calls `load_support_images()`, `predict()`, and `correct()`.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
