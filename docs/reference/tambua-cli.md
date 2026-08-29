# Tambua: command line and configuration reference

> **For:** someone writing or debugging a Tambua configuration, or scripting its launch. For the walkthrough, see [use Tambua](../how-to/use-tambua.md).

## Command line

```text
tambua [--config CONFIG] [--list-configs] [--port PORT] [--host HOST] [--share]
```

| option | meaning | default |
|---|---|---|
| `--config` | a bundled configuration name (`maize`, `solar_panel`) or a path to a YAML file | `maize` |
| `--list-configs` | print the bundled configurations and exit | — |
| `--port` | port to serve on | `7860` |
| `--host` | interface to bind; `0.0.0.0` for other machines on the network | `127.0.0.1` |
| `--share` | a temporary public link through Gradio's tunnel (needs the network; no authentication) | off |

The configuration is validated before anything starts. A mistake is reported with the file and line, and a renamed key from an earlier version is named:

```text
mine.yaml:14: unknown key 'diseases' under domain 'maize' -- it was renamed to 'classes'
```

## Configuration file

Five top-level sections; `application` and `domains` are required.

```yaml
application:
  name: "MziziGuard"            # shown in the page title
  version: "0.2.0"
  description: "…"

engine:                          # any AdaptShotConfig field; defaults are the library's
  backbone: "mobilenet_v3_small" # the bundled one; resnet18 needs the torch extra
  device: "cpu"
  seed: 42
  inference_mode: "prototypical" # prototypical | nearest_neighbor | contrastive
  similarity_metric: "euclidean" # euclidean | cosine
  eco_mode: true
  enable_ood_detection: true
  conformal_alpha: 0.1           # 90% promise; needs 9 photographs per teaching set
  conformal_mode: "split"

domains:                         # at least one; each needs at least two classes
  maize:
    local_name: "mahindi"
    classes:
      healthy_maize:
        local_name: "mahindi yenye afya"
        action: "…"              # required: the advice shown when this class is identified
        description: "…"         # required
        severity: "low"          # low | moderate | high | critical

localization:
  language: "sw"                 # the language local_name is written in
  fallback: "en"                 # used when a class has no local_name

paths:
  model_dir: "models"            # where taught learners are saved, relative to this file
  sample_data: "samples"
```

### Keys

| section | keys | required |
|---|---|---|
| `application` | `name`, `version`, `description` | section |
| `engine` | any `AdaptShotConfig` field — see the [config reference](config-reference.md) | — |
| `domains.<key>` | `local_name`, `classes` | `classes` |
| `domains.<key>.classes.<key>` | `local_name`, `action`, `description`, `severity` | `action`, `description` |
| `localization` | `language`, `fallback` | — |
| `paths` | `model_dir`, `sample_data` | — |

"MziziGuard" is the application name in the bundled maize configuration. It was a prototype whose earlier documentation used synthetic, Pillow-drawn images rather than photographs, and it has never been put into use; the name survives only as a config value. Renamed since those files, and recognised in error messages: `crops` → `domains`, `diseases` → `classes`, `swahili` → `local_name`.

Class keys are what the learner is taught with, so they must match the folder names of your teaching photographs. Local names and actions are what the person using the page sees.

## What the page returns

Each identification the page shows is an `Identification` with: the prediction set (as class keys) and whether it is calibrated, α, the empirical coverage so far and the calibration size, the class key, its local name, calibrated and raw confidence, the class's `action` and `severity`, and the "ask a person" decision. The same object is available from `TambuaEngine` in Python for scripting a batch.
