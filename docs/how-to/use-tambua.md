# Use Tambua, the web application

> **For:** someone who wants a point-and-click interface — for themselves, or for people who will never open a terminal — rather than Python scripts. Assumes AdaptShot is installed; Tambua is a separate, small install on top.

## What Tambua is

*Tambua* is Swahili for *identify*. It is a general few-shot classification application built on AdaptShot: you point it at a folder of example photographs, it learns them, and it identifies new ones in a browser page — with the prediction set, the confidence, and the "ask a person" signal shown rather than hidden. The domain comes from a configuration file, not from the code: the same application runs maize disease and solar-panel inspection from two YAML files, and yours from a third.

## Install and start

```bash
pip install tambua
tambua --list-configs
tambua --config maize
```

Then open the address it prints (by default `http://127.0.0.1:7860`) in a browser. `--port` and `--host` change where it listens; `--share` creates a temporary public link through Gradio's tunnel — which needs the network, and which you should not use for anything you would not put on the internet.

Tambua is installed from the repository during 0.3.0 (`pip install ./apps/tambua`); the PyPI name is reserved for it.

## The two bundled configurations

| config | domain | classes |
|---|---|---|
| `maize` | maize leaf disease | healthy, gray leaf spot, northern leaf blight |
| `solar_panel` | photovoltaic module inspection | as named in the file |

Each is a YAML file under `tambua/configs/`. A configuration names the domain, the classes with a local-language label and an advice line for each, the backbone and α, and where photographs live.

## Write your own configuration

Copy `maize.yaml`, change the class keys and their labels, and point `paths` at your folders. Tambua validates the file before starting and reports the exact line of any mistake:

```text
tambua/configs/mine.yaml:14: unknown key 'diseases' -- did you mean 'classes'?
```

Renamed keys from earlier versions (`crops`, `diseases`, `swahili`) are recognised and named in the error. The [Tambua reference](../reference/tambua-cli.md) lists every key.

## What the page shows

- The prediction and its calibrated confidence.
- The prediction set, and whether it is calibrated — before the learner has enough photographs the page says *no guarantee yet* rather than showing a set that has none.
- The action: accept, or ask a person, with the reason.
- A correction control that feeds `correct()` — the same loop as [tutorial 5](../tutorials/05-teaching-corrections.md).

## Offline

Tambua needs the network for `pip install` and for nothing else. The model is bundled; the page is served from your own machine. Switching the wifi off after install changes nothing except that `--share` stops working.
