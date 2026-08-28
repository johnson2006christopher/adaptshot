# Tambua

*Tambua* is Swahili for **identify**. Show it a handful of labelled examples per class,
and it identifies the rest — on a CPU, offline, with a human in the loop.

It is an application built on [AdaptShot](https://github.com/johnson2006christopher/adaptshot).
The library does the few-shot learning; Tambua is the interface a person actually uses.

## Install

```bash
pip install tambua        # pulls in adaptshot automatically
tambua                    # the flagship config
tambua --list-configs     # what else ships
```

The UI binds to `127.0.0.1` by default — this machine only. To reach it from a phone
on the same network, pass `--host 0.0.0.0` deliberately.

## The domain lives in the config, not the code

Loaded with `maize.yaml`, Tambua is **MziziGuard**: a crop-disease tool that speaks
Swahili and gives treatment advice. Loaded with `solar_panel.yaml` it is **SolarCheck**,
triaging photovoltaic modules for an off-grid technician. Same code, same loop.

Nothing in the application knows about maize or solar panels — a test asserts it,
by loading both configs and failing if any class or domain name appears in a `.py`
file outside the one constant that names the default config.

```yaml
domains:
  maize:
    local_name: "mahindi"
    classes:
      gray_leaf_spot:
        local_name: "ugonjwa wa mabaka ya kijivu"
        action: "Tumia dawa ya kuvu mapema..."
        severity: "high"
```

Every value is validated on load, and every problem is reported at once with the
file, the line and the fix:

```
maize.yaml, line 34: severity is "hihg"
  must be one of: low, moderate, high, critical
```

## Tambua ships no images

It used to generate them — coloured shapes drawn with `ImageDraw`, offered through the
interface as "sample data". That was removed in
[#53](https://github.com/johnson2006christopher/adaptshot/issues/53). Drawn patterns
are not data, and a model that separates a blue ring from a green blob at 95% has told
you nothing about maize.

The premise is few-shot: **five photographs per class**. Anyone who wants this has
photographs — that is why they want it. Arrange them one folder per class:

```
your_photos/
    healthy_maize/
        photo_01.jpg
    northern_leaf_blight/
        photo_02.jpg
```

**Check folder** tells you whether the folder is usable *before* you spend a training
run finding out — too few images in a class, a folder the config does not define,
unreadable files, images below the backbone's input size:

```
2 problems with this folder:

your_photos/northern_leaf_blight: has 2 usable images
  at least 3 are needed; five or more gives the prototype something to average

your_photos/healthy_maze: "healthy_maze" is not a class in the loaded configuration
  rename it to one of: gray_leaf_spot, healthy_maize, northern_leaf_blight
```

For reproducible benchmarking, `scripts/fetch_plantvillage.py` downloads real maize
photographs, pinned to a commit SHA and checksummed. It is a script, not part of the
distribution, and the images it fetches are CC BY-SA 3.0 — attribution required, and
the reason they are never bundled into the wheel.

**Tambua has never been deployed and no farmer has used it.** The first real
measurement is [#18](https://github.com/johnson2006christopher/adaptshot/issues/18);
the wider plan is [#45](https://github.com/johnson2006christopher/adaptshot/issues/45).

## Development

```bash
pip install -e apps/tambua[dev]
pytest apps/tambua -v
ruff check apps/
mypy apps/tambua/src/tambua --strict
```
