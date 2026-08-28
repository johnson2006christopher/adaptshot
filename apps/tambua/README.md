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

## The sample images are drawn, not photographed

`tambua.data` generates its demo images procedurally: one deterministic pattern per
configured class, derived from a hash of the class name. They are coloured shapes. Any
accuracy measured on them tells you the model can distinguish drawn patterns, and
nothing more.

They deliberately look like nothing in particular. An earlier version drew maize
leaves, which invited a viewer to believe the model was analysing foliage when it was
separating drawings.

**Tambua has never been deployed and no farmer has used it.** Evaluation against real
photographs is [#18](https://github.com/johnson2006christopher/adaptshot/issues/18);
the wider plan is [#45](https://github.com/johnson2006christopher/adaptshot/issues/45).

## Development

```bash
pip install -e apps/tambua[dev]
pytest apps/tambua -v
ruff check apps/
mypy apps/tambua/src/tambua --strict
```
