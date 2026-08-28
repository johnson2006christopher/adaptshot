# Tambua

*Tambua* is Swahili for **identify**. Show it a handful of labelled examples per class,
and it identifies the rest — on a CPU, offline, with a human in the loop.

It is an application built on [AdaptShot](https://github.com/johnson2006christopher/adaptshot).
The library does the few-shot learning; Tambua is the interface a person actually uses.

## Install

```bash
pip install tambua        # pulls in adaptshot automatically
tambua --config configs/maize.yaml
```

The UI binds to `127.0.0.1` by default — this machine only. To reach it from a phone
on the same network, pass `--host 0.0.0.0` deliberately.

## The domain lives in the config, not the code

Loaded with `configs/maize.yaml`, Tambua is **MziziGuard**: a crop-disease tool that
speaks Swahili and gives treatment advice. Loaded with a different config, the same
application classifies something else. Nothing in the code knows about maize.

```yaml
crops:
  maize:
    swahili: "mahindi"
    diseases:
      gray_leaf_spot:
        swahili: "ugonjwa wa mabaka ya kijivu"
        action: "Tumia dawa ya kuvu mapema..."
        severity: "high"
```

> Generalising `crops:`/`diseases:` to `domains:`/`classes:` is
> [#47](https://github.com/johnson2006christopher/adaptshot/issues/47).

## The sample images are drawn, not photographed

`tambua.data` generates its demo images procedurally with `ImageDraw.ellipse()`. They
are coloured shapes. Any accuracy measured on them tells you the model can distinguish
drawn ovals, and nothing more.

**Tambua has never been deployed and no farmer has used it.** Evaluation against real
photographs is [#18](https://github.com/johnson2006christopher/adaptshot/issues/18);
the wider plan is [#45](https://github.com/johnson2006christopher/adaptshot/issues/45).

## Development

```bash
pip install -e apps/tambua[dev]
pytest apps/tambua/tests -v
ruff check apps/
mypy apps/tambua/src/tambua --strict
```
