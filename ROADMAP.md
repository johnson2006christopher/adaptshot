# AdaptShot roadmap

Where the project is, what the next release is for, and the direction after that.
Shipped work is in [CHANGELOG.md](CHANGELOG.md); how to help is in
[CONTRIBUTING.md](CONTRIBUTING.md). Dates are not promised; issues are. Every item
below that is committed has an issue number, and an item without one is direction,
not a plan.

---

## v0.3.0 — *make it provable* (current)

The release that turned claims into measurements. Every number in the README and the
documentation is formatted from a committed artifact and held to it by a test.

- The first real result: PlantVillage 5-way 5-shot, 100 episodes, against three
  baselines on the same episodes and embeddings.
- The conformal guarantee measured — in distribution, under distribution shift, and
  after in-situ corrections — with the quantile bug that broke it fixed.
- Torch-free inference: the backbone ships as ONNX inside a 3.5 MB wheel; the core
  install is numpy, Pillow and onnxruntime, and one cycle peaks at about 120 MB.
- Validated on ARM in CI (a Neoverse-N2 server core), with the ONNX export path
  verified end to end there.
- Latency and memory by stage, median and p95, with the CPU named.
- A stable / experimental split of the public API and a deprecation policy.
- Documentation rebuilt: tutorials that the test suite executes, how-to guides,
  explanations, reference, contributor pages.

---

## v0.4.0 — *make it usable where it is meant to be used*

Provability first, then reach. These are the open issues:

- **Swahili localisation** (#31). Externalise the user-facing strings — errors,
  warnings, labels — behind a language setting in `AdaptShotConfig`, ship Swahili for
  the strings a field user meets, and make a third language a data change rather than
  a code change. The API stays English. The Swahili itself is reviewed by a native
  reader before it ships.
- **Phone-class measurement** (#31). The lowest device measured so far is an ARM
  server core. `python -m benchmarks.run_device` on a mid-range Android phone or a
  Raspberry Pi 4 gives the number the README's hardware claim is still waiting for —
  and if it does not fit, the README will say what the real minimum is.
- **What strangers found** (#76). Five people who have never seen the project follow
  the README; every place they got stuck becomes a `first-five-minutes` issue and is
  fixed here.
- **Citable** (#24). The Zenodo DOI and its badge, once the first release is
  published.

---

## v1.0 — *the promise becomes a contract*

Direction, not a dated plan. 1.0 means the stable API in `adaptshot.api` is frozen
as semver-major and the deprecation policy binds; nothing else on this list is a
precondition for it.

- The stable tier frozen; experimental features either promoted with measurements
  or removed.
- A methodology paper with ablations in a peer-reviewed venue, so the results have
  been checked by people outside the project.
- Field results from real partners — this project has none yet, and will not describe
  any until it does.
- A backend protocol so alternative runtimes (OpenVINO, Core ML) can be added
  without touching the learner.

---

## Beyond — *ideas, kept honest*

Things the project would like to be true one day and has no work scheduled for:
event-based (DVS) cameras and neuromorphic backends when that ecosystem matures;
privacy-preserving sharing of replay buffers between devices in one community;
low-literacy, icon-driven interfaces. None of these appears in a release until it
has an issue, a measurement, and a person who will use it.

---

## How priorities are set

Priorities follow the [project constitution](.openproject.md). For any proposed
feature:

1. Does it work on a CPU someone already owns, offline?
2. Does it keep the core install small and the cycle under 250 MB?
3. Can its benefit be measured, and will the measurement be committed?
4. Does it make the library more honest about what it does not know?

A feature that answers yes to all four is prioritised. One that fails the first is
not built.

---

*"The future of AI is not bigger — it's smarter, humbler, and more human."*
