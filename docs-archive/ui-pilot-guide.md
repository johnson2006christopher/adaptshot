# The Pilot Dashboard Moved Out

This guide documented the Gradio pilot dashboard that shipped inside AdaptShot.
The library shipped **two** graphical interfaces at once — `adaptshot.ui.app` and
`adaptshot.studio` — which is one more than the maximum. Both are gone:
`adaptshot.ui` in [#22], `adaptshot.studio` in [#21].

`tests/test_library_ships_no_gui.py` keeps it that way: it fails if a GUI
reappears anywhere under `src/adaptshot/`, if a `ui` or `gui` extra comes back,
or if a live document points at a removed entrypoint.

## What replaces it

**[Tambua](tambua-complete-guide.md)**, a separate distribution built on the
library:

```bash
pip install tambua
tambua
```

It covers everything the pilot dashboard did — loading a support set, predicting,
routing a correction back into the learner — and rather more: conformal
prediction sets instead of a bare confidence number, folder validation before
training, and a domain that comes from a configuration file rather than the code.

## Related

- [Chapter 11](../tutorials/11_ui_pilot_dashboard.md) — the earlier removal
- [Chapter 12](../tutorials/12_studio_guide.md) — the studio extraction
- [Studio Moved Out](studio-complete-guide.md) — where that history lives

[#21]: https://github.com/johnson2006christopher/adaptshot/issues/21
[#22]: https://github.com/johnson2006christopher/adaptshot/issues/22
