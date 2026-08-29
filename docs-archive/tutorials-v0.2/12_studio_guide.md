# 12. Studio Moved Out of the Library

This chapter documented `adaptshot.studio`, a Gradio desktop application that
shipped inside the library. It has been extracted to its own project ([#21]),
together with `adaptshot.ui` before it ([#22]).

The library now ships **no** graphical interface, and a test enforces that: see
`tests/test_library_ships_no_gui.py`, which fails if a GUI reappears anywhere
under `src/adaptshot/`.

## Why it matters more than tidiness

AdaptShot's premise is that it runs on a CPU, in a field, with no internet. A web
framework in the dependency graph of a library making that claim is a
contradiction someone will eventually notice — and it should be us who notice
first.

## Where the interface lives now

**[Tambua](../guides/tambua-complete-guide.md)** — a separate distribution built
on AdaptShot:

```bash
pip install tambua
tambua --list-configs
```

The studio's own history is preserved on the
[`studio-extract`](https://github.com/johnson2006christopher/adaptshot/tree/studio-extract)
branch, ready to become its own repository.

See also [chapter 11](11_ui_pilot_dashboard.md), which covers the earlier
removal.

[#21]: https://github.com/johnson2006christopher/adaptshot/issues/21
[#22]: https://github.com/johnson2006christopher/adaptshot/issues/22
