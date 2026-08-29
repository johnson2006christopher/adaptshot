# How this documentation is tested

> **For:** someone editing a page and wondering what will fail if they get it wrong. Everything on this page runs on every pull request.

## Why documentation has tests

Documentation drifts silently. Nothing fails, nothing goes red — a sentence stops being true and stays on the site. This repository once shipped, at the same time, an application described as deployed that had never run outside a test, a claim of torch-free inference when the data directory held only `__init__.py`, a latency figure whose own artifact recorded almost double, and links to a filesystem path on a machine that no longer existed. The remedy is not to check more carefully. It is to make the document and the thing it describes the same source, so a stale claim fails a test.

## The four guards

### 1. Every tutorial and how-to runs as written — `tests/test_docs_tutorials_run.py`

Each page under `docs/tutorials/` and `docs/how-to/` has its ```python blocks concatenated in order and executed in a fresh interpreter, in an empty directory, with outbound sockets disabled and — unless the page says it needs torch — torch blocked at the import system. The page must exit 0. It must also open with a `> **For:**` line saying who it is for.

Conventions: shell commands go in ```bash blocks (shown, not run); a block containing `# docs: not run -- <reason>` is shown and skipped; a page that needs the torch extra carries `<!-- needs: torch -->` and is skipped where torch is absent. Pages must create anything they read — [tutorial 3](../tutorials/03-your-own-photos.md) builds its folder from the bundled photographs for exactly this reason.

### 2. Every number traces to an artifact — `tests/test_docs_claims.py`

Figures in the README, the technical note and the guarantee page are *formatted from* `results/*.json` and asserted to appear verbatim: accuracies with intervals, coverage, set size, the baseline comparison, latency by stage, cold start, single-cycle memory, the shift curve. Edit either side alone and the suite fails. It caught a misquoted interval on its first run.

The same file forbids absolute filesystem links, and it guards the MziziGuard prototype: no live page may claim it was ever put into use — it was not, see #17 — and any page that mentions it must disclose that the images its earlier documentation used were synthetic, drawn with Pillow, not photographs. Pages retired to `docs-archive/` at the repository root are outside the site: not built, not linked, not tested. They are history.

### 3. The API reference matches the code — `tests/test_api_surface.py`

Every name in `adaptshot.api`'s tiers must appear in `docs/reference/api.md` under the matching heading — stable names in the Stable section, experimental in Experimental — and every experimental object's docstring must open with the marker. Rename a class and forget the reference and this fails.

### 4. Nothing is orphaned or broken — `mkdocs build --strict`

Every page under `docs/` must be in the `nav` in `mkdocs.yml` or listed under `not_in_nav`; every relative link must resolve; the site must build without a warning. This is stage 5 of the [validation gate](development-setup.md) and runs as its own CI job on every pull request, before the deploy that only happens after merge.

## Adding a page

1. Decide what it is — tutorial, how-to, explanation, reference — and put it in that folder. One page cannot be two of these; that is the whole technique.
2. Open with `# Title` and a `> **For:**` line.
3. If it has Python, make it run from an empty directory with no network; use the bundled photographs.
4. If it quotes a number, add it to the tracing test, or link to the page that already traces it.
5. Add it to `nav`.
6. Run the gate.

## Editing a page

Every page has an edit link (the pencil icon) that opens it on GitHub. A pull request that fixes one sentence is welcome and will run the same four guards.
