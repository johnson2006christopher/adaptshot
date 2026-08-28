#!/usr/bin/env python3
"""Fetch real maize photographs from the PlantVillage dataset.

Tambua ships no images (#53). This downloads real ones so that a benchmark can
be run against photographs instead of drawings (#18). It is a script, not part
of any distribution: nothing here is installed by `pip install tambua`, and no
image it downloads is ever committed.

Licence
-------
The licence is genuinely contested across sources, so it is stated here rather
than assumed:

* `github.com/spMohanty/PlantVillage-Dataset`, the canonical repository this
  script downloads from, has **no LICENSE file**.
* `huggingface.co/datasets/mohanty/PlantVillage`, published under the original
  first author's own namespace, states **CC BY-SA 3.0**. This is the most
  authoritative statement available and is what this script relies on.
* `zenodo.org/records/1204914` states CC BY 4.0, but is a third-party re-upload
  by different authors, "modified from the original" -- a stranger's claim about
  someone else's data, and weaker evidence despite the formal archive.

CC BY-SA 3.0 means attribution and share-alike. Two consequences:

1. The citation below must accompany any published result. The script prints it.
2. Share-alike is a further reason not to bundle these images into the wheel.
   They are downloaded to the user's machine and stay there.

Citation::

    Mohanty, S.P., Hughes, D.P., Salathé, M. (2016). Using deep learning for
    image-based plant disease detection. Frontiers in Plant Science 7:1419.
    https://doi.org/10.3389/fpls.2016.01419

Reproducibility
---------------
The download is pinned to a commit SHA, so it is content-addressed: the same
invocation fetches the same bytes forever, regardless of what lands on the
default branch later. Every file's SHA-256 is written to a manifest, and
``--verify`` re-checks an existing download against it.

Usage::

    python scripts/fetch_plantvillage.py --out data/plantvillage_maize
    python scripts/fetch_plantvillage.py --out data/plantvillage_maize --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

#: Pinned so the download is reproducible. Content-addressed by git, so the same
#: SHA can never yield different bytes.
COMMIT = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"

REPO = "spMohanty/PlantVillage-Dataset"
LICENCE = "CC BY-SA 3.0 (per huggingface.co/datasets/mohanty/PlantVillage)"
CITATION = (
    "Mohanty, S.P., Hughes, D.P., Salathe, M. (2016). Using deep learning for "
    "image-based plant disease detection. Frontiers in Plant Science 7:1419. "
    "https://doi.org/10.3389/fpls.2016.01419"
)

#: PlantVillage's directory names mapped onto the class keys in
#: `apps/tambua/src/tambua/configs/maize.yaml`. The mapping lives here, in a
#: script, and not in the application -- Tambua knows nothing about maize, and a
#: test enforces that.
CLASS_MAP = {
    "Corn_(maize)___healthy": "healthy_maize",
    "Corn_(maize)___Northern_Leaf_Blight": "northern_leaf_blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "gray_leaf_spot",
}

#: A wider pool for the 5-way benchmark (#18), which needs more than five
#: classes to draw from: with exactly five, every episode is the same episode
#: and only the image sampling varies. Twenty gives C(20,5) = 15,504 possible
#: class combinations, so 100 episodes are genuinely different problems.
#:
#: Chosen to span crops rather than to flatter the result. Several sets are
#: deliberately hard -- six tomato diseases that look alike under the same lab
#: conditions, and a "healthy" class for six different crops, so the frozen
#: backbone cannot separate episodes on plant species alone.
#:
#: The class names are kept verbatim from PlantVillage rather than renamed. The
#: maize mapping above exists because Tambua's config uses its own keys; the
#: benchmark has no such constraint, and renaming would only obscure which
#: directory a number came from.
BENCHMARK_CLASSES = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Peach___Bacterial_spot",
    "Pepper,_bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
)


def _slug(source_dir: str) -> str:
    """A filesystem-safe directory name for a PlantVillage class."""

    return "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in source_dir
    )

API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "adaptshot-fetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data: bytes = response.read()
    return data


def _listing(source_dir: str) -> list[str]:
    """Filenames in one PlantVillage class directory, sorted for determinism."""

    url = API.format(
        repo=REPO, path=urllib.parse.quote(f"raw/color/{source_dir}"), ref=COMMIT
    )
    entries = json.loads(_get(url))
    return sorted(entry["name"] for entry in entries if entry["type"] == "file")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(
    out_dir: Path, per_class: int, classes: dict[str, str] | None = None
) -> dict[str, str]:
    """Download `per_class` images for each mapped class. Returns path -> sha256."""

    manifest: dict[str, str] = {}
    for source_dir, class_key in (classes or CLASS_MAP).items():
        names = _listing(source_dir)[:per_class]
        if len(names) < per_class:
            print(f"  ! only {len(names)} images available for {class_key}", file=sys.stderr)
        target = out_dir / class_key
        target.mkdir(parents=True, exist_ok=True)
        print(f"  {class_key}: {len(names)} images")
        for name in names:
            path = urllib.parse.quote(f"raw/color/{source_dir}/{name}")
            data = _get(RAW.format(repo=REPO, ref=COMMIT, path=path))
            (target / name).write_bytes(data)
            manifest[f"{class_key}/{name}"] = _sha256(data)
    return manifest


def verify(out_dir: Path) -> int:
    """Re-check a previous download against its manifest. Returns an exit code."""

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}; nothing to verify against", file=sys.stderr)
        return 1

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("commit") != COMMIT:
        print(
            f"manifest was written from commit {record.get('commit')}, "
            f"this script is pinned to {COMMIT}",
            file=sys.stderr,
        )
        return 1

    bad = []
    for relative, expected in record["files"].items():
        path = out_dir / relative
        if not path.is_file():
            bad.append(f"{relative}: missing")
        elif _sha256(path.read_bytes()) != expected:
            bad.append(f"{relative}: checksum mismatch")

    if bad:
        print(f"{len(bad)} problem(s):", file=sys.stderr)
        for line in bad[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"verified {len(record['files'])} files against {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into")
    parser.add_argument(
        "--per-class", type=int, default=30,
        help="Images per class (default: 30, enough for 5-shot with a query set)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Re-check an existing download against its manifest, and exit",
    )
    parser.add_argument(
        "--preset", choices=("maize", "benchmark"), default="maize",
        help=(
            "Which classes to fetch. 'maize' is the three classes Tambua's "
            "config names; 'benchmark' is the 20-class pool the 5-way benchmark "
            "samples episodes from (#18)."
        ),
    )
    args = parser.parse_args(argv)

    classes = (
        CLASS_MAP
        if args.preset == "maize"
        else {source: _slug(source) for source in BENCHMARK_CLASSES}
    )

    if args.verify:
        return verify(args.out)

    print(f"PlantVillage, pinned to {COMMIT[:12]}")
    print(f"Licence: {LICENCE}")
    print("These images are not ours. Cite them:")
    print(f"  {CITATION}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        files = fetch(args.out, args.per_class, classes)
    except urllib.error.URLError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print("This script needs network access. Nothing else in the project does.", file=sys.stderr)
        return 1

    (args.out / "manifest.json").write_text(
        json.dumps(
            {"repo": REPO, "commit": COMMIT, "licence": LICENCE,
             "citation": CITATION, "preset": args.preset, "files": files},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(files)} images in {args.out}")
    print(f"Re-check any time: python {sys.argv[0]} --out {args.out} --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
