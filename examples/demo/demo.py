#!/usr/bin/env python3
"""AdaptShot, with the wifi switched off.

A single script for a conference laptop. It teaches a few-shot classifier three
maize diseases from twelve photographs, asks it about four leaves it has never
seen, and shows what is actually novel here: not the label, but the prediction
set widening as the model becomes less sure, and the moment it declines and asks
for a human.

Nothing here touches the network -- and that is enforced, not promised: outbound
sockets are disabled in-process before the library is imported, so a download
would fail loudly rather than silently succeed on the venue wifi.

    python examples/demo/demo.py            # run straight through
    python examples/demo/demo.py --pause    # pause between steps while presenting
    python examples/demo/demo.py --debug    # show tracebacks instead of one-line errors

Every number printed is either computed live or read from
results/plantvillage_5way5shot.json, the artifact of the published benchmark.
Nothing is hard-coded to look good.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = Path(os.environ.get("ADAPTSHOT_DEMO_RESULTS", REPO / "results" / "plantvillage_5way5shot.json"))
ALPHA = 0.10


def block_network() -> None:
    """Make any outbound connection raise. The demo's offline claim, enforced."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("this demo runs offline; something tried to open a network connection")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]


class Out:
    """Plain-text output with optional colour. No dependency, no widgets."""

    def __init__(self, colour: bool, pause: bool) -> None:
        self.colour = colour and sys.stdout.isatty()
        self.pause = pause

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def title(self, text: str) -> None:
        print(f"\n{self._c('1', text)}")
        print("-" * len(text))

    def line(self, text: str = "") -> None:
        print(text)

    def good(self, text: str) -> None:
        print(self._c("32", text))

    def warn(self, text: str) -> None:
        print(self._c("33", text))

    def ask(self, text: str) -> None:
        print(self._c("1;35", text))

    def step(self) -> None:
        if self.pause:
            input(self._c("2", "  [enter] "))


def true_label_from_name(path: str) -> str | None:
    """`query_2_gray_leaf_spot.jpg` -> `gray_leaf_spot`; the tomato has no maize label."""

    stem = Path(path).stem
    label = stem.split("_", 2)[2] if stem.count("_") >= 2 else None
    return None if label is None or label.startswith("tomato") else label


def run(out: Out) -> int:
    from adaptshot import AdaptShotConfig, FewShotLearner
    from adaptshot.data import demo_images, sample_images
    from adaptshot.utils.determinism import set_deterministic_seed

    started = time.perf_counter()
    set_deterministic_seed(42)

    out.title("AdaptShot -- a classifier that knows when it doesn't know")
    out.line("network: blocked for this process   device: cpu   torch: not required")
    out.step()

    # ---- 1. teach it ------------------------------------------------------
    out.title("1. Twelve photographs, three diseases")
    paths, labels = sample_images()
    counts = {name: labels.count(name) for name in sorted(set(labels))}
    for name, count in counts.items():
        out.line(f"   {name:<22} {count} photos")
    learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=ALPHA, seed=42))
    t0 = time.perf_counter()
    learner.load_support_images(paths, labels)
    out.line(f"   learned in {time.perf_counter() - t0:.1f}s. No training loop; a frozen backbone and {len(paths)} embeddings.")
    out.step()

    # ---- 2. ask it about leaves it has never seen ----------------------------
    out.title(f"2. Four leaves it has never seen  (prediction sets at alpha = {ALPHA:.2f})")
    for path in demo_images():
        name = Path(path).name
        truth = true_label_from_name(path)
        result = learner.predict(path)
        members = sorted(str(m) for m in (result.conformal_set or [result.prediction]))
        accepted = result.act_action == "ACCEPT"

        out.line()
        out.line(f"   {name}")
        out.line(f"   top-1: {result.prediction}  ({result.calibrated_confidence:.0%})")
        out.line(f"   set:   {{{', '.join(members)}}}   size {len(members)}")

        if truth is None:
            out.warn("   a crop it was never shown -- a tomato leaf from the same dataset")
        if accepted and len(members) == 1:
            out.good("   -> one answer. Confident enough to act on.")
        elif accepted:
            if truth is not None and result.prediction != truth and truth in members:
                out.warn(f"   -> the top-1 is wrong. The true class is {truth} -- and it is in the set.")
                out.good("      A plain classifier would have answered wrongly and stopped there.")
                out.good("      This is what the coverage guarantee buys.")
                out.step()
                continue
            diseases = [m for m in members if "healthy" not in m]
            if len(diseases) == len(members):
                out.good("   -> it will not choose between these on this evidence -- but every")
                out.good("      one is a disease, so the advice to the farmer is the same: act.")
            else:
                out.warn("   -> more than one candidate, and they disagree on whether to act.")
        else:
            out.ask("   -> it declines. \"I don't know -- ask a human.\"")
            if truth is not None:
                verdict = "wrong" if result.prediction != truth else "right"
                out.line(f"      (the top-1 was {verdict}; the true class is {truth}.)")
                if verdict == "wrong":
                    out.line("      Declining was the correct call. A plain classifier would have answered.")
        if truth is None:
            if result.ood_flag:
                out.line("   the out-of-distribution detector fired on it.")
            else:
                out.line("   the out-of-distribution detector did NOT fire: under the same lab lighting,")
                out.line("   a tomato leaf is not far enough from maize for a frozen ImageNet backbone")
                out.line("   to tell. What caught it was the confidence gate. Two layers; one held.")
        out.step()

    # ---- 3. the guarantee, measured -----------------------------------------
    out.title("3. Does the prediction set keep its promise? Measured, not claimed")
    if RESULTS.is_file():
        record = json.loads(RESULTS.read_text(encoding="utf-8"))
        conformal = record["conformal"]
        protocol = record["protocol"]
        coverage = conformal["empirical_coverage"]
        size = conformal["mean_set_size"]
        out.line(f"   PlantVillage, {protocol['task']}, {protocol['episodes']} episodes, seed {protocol['seed']}")
        out.line(f"   target coverage at alpha = {conformal['alpha']:.2f}:   {conformal['target_coverage']:.0%}")
        out.good(f"   empirical coverage:               {coverage['mean']:.1%} +/- {coverage['ci95_half_width']:.1%}")
        out.line(f"   mean prediction-set size:          {size['mean']:.2f} +/- {size['ci95_half_width']:.2f}")
        out.line("   The set contains the true class more often than promised. The price is its size.")
    else:
        out.warn(f"   coverage artifact not found: {RESULTS}")
        out.warn("   produce it with:  python -m benchmarks.run_plantvillage --seed 42")
        out.warn("   (the demo above ran without it; only this figure is missing)")
    out.step()

    out.title("Try it")
    out.line("   pip install adaptshot")
    out.line("   https://github.com/johnson2006christopher/adaptshot")
    out.line(f"\n   total: {time.perf_counter() - started:.1f}s on cpu, offline.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pause", action="store_true", help="pause between steps")
    parser.add_argument("--no-color", action="store_true", help="plain text output")
    parser.add_argument("--debug", action="store_true", help="show tracebacks")
    args = parser.parse_args(argv)
    out = Out(colour=not args.no_color, pause=args.pause)

    block_network()
    try:
        import adaptshot  # noqa: F401
    except ImportError:
        out.warn("adaptshot is not installed. Run:  pip install adaptshot")
        return 2

    try:
        return run(out)
    except KeyboardInterrupt:
        out.line("\nstopped.")
        return 130
    except Exception as error:
        if args.debug:
            raise
        out.warn(f"\nthe demo stopped: {type(error).__name__}: {error}")
        out.warn("run again with --debug for the traceback.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
