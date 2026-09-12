"""The ``tambua`` command.

pip installs console scripts unconditionally: a core install (``pip install
adaptshot``) gets a ``tambua`` on PATH even though gradio is not there. So this
module must import cleanly, and run ``--help`` and ``--list-configs``, on a
core install -- everything gradio-shaped is imported lazily, inside
:func:`launch`, after argument parsing. A core user who types ``tambua`` gets
one sentence naming the extra to install, not a traceback.
"""

from __future__ import annotations

import argparse
import os
import sys

from adaptshot.app.config import load_config
from adaptshot.app.engine import DEFAULT_CONFIG, bundled_config, bundled_configs

#: What a core user is told when the UI's dependencies are missing.
_INSTALL_HINT = (
    "Tambua's interface needs gradio, which is not installed. "
    'Install the app extra:  pip install "adaptshot[app]"'
)


def launch(argv: list[str] | None = None) -> None:
    """Console-script entry point for ``tambua``.

    Args:
        argv: Command-line arguments. ``None`` reads them from ``sys.argv``.
    """

    parser = argparse.ArgumentParser(
        prog="tambua",
        description="Tambua — few-shot image classification, powered by AdaptShot",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to a domain config. Defaults to the bundled "
            f"{DEFAULT_CONFIG!r} configuration; run with --list-configs to see "
            "what else ships."
        ),
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List the configurations bundled with this installation, and exit.",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio server (default: 7860)",
    )
    parser.add_argument(
        # Previously hardcoded to 0.0.0.0, which serves the UI to every machine on
        # the network the moment the app starts. That is a reasonable thing to ask
        # for -- a phone reaching a laptop over shared wifi -- but not a reasonable
        # default for a `pip install`-able app that accepts file uploads and writes
        # model files. Opt in explicitly.
        "--host", type=str, default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1, this machine only). "
             "Pass 0.0.0.0 to serve other devices on your network.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable link",
    )
    args = parser.parse_args(argv)

    if args.list_configs:
        for name in bundled_configs():
            cfg = load_config(bundled_config(name))
            print(f"{name:16} {cfg.application.name} — {', '.join(cfg.domains)}")
        return

    # The page header says "Offline". Gradio 6 phones home on launch -- a PyPI
    # version check and two telemetry posts -- unless told not to, and the flag
    # must be in the environment *before* gradio is first imported (#106).
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    try:
        from adaptshot.app import ui
    except ImportError as exc:
        if getattr(exc, "name", None) == "gradio":
            print(_INSTALL_HINT, file=sys.stderr)
            raise SystemExit(1) from None
        raise

    ui.serve(
        config_path=args.config,
        host=args.host,
        port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    launch()
