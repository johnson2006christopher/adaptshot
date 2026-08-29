"""The public surface is a classified list, and this keeps it one (#23).

Every name in ``adaptshot.__all__`` is stable or experimental, per
``adaptshot.api``. That distinction is only worth having if it cannot drift:
a name added to ``__all__`` without a tier, an experimental docstring that
stops saying so, or a reference page that documents a name under the wrong
heading would each quietly turn the classification back into a comment.

So each of those is a test. Hyrum's law says every observable behaviour will
be depended on eventually; the least the library can do is say, per name,
whether that is a good idea.
"""

from __future__ import annotations

import re
import typing
import warnings
from pathlib import Path

import numpy as np
import pytest

import adaptshot
from adaptshot.api import EXPERIMENTAL, STABLE

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "docs" / "reference" / "api.md"
TESTS_DIR = REPO_ROOT / "tests"

MARKER = "**Experimental.**"


def _region(text: str, heading: str) -> str:
    """The body of one ``## heading`` up to the next ``## `` heading."""

    start = text.index(f"\n## {heading}\n")
    rest = text[start + 1 :]
    following = re.search(r"^## ", rest[len(heading) + 4 :], flags=re.MULTILINE)
    return rest if following is None else rest[: following.start() + len(heading) + 4]


def _is_type_alias(obj: object) -> bool:
    return typing.get_origin(obj) is typing.Literal


# ---------------------------------------------------------------------------
# The classification itself
# ---------------------------------------------------------------------------


def test_every_export_is_classified_exactly_once() -> None:
    """A name in ``__all__`` with no tier is a promise nobody decided to make."""

    classified = set(STABLE) | set(EXPERIMENTAL)
    exported = set(adaptshot.__all__)

    assert not (set(STABLE) & set(EXPERIMENTAL)), "a name is in both tiers"
    assert classified == exported, (
        f"unclassified exports: {sorted(exported - classified)}; "
        f"classified but not exported: {sorted(classified - exported)}"
    )


def test_every_classified_name_resolves() -> None:
    for name in (*STABLE, *EXPERIMENTAL):
        assert hasattr(adaptshot, name), f"{name} is classified but not importable"


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPERIMENTAL)
def test_experimental_names_say_so_where_they_are_used(name: str) -> None:
    """The status has to be visible at the point of use, not only in a list."""

    doc = (getattr(adaptshot, name).__doc__ or "").lstrip()
    assert doc.startswith(MARKER), (
        f"{name} is experimental but its docstring does not open with {MARKER!r}"
    )


@pytest.mark.parametrize("name", STABLE)
def test_stable_names_do_not_claim_to_be_experimental(name: str) -> None:
    """Guards the copy-paste in the other direction."""

    obj = getattr(adaptshot, name)
    if _is_type_alias(obj):
        return
    doc = (obj.__doc__ or "").lstrip()
    assert not doc.startswith(MARKER), f"{name} is stable but its docstring says experimental"


# ---------------------------------------------------------------------------
# The reference page
# ---------------------------------------------------------------------------


def test_reference_documents_every_name_under_its_own_heading() -> None:
    """``docs/reference/api.md`` reflects the supported surface, by construction."""

    text = REFERENCE.read_text(encoding="utf-8")
    stable_region = _region(text, "Stable")
    experimental_region = _region(text, "Experimental")

    missing_stable = [n for n in STABLE if not re.search(rf"\b{re.escape(n)}\b", stable_region)]
    missing_experimental = [
        n for n in EXPERIMENTAL if not re.search(rf"\b{re.escape(n)}\b", experimental_region)
    ]
    assert not missing_stable, f"stable names absent from the Stable section: {missing_stable}"
    assert not missing_experimental, (
        f"experimental names absent from the Experimental section: {missing_experimental}"
    )


# ---------------------------------------------------------------------------
# What "stable" is required to mean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", STABLE)
def test_stable_classes_are_named_in_at_least_one_test(name: str) -> None:
    """Stable means tested, in the maintainer's definition. So it is checked.

    Type aliases are exempt: a ``Literal`` has nothing to test beyond the config
    validation that already covers its values.
    """

    if _is_type_alias(getattr(adaptshot, name)):
        return
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    referenced = any(
        pattern.search(path.read_text(encoding="utf-8"))
        for path in TESTS_DIR.glob("test_*.py")
        if path.name != Path(__file__).name
    )
    assert referenced, (
        f"{name} is classified stable but no test names it. Either add one or move it to "
        "EXPERIMENTAL -- stable is a promise about tests, not about intent."
    )


# ---------------------------------------------------------------------------
# The deprecation policy, applied
# ---------------------------------------------------------------------------


def test_deprecated_uncertainty_methods_warn_and_still_work() -> None:
    """Deprecated, not deleted. The warning names the version and the reason."""

    quantifier = adaptshot.UncertaintyQuantifier()
    versions = r"0\.3\.0.*0\.4\.0"

    with pytest.warns(DeprecationWarning, match=versions):
        quantifier.get_ood_summary()
    with pytest.warns(DeprecationWarning, match=versions):
        quantifier.get_class_statistics()
    with pytest.warns(DeprecationWarning, match=versions):
        quantifier.compute_perturbation_variance([np.ones(4, dtype=np.float32)] * 2)


def test_the_old_contrastive_import_path_warns_and_still_works() -> None:
    """Moved in 0.3.0; the alias is the first use of the deprecation policy."""

    import importlib
    import sys

    sys.modules.pop("adaptshot.core.contrastive", None)
    with pytest.warns(DeprecationWarning, match=r"training\.contrastive.*0\.4\.0"):
        legacy = importlib.import_module("adaptshot.core.contrastive")

    assert legacy.ContrastivePrototypeLearner is adaptshot.ContrastivePrototypeLearner
    assert legacy.ContrastiveConfig is adaptshot.ContrastiveConfig


def test_importing_the_package_emits_no_deprecation_warning() -> None:
    """The library must not trip its own alias -- that is how #23 found learner.py."""

    import importlib

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(adaptshot)
