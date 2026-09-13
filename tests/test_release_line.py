"""The README's "Current release" line must track the changelog (#128).

The original drift: the README said v0.2.0 for weeks after 0.3.0 shipped, on
the one page every visitor reads first. `tests/test_release_metadata.py`
already pins `__version__` to pyproject, but pyproject moves at the *start* of
a cycle while the README's release line describes what has actually shipped —
so the right anchor is the newest dated entry in CHANGELOG.md, which only
gains a date at release time.

The ratchet this creates on release day is deliberate: dating the 0.x.y
changelog entry without bumping the README fails this test, which is exactly
the drift it exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Matches "## [0.3.0] - 2026-08-29" but not "## [0.3.1] - unreleased".
_DATED_RELEASE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)

_CURRENT_RELEASE = re.compile(r"\*\*Current release: v(\d+\.\d+\.\d+)\.\*\*")


def _latest_shipped_version() -> str:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = _DATED_RELEASE.search(changelog)
    assert match is not None, (
        "CHANGELOG.md has no dated release entry (## [x.y.z] - YYYY-MM-DD); "
        "the README's release line has nothing to be checked against"
    )
    return match.group(1)


def test_readme_current_release_matches_the_newest_dated_changelog_entry() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = _CURRENT_RELEASE.search(readme)
    assert match is not None, (
        'README.md no longer contains a "**Current release: vX.Y.Z.**" line; '
        "move this test to wherever that claim went rather than deleting it"
    )
    shipped = _latest_shipped_version()
    assert match.group(1) == shipped, (
        f"README.md says the current release is v{match.group(1)}, but the "
        f"newest dated CHANGELOG.md entry is {shipped}. This is the v0.2.0 "
        "drift (#128) happening again — update the README's release line in "
        "the same change that dates the changelog entry."
    )
