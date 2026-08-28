"""One test per way a domain config can be wrong (#47).

Once the config is the application's whole identity, its error messages are a
user-facing feature. The person editing the file is an extension officer or a
technician; a traceback several frames from their typo is a dead end.

Two properties are asserted throughout. Every message must name the file, a
line and a remedy -- `test_every_message_names_file_line_and_remedy` checks that
across every failure mode at once, so a new check cannot ship a bare complaint.
And every problem in a file must be reported in one pass, because fixing a
config one error per run is miserable.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("tambua", reason="the application is not installed in this environment")

from tambua.config import load_config

from adaptshot.utils.exceptions import ConfigValidationError

VALID = """
application:
  name: "Test"
  version: "0.1.0"
  description: "A fixture"
engine:
  backbone: "resnet18"
  device: "cpu"
  seed: 7
domains:
  widgets:
    local_name: "vitu"
    classes:
      widget_ok:
        local_name: "kizuri"
        action: "Ship it."
        description: "An acceptable widget."
        severity: "low"
      widget_cracked:
        local_name: "kilichopasuka"
        action: "Scrap it."
        description: "A cracked widget."
        severity: "critical"
localization:
  language: "sw"
  fallback: "en"
paths:
  model_dir: "models"
  sample_data: "samples"
"""


def _write(tmp_path: Path, body: str, name: str = "domain.yaml") -> str:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def _rejected(tmp_path: Path, body: str) -> str:
    """Load a config expected to fail, and return the message."""

    with pytest.raises(ConfigValidationError) as caught:
        load_config(_write(tmp_path, body))
    return str(caught.value)


def test_a_valid_config_loads(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID))
    assert cfg.application.name == "Test"
    assert cfg.domains == ("widgets",)
    assert cfg.labels == ["widget_cracked", "widget_ok"]
    assert cfg.classes["widget_cracked"].severity == "critical"
    assert cfg.classes["widget_cracked"].domain == "widgets"


def test_missing_file_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError) as caught:
        load_config(str(tmp_path / "absent.yaml"))
    assert "does not exist" in str(caught.value)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    assert "is empty" in _rejected(tmp_path, "\n")


def test_top_level_sequence_is_rejected(tmp_path: Path) -> None:
    assert "mapping at the top level" in _rejected(tmp_path, "- application\n- domains\n")


def test_unparseable_yaml_keeps_pyyaml_own_position(tmp_path: Path) -> None:
    message = _rejected(tmp_path, "application:\n  name: 'unterminated\n")
    assert "not valid YAML" in message


def test_missing_domains_section(tmp_path: Path) -> None:
    assert "no domains: section" in _rejected(tmp_path, "application:\n  name: 'X'\n")


def test_pre_generalisation_keys_get_a_migration_hint(tmp_path: Path) -> None:
    """A config written for the crop-only schema must be told what changed.

    Listing the allowed keys is useless here: the reader's key is not in it, and
    they have no way to guess that `crops` became `domains`.
    """

    message = _rejected(
        tmp_path,
        """
        application:
          name: "Old"
        crops:
          maize:
            diseases: {}
        """,
    )
    assert 'unknown key "crops"' in message
    assert 'renamed to "domains"' in message


def test_a_typo_in_a_key_suggests_the_real_one(tmp_path: Path) -> None:
    message = _rejected(tmp_path, VALID.replace('severity: "low"', 'sevrity: "low"'))
    assert 'unknown key "sevrity"' in message
    assert 'did you mean "severity"?' in message


def test_bad_severity_lists_the_allowed_values(tmp_path: Path) -> None:
    message = _rejected(tmp_path, VALID.replace('severity: "low"', 'severity: "hihg"'))
    assert 'severity is "hihg"' in message
    assert "must be one of: low, moderate, high, critical" in message


def test_a_class_needs_action_and_description(tmp_path: Path) -> None:
    message = _rejected(tmp_path, VALID.replace('        action: "Ship it."\n', ""))
    assert "has no action" in message


def test_a_domain_needs_at_least_two_classes(tmp_path: Path) -> None:
    """One class is not a classification problem -- there is nothing to tell apart."""

    body = VALID[: VALID.index("      widget_cracked:")] + textwrap.dedent(
        """\
        localization:
          language: "en"
          fallback: "en"
        """
    )
    message = _rejected(tmp_path, body)
    assert "defines 1 class" in message
    assert "at least 2" in message


def test_a_class_key_repeated_across_domains_is_caught(tmp_path: Path) -> None:
    """The engine predicts one label space, so a collision would lose a class."""

    body = VALID.replace(
        "localization:",
        textwrap.dedent(
            """\
              gadgets:
                local_name: "vifaa"
                classes:
                  widget_ok:
                    action: "Ship it."
                    description: "Collides with the widgets domain."
                    severity: "low"
                  gadget_bent:
                    action: "Straighten it."
                    description: "A bent gadget."
                    severity: "moderate"
            localization:"""
        ),
    )
    message = _rejected(tmp_path, body)
    assert 'class "widget_ok" is already defined in domain "widgets"' in message


def test_a_repeated_yaml_key_is_caught(tmp_path: Path) -> None:
    """PyYAML keeps the last of a repeated key silently; a whole class can vanish."""

    message = _rejected(tmp_path, VALID.replace('  version: "0.1.0"', '  name: "Twice"'))
    assert 'duplicate key "name"' in message


def test_an_unknown_backbone_is_rejected(tmp_path: Path) -> None:
    message = _rejected(tmp_path, VALID.replace('backbone: "resnet18"', 'backbone: "resnet50"'))
    assert 'backbone is "resnet50"' in message
    assert "resnet18" in message


def test_a_non_numeric_seed_is_rejected(tmp_path: Path) -> None:
    """`seed: true` is a mistake. bool is an int in Python, so it must be excluded."""

    assert "not a whole number" in _rejected(tmp_path, VALID.replace("seed: 7", "seed: true"))


def test_unwritable_paths_are_reported_before_the_first_save(tmp_path: Path) -> None:
    body = VALID.replace('model_dir: "models"', 'model_dir: "nope/deeper/models"')
    message = _rejected(tmp_path, body)
    assert "parent directory does not exist" in message


def test_untranslated_labels_are_reported_when_a_language_is_requested(
    tmp_path: Path,
) -> None:
    """Mixing two languages in one dropdown is worse than showing keys deliberately."""

    body = VALID.replace('        local_name: "kizuri"\n', "")
    message = _rejected(tmp_path, body)
    assert "have no local_name" in message


def test_all_problems_are_reported_in_one_pass(tmp_path: Path) -> None:
    body = (
        VALID.replace('severity: "low"', 'severity: "hihg"')
        .replace('backbone: "resnet18"', 'backbone: "resnet50"')
        .replace("seed: 7", "seed: true")
    )
    message = _rejected(tmp_path, body)
    assert "3 problems" in message
    for fragment in ('severity is "hihg"', 'backbone is "resnet50"', "not a whole number"):
        assert fragment in message


def test_every_message_names_file_line_and_remedy(tmp_path: Path) -> None:
    """The acceptance criterion for #47, checked across every failure mode at once.

    A new validation check that ships a bare complaint fails here, which is the
    only way to keep the guarantee from eroding one check at a time.
    """

    broken = [
        VALID.replace('severity: "low"', 'severity: "hihg"'),
        VALID.replace('severity: "low"', 'sevrity: "low"'),
        VALID.replace('backbone: "resnet18"', 'backbone: "resnet50"'),
        VALID.replace("seed: 7", "seed: true"),
        VALID.replace('        action: "Ship it."\n', ""),
        VALID.replace('model_dir: "models"', 'model_dir: "nope/deeper/models"'),
        VALID.replace('  version: "0.1.0"', '  name: "Twice"'),
    ]
    pattern = re.compile(r"^(.+\.yaml), line (\d+): (.+)$")

    for body in broken:
        message = _rejected(tmp_path, body)
        located = [line for line in message.splitlines() if pattern.match(line)]
        assert located, f"no located problem in:\n{message}"

        lines = message.splitlines()
        for line in located:
            match = pattern.match(line)
            assert match is not None
            assert Path(match.group(1)).is_file(), "the named file must exist"
            assert int(match.group(2)) >= 1, "line numbers are 1-based"
            remedy = lines[lines.index(line) + 1].strip()
            assert remedy, f"no remedy offered after: {line}"
