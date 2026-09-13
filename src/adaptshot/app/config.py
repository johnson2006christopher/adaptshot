"""Domain configuration: the file that decides what Tambua is.

Tambua has no built-in subject matter. A YAML config supplies the domains, the
classes inside them, the advice shown for each, and the language the interface
speaks. Swap the file and the same application diagnoses solar panels instead of
maize.

That makes the config the application's entire identity, which in turn makes
validation a user-facing feature rather than defensive programming. The person
editing this file is an extension officer or a technician, not a Python
programmer: a one-character typo must produce a message naming the file, the
line, and the remedy -- never a traceback several frames from the mistake.

Every problem in a file is reported at once, sorted by line, because fixing a
config one error per run is miserable.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast, get_args

import yaml

from adaptshot import (
    Backbone,
    ConformalMode,
    Device,
    InferenceMode,
    SimilarityMetric,
)
from adaptshot.utils.exceptions import ConfigValidationError

SEVERITIES = ("low", "moderate", "high", "critical")

#: Preserves the specific literal type of whatever set of choices is being
#: checked, so a validated value can be handed on without widening to `str`.
_Choice = TypeVar("_Choice", bound=str)

# Read out of AdaptShot's own annotations rather than restated here. Two hand-kept
# lists drift: the previous copy omitted "contrastive", so a valid AdaptShot
# inference mode was rejected by the app for no reason anyone had decided on.
# Deriving them means a backbone added upstream is accepted the day it lands.


BACKBONES: tuple[Backbone, ...] = get_args(Backbone)
DEVICES: tuple[Device, ...] = get_args(Device)
INFERENCE_MODES: tuple[InferenceMode, ...] = get_args(InferenceMode)
SIMILARITY_METRICS: tuple[SimilarityMetric, ...] = get_args(SimilarityMetric)
CONFORMAL_MODES: tuple[ConformalMode, ...] = get_args(ConformalMode)

#: 0.1, not AdaptShot's own 0.05. With three or four configured classes, a 95%
#: target routinely returns every class -- a "prediction set" containing the
#: whole label space tells the person holding the phone nothing. 90% coverage
#: keeps the set small enough to act on, and the trade is stated in the config
#: rather than buried.
DEFAULT_ALPHA = 0.1

# A domain with one class cannot be classified into -- there is nothing to tell
# it apart from. Two is the smallest config that means anything.
MIN_CLASSES_PER_DOMAIN = 2



_TOP_LEVEL = {"application", "engine", "domains", "localization", "paths"}
_REQUIRED_TOP = ("application", "domains")
_APPLICATION_KEYS = {"name", "version", "description"}
_ENGINE_KEYS = {
    "backbone",
    "device",
    "seed",
    "inference_mode",
    "similarity_metric",
    "eco_mode",
    "enable_ood_detection",
    "conformal_alpha",
    "conformal_mode",
}
_LOCALIZATION_KEYS = {"language", "fallback"}
_PATHS_KEYS = {"model_dir", "sample_data"}
_DOMAIN_KEYS = {"local_name", "classes"}
_CLASS_KEYS = {"local_name", "action", "description", "severity"}
_REQUIRED_CLASS = ("action", "description")

# The vocabulary this schema replaced. Anyone holding a config written against
# the crop-only version gets told what it became, rather than a list of keys
# that does not contain the one they used.
_RENAMED = {
    "crops": "domains",
    "diseases": "classes",
    "swahili": "local_name",
}


# ---------------------------------------------------------------------------
# The shapes the rest of the application consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassInfo:
    """One thing the application can recognise, with the advice that follows it.

    Deliberately domain-neutral: *name, local label, what to do, what it looks
    like, how urgent*. That set describes a crop disease, a solar panel fault and
    a textile defect equally well, which is the whole claim #47 has to support.
    """

    key: str
    local_name: str
    action: str
    description: str
    severity: str
    domain: str


@dataclass(frozen=True)
class ApplicationInfo:
    """What the interface calls itself."""

    name: str
    version: str
    description: str


@dataclass(frozen=True)
class EngineSettings:
    """AdaptShot settings, validated before they reach `AdaptShotConfig`.

    The four choice fields reuse AdaptShot's own `Literal` types rather than
    widening to `str`. Validation has already established the value is one of
    them, and saying so keeps that guarantee visible to the type checker all the
    way to the `AdaptShotConfig` constructor.
    """

    backbone: Backbone
    device: Device
    seed: int
    inference_mode: InferenceMode
    similarity_metric: SimilarityMetric
    eco_mode: bool
    enable_ood_detection: bool
    conformal_alpha: float
    conformal_mode: ConformalMode


@dataclass(frozen=True)
class Localization:
    """Which language the class labels are shown in, and what to fall back to."""

    language: str
    fallback: str


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, resolved relative to the config file."""

    model_dir: str
    sample_data: str


@dataclass(frozen=True)
class TambuaConfig:
    """A validated domain configuration.

    Construction is the validation: if an instance exists, every check in this
    module passed, so the engine and the UI can read fields directly instead of
    defending against a malformed file at each use.
    """

    path: str
    application: ApplicationInfo
    engine: EngineSettings
    localization: Localization
    paths: Paths
    domains: tuple[str, ...]
    classes: Mapping[str, ClassInfo]

    @property
    def labels(self) -> list[str]:
        """Every class key the configuration defines, sorted."""
        return sorted(self.classes)

    def classes_in(self, domain: str) -> list[ClassInfo]:
        """The classes belonging to one domain, in config order."""
        return [c for c in self.classes.values() if c.domain == domain]


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigProblem:
    """One thing wrong with a config file, and how to fix it."""

    line: int
    problem: str
    remedy: str


def _did_you_mean(unknown: str, allowed: Sequence[str]) -> str:
    if unknown in _RENAMED:
        return f'renamed to "{_RENAMED[unknown]}" when the schema stopped being crop-specific'
    close = difflib.get_close_matches(unknown, allowed, n=1, cutoff=0.6)
    if close:
        return f'did you mean "{close[0]}"?'
    return "allowed keys: " + ", ".join(sorted(allowed))


def _render(path: str, problems: Sequence[ConfigProblem]) -> str:
    ordered = sorted(problems, key=lambda p: (p.line, p.problem))
    header = (
        f"{path} is not a valid Tambua configuration "
        f"({len(ordered)} problem{'s' if len(ordered) != 1 else ''}):"
    )
    body = "\n".join(
        f"\n{path}, line {p.line}: {p.problem}\n  {p.remedy}" for p in ordered
    )
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Line tracking
# ---------------------------------------------------------------------------

_Path = tuple[str, ...]


def _index_lines(
    node: yaml.Node, prefix: _Path, lines: dict[_Path, int], dupes: list[ConfigProblem]
) -> None:
    """Record the source line of every key, and flag duplicates as we go.

    PyYAML resolves a repeated key silently by keeping the last one, so a config
    can lose a whole class to a copy-paste slip without a word. The node tree
    still holds both, which is the only place the duplicate is visible.
    """
    lines[prefix] = node.start_mark.line + 1
    if not isinstance(node, yaml.MappingNode):
        return
    seen: dict[str, int] = {}
    for key_node, value_node in node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            continue
        key = str(key_node.value)
        line = key_node.start_mark.line + 1
        if key in seen:
            dupes.append(
                ConfigProblem(
                    line=line,
                    problem=f'duplicate key "{key}", first defined on line {seen[key]}',
                    remedy="remove one of them; YAML silently keeps only the last",
                )
            )
        seen[key] = line
        child = (*prefix, key)
        _index_lines(value_node, child, lines, dupes)
        # After the recursion, so the key's own line wins over its value's.
        lines[child] = line


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class _Validator:
    """Accumulates every problem in one file rather than raising at the first."""

    def __init__(self, path: str, lines: dict[_Path, int]) -> None:
        self.path = path
        self.lines = lines
        self.problems: list[ConfigProblem] = []

    def line(self, where: _Path) -> int:
        """The line for a config path, falling back to its nearest known parent."""
        probe = where
        while probe and probe not in self.lines:
            probe = probe[:-1]
        return self.lines.get(probe, 1)

    def fail(self, where: _Path, problem: str, remedy: str) -> None:
        self.problems.append(ConfigProblem(self.line(where), problem, remedy))

    def mapping(self, value: Any, where: _Path, what: str) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        self.fail(
            where,
            f"{what} is {type(value).__name__}, not a mapping",
            f"write {what} as indented `key: value` pairs",
        )
        return None

    def unknown_keys(self, found: Mapping[str, Any], allowed: set[str], where: _Path) -> None:
        for key in found:
            if key not in allowed:
                self.fail(
                    (*where, key),
                    f'unknown key "{key}"',
                    _did_you_mean(key, sorted(allowed)),
                )

    def text(self, found: Mapping[str, Any], key: str, where: _Path, default: str = "") -> str:
        value = found.get(key, default)
        if not isinstance(value, str):
            self.fail(
                (*where, key),
                f"{key} is {type(value).__name__}, not text",
                f'quote it, e.g. {key}: "..."',
            )
            return default
        return value

    def choice(
        self, found: Mapping[str, Any], key: str, allowed: Sequence[_Choice], where: _Path
    ) -> _Choice:
        value = found.get(key, allowed[0])
        if value not in allowed:
            shown = f'"{value}"' if isinstance(value, str) else repr(value)
            self.fail(
                (*where, key),
                f"{key} is {shown}",
                "must be one of: " + ", ".join(allowed),
            )
            return allowed[0]
        # Membership in `allowed` was just established, which is exactly the
        # premise the Literal type encodes; the checker cannot see that through
        # an untyped YAML mapping.
        return cast(_Choice, value)

    def flag(self, found: Mapping[str, Any], key: str, where: _Path, default: bool) -> bool:
        value = found.get(key, default)
        if not isinstance(value, bool):
            self.fail(
                (*where, key),
                f"{key} is {type(value).__name__}, not a true/false value",
                f"write {key}: true or {key}: false",
            )
            return default
        return value

    def fraction(
        self, found: Mapping[str, Any], key: str, where: _Path, default: float
    ) -> float:
        """A number strictly between 0 and 1.

        Both endpoints are excluded deliberately: alpha=0 asks for a set that
        contains the truth every time, which means every class; alpha=1 asks for
        no guarantee at all. Neither is a setting, they are ways of switching the
        feature off while appearing to configure it.
        """

        value = found.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(
                (*where, key),
                f"{key} is {type(value).__name__}, not a number",
                f"write {key}: {default}",
            )
            return default
        if not 0.0 < float(value) < 1.0:
            self.fail(
                (*where, key),
                f"{key} is {value}",
                f"must be between 0 and 1, exclusive -- {default} means "
                f"{(1 - default) * 100:.0f}% target coverage",
            )
            return default
        return float(value)

    def whole_number(self, found: Mapping[str, Any], key: str, where: _Path, default: int) -> int:
        value = found.get(key, default)
        # bool is an int in Python; `seed: true` is a mistake, not a seed.
        if isinstance(value, bool) or not isinstance(value, int):
            self.fail(
                (*where, key),
                f"{key} is {type(value).__name__}, not a whole number",
                f"write {key}: {default}",
            )
            return default
        return value


def _validate_domains(v: _Validator, raw: Mapping[str, Any]) -> dict[str, ClassInfo]:
    """Check every domain and class, and flatten them into one label map.

    The flattening is why duplicate class keys across domains matter: the engine
    predicts a single label space, so `maize.healthy` and `beans.healthy` would
    collide and one would silently disappear.
    """
    classes: dict[str, ClassInfo] = {}
    origin: dict[str, str] = {}

    if "domains" not in raw:
        # Already reported as a missing required section; saying it twice in two
        # different vocabularies makes the report harder to read, not clearer.
        return classes

    domains = v.mapping(raw.get("domains"), ("domains",), "domains")
    if domains is None:
        return classes
    if not domains:
        v.fail(
            ("domains",),
            "domains is empty",
            "define at least one domain, each with two or more classes",
        )
        return classes

    for domain_name, domain_raw in domains.items():
        at = ("domains", domain_name)
        domain = v.mapping(domain_raw, at, f'domain "{domain_name}"')
        if domain is None:
            continue
        v.unknown_keys(domain, _DOMAIN_KEYS, at)

        entries = v.mapping(domain.get("classes"), (*at, "classes"), "classes")
        if entries is None:
            continue
        if len(entries) < MIN_CLASSES_PER_DOMAIN:
            v.fail(
                (*at, "classes"),
                f'domain "{domain_name}" defines {len(entries)} class'
                f"{'' if len(entries) == 1 else 'es'}",
                f"at least {MIN_CLASSES_PER_DOMAIN} are needed; "
                "one class gives the model nothing to tell apart",
            )

        for key, entry_raw in entries.items():
            spot = (*at, "classes", key)
            if key in origin:
                v.fail(
                    spot,
                    f'class "{key}" is already defined in domain "{origin[key]}"',
                    "class keys share one label space across all domains, "
                    f'so make it unique -- e.g. "{domain_name}_{key}"',
                )
                continue
            entry = v.mapping(entry_raw, spot, f'class "{key}"')
            if entry is None:
                continue
            v.unknown_keys(entry, _CLASS_KEYS, spot)

            for required in _REQUIRED_CLASS:
                if not str(entry.get(required, "")).strip():
                    v.fail(
                        spot,
                        f'class "{key}" has no {required}',
                        f"add {required}: and describe it for the person "
                        "reading the result",
                    )

            origin[key] = domain_name
            classes[key] = ClassInfo(
                key=key,
                local_name=v.text(entry, "local_name", spot, key),
                action=v.text(entry, "action", spot),
                description=v.text(entry, "description", spot),
                severity=v.choice(entry, "severity", SEVERITIES, spot),
                domain=domain_name,
            )

    return classes


def _validate_localization(
    v: _Validator, raw: Mapping[str, Any], classes: Mapping[str, ClassInfo]
) -> Localization:
    """Check the language settings, and that the labels actually exist in them.

    A missing `local_name` is not an error -- the class key is a reasonable
    fallback. It *is* worth reporting when the config asks for a language other
    than the fallback and then leaves labels untranslated, because the interface
    would silently mix two languages in one list.
    """
    at = ("localization",)
    section = v.mapping(raw.get("localization", {}), at, "localization") or {}
    v.unknown_keys(section, _LOCALIZATION_KEYS, at)

    language = v.text(section, "language", at, "en")
    fallback = v.text(section, "fallback", at, "en")

    if language != fallback:
        untranslated = [c.key for c in classes.values() if c.local_name == c.key]
        if untranslated and len(untranslated) != len(classes):
            shown = ", ".join(sorted(untranslated)[:3])
            more = "" if len(untranslated) <= 3 else f" (+{len(untranslated) - 3} more)"
            v.fail(
                (*at, "language"),
                f'language is "{language}" but {len(untranslated)} of '
                f"{len(classes)} classes have no local_name: {shown}{more}",
                f'add local_name: to each, or set language: "{fallback}" '
                "to show the keys deliberately",
            )
    return Localization(language=language, fallback=fallback)


def _validate_paths(v: _Validator, raw: Mapping[str, Any], config_path: str) -> Paths:
    """Check that the configured directories can actually be written to.

    They are outputs, so they need not exist yet -- but their parent must, or the
    first save fails long after the config was accepted.
    """
    at = ("paths",)
    section = v.mapping(raw.get("paths", {}), at, "paths") or {}
    v.unknown_keys(section, _PATHS_KEYS, at)

    base = os.path.dirname(os.path.abspath(config_path))
    resolved: dict[str, str] = {}
    for key, default in (("model_dir", "models"), ("sample_data", "samples")):
        value = v.text(section, key, at, default)
        full = value if os.path.isabs(value) else os.path.join(base, value)
        parent = os.path.dirname(os.path.normpath(full))
        if parent and not os.path.isdir(parent):
            v.fail(
                (*at, key),
                f'{key} resolves to "{full}", whose parent directory does not exist',
                f"create {parent}, or point {key} somewhere that exists",
            )
        resolved[key] = full
    return Paths(model_dir=resolved["model_dir"], sample_data=resolved["sample_data"])


def load_config(path: str) -> TambuaConfig:
    """Load and fully validate a domain configuration.

    Args:
        path: Path to the YAML config file.

    Returns:
        A validated `TambuaConfig`. Every field is present and well-typed.

    Raises:
        ConfigValidationError: If the file is missing, unparseable, or fails any
            validation check. The message names the file, the line, and the
            remedy for every problem found, not just the first.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError as exc:
        raise ConfigValidationError(
            f"{path} does not exist. Pass --config with a path to a domain "
            "configuration, or copy one of the shipped configs to start from."
        ) from exc
    except OSError as exc:
        raise ConfigValidationError(f"{path} could not be read: {exc}") from exc

    try:
        loaded = yaml.safe_load(text)
        composed = yaml.compose(text)
    except yaml.YAMLError as exc:
        # PyYAML's own marks are precise; quoting them beats paraphrasing.
        raise ConfigValidationError(f"{path} is not valid YAML.\n{exc}") from exc

    if loaded is None:
        raise ConfigValidationError(
            f"{path} is empty. A domain configuration needs at least "
            "`application:` and `domains:`."
        )
    if not isinstance(loaded, dict):
        raise ConfigValidationError(
            f"{path} must contain a mapping at the top level, found "
            f"{type(loaded).__name__}. The file should begin with a key such as "
            "`application:`, not a list item."
        )

    lines: dict[_Path, int] = {}
    duplicates: list[ConfigProblem] = []
    if composed is not None:
        _index_lines(composed, (), lines, duplicates)
    # `()` is only ever the location of a *missing* top-level section, which is a
    # property of the whole file. Reporting it at the line the first key happens
    # to sit on would point the reader at something unrelated.
    lines[()] = 1

    v = _Validator(path, lines)
    v.problems.extend(duplicates)

    raw = {str(k): val for k, val in loaded.items()}
    v.unknown_keys(raw, _TOP_LEVEL, ())
    for required in _REQUIRED_TOP:
        if required not in raw:
            v.fail((), f"no {required}: section", f"add a top-level {required}: block")

    app_at = ("application",)
    app_raw = v.mapping(raw.get("application", {}), app_at, "application") or {}
    v.unknown_keys(app_raw, _APPLICATION_KEYS, app_at)
    application = ApplicationInfo(
        name=v.text(app_raw, "name", app_at, "Tambua"),
        version=v.text(app_raw, "version", app_at, "0.0.0"),
        description=v.text(app_raw, "description", app_at),
    )

    eng_at = ("engine",)
    eng_raw = v.mapping(raw.get("engine", {}), eng_at, "engine") or {}
    v.unknown_keys(eng_raw, _ENGINE_KEYS, eng_at)
    engine = EngineSettings(
        backbone=v.choice(eng_raw, "backbone", BACKBONES, eng_at),
        device=v.choice(eng_raw, "device", DEVICES, eng_at),
        seed=v.whole_number(eng_raw, "seed", eng_at, 42),
        inference_mode=v.choice(eng_raw, "inference_mode", INFERENCE_MODES, eng_at),
        similarity_metric=v.choice(eng_raw, "similarity_metric", SIMILARITY_METRICS, eng_at),
        eco_mode=v.flag(eng_raw, "eco_mode", eng_at, True),
        enable_ood_detection=v.flag(eng_raw, "enable_ood_detection", eng_at, True),
        conformal_alpha=v.fraction(eng_raw, "conformal_alpha", eng_at, DEFAULT_ALPHA),
        conformal_mode=v.choice(eng_raw, "conformal_mode", CONFORMAL_MODES, eng_at),
    )

    classes = _validate_domains(v, raw)
    localization = _validate_localization(v, raw, classes)
    paths = _validate_paths(v, raw, path)

    if v.problems:
        raise ConfigValidationError(_render(path, v.problems))

    domain_order: list[str] = []
    for info in classes.values():
        if info.domain not in domain_order:
            domain_order.append(info.domain)

    return TambuaConfig(
        path=os.path.abspath(path),
        application=application,
        engine=engine,
        localization=localization,
        paths=paths,
        domains=tuple(domain_order),
        classes=classes,
    )
