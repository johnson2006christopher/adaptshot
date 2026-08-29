# Release checklist

Releases are automated on tags (`.github/workflows/release.yml`, #25). This page
is what a human still does, and the one-time setup that lets the automation
publish without a token.

## One-time setup: Trusted Publishing

The workflow publishes with a short-lived OIDC token that PyPI checks against a
publisher registered for this exact repository, workflow file and environment.
No API token is stored anywhere. Until the publisher exists, the publish jobs
fail with a clear "invalid or non-existent authentication" error and nothing
is uploaded.

On **pypi.org** → account → *Publishing* → *Add a new pending publisher*:

| field | value |
|---|---|
| PyPI project name | `adaptshot` |
| Owner | `johnson2006christopher` |
| Repository name | `adaptshot` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

On **test.pypi.org**, the same with environment name `testpypi`.

On **GitHub** → repository *Settings* → *Environments*: create `pypi` and
`testpypi`. Adding yourself as a required reviewer on `pypi` means every
production publish waits for one click from you after the gate has passed.
Recommended: PyPI releases cannot be overwritten.

## Every release

1. **Bump** `version` in `pyproject.toml`. It is the only place. `__version__`
   is read from the installed metadata, so run `pip install -e .` afterwards or
   `tests/test_release_metadata.py` will report the stale value and tell you
   the same thing.
2. **Write the CHANGELOG entry.** `test_changelog_documents_current_version`
   fails without one.
3. **Run the gate** locally: `ruff`, `mypy --strict`, `pytest`, the smoke
   benchmark, and `mkdocs build --strict`.
4. **Merge the version branch into `main`.** That step is the maintainer's.
5. **Tag a release candidate first**: `git tag v0.3.0rc1 && git push origin v0.3.0rc1`.
   The tag containing `rc` routes to TestPyPI. Check the page renders, then
   `pip install -i https://test.pypi.org/simple/ adaptshot==0.3.0rc1` somewhere
   clean.
6. **Tag the release**: `git tag v0.3.0 && git push origin v0.3.0`. Routes to PyPI.

## What the workflow checks before it publishes

- The tag matches `pyproject.toml`'s version, or the build stops.
- The wheel carries `py.typed`, the ONNX backbone, and the sample photographs.
- The full gate passes on the tagged commit.
- **The wheel installs in a clean `python:3.12-slim` container and the README
  quickstart runs against it** with the network blocked. This is the only test
  that proves the artifact rather than the tree; `src/` is not on the path.

If any of those fail, nothing is uploaded, and the tag can be deleted and
re-pushed after a fix -- which is exactly what the release candidate is for.
