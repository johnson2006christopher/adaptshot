# v0.1.1 Release Checklist

Follow these steps in order to publish AdaptShot v0.1.1 to PyPI and GitHub.

## Pre-Release Verification

- [ ] All tests pass: `pytest tests/ -v` (52 tests expected)
- [ ] Type check clean: `mypy src/adaptshot --strict`
- [ ] Lint clean: `ruff check src/ tests/`
- [ ] Smoke test passes: `python -m benchmarks.run_benchmark --smoke-test --seed 42`
- [ ] Docs build clean: `mkdocs build --strict`
- [ ] Version is `0.1.1` in both `src/adaptshot/__init__.py` and `pyproject.toml`
- [ ] `CHANGELOG.md` has complete `[0.1.1]` entry
- [ ] `ROADMAP.md` is populated
- [ ] No "unreleased" language remains in docs or README
- [ ] All links in README point to live URLs (GitHub, PyPI, Docs)

## Build

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build source and wheel distributions
python -m build
```

- [ ] `dist/adaptshot-0.1.1.tar.gz` exists
- [ ] `dist/adaptshot-0.1.1-py3-none-any.whl` exists

## Check The Package

```bash
# Verify metadata and long description rendering
twine check dist/*
```

- [ ] No warnings or errors from `twine check`

## Test PyPI (Recommended)

```bash
# Upload to Test PyPI first
twine upload -r testpypi dist/*

# Verify install from Test PyPI
pip install -i https://test.pypi.org/simple/ adaptshot

# Smoke test the installed package
python -c "
from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig
config = AdaptShotConfig(backbone='resnet18', device='cpu', seed=42)
learner = FewShotLearner(config=config)
print('Test PyPI install verified.')
"
```

- [ ] Test PyPI install works
- [ ] Import and basic initialization succeed

## Production PyPI

```bash
# Upload to PyPI
twine upload dist/*
```

- [ ] Upload succeeds
- [ ] `pip install adaptshot` installs v0.1.1
- [ ] PyPI page shows correct README, version, and metadata

## GitHub Release

```bash
# Tag the release
git tag -a v0.1.1 -m "AdaptShot v0.1.1: Eco mode, OOD detection, prototypical inference, 52 tests"

# Push the tag
git push origin v0.1.1

# Push any remaining commits
git push origin main
```

- [ ] Go to GitHub Releases: https://github.com/johnson2006christopher/adaptshot/releases/new
- [ ] Choose the `v0.1.1` tag
- [ ] Title: `AdaptShot v0.1.1`
- [ ] Copy the `[0.1.1]` section from `CHANGELOG.md` as release notes
- [ ] Attach `dist/adaptshot-0.1.1-py3-none-any.whl` as a release asset
- [ ] Publish the release

## Post-Release

- [ ] Verify `pip install adaptshot` pulls v0.1.1 (not cached v0.1.0)
- [ ] Verify GitHub Pages docs update: https://johnson2006christopher.github.io/adaptshot/
- [ ] Verify PyPI badge shows v0.1.1: https://pypi.org/project/adaptshot/
- [ ] Announce on relevant channels (Twitter/LinkedIn, Tanzania AI community)

## Rollback (If Needed)

If something goes wrong:

```bash
# Delete the tag
git tag -d v0.1.1
git push origin :refs/tags/v0.1.1

# Yank the PyPI release (cannot be re-uploaded with same version)
# Go to https://pypi.org/manage/project/adaptshot/releases/ and delete v0.1.1
# If patching, bump to 0.1.2 and repeat checklist
```
