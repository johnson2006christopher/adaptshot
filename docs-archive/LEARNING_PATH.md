# AdaptShot — 12-Week Learning & Improvement Path

> A quarter-long plan to go from *"I vibe-coded this"* to *"I can defend every line of this."*
> Each week pairs **one concept** with **one module of your repo** and **one shipped improvement**.
> By week 12 you will understand your own system, and the repo will be defensible to a reviewer.

**Owner:** Johnson Christopher Hassan · **Start date:** `____________` · **Target end:** `____________`

---

## The one sentence everything serves

> **AdaptShot is a few-shot image classifier that knows when it doesn't know — with a coverage guarantee, on hardware people already own.**

Every week below either *deepens* that claim or *removes* something that distracts from it. When you're unsure whether a task matters, ask: **does this make the sentence more true, or more provable?**

---

## How to use this document

**Time budget:** ~10 hours/week — 2h × 5 weekdays. Weekends off (you will burn out otherwise; this is a 12-week plan, not a 12-day one).

**The daily rhythm:**

| Block | Minutes | What |
| :-- | :-- | :-- |
| Learn | 45 | Read the concept source. Take notes **in your own words**. |
| Read | 45 | Read the repo code with that concept in mind. |
| Ship | 30 | Make one small, committed improvement. |

**Four rules that make this work:**

1. **Write notes in `notes/weekNN.md`.** If you can't explain it in writing, you don't know it. This folder is your real deliverable — the code is secondary.
2. **Commit every day, even if tiny.** A daily green square is evidence of discipline to anyone who looks at your profile.
3. **Never mark a week done until its "Prove it" questions are answered from memory.** Looking things up is fine while learning; the test is answering *closed-book*.
4. **If a week takes 2 weeks, take 2 weeks.** The order matters far more than the pace.

**Set up before Week 1:**

```bash
mkdir -p notes
echo "notes/" >> .gitignore        # optional: keep your notes private
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,torch]"
```

---

## The arc at a glance

| Phase | Weeks | Goal |
| :-- | :-- | :-- |
| **I — Foundations** | 1–2 | Get CI honest. Learn the tools that let you change code safely. |
| **II — The core** | 3–6 | Understand few-shot, calibration, conformal prediction, uncertainty. This is your thesis. |
| **III — Structure** | 7–8 | Tame the god object. Understand continual learning. |
| **IV — Truth** | 9–10 | Delete every unverifiable claim. Produce one real benchmark on real data. |
| **V — Focus** | 11–12 | Split out what isn't the library. Release something you can stand behind. |

**Progress tracker** — tick as you go:

- [ ] W1 Orientation & green CI
- [ ] W2 Testing & type safety
- [ ] W3 Embeddings & similarity
- [ ] W4 Calibration
- [ ] W5 Conformal prediction ★
- [ ] W6 Uncertainty & OOD
- [ ] W7 Refactoring `learner.py`
- [ ] W8 Continual learning
- [ ] W9 The truth pass
- [ ] W10 Real data, real numbers
- [ ] W11 Narrowing the scope
- [ ] W12 Release & write-up

---

# Phase I — Foundations

## Week 1 — Orientation & green CI

> **Thesis link:** You cannot prove any claim about your library until the machine can check it for you.

| | |
| :-- | :-- |
| **Concept goal** | How Python packaging, imports, and CI actually work |
| **Repo focus** | `pyproject.toml`, `.github/workflows/ci.yml`, `src/adaptshot/__init__.py` (43 lines) |
| **Ship** | CI passing green on all 4 Python versions |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Python packaging: `src/` layout, why it exists, editable installs, extras (`[dev]`, `[torch]`) | `notes/week01.md` — explain why `src/` layout prevents a whole class of bug |
| Tue | Read | Trace one import end to end: `from adaptshot import FewShotLearner`. Every file it touches, in order. | A hand-drawn import graph |
| Wed | Ship | Fix ruff failures in `benchmarks/` (`ruff check --fix benchmarks/`, then fix the rest by hand) | Lint job green |
| Thu | Ship | Make `test_extractor.py` + `test_studio_utils.py` skip cleanly without torch via `pytest.importorskip("torch")` | Test job green |
| Fri | Ship | Split CI: a full job (`.[dev,torch]`, all tests) + a core-only job (`.[dev]`, proves torch-free) | All jobs green |

### Prove it (closed-book, in `notes/week01.md`)

1. Why does `src/adaptshot/` exist instead of `adaptshot/` at the repo root?
2. What exactly does `pip install -e ".[dev]"` do that `pip install .` doesn't?
3. Your tests import `src.adaptshot`, not `adaptshot`. What are the consequences of that choice? (Hint: it's a real design smell — flag it, you'll fix it in W12.)
4. Why did a misnamed `.github` folder hide a real bug for months?

### Ship — Definition of done

- [ ] `gh run list` shows a green CI run on your branch
- [ ] All four Python versions pass
- [ ] A core-only job proves the library imports without torch

**Skill unlocked:** You can read a CI failure log and fix it without guessing.

---

## Week 2 — Testing & type safety

> **Thesis link:** A guarantee you can't test is a marketing claim.

| | |
| :-- | :-- |
| **Concept goal** | pytest properly; what `mypy --strict` is actually enforcing |
| **Repo focus** | `tests/` (all 14 modules, ~1,500 lines), `src/adaptshot/utils/exceptions.py` (21 lines) |
| **Ship** | `mypy --strict` enforced and passing; coverage measured |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | pytest: fixtures, `parametrize`, `raises`, `monkeypatch`, markers | Notes + rewrite one existing test to use `parametrize` |
| Tue | Learn | Type theory basics: `Optional`, `Union`, generics, `Protocol`, why `Any` defeats the point | Notes: what does `--strict` add over plain mypy? |
| Wed | Read | Read all of `tests/`. For each module, write one line: *what is it actually protecting?* | A 14-line inventory |
| Thu | Ship | Run `mypy src/adaptshot --strict`. Fix every error. Do **not** use `# type: ignore` without a comment explaining why. | mypy green |
| Fri | Ship | Generate a coverage report. Find the 3 least-covered core modules. Write one meaningful test for each. | Coverage number recorded in notes |

### Prove it

1. What's the difference between a test that *protects behaviour* and one that *describes implementation*? Give an example of each from your own suite.
2. What does `--strict` catch that ordinary mypy misses?
3. Your `exceptions.py` defines 5 exception types. Where is each raised, and is any of them raised *nowhere*? (Dead exception types are a real finding — write it down.)

### Ship — Definition of done

- [ ] `mypy src/adaptshot --strict` exits 0
- [ ] Coverage % recorded as your baseline for the quarter
- [ ] Every `# type: ignore` carries a reason

**Skill unlocked:** You can change code and know within 60 seconds whether you broke something.

---

# Phase II — The core

## Week 3 — Embeddings & similarity

> **Thesis link:** This is the "few-shot" half. Everything downstream is built on the embedding space.

| | |
| :-- | :-- |
| **Concept goal** | Metric learning, prototypes, nearest-neighbour classification |
| **Repo focus** | `core/extractor.py` (273), `core/similarity.py` (240) |
| **Ship** | An embedding-space diagnostic tool |
| **Read** | 📄 `research/Prototypical Networks for Few-shot Learning.pdf` — you already own it |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Prototypical Networks §1–3. A prototype = mean embedding of a class's support examples. | Notes: write the prototype formula from memory |
| Tue | Learn | Distance metrics: Euclidean vs cosine. Why ProtoNets found squared Euclidean beat cosine. | Notes: when does the choice matter? |
| Wed | Read | `extractor.py` — how does a JPEG become a 512-d vector? Preprocessing, backbone, pooling, normalisation. | Trace one image through, step by step |
| Thu | Read | `similarity.py` — which metrics are implemented? Where is FAISS used, and what happens without it? | Notes |
| Fri | Ship | Write `benchmarks/embedding_diagnostics.py`: given a support set, report intra-class vs inter-class distance, and flag classes that overlap. | A runnable script |

### Prove it

1. Write the prototypical-network classification rule as an equation.
2. Why is a frozen ImageNet backbone a reasonable feature extractor for maize leaves? Where does that assumption break?
3. If two classes have overlapping embedding clusters, what will calibration do about it — and what *can't* it fix?
4. What does L2-normalising an embedding do to Euclidean distance?

**Skill unlocked:** You can explain, to a skeptic, why 5 images per class can work at all.

---

## Week 4 — Calibration

> **Thesis link:** "Knows when it doesn't know" starts here. A confidence number that doesn't mean anything is worse than no number.

| | |
| :-- | :-- |
| **Concept goal** | ECE, temperature scaling, why the standard ECE estimator is biased |
| **Repo focus** | `core/calibration.py` (304), `core/act.py` (152) |
| **Ship** | A reliability diagram generator |
| **Read** | 📄 `research/On Calibration of Modern Neural Networks.pdf` (Guo et al. 2017)<br>📄 `research/Verified Uncertainty Calibration.pdf` (Kumar et al. 2019) |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Guo et al. §1–3. Definition of calibration; why modern nets are overconfident. | Notes: define calibration in one sentence |
| Tue | Learn | Temperature scaling: divide logits by scalar T. **It cannot change accuracy** — prove to yourself why. | Notes: why is argmax preserved? |
| Wed | Learn | Kumar et al.: the plug-in ECE estimator is **biased**; scaling-binning fixes it. This is why your code has `debiased_ece` and `scaling_binning`. | Notes: what is the bias, and why does binning cause it? |
| Thu | Read | `calibration.py` — find ECE, debiased ECE, temperature fitting. Match each to the paper section it came from. | Annotated notes |
| Fri | Ship | Write a reliability-diagram generator (matplotlib optional-extra, or ASCII to stay dependency-free). Run it on synthetic data. | A diagram you can put in the README |

### Prove it

1. A model outputs confidence 0.9 on 100 predictions and gets 70 right. What is its calibration error on that bin? Is it over- or under-confident?
2. Why does temperature scaling leave accuracy untouched?
3. Why is the naïve ECE estimate biased, and in which direction?
4. `act.py` adapts thresholds over time. What breaks if the threshold drifts monotonically? (Your CHANGELOG says v0.2.0 fixed exactly this — find the fix.)

**Skill unlocked:** You can look at a confidence score and say whether it's trustworthy — and prove it with a diagram.

---

## Week 5 — Conformal prediction ★

> **Thesis link:** This is the week that matters most. Conformal prediction is your actual scientific contribution. Everything else is supporting cast.

| | |
| :-- | :-- |
| **Concept goal** | Distribution-free, finite-sample coverage guarantees |
| **Repo focus** | `core/conformal.py` (436 lines, 14 tests — your best-tested module) |
| **Ship** | An empirical coverage validation experiment |
| **Read** | 📄 Angelopoulos & Bates, *"A Gentle Introduction to Conformal Prediction"* (arXiv 2107.07511) — free, and the best entry point that exists |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Gentle Intro §1–2. Nonconformity scores. The split-conformal recipe. | Notes: the 4-step recipe from memory |
| Tue | Learn | The quantile: `q̂ = ⌈(n+1)(1−α)⌉/n` empirical quantile of calibration scores. Why the `+1`? | Notes: explain the finite-sample correction |
| Wed | Learn | **Exchangeability** — the one assumption. And **marginal ≠ conditional** coverage: the single most misunderstood caveat. | Notes: give a concrete example where marginal coverage is satisfied but a subgroup is badly served |
| Thu | Read | `conformal.py` — find the score function, the quantile computation, LOO calibration, and the cross-conformal K-fold path. | Map each to the paper |
| Fri | Ship | `benchmarks/conformal_coverage.py`: run many trials at α = 0.1, count how often the true label lands in the set. **Empirical coverage should be ≥ 90%.** | A number that either validates your implementation or exposes a bug |

### Prove it

1. State the conformal guarantee precisely. What is it a probability *over*?
2. What breaks the guarantee? (Name the assumption and a realistic scenario that violates it.)
3. Why does leave-one-out calibration matter more in few-shot than in ordinary ML?
4. Your prediction set is empty. What does that mean, and what should the system do?
5. Marginal coverage is 90% overall but only 60% for one disease class. Is the guarantee violated? Is the system acceptable?

> ⚠️ **If Friday's empirical coverage comes out below 1−α, stop the roadmap and debug it.** That number is the foundation of the whole project. Everything else can wait.

**Skill unlocked:** You can defend a mathematical guarantee under questioning. This is what a reviewer will probe hardest.

---

## Week 6 — Uncertainty & OOD

> **Thesis link:** Coverage tells you *how often* you're right. This tells you *why* you're unsure.

| | |
| :-- | :-- |
| **Concept goal** | Aleatoric vs epistemic vs distributional uncertainty; Mahalanobis OOD |
| **Repo focus** | `core/uncertainty.py` (569 lines, 10 tests) |
| **Ship** | An OOD detection benchmark |
| **Read** | Lee et al. 2018, *"A Simple Unified Framework for Detecting Out-of-Distribution Samples"* (arXiv 1807.03888) |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Aleatoric (irreducible data noise) vs epistemic (reducible model ignorance). More data fixes one, not the other. | Notes: one real example of each from crop disease |
| Tue | Learn | Mahalanobis distance; why covariance matters; why shrinkage regularisation is needed when n < d. | Notes: what goes wrong with a raw covariance estimate on 15 samples? |
| Wed | Read | `uncertainty.py` — locate all three signals. How are they fused? What is "mode-gated computation"? | Notes |
| Thu | Read | Find the OOD threshold. How is it chosen? Is it principled or hand-tuned? **Be honest in your notes.** | An honest assessment |
| Fri | Ship | `benchmarks/ood_detection.py`: feed in-distribution and clearly-OOD images, report AUROC. | A real detection number |

### Prove it

1. A blurry photo of a leaf vs a photo of a car. Which uncertainty type spikes for each?
2. Why does Mahalanobis need shrinkage when your support set is 15 images and embeddings are 512-d?
3. Your OOD detector flags 20% of valid inputs. What tradeoff are you looking at, and who decides where it sits?

**Skill unlocked:** You can distinguish "the model is guessing" from "this input doesn't belong here" — and measure both.

---

# Phase III — Structure

## Week 7 — Refactoring `learner.py`

> **Thesis link:** 1,643 lines in one class is the thing a reviewer points at first. It also stops *you* from reasoning about your own system.

| | |
| :-- | :-- |
| **Concept goal** | Separation of concerns, dependency injection, safe refactoring |
| **Repo focus** | `core/learner.py` (1,643 lines — 21% of your library, 7 tests) |
| **Ship** | `learner.py` under 800 lines with behaviour unchanged |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Read | List every method in `FewShotLearner` and group by responsibility: state, inference, calibration, persistence, feedback. | A grouped inventory |
| Tue | Learn | The "god object" anti-pattern. Extract Class, Extract Method. **Characterisation tests** — pin current behaviour before touching anything. | Notes |
| Wed | Ship | Write characterisation tests for the paths you're about to move. These are your safety net. | Tests that pass now |
| Thu | Ship | Extract the largest cohesive group into its own collaborator class. Run the suite after **every** extraction. | Green suite |
| Fri | Ship | Extract a second group. Update docstrings. Confirm the public API is byte-for-byte unchanged. | `learner.py` < 800 lines |

### Prove it

1. Which responsibilities did `FewShotLearner` hold? Name at least four.
2. What is a characterisation test, and why must it come *before* the refactor?
3. How do you refactor without changing the public API — and why does that matter to users?

> 🛟 **Safety rule:** commit after every single extraction. If the suite goes red, `git reset --hard` and take a smaller bite. Never refactor for more than 30 minutes without a green run.

**Skill unlocked:** You can safely restructure code you didn't originally write — the single most valuable skill on this list.

---

## Week 8 — Continual learning

> **Thesis link:** "Learns from every human correction" is a headline claim. This week you find out whether it's true.

| | |
| :-- | :-- |
| **Concept goal** | Catastrophic forgetting, Fisher information, EWC |
| **Repo focus** | `training/feedback_router.py` (139), `training/finetune.py` (207), `training/up_ugf.py` (160) |
| **Ship** | A forgetting-measurement experiment |
| **Read** | Kirkpatrick et al. 2017, *"Overcoming catastrophic forgetting in neural networks"* (PNAS, arXiv 1612.00796) |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Catastrophic forgetting: why learning task B destroys task A. | Notes |
| Tue | Learn | Fisher information as a measure of parameter importance; the EWC quadratic penalty. | Notes: write the EWC loss from memory |
| Wed | Read | `finetune.py` — note the scope comment: **head-only, ~2K params, not full-network EWC.** Is the README as honest as the code? | An honesty check |
| Thu | Read | `up_ugf.py` — uncertainty-guided forgetting, LSH redundancy scoring. What does LSH approximate, and what's the failure mode? | Notes |
| Fri | Ship | `benchmarks/forgetting.py`: train on classes A+B, add C via corrections, re-measure A+B accuracy. **Quantify the forgetting.** | A real number |

### Prove it

1. Why does the Fisher diagonal indicate which weights matter?
2. Your EWC is head-only (~2K params). What can it *not* protect against, and is that limitation stated in your docs?
3. What does UP-UGF discard, and what's the risk of discarding the wrong example?

**Skill unlocked:** You can measure whether a learning system actually learns — or just appears to.

---

# Phase IV — Truth

## Week 9 — The truth pass

> **Thesis link:** Every unverifiable claim in your docs is a loaded gun pointed at your credibility. This week you unload them.

| | |
| :-- | :-- |
| **Concept goal** | Scientific honesty; the difference between a demo and a deployment |
| **Repo focus** | `docs/` (50 files), `README.md`, `examples/mziziguard/` |
| **Ship** | Zero unverifiable claims in the repo |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Audit | Grep every number in the docs: `%`, `ms`, `MB`, accuracy. For each, find the benchmark that produced it — **or mark it for deletion.** | A claims spreadsheet |
| Tue | Ship | Fix MziziGuard: remove "deployed", remove the `~150ms` latency, remove implied farmers. Relabel as *"an illustrative demo on synthetic images"*. | Honest docs |
| Wed | Ship | Add a banner to the demo docs stating plainly that the sample data is procedurally generated, not photographs. | An honest demo |
| Thu | Ship | Rewrite the README opening around the thesis sentence. Cut superlatives. Replace every claim with a measured number or delete it. | A README you'd defend |
| Fri | Ship | Docs structure: merge `architecture.md` + `architecture-deep-dive.md`; unpublish `AUDIT_REPORT.md`, `v0.1.1-docs-roadmap.md`, `release-checklist-v0.1.1.md`; fix the two tutorials both numbered 13. | Clean nav |

### Prove it

1. What is the difference between "MziziGuard is deployed" and "MziziGuard is an example application"? Who is harmed by the first?
2. Which of your numbers can you reproduce **right now**, on demand? That set is your real evidence base.

> 💡 This is the hardest week emotionally. Deleting your own claims feels like going backwards. It isn't — it's the week your project becomes trustworthy. **A small true claim beats a large unprovable one, every time.**

**Skill unlocked:** Scientific integrity as a working practice, not a slogan.

---

## Week 10 — Real data, real numbers

> **Thesis link:** One honest benchmark on real data is worth more than fifty pages of documentation.

| | |
| :-- | :-- |
| **Concept goal** | Benchmark design, baselines, reproducibility, error bars |
| **Repo focus** | `benchmarks/run_benchmark.py` (394), `results/` |
| **Ship** | Your first real, reproducible, published result |
| **Data** | **PlantVillage** — free, public, ~54k real crop-disease leaf photographs |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Few-shot evaluation protocol: N-way K-shot episodes, why you average over many episodes, why you report confidence intervals. | Notes |
| Tue | Ship | Write a PlantVillage loader. Keep it out of the package — a benchmark script, not a dependency. | Loader + a documented download step |
| Wed | Ship | Run 5-way 5-shot over ≥100 episodes. Report mean accuracy **± 95% CI**. | A real accuracy number |
| Thu | Ship | Run the conformal path on the same data. Report **empirical coverage** at α = 0.1, plus mean prediction-set size. | Your headline result |
| Fri | Ship | Commit `results/plantvillage_5way5shot.json` with seed, hardware, OS, Python version, timestamp, and library version. Write it up in `docs/getting-started/benchmarks.md`. | Reproducible evidence |

### Prove it

1. Why report a confidence interval instead of a single accuracy number?
2. What must a reader know to reproduce your result exactly?
3. Your coverage is 88% when you targeted 90%. Is that a bug, or noise? **How would you tell?** (This question is the whole week.)

> 🎯 **This is the most valuable week in the plan.** After Friday, you own something almost no solo project has: a real number, on real data, that a stranger can reproduce. Everything before this was preparation.

**Skill unlocked:** You can design and report an experiment that survives scrutiny.

---

# Phase V — Focus

## Week 11 — Narrowing the scope

> **Thesis link:** A library is judged by what it refuses to include.

| | |
| :-- | :-- |
| **Concept goal** | API surface design, semantic versioning, deprecation |
| **Repo focus** | `studio/` (1,822 lines, 4 tests), `ui/app.py` (151), `core/contrastive.py` (512), `core/explain.py` (586) |
| **Ship** | A library ~40% smaller, arguing one point |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Learn | Semantic versioning, deprecation policy, Hyrum's Law (*every observable behaviour will be depended on by someone*). | Notes: what does removing a public name cost? |
| Tue | Ship | Extract `studio/` to its own repo **with history preserved** (`git subtree split`). Nothing is lost — it becomes `adaptshot-studio`. | A new repo |
| Wed | Ship | Remove `studio/` + `ui/` from the library. Drop the `gui`/`ui` extras. Fix the tests and docs that referenced them. | −1,973 lines |
| Thu | Ship | Move `contrastive.py` under `training/`. Mark `explain.py` experimental in its docstring and docs. Drop both from the README's headline features. | An honest feature list |
| Fri | Ship | Trim `__all__` to the names you will actually support. Every export is a promise. | A defensible API |

### Prove it

1. Why is a smaller public API easier to maintain *and* more credible?
2. What does `git subtree split` preserve that copying files would destroy?
3. Which exports would you now refuse to add, and why?

**Skill unlocked:** You can say no to your own code — the skill that separates maintainers from accumulators.

---

## Week 12 — Release & write-up

> **Thesis link:** Work nobody can find or cite doesn't compound.

| | |
| :-- | :-- |
| **Concept goal** | Release engineering, citation, technical writing |
| **Repo focus** | `pyproject.toml`, `CHANGELOG.md`, `.github/workflows/` |
| **Ship** | v0.3.0 released, citable, with an honest write-up |

### Daily

| Day | Focus | Task | Output |
| :-- | :-- | :-- | :-- |
| Mon | Ship | Fix the `src.adaptshot` import prefix you flagged in W1 — tests should import the package the way users do. | A normal test suite |
| Tue | Ship | Add `CITATION.cff` so people can cite you properly. Add a Zenodo DOI hook if you want permanence. | A citable project |
| Wed | Ship | Automate release: a workflow that builds and publishes to PyPI on tag. Flip the docs build to `--strict`. | One-command releases |
| Thu | Ship | Write `CHANGELOG` for v0.3.0 — including, honestly, what you **removed** and why. | An honest changelog |
| Fri | Write | Draft a 2-page technical note: the problem, the method, the W10 numbers, the limitations. **Be specific about limitations.** | The seed of a paper |

### Prove it

1. What makes a result citable rather than merely public?
2. Why does a changelog that documents removals build more trust than one that only lists additions?
3. What are the three most important limitations of AdaptShot? (If you can't name three, you don't understand it yet.)

**Skill unlocked:** You can ship, version, and communicate research-grade software.

---

# Appendices

## A. Reading list

**You already own these** (in `research/`):

| Paper | Week | Why |
| :-- | :-- | :-- |
| Prototypical Networks (Snell et al. 2017) | W3 | The algorithm your library is built on |
| On Calibration of Modern Neural Networks (Guo et al. 2017) | W4 | ECE and temperature scaling |
| Verified Uncertainty Calibration (Kumar et al. 2019) | W4 | Why your code has debiased ECE and scaling-binning |

**Get these** (all free on arXiv):

| Paper | Week | Why |
| :-- | :-- | :-- |
| A Gentle Introduction to Conformal Prediction (Angelopoulos & Bates, 2107.07511) | W5 | The best entry point to your core contribution |
| A Simple Unified Framework for Detecting OOD Samples (Lee et al., 1807.03888) | W6 | The Mahalanobis OOD method |
| Overcoming Catastrophic Forgetting (Kirkpatrick et al., 1612.00796) | W8 | EWC |
| Matching Networks (Vinyals et al., 1606.04080) | W3 opt. | The other classic few-shot baseline |

> `research/Ordinal Depth Supervision for 3D Human Pose Estimation.pdf` is unrelated to this project — move it out of `research/`.

## B. Glossary — you must be able to define all of these by Week 12

Few-shot · support set · query · prototype · embedding · backbone · N-way K-shot episode · calibration · ECE · debiased ECE · temperature scaling · reliability diagram · conformal prediction · nonconformity score · coverage · marginal vs conditional coverage · exchangeability · prediction set · leave-one-out calibration · cross-conformal · aleatoric · epistemic · distributional uncertainty · Mahalanobis distance · shrinkage · OOD · AUROC · catastrophic forgetting · Fisher information · EWC · InfoNCE · LSH · characterisation test · god object · semantic versioning

## C. When you get stuck

| Situation | Do this |
| :-- | :-- |
| Concept won't click | Explain it aloud to an empty room. The gap where you stumble is the thing you don't know. |
| Code is impenetrable | Add a breakpoint or print, run a test, watch the values move. Reading alone is slower than watching. |
| Refactor went red | `git reset --hard`, take a smaller bite. Never debug a big refactor. |
| Week is overrunning | Take two. The order matters more than the pace. |
| Motivation gone | Re-read your W10 result. That number is real, and you made it. |

## D. Self-assessment — score yourself 1–5 at weeks 1, 6, and 12

| Skill | W1 | W6 | W12 |
| :-- | :-: | :-: | :-: |
| I can explain few-shot learning to a non-expert | | | |
| I can explain conformal prediction to an expert | | | |
| I can change this code without fear | | | |
| I can tell a real result from a plausible one | | | |
| I can defend every claim in my README | | | |
| I could rebuild this library from scratch | | | |

## E. After week 12

1. **Deploy for real.** Find one agricultural extension officer, one clinic, one partner. Real users beat any benchmark.
2. **Write the paper.** You'll have the numbers and the limitations. Target a workshop first — they're friendlier and faster.
3. **Preprint on arXiv.** Free, permanent, citable.
4. **Then, and only then, talk about impact.** By that point you won't need to — the work will do it.

---

> **The honest summary:** in 12 weeks you will not have the most impactful AI library ever built. You will have something better: a small, correct, well-understood library that does one hard thing provably well, and a maintainer who can defend every line of it. That is what gets cited, funded, and deployed. Everything else follows from it.
