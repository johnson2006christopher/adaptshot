# What AdaptShot is, in plain words

> **For:** someone who has never written a line of code and does not know what "machine learning" means. Nothing to install on this page. Ten minutes of reading.

## The problem it solves

Imagine you grow maize. A leaf on one plant has grey spots. Is it a disease? Which one? Should you spray, and with what? Getting that wrong costs money either way.

There are computer programs that can look at a photograph of a leaf and name the disease. Almost all of them need three things you probably do not have: a powerful computer with a special graphics chip, a fast internet connection, and thousands of labelled example photographs to learn from.

AdaptShot is built for the opposite situation. It runs on an ordinary laptop, it works with the internet switched off, and it learns from **a handful** of example photographs — say, five of a healthy leaf and five of each disease. That is what "few-shot" means: few examples.

## What it gives you back

Most programs of this kind give you one answer: *"northern leaf blight."* You have no way of knowing how sure it is.

AdaptShot gives you three things, and the second and third are the point:

1. **A best guess** — *northern leaf blight*, with a confidence number.
2. **A short list it is prepared to stand behind** — for example `{gray leaf spot, northern leaf blight}`. This list comes with a promise: over many photographs, the true answer will be inside the list at least a chosen fraction of the time (you choose the fraction; 90% is typical). When the program is sure, the list has one entry. When it is torn between two diseases, it has two — and it tells you so instead of picking one and hoping.
3. **Sometimes, a refusal.** If the photograph does not look like anything it was taught, it says *"I don't know — ask a person."* A tool that admits this is safer than one that always answers.

Two honest limits, stated now rather than discovered later. The promise in point 2 holds only when the photographs you ask about look like the ones you taught it with — same kind of camera, similar light. Take the teaching photos in the field where you will use it. And the best guess is only as good as the examples: it cannot recognise a disease you never showed it.

## What you need

- An ordinary computer — Windows, Mac or Linux. No special graphics chip.
- About 200 MB of free space and, once, an internet connection to install it. After that it works offline.
- A few photographs of each thing you want it to recognise. Five each is enough to start; more is better.

You do **not** need to know Python to follow the next five pages. You will type a few commands and run a few short scripts, each explained line by line. If you want to learn Python properly afterwards, the [official beginner's guide](https://docs.python.org/3/tutorial/) is the place — but you can use AdaptShot before that.

## The five pages after this one

1. [Install it](01-install.md) — and check what your computer can do.
2. [Your first prediction](02-first-prediction.md) — on photographs that come with the program, so nothing can go wrong.
3. [Your own photographs](03-your-own-photos.md) — a folder per thing you want recognised.
4. [Reading the answer](04-reading-the-answer.md) — what each part means and what to do about it.
5. [Teaching it when it is wrong](05-teaching-corrections.md) — and saving what it has learned.

Each page's commands are run automatically by the project's tests on every change, so they cannot quietly stop working. If one does not work for you, that is a bug in the documentation, not in you — [tell us](https://github.com/johnson2006christopher/adaptshot/issues).
