# Why CPU-only, why offline, and why it never picks a GPU

> **For:** anyone who wonders why a machine-learning library would refuse to use a graphics card sitting right there. No code.

## The person it is built for

AdaptShot began with a specific person in mind: someone in a field in Tanzania with a phone or a second-hand laptop, a patchy connection that costs money per megabyte, and a plant they need to identify today. Every design constraint follows from taking that person seriously rather than as a marketing story.

- **No GPU**, because they do not have one, and a tool that is fast only on hardware they will never own is not a tool for them.
- **Offline**, because the connection is the thing they cannot count on — and because a tool that phones home is a tool that stops working, and a tool that stops working at the moment it is needed teaches people never to trust it again.
- **Small**, because 200 MB is a real cost on a metered connection and a real fraction of an old phone's storage.
- **Few examples**, because thousands of labelled photographs of *their* crop, in *their* conditions, do not exist.

## What the constraints buy

A fixed target makes claims checkable. Because the target is an ordinary CPU, the project can state one latency, one memory ceiling, and one install size, measure them, and be held to them: about 8 ms per photograph and 120 MB for a full cycle on the laptop the numbers were taken on, 3.5 MB for the wheel, five seconds from `pip install` to a correct answer. "Depends on your hardware" is not a specification; "on this machine, this" is.

Because it is offline, the test suite can enforce it. On every change, the wheel is installed into a clean environment inside a network namespace with no interfaces, and the quickstart, the demo, the conformal suite and the benchmark run there. Any dependency that adds a download, a telemetry call or a version check fails the build. *The test suite fails if the library touches the network* is a sentence the project can say because it made itself unable to say anything else.

Because it learns from few examples, it has to say when it does not know — which is where the [prediction set and its guarantee](the-guarantee.md) come from. A tool that is right 91% of the time and silent about the other 9% sends someone to spray the wrong thing one time in eleven.

## Why a GPU is mentioned and never selected

`check_environment()` will tell you a GPU is present and that `device="cuda"` is available. It will not use it, and the defaults never will. This is deliberate, and the reasons are strategic rather than technical:

- Every machine-learning library selects a GPU when it finds one. It is table stakes, not a contribution.
- It invites a comparison the project loses. If AdaptShot is fast on a GPU, the honest next question is why not use PyTorch or timm directly — and the honest answer is that you should. The argument holds only on the ground the project chose.
- It dissolves the constraint that makes the numbers mean anything.
- GPU determinism is materially harder to guarantee, and reproducibility at a fixed seed is a stated constraint of the project.

GPU support exists and stays opt-in: set `device="cuda"` yourself if you have one and want it.

## What is *not* claimed

CPU-only does not mean "as accurate as the big models". On the published benchmark AdaptShot's accuracy is exactly a nearest-centroid classifier's on a frozen ImageNet encoder — the layers above it change what accompanies the answer, not which answer comes out. A task the encoder cannot separate is a task AdaptShot cannot separate either, without the optional fine-tuning extra. The [technical note](technical-note.md) says this in its results section, deliberately, because a reader deciding whether to adopt the library deserves the unflattering figure before they install it.

## The argument, in one line

Most tools address the missing GPU and the missing dataset. The missing connection is the hard one, and it is the one this project made a specification.
