# About AdaptShot

<div align="center" style="margin: 40px 0;">
<img src="../images/johnson.jpeg" alt="Johnson Christopher Hassan" width="200" height="200" style="border-radius: 50%; border: 4px solid #4ADE80; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</div>

## Built in Tanzania, for the World

**AdaptShot was born from a simple but powerful realization:**  
*The world doesn't need another AI that requires a supercomputer to run. It needs AI that can run on a laptop in a rural clinic, a phone in a farmer's pocket, or a tablet in an offline classroom.*

---

## 🌍 The Story Behind the Code

### The Moment of Clarity

In 2025, while working in Mbeya, Tanzania, I watched a healthcare worker struggle to diagnose crop diseases from photos. She had a smartphone, dedication, and knowledge—but the AI tools available to her required:
- Cloud connectivity (unavailable in her region)
- High-end GPUs (costing more than her annual salary)
- Thousands of labeled images (impossible for rare, localized diseases)
- Constant internet access (a luxury, not a reality)

**Every "state-of-the-art" solution failed her.**

That's when I realized: *We've been building AI for Silicon Valley, not for the world.*

### The Problem We're Solving

Modern AI has become **obsessed with scale**:
- "Bigger models"
- "More data"
- "Higher accuracy on benchmark leaderboards"

But this obsession has created a dangerous blind spot:
- ❌ **Energy waste**: Training one large model emits as much CO₂ as five cars over their lifetimes
- ❌ **Exclusion**: 80% of the world cannot access these tools due to infrastructure constraints
- ❌ **Overconfidence**: Models that guess with 99% confidence while being completely wrong
- ❌ **Forgetfulness**: Every new task requires retraining from scratch, wasting prior knowledge
- ❌ **Black boxes**: Users have no idea when to trust the model and when to seek human expertise

**This isn't progress. It's a crisis of accessibility, sustainability, and trust.**

---

## 🎯 The AdaptShot Philosophy

### Mission
> **Democratize trustworthy, CPU-first, human-in-the-loop few-shot vision AI for resource-constrained environments.**

We believe that:
1. **Accuracy shouldn't require abundance** – A model should work with 10 images, not 10 million
2. **Uncertainty is a feature, not a bug** – A model that knows when it doesn't know is more valuable than one that always guesses
3. **Humans are the ultimate teachers** – One expert correction is worth a thousand synthetic augmentations
4. **Carbon footprint matters** – Every joule of energy counts when deploying at scale
5. **Open science is the only science** – If it can't be reproduced on a laptop in Tanzania, it's not real progress

### Vision
> **To become the most impactful, energy-efficient, and scientifically rigorous AI library ever built—shifting the paradigm from "bigger models, more data, higher carbon" to "smarter architectures, human alignment, minimal footprint."**

We're not competing on benchmarks. We're competing on **real-world impact**.

---

## 🌱 Core Values

### 1. **Truth Over Hype**
We document what works, what doesn't, and why. No exaggerated claims. No "state-of-the-art" without peer review. No benchmarks without hardware specs.

### 2. **Constraint-First Engineering**
We start with the hardest constraints (CPU, <250MB RAM, offline, <50 images) and build up. If it works there, it'll work anywhere.

### 3. **Human Dignity**
Every feature is designed to **augment human expertise**, not replace it. A doctor in a rural clinic deserves AI that says "I'm 60% confident—please review" rather than "I'm 99% sure" when wrong.

### 4. **Environmental Responsibility**
We track Joules per inference. We optimize for carbon efficiency. We publish our methodology transparently. AI should help solve climate change, not accelerate it.

### 5. **Global South First**
We build for the user in Mbeya before the user in Menlo Park. If it works offline on a 3-year-old Android phone, it's ready for release.

---

## 🔬 The Technical Journey

### Phase 1: The Foundation (Early 2025)
**Question:** *Can we achieve reasonable accuracy with <50 images per class using only CPU?*

**Answer:** Yes—by freezing a pretrained backbone (ResNet-18/MobileNetV3), extracting embeddings, and using cosine similarity. No fine-tuning required. Just metric-based retrieval.

**Result:** 68% accuracy on CIFAR-10 subset with 10 images/class, 90ms latency on Intel i5 CPU.

### Phase 2: Calibration & Trust (Mid 2025)
**Question:** *How do we know when the model is overconfident?*

**Answer:** Online temperature scaling with sliding-window Expected Calibration Error (ECE) tracking. No held-out validation set required—critical for few-shot scenarios.

**Result:** ECE < 0.05 after 15 predictions. Model knows when to say "I'm not sure."

### Phase 3: Human-in-the-Loop (Late 2025)
**Question:** *How do we incorporate expert corrections without retraining from scratch?*

**Answer:** Adaptive Confidence Thresholding (ACT) + Correction-Aware Elastic Weight Consolidation (CA-EWC). ACT decides when to request human feedback. CA-EWC fine-tunes the classification head while preserving prior knowledge.

**Result:** Model improves with every correction. No catastrophic forgetting.

### Phase 4: Memory & Sustainability (Early 2026)
**Question:** *How do we keep memory bounded on edge devices?*

**Answer:** Uncertainty-Guided Forgetting (UP-UGF). Score embeddings by uncertainty × recency × (1−redundancy). Evict low-value examples when buffer exceeds capacity.

**Result:** <250MB RAM footprint indefinitely. No OOM crashes.

### Phase 5: Carbon Awareness (Mid 2026 - v0.1.1)
**Question:** *How do we make energy efficiency a first-class citizen?*

**Answer:** `eco_mode` configuration, energy profiling benchmarks, CO₂ estimation, early-exit thresholds. Users can optimize for carbon without sacrificing core functionality.

**Result:** 15–30% energy reduction with minimal accuracy loss. Transparent carbon reporting.

---

## 🏆 What Makes AdaptShot Different

| Feature | Conventional AI | AdaptShot |
|---------|----------------|-----------|
| **Hardware** | GPU cluster required | CPU-only, <250MB RAM |
| **Data** | Millions of images | <50 images per class |
| **Connectivity** | Cloud-dependent | Fully offline |
| **Confidence** | Overconfident softmax | Calibrated ECE tracking |
| **Learning** | Retrain from scratch | Continual via corrections |
| **Memory** | Unbounded growth | Bounded via UP-UGF |
| **Carbon** | Hidden cost | Transparent reporting |
| **Trust** | Black box | Human-in-the-loop by design |

---

## 👨‍💻 Meet the Creator

<div align="center" style="margin: 30px 0;">
<img src="../images/johnson.jpeg" alt="Johnson Christopher Hassan" width="280" style="border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
</div>

**Johnson Christopher Hassan**  
📍 Mbeya, Tanzania 🇹  
✉️ johnson2006christopher@gmail.com

### My Journey
I grew up watching brilliant people in my community solve impossible problems with limited resources. A farmer diagnosing crop disease by squinting at leaves. A nurse identifying pneumonia from a single X-ray. A teacher recognizing dyslexia from handwriting patterns.

**They didn't need millions of examples. They needed wisdom.**

My mission is to encode that wisdom into AI that respects their constraints, honors their expertise, and amplifies their impact—without requiring them to become machine learning engineers.

### Why I Built AdaptShot
1. **Frustration**: Every "edge AI" library still assumed cloud connectivity or GPU acceleration
2. **Responsibility**: As an AI engineer, I owe it to my community to build tools that actually work for them
3. **Vision**: I believe the future of AI isn't bigger—it's **smarter, humbler, and more human**

### What Drives Me
- **The farmer** who needs to identify cassava mosaic virus before the harvest fails
- **The nurse** who needs to triage pneumonia cases without sending X-rays to a distant radiologist
- **The teacher** who needs to identify learning disabilities early, without expensive assessments
- **The conservationist** who needs to identify endangered species from camera traps in remote areas

**These are not "edge cases." They are the majority.**

---

## 🌍 The Bigger Picture

### The Climate Crisis
AI currently accounts for **~2–3% of global electricity consumption**—and growing. By 2030, it could consume more energy than all of Sweden.

**AdaptShot is part of the solution:**
- CPU-first design reduces energy by 10–100× vs. GPU inference
- Few-shot learning eliminates the need for massive training runs
- Carbon-aware configuration lets users optimize for efficiency
- Transparent reporting holds us accountable

### The Accessibility Crisis
80% of the world's population lives in regions with unreliable internet, limited electricity, or both. Yet 95% of AI research assumes cloud connectivity and high-end hardware.

**AdaptShot flips this:**
- Offline-first by default
- Runs on a Raspberry Pi 4
- <50 images per class (achievable in real-world settings)
- Human corrections valued over synthetic data augmentation

### The Trust Crisis
AI systems are deployed in high-stakes domains (healthcare, criminal justice, finance) while being fundamentally uncalibrated—overconfident on wrong answers, uncertain on right ones.

**AdaptShot addresses this:**
- Calibrated uncertainty (ECE tracking)
- Adaptive thresholding (ACT) knows when to request human review
- Transparent predictions (returns nearest neighbor, confidence, decision rationale)
- Human-in-the-loop by design, not as an afterthought

---

##  The Road Ahead

### v0.1.1 (Current - May 2026)
✅ Production-ready tutorials  
✅ Energy profiling & eco mode  
✅ Robust error handling  
✅ Complete documentation  

### v0.2.0 (Q3 2026)
- 🔜 ONNX export for mobile deployment
- 🔜 Swahili documentation (my first language)
- 🔜 Federated buffer sharing for multi-device deployments
- 🔜 Plugin architecture for experimental backends

### v1.0.0 (Q1 2027)
- 🔜 Full ablation studies published in peer-reviewed venue
- 🔜 Field pilot results from 3+ NGOs
- 🔜 Carbon-neutral CI/CD pipeline
- 🔜 Community governance board established

### v2.0+ (2027+)
- 🔜 Neuromorphic backend support (when hardware matures)
- 🔜 Event-based vision for DVS cameras
- 🔜 Multilingual, low-literacy UI extensions
- 🔜 Integration with national healthcare/agriculture systems

---

## 💌 A Personal Note

If you're reading this, you're part of the solution.

Maybe you're a researcher who wants to reproduce our results.  
Maybe you're a developer who wants to deploy this in your community.  
Maybe you're a student who wants to learn how to build responsible AI.  
Maybe you're a policymaker who wants to understand what sustainable AI looks like.

**Whoever you are, wherever you are—thank you.**

This project exists because people like you believe that technology should serve humanity, not the other way around. That progress should be measured in lives improved, not benchmarks topped. That the future of AI should be built **with** the Global South, not **for** it.

I'm just one person in Mbeya, Tanzania, writing Python code at night after work. But with your help, AdaptShot can become something far bigger than me.

**Let's build the future of AI together.**  
One correction at a time. One joule at a time. One life at a time.

---

## 🤝 Get Involved

- **Use it**: Deploy AdaptShot in your community. Share your results.
- **Contribute**: Submit a PR, write a tutorial, translate documentation.
- **Fund it**: Support sustainable AI development through GitHub Sponsors.
- **Teach it**: Use AdaptShot in your classroom or workshop.
- **Research it**: Publish ablation studies, comparisons, or extensions.

**The future is open. The future is collaborative. The future is human.**

---

<div align="center" style="margin-top: 60px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white;">

### "The best AI doesn't guess confidently.  
### It learns humbly, admits uncertainty,  
### and improves through every human correction."

**— Johnson Christopher Hassan, 2026**

[⭐ Star on GitHub](https://github.com/johnson2006christopher/adaptshot) · [📖 Read the Docs](https://johnson2006christopher.github.io/adaptshot/) · [💬 Join the Community](https://github.com/johnson2006christopher/adaptshot/discussions)

</div>