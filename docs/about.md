# About AdaptShot

<div align="center" style="margin: 40px 0;">
<img src="../images/johnson.jpeg" alt="Johnson Christopher Hassan" width="200" height="200" style="border-radius: 50%; border: 4px solid #4ADE80; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</div>

## Built in Tanzania with Nothing But Determination

**This is not just a library. This is proof that innovation doesn't require Silicon Valley.**  
**It requires vision. It requires pain. It requires refusing to accept that "impossible" is a final answer.**

---

## 🌍 The Story Behind the Code

### Who I Am

My name is **Johnson Christopher Hassan**.

I am a **diploma student in Computer Engineering** at **Mbeya University of Science and Technology**, Tanzania.

I am **self-taught** in AI and machine learning.

I have **no fancy GPU cluster**.  
I have **no research lab**.  
I have **no corporate funding**.  
I have **zero dollars** in my pocket most days.

What I **do** have:
- A laptop that struggles to run modern software
- Unreliable electricity that cuts out daily
- Internet that costs more than I can afford
- A community that needs AI solutions **now**, not in 10 years
- An unshakable belief that **African engineers can build world-changing technology**

### The Moment Everything Changed

In early 2025, I watched a healthcare worker in my community try to diagnose crop disease from photos on her phone. She had:
- **Knowledge**: Years of agricultural training
- **Dedication**: Walking miles to reach remote farms
- **A smartphone**: Her only tool

But the AI tools available to her required:
- **Cloud connectivity**: Unavailable in rural Tanzania
- **High-end GPUs**: Costing more than her annual salary
- **Thousands of labeled images**: Impossible for rare, localized diseases
- **Constant internet**: A luxury we don't have

**Every "state-of-the-art" solution failed her. Every single one.**

That's when I realized:  
> *"We've been building AI for Silicon Valley, not for the world."*

### The Pain That Fuels This

Let me tell you what it's like to build AI in Tanzania:

**Electricity**: I code by candlelight when the power goes out (which is often). My laptop battery dies mid-training run. I lose hours of work.

**Internet**: I download research papers at 2 AM when bandwidth is cheaper. I wait 3 hours for a 50MB file. I use public Wi-Fi at cafes, counting every megabyte.

**Hardware**: My laptop has 4GB RAM. It overheats constantly. I can't run PyTorch with CUDA because I don't have a GPU. I optimize every line of code because memory is precious.

**Money**: I spend my lunch money on internet bundles. I can't afford cloud compute credits. I can't buy datasets. I build everything from scratch with whatever I can find.

**Isolation**: I have no professor mentoring me. No research group to bounce ideas off. No conference travel budget. I learn from arXiv papers at 3 AM, alone, figuring it out line by line.

**But here's what that pain taught me:**

> **Constraints breed creativity.**  
> **Necessity forces innovation.**  
> **Limitations reveal what truly matters.**

When you have nothing, you learn to make everything count.

---

## 🎯 Why AdaptShot Exists

### The Problem Nobody Is Solving

Modern AI has become **obsessed with scale**:
- "Bigger models" (GPT-4, PaLM, Claude)
- "More data" (trillions of tokens)
- "Higher accuracy on benchmark leaderboards" (by 0.1%)

But this obsession has created a **dangerous blind spot**:

❌ **Energy waste**: Training one large model emits as much CO₂ as five cars over their lifetimes  
❌ **Exclusion**: 80% of the world cannot access these tools due to infrastructure constraints  
❌ **Overconfidence**: Models that guess with 99% confidence while being completely wrong  
❌ **Forgetfulness**: Every new task requires retraining from scratch, wasting prior knowledge  
❌ **Black boxes**: Users have no idea when to trust the model and when to seek human expertise

**This isn't progress. It's a crisis of accessibility, sustainability, and trust.**

### My Mission

> **Democratize trustworthy, CPU-first, human-in-the-loop few-shot vision AI for resource-constrained environments.**

I'm not building AI for:
- Tech giants with billion-dollar compute budgets
- Universities with GPU clusters
- Startups chasing unicorn valuations

I'm building AI for:
- **The farmer** in rural Tanzania who needs to identify cassava mosaic virus before the harvest fails
- **The nurse** in a rural clinic who needs to triage pneumonia cases without sending X-rays to a distant radiologist
- **The teacher** in an offline classroom who needs to identify learning disabilities early
- **The conservationist** monitoring endangered species from camera traps in remote areas
- **The student** like me, coding by candlelight, who believes they can change the world

**These are not "edge cases." They are the majority.**

---

## 🔬 The Journey: From v0.0.1 to v0.1.1

### v0.0.1 (January - May 19, 2026): The Foundation

**Question:** *Can I achieve reasonable accuracy with <50 images per class using only a CPU?*

**Answer:** Yes.

I froze a pretrained backbone (ResNet-18/MobileNetV3), extracted embeddings, and used cosine similarity. No fine-tuning required. Just metric-based retrieval.

**Result:** 68% accuracy on CIFAR-10 subset with 10 images/class, 90ms latency on my overheating Intel i5 CPU.

**What it cost me:**
- 3 months of sleepless nights
- Countless failed experiments
- Internet bills I couldn't afford
- Meals I skipped to save money for data bundles

But **it worked**.

### v0.1.0 (May 15, 2026): The Breakthrough

I realized accuracy wasn't enough. **Trust** was the real problem.

**Innovation 1: Calibration**  
Online temperature scaling with sliding-window Expected Calibration Error (ECE) tracking. No held-out validation set required—critical for few-shot scenarios.

**Result:** ECE < 0.05 after 15 predictions. Model knows when to say "I'm not sure."

**Innovation 2: Human-in-the-Loop**  
Adaptive Confidence Thresholding (ACT) + Correction-Aware Elastic Weight Consolidation (CA-EWC). ACT decides when to request human feedback. CA-EWC fine-tunes the classification head while preserving prior knowledge.

**Result:** Model improves with every correction. No catastrophic forgetting.

**Innovation 3: Memory Management**  
Uncertainty-Guided Forgetting (UP-UGF). Score embeddings by uncertainty × recency × (1−redundancy). Evict low-value examples when buffer exceeds capacity.

**Result:** <250MB RAM footprint indefinitely. No OOM crashes on my 4GB laptop.

### v0.1.1 (May 20, 2026 - Today): Release Candidate

I added:
- ✅ Complete 5-phase tutorial suite (beginner to production)
- ✅ Energy profiling & carbon-aware configuration
- ✅ Robust error handling with custom exceptions
- ✅ Comprehensive documentation
- ✅ This story

AdaptShot v0.1.1 is now released on PyPI. The focus remains on standard release quality: keep the native API stable, keep the docs honest, and let Studio remain an optional convenience layer.

**Why?**  
Because code without documentation is code that nobody can use.  
And AI that nobody can use is AI that doesn't matter.

---

## 🌱 Core Values

### 1. **Truth Over Hype**
I document what works, what doesn't, and why. No exaggerated claims. No "state-of-the-art" without peer review. No benchmarks without hardware specs.

**Because** when you're building for life-or-death scenarios (healthcare, agriculture), lies kill.

### 2. **Constraint-First Engineering**
I start with the hardest constraints (CPU, <250MB RAM, offline, <50 images) and build up. If it works on my laptop in Tanzania, it'll work anywhere.

**Because** the world's hardest problems don't have GPU clusters.

### 3. **Human Dignity**
Every feature is designed to **augment human expertise**, not replace it. A doctor in a rural clinic deserves AI that says "I'm 60% confident—please review" rather than "I'm 99% sure" when wrong.

**Because** AI should serve humans, not the other way around.

### 4. **Environmental Responsibility**
I track Joules per inference. I optimize for carbon efficiency. I publish my methodology transparently. AI should help solve climate change, not accelerate it.

**Because** the planet doesn't have time for wasteful AI.

### 5. **Global South First**
I build for the user in Mbeya before the user in Menlo Park. If it works offline on a 3-year-old Android phone, it's ready for release.

**Because** innovation should serve the many, not the few.

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
| **Cost** | $10,000+ in compute | $0 (runs on your laptop) |
| **Accessibility** | Silicon Valley only | Anywhere with electricity |

---

## 🔮 The Future: What AdaptShot Will Become

### The Next 5 Years

**v0.2.0 (Q3 2026)**
- ONNX export for mobile deployment
- Swahili documentation (my first language)
- Federated buffer sharing for multi-device deployments
- Plugin architecture for experimental backends

**v1.0.0 (Q1 2027)**
- Full ablation studies published in peer-reviewed venue
- Field pilot results from 3+ NGOs in Tanzania, Kenya, and Uganda
- Carbon-neutral CI/CD pipeline
- Community governance board established

**v2.0+ (2027+)**
- Neuromorphic backend support (when hardware matures)
- Event-based vision for DVS cameras
- Multilingual, low-literacy UI extensions
- Integration with national healthcare/agriculture systems

### The Vision: Bringing Impossible to Possible

**What if** a farmer in rural Tanzania could:
- Identify crop diseases from 10 photos on a $50 Android phone
- Get instant diagnosis without internet connectivity
- Teach the model new diseases by showing it examples
- Trust the AI because it admits when it's uncertain

**What if** a nurse in a remote clinic could:
- Triage pneumonia cases from X-rays without a radiologist
- Work offline for weeks, then sync when connectivity returns
- Correct the model when it's wrong, making it smarter for everyone
- Deploy on hardware that costs less than a stethoscope

**What if** a student like me could:
- Build world-class AI with zero funding
- Learn from documentation written in their language
- Contribute to research that matters to their community
- Prove that innovation doesn't require Silicon Valley

**This is not a dream. This is the roadmap.**

### The Impact: Africa and Beyond

**On Research:**
- Prove that rigorous AI research can happen outside elite institutions
- Publish methodologies optimized for constraints, not benchmarks
- Create a new paradigm: "sustainable AI" vs. "bigger is better"
- Inspire the next generation of African AI researchers

**On Africa:**
- Deploy in 10+ countries across healthcare, agriculture, education
- Train 1,000+ African engineers in responsible AI development
- Create open-source tools that solve African problems
- Shift the narrative: Africa as AI innovator, not just AI consumer

**On the World:**
- Challenge Big Tech: "If a student in Tanzania can build this with nothing, what's your excuse?"
- Reduce AI carbon footprint by 10–100× through constraint-driven design
- Make AI accessible to the 80% of the world currently excluded
- Prove that **the future of AI is not bigger—it's smarter, humbler, and more human**

---

## 💌 A Message to Big Tech

To the engineers at Google, Meta, Microsoft, OpenAI:

**You have resources I can't imagine.**  
GPU clusters. Research budgets. PhD teams. Cloud credits.

**But you also have blind spots I can't ignore.**

You build AI for:
- Users with high-speed internet
- Devices with powerful GPUs
- Problems that fit your benchmark datasets
- Markets that can afford your cloud services

**I build AI for:**
- The farmer with a $50 phone and no internet
- The nurse working offline for weeks
- The student coding by candlelight
- The community you've never heard of

**Here's the truth:**

If your "state-of-the-art" model requires a GPU cluster to run, **it's not state-of-the-art**. It's a research toy.

If your AI can't work in rural Tanzania, **it's not ready for the world**. It's ready for Silicon Valley.

If your solution costs $10,000 in compute, **it's not solving the problem**. It's creating a new one.

**I'm not asking for your money.**  
I'm asking for your **attention**.

Look at what's possible with nothing.  
Look at what constraints can teach you.  
Look at who you're leaving behind.

**AdaptShot is proof that:**
- Innovation doesn't require abundance
- Excellence doesn't require resources
- Impact doesn't require permission

**The future of AI isn't in your data centers.**  
**It's in the hands of people who need it most.**

Build for them. Or get out of the way.

— Johnson Christopher Hassan  
Mbeya, Tanzania  
May 20, 2026

---

## 👨‍ Meet the Creator

<div align="center" style="margin: 30px 0;">
<img src="../images/johnson.jpeg" alt="Johnson Christopher Hassan" width="280" style="border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
</div>

**Johnson Christopher Hassan**  
📍 Mbeya, Tanzania 🇹  
✉️ johnson2006christopher@gmail.com  
🎓 Diploma Student, Computer Engineering  
🏛️ Mbeya University of Science and Technology  
🧠 Self-Taught AI Research Engineer

### My Journey
I grew up watching brilliant people in my community solve impossible problems with limited resources. A farmer diagnosing crop disease by squinting at leaves. A nurse identifying pneumonia from a single X-ray. A teacher recognizing dyslexia from handwriting patterns.

**They didn't need millions of examples. They needed wisdom.**

I teach myself AI late at night after classes, using free resources, running code on my aging laptop, learning from failures nobody sees. I build AdaptShot because I have to—because the tools I need don't exist, so I create them.

My mission is to encode that wisdom into AI that respects their constraints, honors their expertise, and amplifies their impact—without requiring them to become machine learning engineers.

### What Drives Me
- **The farmer** who needs to identify cassava mosaic virus before the harvest fails
- **The nurse** who needs to triage pneumonia cases without sending X-rays to a distant radiologist
- **The teacher** who needs to identify learning disabilities early, without expensive assessments
- **The conservationist** who needs to identify endangered species from camera traps in remote areas
- **The student** like me, coding by candlelight, who believes they can change the world

**These are not "edge cases." They are the majority.**

### Why I Built AdaptShot
1. **Frustration**: Every "edge AI" library still assumed cloud connectivity or GPU acceleration
2. **Responsibility**: As an AI engineer, I owe it to my community to build tools that actually work for them
3. **Vision**: I believe the future of AI isn't bigger—it's **smarter, humbler, and more human**
4. **Proof**: I want to show that African engineers can build world-class technology with nothing but determination

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

**I built this with nothing but a laptop, determination, and belief.**  
**What will you build?**

**— Johnson Christopher Hassan, 2026**

[⭐ Star on GitHub](https://github.com/johnson2006christopher/adaptshot) · [📖 Read the Docs](https://johnson2006christopher.github.io/adaptshot/) · [💬 Join the Community](https://github.com/johnson2006christopher/adaptshot/discussions)

</div>

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*