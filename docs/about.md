# About AdaptShot

<div align="center" style="margin: 40px 0;">
<img src="../images/johnson.jpeg" alt="Johnson Christopher Hassan" width="200" height="200" style="border-radius: 50%; border: 4px solid #4ADE80; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</div>

## Built in Tanzania for the World

**AdaptShot is proof that world-class AI can be built anywhere, with any resources, when guided by a clear mission: make AI work for the people who need it most.**

---

## 🌍 The Origin Story

### The Problem

In 2025, I watched an agricultural extension worker in Mbeya, Tanzania try to diagnose crop disease from phone photos. She had years of training and a smartphone—but every AI tool available to her required cloud connectivity, GPU acceleration, or thousands of labeled images. None of those exist in rural Tanzania.

**Every "state-of-the-art" solution failed her.**

That moment crystallized a truth: modern AI has been optimized for Silicon Valley, not for the world. 80% of the global population lives in regions with unreliable internet, limited electricity, or both—yet 95% of AI research assumes the opposite.

### The Response

AdaptShot was built to flip this paradigm. It is:

- **CPU-first**: Runs on laptops and single-board computers, no GPU required
- **Offline-capable**: Fully functional without internet after initial installation
- **Few-shot**: Learns from as few as 5 examples per class
- **Memory-efficient**: Operates within 250MB RAM indefinitely
- **Human-aligned**: Every prediction is calibrated and every correction improves the model

---

## 🎯 Mission & Vision

### Our Mission

> Democratize trustworthy, CPU-first, human-in-the-loop few-shot vision AI for resource-constrained environments worldwide.

### Who We Build For

| User | Use Case | Why AdaptShot |
|------|----------|---------------|
| 🧑‍🌾 **Small-scale farmers** | Crop disease identification from 10 leaf photos | Works offline on low-cost Android phones |
| 🏥 **Rural healthcare workers** | Triage pneumonia from chest X-rays without a radiologist | Calibrated confidence prevents dangerous overconfidence |
| 🦁 **Wildlife conservationists** | Classify camera trap images in remote areas | Runs on battery-powered Raspberry Pi |
| 🏭 **Quality control inspectors** | Detect manufacturing defects from few reference samples | Learns new defect types from human corrections |
| 🎓 **Students & educators** | Learn AI concepts with real, runnable code | No GPU, no cloud, no expensive hardware |

### Core Values

1. **Truth Over Hype** — We document what works, what doesn't, and why. No exaggerated claims.
2. **Constraint-First Engineering** — We design for the hardest constraints first. If it runs on a $50 phone in rural Tanzania, it runs anywhere.
3. **Human Dignity** — AI should augment human expertise, not replace it. Uncertain predictions are flagged for review, never silently wrong.
4. **Environmental Responsibility** — CPU-first inference uses 10–100× less energy than GPU alternatives. Carbon-aware configuration lets users optimize further.
5. **Global South First** — We build for the user in Mbeya before the user in Menlo Park. Innovation should serve the many, not the few.

---

## 📜 Technical Journey

### v0.1.0 — The Foundation (May 2024)
Frozen backbone extraction (ResNet-18, MobileNetV3-Small) with cosine similarity search. 68% accuracy on CIFAR-10 with 10 images per class at 90ms latency on a consumer CPU. Proved that useful few-shot inference doesn't require GPUs.

### v0.1.1 — Trust & Continual Learning (June 2025)
Introduced online temperature scaling with sliding-window ECE tracking, Adaptive Confidence Thresholding (ACT), CA-EWC head fine-tuning, and UP-UGF buffer pruning. Added energy profiling, embedding caching, and OOD detection.

### v0.2.0 — Production Hardening (Current)
Comprehensive algorithmic and production hardening across all subsystems:

- **Contrastive Learning**: Gradient-trained InfoNCE projection head with 2-layer MLP and full backpropagation
- **Conformal Prediction**: True leave-one-out prototype recomputation for valid finite-sample coverage
- **Uncertainty**: Shrinkage-regularized Mahalanobis covariance with adaptive alpha = d/(d+n_k)
- **ACT**: Symmetric threshold updates with mean-reversion to prevent monotonic drift
- **UP-UGF**: Random projection LSH for O(N log N) approximate redundancy scoring
- **Explainability**: Historical penalty tracking replacing magic-number fallbacks
- **Calibration**: Bootstrap temperature estimation for cold-start autonomous operation
- **Production**: ONNX export, memory profiling, miniImageNet benchmarks, baseline comparisons

---

## 🏆 What Sets AdaptShot Apart

| Dimension | Conventional AI | AdaptShot |
|-----------|----------------|-----------|
| **Hardware** | GPU cluster required | CPU-only, <250MB RAM |
| **Data per class** | Thousands to millions | 5–50 images |
| **Connectivity** | Cloud-dependent | Fully offline |
| **Confidence** | Overconfident softmax | Calibrated ECE with conformal guarantees |
| **Learning** | Retrain from scratch | Continual via human corrections |
| **Memory** | Unbounded growth | Bounded via UP-UGF with LSH acceleration |
| **Energy** | Hidden carbon cost | Transparent per-inference Joule tracking |
| **Trust** | Black-box output | Conformal sets + uncertainty report + explanation |
| **Cost** | $10,000+ in compute | $0 (runs on existing hardware) |
| **Quality Gates** | Often absent | ruff=0, mypy strict=32 files clean, pytest=92 passed |

---

## 🔮 Roadmap

### v0.2.0 (Current)
- [x] ONNX export for torch-free inference
- [x] Memory profiling with tracemalloc + psutil
- [x] miniImageNet benchmarks with baseline comparisons
- [x] Full production hardening of all algorithmic subsystems
- [ ] Swahili documentation translation
- [ ] Federated buffer sharing for multi-device deployments

### v1.0.0 (Target: 2027)
- Field pilot results from 3+ NGOs in Tanzania, Kenya, and Uganda
- Peer-reviewed ablation studies
- Carbon-neutral CI/CD pipeline
- Community governance board

### v2.0+ (Vision)
- Neuromorphic backend support
- Event-based vision for DVS cameras
- Multilingual, low-literacy UI extensions
- Integration with national healthcare/agriculture systems

---

## 👨‍💻 About the Creator

**Johnson Christopher Hassan** is a self-taught AI research engineer and diploma student in Computer Engineering at Mbeya University of Science and Technology, Tanzania. He built AdaptShot with a standard laptop, unreliable electricity, and determination—proving that world-class engineering doesn't require Silicon Valley resources.

- 📍 Mbeya, Tanzania 🇹🇿
- ✉️ [johnson2006christopher@gmail.com](mailto:johnson2006christopher@gmail.com)
- 🐙 [GitHub](https://github.com/johnson2006christopher)
- 💼 [LinkedIn](https://www.linkedin.com/in/johnson-christopher-hassan)

---

## 🤝 Get Involved

- **Use it**: Deploy AdaptShot in your community. Share your results.
- **Contribute**: Submit a PR, write a tutorial, or translate documentation.
- **Research it**: Publish ablation studies, comparisons, or extensions.
- **Teach it**: Use AdaptShot in your classroom or workshop.

---

<div align="center" style="margin-top: 60px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white;">

### "The best AI doesn't guess confidently.
### It learns humbly, admits uncertainty,
### and improves through every human correction."

[⭐ Star on GitHub](https://github.com/johnson2006christopher/adaptshot) · [📖 Read the Docs](https://johnson2006christopher.github.io/adaptshot/) · [💬 GitHub Discussions](https://github.com/johnson2006christopher/adaptshot/discussions) · [📱 WhatsApp Community](https://chat.whatsapp.com/J6AbrvbjmBc5XXX2fnN6RK)

</div>
