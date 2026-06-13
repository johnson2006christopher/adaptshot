# Deployment Guide — From Development to the Field

**Step-by-step instructions for deploying AdaptShot and MziziGuard in real-world environments — laptops, field offices, mobile workstations, and embedded devices.**

---

## Deployment Scenarios

| Scenario | Hardware | Connectivity | Guide Section |
|----------|----------|-------------|---------------|
| Field office laptop | Standard laptop | Offline | [Section 1](#scenario-1-field-office-laptop) |
| Extension officer tablet | Low-power device | Occasional WiFi | [Section 2](#scenario-2-extension-officer-tablet) |
| Multi-user server | Desktop/server | LAN | [Section 3](#scenario-3-multi-user-server) |
| Raspberry Pi / edge device | ARM SBC | Offline | [Section 4](#scenario-4-raspberry-pi-edge-device) |
| Demo/presentation | Any laptop | Internet (for share) | [Section 5](#scenario-5-demo-presentation) |

---

## Scenario 1: Field Office Laptop

**Target**: Agricultural extension office with a standard laptop. No internet. Multiple officers sharing one machine.

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| CPU | Intel Core i3 (2015+) | Intel Core i5 (2018+) |
| RAM | 4GB (250MB for AdaptShot) | 8GB |
| Storage | 500MB free | 2GB+ (for photos) |
| OS | Ubuntu 20.04+ / Windows 10+ | Ubuntu 22.04 LTS |

### Step-by-Step Setup

#### 1. Install Python and AdaptShot

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 -m venv adaptshot_env
source adaptshot_env/bin/activate
pip install "adaptshot[ui]"
```

#### 2. Prepare Training Images

Create a USB stick or local folder with crop photos:

```
training_images/
├── healthy_maize/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── img_003.jpg
├── northern_leaf_blight/
│   ├── img_001.jpg
│   └── img_002.jpg
└── gray_leaf_spot/
    └── img_001.jpg
```

Copy to the laptop's local storage.

#### 3. Configure MziziGuard

Edit `examples/mziziguard/config.yaml` to match your crops and local paths.

#### 4. Launch the Application

```bash
# Start the web server
python -m examples.mziziguard.app --port 7860

# Open browser: http://localhost:7860
```

#### 5. Train and Test

1. Go to **Setup** tab → **Load from Folder**
2. Enter the path to your training images
3. Click **Load from Folder**
4. Go to **Diagnose** tab → upload a test photo
5. Verify the diagnosis is reasonable

#### 6. Save the Trained Model

```python
from examples.mziziguard import MziziGuard
guard = MziziGuard()
guard.load_images_from_dir("training_images/")
guard.save_model("models/field_model.json")
```

#### 7. Create a Desktop Launcher (Ubuntu)

```bash
cat > ~/Desktop/mziziguard.desktop << 'EOF'
[Desktop Entry]
Name=MziziGuard
Comment=Crop Disease Detection
Exec=bash -c "cd /path/to/adaptshot && source adaptshot_env/bin/activate && python -m examples.mziziguard.app"
Type=Application
Terminal=true
EOF
chmod +x ~/Desktop/mziziguard.desktop
```

---

## Scenario 2: Extension Officer Tablet

**Target**: Low-power device (tablet, netbook) for individual extension officers in the field. Battery-powered. Occasional WiFi for sync.

### Hardware

| Requirement | Minimum |
|-------------|---------|
| CPU | Intel Atom / ARM Cortex-A72 |
| RAM | 2GB (500MB free for AdaptShot) |
| Storage | 1GB free |
| Battery | 4+ hours |

### Optimization for Low-Power

```python
from adaptshot import AdaptShotConfig, FewShotLearner

config = AdaptShotConfig(
    backbone="mobilenet_v3_small",    # Smallest backbone (~10MB)
    device="cpu",
    eco_mode=True,                      # Battery saving
    early_exit_threshold=0.85,          # More aggressive early exit
    use_faiss=False,                    # Skip FAISS memory overhead
    max_buffer_size=30,                 # Minimal buffer
    enable_ood_detection=True,          # Keep OOD for safety
)
```

### Startup Script (auto-resume)

```python
#!/usr/bin/env python3
"""MziziGuard field launcher — auto-resumes from last session."""
from pathlib import Path
from examples.mziziguard import MziziGuard

MODEL_PATH = "field_model.json"

guard = MziziGuard()

if Path(MODEL_PATH).exists():
    count = guard.load_model(MODEL_PATH)
    print(f"Resumed from {MODEL_PATH} ({count} images)")
else:
    print("No saved model found. Training from samples...")
    guard.initialize_with_samples(n_support=5)

# Now use guard.diagnose() or launch the app
```

### Periodic Sync (When WiFi Available)

```python
# Officer A: save their corrections
guard.save_model("officer_a_corrections.json")

# Officer B: load Officer A's corrections
guard.load_model("officer_a_corrections.json")
# Now both officers benefit from each other's corrections
```

---

## Scenario 3: Multi-User Server

**Target**: Shared server in an extension office LAN. Multiple officers access via web browser.

### Architecture

```
                  ┌─────────────────┐
                  │   Web Browser   │  Officer A (port 7861)
                  └────────┬────────┘
                           │
┌──────────────┐    ┌──────┴──────┐    ┌──────────────┐
│ Web Browser  │────│   Server    │────│ Web Browser  │
│ Officer B    │    │  LAN IP     │    │ Officer C    │
│ (port 7862)  │    └─────────────┘    │ (port 7863)  │
└──────────────┘                       └──────────────┘
```

### Launch Multiple Instances

```bash
# Officer A — port 7861
python -m examples.mziziguard.app --port 7861 &

# Officer B — port 7862
python -m examples.mziziguard.app --port 7862 &

# Officer C — port 7863
python -m examples.mziziguard.app --port 7863 &
```

Each instance is independent. Share models by saving to a shared directory:

```python
guard.save_model("/shared/models/officer_a_latest.json")
```

### Shared Model Aggregation

```python
"""Merge corrections from multiple officers into a master model."""
from examples.mziziguard import MziziGuard

master = MziziGuard()
master.initialize_with_samples(n_support=5)  # Base training

# Absorb each officer's corrections
for officer in ["officer_a", "officer_b", "officer_c"]:
    temp = MziziGuard()
    temp.load_model(f"/shared/{officer}_latest.json")
    # Corrections are embedded in the buffer — load them all
    # (Simplified: in practice, merge embeddings/labels)

master.save_model("/shared/master_latest.json")
```

### Systemd Service (Ubuntu)

```ini
# /etc/systemd/system/mziziguard.service
[Unit]
Description=MziziGuard Crop Disease Detection
After=network.target

[Service]
Type=simple
User=extension
WorkingDirectory=/opt/adaptshot
ExecStart=/opt/adaptshot_env/bin/python -m examples.mziziguard.app --port 7860
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mziziguard
sudo systemctl start mziziguard
```

---

## Scenario 4: Raspberry Pi / Edge Device

**Target**: ARM-based single-board computer. Ultra low-power, fully offline. Could be deployed in a weatherproof enclosure in a village.

### Hardware

| Component | Recommendation |
|-----------|---------------|
| Board | Raspberry Pi 4 (4GB) or Pi 5 |
| Storage | 32GB+ microSD (or USB SSD) |
| Display | 7" touchscreen (or headless + phone browser) |
| Power | Solar + battery or mains |

### Setup for ARM

```bash
# Raspberry Pi OS (64-bit)
sudo apt update
sudo apt install python3-pip python3-venv libatlas-base-dev

# Create venv and install
python3 -m venv adaptshot_env
source adaptshot_env/bin/activate

# PyTorch for ARM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# AdaptShot
pip install "adaptshot[ui]"
```

### Optimized Config for Pi

```python
config = AdaptShotConfig(
    backbone="mobilenet_v3_small",    # Must: ResNet too heavy for Pi
    device="cpu",
    eco_mode=True,
    early_exit_threshold=0.80,         # Aggressive early exit
    max_buffer_size=20,                # Minimal buffer
    use_faiss=False,
    enable_ood_detection=True,
    verbose=False,                     # Reduce I/O
)
```

### Auto-Start on Boot

```bash
# ~/.config/autostart/mziziguard.desktop
[Desktop Entry]
Name=MziziGuard
Exec=bash -c "cd /home/pi/adaptshot && source adaptshot_env/bin/activate && python -m examples.mziziguard.app"
Type=Application
```

### Expected Pi 4 Performance

| Metric | Value |
|--------|-------|
| Backbone load | ~15s (first time, cached after) |
| Inference latency | ~500–800ms per image |
| Memory usage | ~80–120MB |
| Power draw | ~5–8W (Pi 4 + display) |

---

## Scenario 5: Demo / Presentation

**Target**: Any laptop. Public shareable link for remote audience. Time-limited session.

### Quick Launch

```bash
pip install "adaptshot[ui]"
python -m examples.mziziguard.app --share
```

The `--share` flag creates a public Gradio link (valid for 72 hours). Share the URL with your audience.

### Presentation Tips

1. **Pre-train** the model before the demo starts
2. **Prepare sample images** (healthy vs. diseased leaves) on your desktop
3. **Walk through each tab** in order: Setup → Diagnose → Teach → Health → Batch
4. **The Teach tab is your climax** — show a wrong prediction, then correct it live
5. **The Health tab shows progress** — show how ECE drops after corrections

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install "adaptshot[ui]"

COPY examples/mziziguard/ /app/examples/mziziguard/

# Pre-download backbone weights
RUN python -c "from adaptshot import FewShotLearner; \
    from adaptshot.config.settings import AdaptShotConfig; \
    FewShotLearner(config=AdaptShotConfig(device='cpu'))"

EXPOSE 7860

CMD ["python", "-m", "examples.mziziguard.app", "--port", "7860"]
```

### Build and Run

```bash
docker build -t mziziguard .
docker run -p 7860:7860 mziziguard
```

---

## Update & Maintenance

### Upgrading AdaptShot

```bash
pip install --upgrade adaptshot

# Verify
python -c "from adaptshot import FewShotLearner; print('OK')"
```

### Upgrading MziziGuard

```bash
# Pull latest from GitHub
cd /path/to/adaptshot
git pull origin main
```

### Backup Checklist

- [ ] `models/*.json` — Model checkpoints
- [ ] `models/*.embeddings.npy` — Embedding arrays
- [ ] `models/*.head.pt` — Fine-tuned head weights
- [ ] `examples/mziziguard/config.yaml` — Custom crop config
- [ ] Training images (if custom)
- [ ] Correction/feedback logs (if applicable)

### Recovery from Backup

```python
from examples.mziziguard import MziziGuard

guard = MziziGuard()
guard.load_model("backup/model.json")

# Verify
result = guard.diagnose("test_photo.jpg")
print(f"Model loaded: {result.swahili} ({result.confidence:.1%})")
```

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
