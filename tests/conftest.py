import sys, os, torch, pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="session")
def device():
    """Returns cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_batch():
    """Generates a random batch: [batch=4, channels=3, H=128, W=128]"""
    return torch.randn(4, 3, 128, 128)

@pytest.fixture
def dummy_labels():
    """Generates random labels for 5 classes."""
    return torch.randint(0, 5, (4,))
