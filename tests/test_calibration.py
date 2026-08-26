"""Unit tests for core/calibration.py.

Validates online temperature fitting, ECE computation, sliding window behavior,
and conformal prediction fallback logic.
"""

import numpy as np

from adaptshot.core.calibration import CalibrationEngine


def test_initial_state():
    """Verify default initialization values."""
    engine = CalibrationEngine(n_bins=10, window_size=50, temperature_init=1.0)
    assert engine.n_bins == 10
    assert engine.window_size == 50
    assert float(engine.temperature) == 1.0
    assert engine.current_ece == 0.0
    assert engine.method == "temperature"


def test_ece_computation_perfect_accuracy_not_perfect_calibration():
    """If confidence is 0.8 but accuracy is 1.0, ECE should reflect a 0.2 gap."""
    engine = CalibrationEngine(n_bins=5)
    # 10 predictions, all 0.8 confident, all correct
    confs = np.full(10, 0.8)
    correct = np.ones(10, dtype=int)
    ece = engine.compute_ece(confs, correct)
    assert np.isclose(ece, 0.2, atol=1e-7)


def test_ece_computation_overconfident():
    """If model is 100% confident but only 50% accurate, ECE should be ~0.5."""
    engine = CalibrationEngine(n_bins=5)
    confs = np.full(10, 1.0)
    correct = np.concatenate([np.ones(5), np.zeros(5)])  # 5 correct, 5 wrong
    ece = engine.compute_ece(confs, correct)
    assert np.isclose(ece, 0.5, atol=1e-6)


def test_sliding_window_updates():
    """Verify that the sliding window enforces fixed capacity."""
    engine = CalibrationEngine(window_size=5)
    for i in range(10):
        engine.update(raw_confidence=0.9, predicted_label=1, true_label=1)
    
    # After 10 updates, window should only hold the last 5
    assert len(engine._window_confidences) == 5
    assert len(engine._window_correct) == 5


def test_temperature_refitting():
    """Ensure temperature updates when window is sufficiently populated."""
    engine = CalibrationEngine(window_size=10, temperature_init=1.0)
    
    # Feed 8 correct high-confidence predictions
    for _ in range(8):
        engine.update(raw_confidence=0.95, predicted_label=1, true_label=1)
    
    initial_temp = float(engine.temperature)
    
    # Feed 4 incorrect high-confidence predictions (should push T > 1.0 to soften confidence)
    for _ in range(4):
        engine.update(raw_confidence=0.95, predicted_label=1, true_label=0)
    
    new_temp = float(engine.temperature)
    assert new_temp != initial_temp, "Temperature should have refitted after window update"


def test_calibration_scaling():
    """Verify temperature scaling transforms raw confidence correctly."""
    engine = CalibrationEngine(temperature_init=2.0)  # T=2 softens confidence
    # Raw cosine sim = 0.8 → norm = 0.9 → logit positive → scaled should be < 0.9
    calibrated = engine.calibrate(raw_confidence=0.8)
    assert 0.0 <= calibrated <= 1.0
    # With T=2, confidence should be pulled toward 0.5
    assert calibrated < 0.9


def test_conformal_stub():
    """Verify conformal method applies conservative lower bound."""
    engine = CalibrationEngine(method="conformal")
    calibrated = engine.calibrate(raw_confidence=0.7)
    assert np.isclose(calibrated, max(0.0, 0.7 - 0.1), atol=1e-7)


def test_empty_input_handling():
    """Ensure ECE computation doesn't crash on empty arrays."""
    engine = CalibrationEngine()
    ece = engine.compute_ece(np.array([]), np.array([]))
    assert ece == 0.0
