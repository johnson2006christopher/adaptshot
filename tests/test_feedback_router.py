"""Unit tests for training/feedback_router.py.

Validates buffer management, fine-tune triggering, calibrator updates,
and edge-case handling for the human-in-the-loop routing logic.
"""

from unittest.mock import MagicMock

from src.adaptshot.training.feedback_router import Correction, FeedbackRouter


def test_initial_state():
    """Verify router initializes with empty buffer and correct defaults."""
    router = FeedbackRouter(buffer_capacity=50, fine_tune_trigger_threshold=5)
    assert len(router.buffer) == 0
    assert len(router.pending_corrections) == 0
    assert router.total_corrections == 0


def test_route_feedback_basic():
    """Routing a single correction should append to buffer and pending queue."""
    router = FeedbackRouter(buffer_capacity=10)
    calibrator = MagicMock()
    router.calibrator = calibrator

    correction = Correction(
        image_path="img1.jpg", predicted_label=0, corrected_label=1, raw_confidence=0.8
    )
    result = router.route_feedback(correction)

    assert result["buffer_size"] == 1
    assert result["pending_corrections"] == 1
    assert result["calibration_updated"] is True
    assert router.total_corrections == 1
    calibrator.update.assert_called_once()


def test_route_feedback_triggers_finetune():
    """Fine-tuning should trigger when pending count hits the threshold."""
    threshold = 3
    router = FeedbackRouter(fine_tune_trigger_threshold=threshold)
    router.finetune_fn = MagicMock()

    # Route threshold - 1 corrections (no trigger yet)
    for i in range(threshold - 1):
        c = Correction(image_path=f"img_{i}.jpg", predicted_label=0, corrected_label=1, raw_confidence=0.9)
        res = router.route_feedback(c)
        assert res["fine_tuned"] is False

    # Route the Nth correction (should trigger)
    c_last = Correction(image_path="final.jpg", predicted_label=0, corrected_label=1, raw_confidence=0.9)
    res_last = router.route_feedback(c_last)

    assert res_last["fine_tuned"] is True
    assert len(router.pending_corrections) == 0  # Queue cleared
    router.finetune_fn.assert_called_once()


def test_buffer_eviction():
    """Buffer should evict oldest entries when capacity is exceeded."""
    capacity = 3
    router = FeedbackRouter(buffer_capacity=capacity)

    for i in range(5):
        c = Correction(image_path=f"img_{i}.jpg", predicted_label=0, corrected_label=1, raw_confidence=0.8)
        router.route_feedback(c)

    assert len(router.buffer) == capacity
    # Check FIFO eviction: oldest images (img_0.jpg, img_1.jpg) should be gone
    paths_in_buffer = [c.image_path for c in router.buffer]
    assert "img_0.jpg" not in paths_in_buffer
    assert "img_2.jpg" in paths_in_buffer


def test_finetune_fn_error_handling():
    """If fine_tune_fn raises an error, routing should not crash; it logs and continues."""
    router = FeedbackRouter(fine_tune_trigger_threshold=1)
    router.finetune_fn = MagicMock(side_effect=RuntimeError("Simulated failure"))

    c = Correction(image_path="fail.jpg", predicted_label=0, corrected_label=1, raw_confidence=0.5)
    result = router.route_feedback(c)

    # Should report failure, not crash
    assert result["fine_tuned"] is False
    assert len(router.pending_corrections) > 0  # Queue not cleared on failure


def test_clear_buffer():
    """Clearing the router should reset all state counters and lists."""
    router = FeedbackRouter(buffer_capacity=10)
    router.route_feedback(
        Correction(image_path="temp.jpg", predicted_label=0, corrected_label=1, raw_confidence=0.7)
    )
    router.clear_buffer()
    
    assert len(router.buffer) == 0
    assert len(router.pending_corrections) == 0
    assert router.total_corrections == 0