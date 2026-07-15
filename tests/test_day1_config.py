"""Day 1 Project Challenge: Test inference_timeout_ms field."""


def test_inference_timeout_ms_default():
    """Verify default value."""
    from adaptshot import AdaptShotConfig
    config = AdaptShotConfig()
    assert config.inference_timeout_ms == 5000.0


def test_inference_timeout_ms_custom_valid():
    """Verify custom valid value is accepted."""
    from adaptshot import AdaptShotConfig
    config = AdaptShotConfig(inference_timeout_ms=10000.0)
    assert config.inference_timeout_ms == 10000.0


def test_inference_timeout_ms_zero_rejected():
    """Verify zero is rejected."""
    import pytest
    from adaptshot import AdaptShotConfig
    with pytest.raises(ValueError, match="must be positive"):
        AdaptShotConfig(inference_timeout_ms=0)


def test_inference_timeout_ms_negative_rejected():
    """Verify negative value is rejected."""
    import pytest
    from adaptshot import AdaptShotConfig
    with pytest.raises(ValueError, match="must be positive"):
        AdaptShotConfig(inference_timeout_ms=-100.0)


def test_inference_timeout_ms_too_large_rejected():
    """Verify value > 60000 is rejected."""
    import pytest
    from adaptshot import AdaptShotConfig
    with pytest.raises(ValueError, match="must be <= 60000"):
        AdaptShotConfig(inference_timeout_ms=120000.0)
