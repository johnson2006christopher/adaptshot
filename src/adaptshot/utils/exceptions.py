"""Custom exceptions for AdaptShot runtime and validation failures."""


class AdaptShotError(Exception):
    """Base exception for all AdaptShot-specific errors."""


class InvalidImageError(AdaptShotError):
    """Raised when an image path, format, or dimensionality is invalid."""


class ConfigValidationError(AdaptShotError):
    """Raised when configuration values violate supported bounds or constraints."""


class CalibrationNotReadyError(AdaptShotError):
    """Raised when calibration requires more observations before reliable use."""


class BufferCapacityError(AdaptShotError):
    """Raised when replay buffer pruning fails to enforce configured capacity."""


class BackboneError(AdaptShotError):
    """Raised when no usable backend exists for the requested backbone.

    Since #36 the answer depends on the install: a core install runs the
    backbones whose ONNX weights are bundled and nothing else, while an install
    with torch runs all of them. This distinguishes that from a typo in the
    backbone name, which is still a ``ValueError``.
    """
