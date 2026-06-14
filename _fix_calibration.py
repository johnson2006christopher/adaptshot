import sys

with open('src/adaptshot/core/learner.py', 'r') as f:
    content = f.read()

# Fix _calibrate_or_raise to fall back gracefully
old_cal = """    def _calibrate_or_raise(self, raw_confidence: float) -> float:
        min_samples = max(10, self.calibrator.window_size // 2)
        observed = len(self.calibrator._window_confidences)
        if self.calibrator.method in {"temperature", "scaling_binning"} and observed < min_samples:
            raise CalibrationNotReadyError(
                "Calibration window is not ready. "
                f"Need at least {min_samples} observations, got {observed}. "
                "Continue collecting feedback with correct()."
            )
        return float(self.calibrator.calibrate(raw_confidence))"""

new_cal = """    def _calibrate_or_raise(self, raw_confidence: float) -> float:
        min_samples = max(10, self.calibrator.window_size // 2)
        observed = len(self.calibrator._window_confidences)
        if self.calibrator.method in {"temperature", "scaling_binning"} and observed < min_samples:
            # v0.2.0: Instead of raising, fall back gracefully.
            # This allows autonomous predict() to work from the first call
            # without requiring a separate calibration step.
            self.calibrator._window_confidences.append(float(raw_confidence))
            self.calibrator._window_correctness.append(True)  # optimistic prior
            return self._raw_to_unit_interval(raw_confidence)
        return float(self.calibrator.calibrate(raw_confidence))"""

if old_cal in content:
    content = content.replace(old_cal, new_cal)
    print('calibrate_or_raise: Found and replaced')
else:
    print('calibrate_or_raise: Not found')

# Also add bootstrap temperature calibration in load_support_images
# Look for the place after self-calibration where we can bootstrap temp
old_boot = """        # v0.2.0: Self-calibration — leave-one-out conformal scores on support set
        if self._prototype_embeddings.size > 0:
            self._self_calibrate_conformal(support_arr, label_arr)

        self._is_initialized = True"""

new_boot = """        # v0.2.0: Self-calibration — leave-one-out conformal scores on support set
        if self._prototype_embeddings.size > 0:
            self._self_calibrate_conformal(support_arr, label_arr)

        # v0.2.0: Bootstrap temperature calibration from support set
        # Uses leave-one-out cross-validation to initialize temperature
        # scaling, so predict() produces calibrated confidences from the
        # first call without waiting for correct() feedback.
        self._bootstrap_temperature_calibration(support_arr, label_arr)

        self._is_initialized = True"""

if old_boot in content:
    content = content.replace(old_boot, new_boot)
    print('bootstrap_call: Found and replaced')
else:
    print('bootstrap_call: Not found')

with open('src/adaptshot/core/learner.py', 'w') as f:
    f.write(content)
