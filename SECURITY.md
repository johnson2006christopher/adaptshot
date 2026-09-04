# Security Policy

## Supported Versions

AdaptShot is pre-1.0. Only the latest released minor version receives security fixes.

| Version | Supported |
| :------ | :-------- |
| 0.3.x   | ✅ |
| 0.2.x   | ❌ |
| 0.1.x   | ❌ |

## Reporting a Vulnerability

**Do not open a public issue for security problems.**

Report privately via [GitHub Security Advisories](https://github.com/johnson2006christopher/adaptshot/security/advisories/new),
or by email to **johnson2006christopher@gmail.com**.

Please include:

- The affected version and Python version
- A minimal reproduction (code or steps)
- What an attacker could achieve

You will receive an acknowledgement within **7 days**. AdaptShot is maintained by a
single developer, so please allow reasonable time for a fix before public disclosure.
A 90-day coordinated disclosure window is requested.

## Scope

Because AdaptShot loads models and images from disk and can persist learner state,
the following are treated as security issues:

- **Deserialization**: unsafe loading of persisted learner state or checkpoints that
  permits arbitrary code execution.
- **Image parsing**: crashes or memory exhaustion triggered by a malicious image file
  reaching the extractor.
- **Path traversal**: file paths supplied to support-set loading or state persistence
  escaping their intended directory.
- **ONNX backbones**: loading a tampered `.onnx` file from `src/adaptshot/data/`
  without SHA-256 verification.

The following are **not** considered vulnerabilities:

- Model accuracy, calibration error, or misclassification on adversarial *inputs*
  (this is a research limitation, documented openly, not a security flaw).
- Resource use that exceeds the 250 MB target under configurations documented as
  requiring more (e.g. very large support sets).
