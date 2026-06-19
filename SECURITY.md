# Security Policy

## Supported versions

yolo-validator is pre-1.0. Security fixes are applied to the latest released
`0.x` version. Please ensure you are on the most recent release before
reporting an issue.

| Version | Supported          |
| ------- | ------------------ |
| 0.x     | :white_check_mark: |

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public
GitHub issue for a suspected vulnerability.

Email **sebastien@au-zone.com** with the subject line
`yolo-validator security`, and include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal proof of concept is ideal).
- Affected version(s) and environment (OS, Python, runtime/backend).
- Any suggested remediation, if known.

You can expect an acknowledgement within **3 business days**. We will work with
you to understand and resolve the issue, and we ask that you give us a
reasonable opportunity to release a fix before any public disclosure
(coordinated disclosure).

## Scope and threat model

yolo-validator processes potentially untrusted inputs — model files
(`.onnx`, `.engine`) and image data — and depends on native libraries
(OpenCV, ONNX Runtime, pycocotools, and optionally TensorRT/PyTorch).
Reports involving malformed or adversarial model/image inputs that cause
crashes, resource exhaustion, or arbitrary code execution are in scope and
welcome.
