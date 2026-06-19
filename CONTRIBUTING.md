# Contributing to yolo-validator

Thank you for your interest in contributing. All contributions are welcome:
bug reports, documentation improvements, new backend implementations, and
test coverage expansions.

## Licensing

By submitting a contribution you agree to license your work under the
**Apache License 2.0**, the same license as this project. No Contributor
License Agreement (CLA) is required. The DCO sign-off (see below)
is sufficient to attest to the provenance of your contribution.

### Licensing boundary — important

This codebase is Apache-2.0 licensed. **Do not copy code from AGPL-licensed
projects** (including Ultralytics) into this repository. Any technique or
algorithm derived from or popularized by such projects must be independently
reimplemented from the published specification or documentation, not
transcribed from their source. If you are unsure whether a contribution
crosses this line, open an issue and ask before submitting a pull request.

---

## Development environment

**Prerequisites**: Python ≥ 3.10, Git.

```bash
# Clone and enter the repository
git clone <repository-url>
cd yolo-validator

# Create and activate the local virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
# .\venv\Scripts\activate  # Windows

# Install the package with ONNX and development dependencies
pip install -e '.[onnx,dev]'
```

Always work inside the local `venv`. Do not install packages into the
global Python environment.

---

## Running the tests

```bash
source venv/bin/activate
pytest
```

The test suite is hermetic; ONNX-model tests are skipped automatically
when no model file is present. All tests must pass before a pull request
is merged.

---

## Coding conventions

- **Match the existing style.** The codebase follows standard PEP 8
  conventions. Keep lines to a reasonable length and use consistent naming.
- **Keep the core path dependency-free.** The ONNX/NumPy inference path
  must not import from `ultralytics`, `torch`, or any other package that is
  not available without the `[torch]` extra. Ultralytics/Torch code belongs
  exclusively behind that optional dependency boundary.
- **Serial execution.** The validator is intentionally un-optimized and
  runs one frame at a time to completion so per-stage timings are additive.
  Do not introduce pipelining, threading, or other optimizations.
- **Dual-path equivalence.** The Torch and NumPy paths must produce the
  same detections within floating-point tolerance.
- **Stats.** Report min / mean / p50 / p95 / p99 / max. Do not trim
  outliers.
- **Type hints.** Add type annotations to all new public functions and
  methods.

---

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short summary>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.

### DCO sign-off

Every commit must include a Developer Certificate of Origin sign-off.
Pass `-s` to `git commit`:

```bash
git commit -s -m "feat(backends): add TFLite backend skeleton"
```

The sign-off line (`Signed-off-by: Your Name <your@email.com>`) certifies
that you have the right to submit the contribution under the project's
Apache-2.0 license.

---

## Pull request process

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes, add or update tests, and ensure `pytest` passes.
3. Commit with a meaningful message and DCO sign-off.
4. Open a pull request against `main` with a clear description of what
   the change does and why.
5. Address any review feedback. Pull requests require at least one
   approving review before merge.

---

## Reporting bugs

Open a GitHub Issue with:
- A minimal, reproducible example.
- The Python and dependency versions (`pip freeze`).
- The full error message or unexpected output.
