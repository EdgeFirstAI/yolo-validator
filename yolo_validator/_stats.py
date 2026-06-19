"""Per-stage timing statistics.

min/mean/p50/p95/p99/max with NO trimming. Percentiles convey the tail
directly, so the old trimmed-mean approach is dropped (design §4.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StageStats:
    count: int
    min_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def stage_stats(samples_ms) -> StageStats:
    """Compute timing statistics for one stage from a list of ms samples."""
    a = np.asarray(samples_ms, dtype=np.float64)
    if a.size == 0:
        raise ValueError("stage_stats requires at least one sample")
    return StageStats(
        count=int(a.size),
        min_ms=float(a.min()),
        mean_ms=float(a.mean()),
        p50_ms=float(np.percentile(a, 50)),
        p95_ms=float(np.percentile(a, 95)),
        p99_ms=float(np.percentile(a, 99)),
        max_ms=float(a.max()),
    )


def format_stage_table(stages: dict[str, StageStats]) -> str:
    """Render a fixed-width per-stage timing table."""
    header = f"{'Stage':<24}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'min':>9}{'max':>9}  ms"
    lines = [header]
    for name, s in stages.items():
        lines.append(
            f"{name:<24}{s.mean_ms:>9.2f}{s.p50_ms:>9.2f}{s.p95_ms:>9.2f}"
            f"{s.p99_ms:>9.2f}{s.min_ms:>9.2f}{s.max_ms:>9.2f}"
        )
    return "\n".join(lines)
