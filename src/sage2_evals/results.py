"""The results.json every step emits as its ``sage2_results`` artifact.

One file per benchmark run, one schema across benchmarks, so a recipe-level
report can join them without knowing any benchmark's internals.
"""

from __future__ import annotations

import json
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from sage2_evals import __version__
from sage2_evals.registry import Benchmark

SCHEMA_VERSION = 1
RESULTS_FILE = "results.json"


def _package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def write_results(
    benchmark: Benchmark,
    outcome: dict[str, Any],
    *,
    started: float,
    served_model_name: str,
) -> Path:
    config = benchmark.config
    record = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark.id,
        "metric": benchmark.metric,
        "value": outcome.pop("value"),
        "n": outcome.pop("n"),
        "repeats": benchmark.repeats,
        "smoke": config.limit is not None,
        "limit": config.limit,
        "model": config.model,
        "served_model_name": served_model_name,
        "dataset": outcome.pop("dataset", config.dataset),
        "dataset_revision": outcome.pop("dataset_revision", config.dataset_revision),
        "options": config.options,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1),
        "versions": {
            "sage2-evals": __version__,
            "python": platform.python_version(),
            **_package_versions(["vllm", "mini-swe-agent", "swebench", "datasets"]),
        },
        "details": outcome,
    }
    path = config.output_dir / RESULTS_FILE
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    return path
