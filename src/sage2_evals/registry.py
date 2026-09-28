"""Benchmark interface and registry.

Every benchmark is one class registered under a stable id (e.g.
``swebench-verified``). The id is what granite.build step templates pass to
``sage2-evals run``, and what the suite files in ``sage2_evals/suites`` list.
"""

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

# Modules that define benchmarks. Imported lazily so that `sage2-evals list`
# works without any benchmark extra installed.
_BENCHMARK_MODULES = [
    "sage2_evals.benchmarks.swebench",
]

_REGISTRY: dict[str, type[Benchmark]] = {}


@dataclass
class RunConfig:
    """Options shared by every benchmark, set from the CLI / step config."""

    model: str
    """Model path or HF id served by vLLM; also the default served name."""
    output_dir: Path
    base_url: str = ""
    """OpenAI-compatible endpoint. Empty means sage2-evals starts vLLM itself."""
    served_model_name: str = ""
    limit: int | None = None
    """Evaluate only the first N examples (smoke mode). None = full dataset."""
    repeats: int | None = None
    """Independent repetitions for pass@1[avg-of-k]. None = benchmark default."""
    workers: int = 8
    seed: int = 0
    dataset: str = ""
    """Override the dataset repo id or local path (default: benchmark's own)."""
    dataset_revision: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    """Benchmark-specific key=value options (``--option k=v``)."""


class Benchmark(abc.ABC):
    """One Sage2 benchmark: fetch data, run the model, score, report."""

    id: ClassVar[str]
    metric: ClassVar[str]
    """Human-readable headline metric, as named in the suite sheet."""
    default_repeats: ClassVar[int] = 1
    extra: ClassVar[str] = ""
    """The pyproject extra this benchmark needs (for error messages)."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.repeats = config.repeats or self.default_repeats

    @abc.abstractmethod
    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        """Run the benchmark against an OpenAI-compatible endpoint.

        Returns a dict with at least ``value`` (headline metric) and ``n``
        (examples scored). Anything else is copied into results.json.
        """

    def needs_server(self) -> bool:
        """Whether the benchmark talks to the served model at all."""
        return True


def register(cls: type[Benchmark]) -> type[Benchmark]:
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate benchmark id {cls.id!r}")
    _REGISTRY[cls.id] = cls
    return cls


def _load_all() -> None:
    for module in _BENCHMARK_MODULES:
        importlib.import_module(module)


def get(benchmark_id: str) -> type[Benchmark]:
    _load_all()
    try:
        return _REGISTRY[benchmark_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise SystemExit(f"unknown benchmark {benchmark_id!r}; implemented: {known}") from None


def all_benchmarks() -> dict[str, type[Benchmark]]:
    _load_all()
    return dict(sorted(_REGISTRY.items()))
