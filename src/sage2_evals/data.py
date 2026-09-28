"""Dataset resolution.

Every benchmark reads its data from a private mirror on the HF hub, one dataset
repo per benchmark, named ``<SAGE2_HF_ORG>/sage2-<benchmark-id>``. Mirrors are
created with ``scripts/mirror_dataset.py`` and pinned by revision, so a
published score always names the exact bytes it was computed on.

``--dataset`` overrides the mirror with any hub id or local path, which is how
a benchmark is brought up before its mirror exists.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

HF_ORG_ENV = "SAGE2_HF_ORG"


def mirror_repo(benchmark_id: str) -> str:
    org = os.environ.get(HF_ORG_ENV)
    if not org:
        raise SystemExit(
            f"no dataset for {benchmark_id!r}: set {HF_ORG_ENV} to the HF org holding the "
            f"sage2 mirrors, or pass --dataset <hub-id-or-path>"
        )
    return f"{org}/sage2-{benchmark_id}"


def load_split(
    benchmark_id: str,
    *,
    dataset: str = "",
    revision: str | None = None,
    split: str = "test",
    name: str | None = None,
) -> tuple[list[dict], str]:
    """Load one split as a list of rows. Returns (rows, resolved source)."""
    from datasets import load_dataset

    source = dataset or mirror_repo(benchmark_id)
    log.info("loading %s split=%s revision=%s", source, split, revision or "default")
    if Path(source).exists():
        ds = load_dataset(source, name=name, split=split)
    else:
        ds = load_dataset(source, name=name, split=split, revision=revision)
    return list(ds), source


def take(rows: list[dict], limit: int | None, *, key: str) -> list[dict]:
    """First ``limit`` rows in a stable order, so smoke runs are comparable."""
    rows = sorted(rows, key=lambda r: r[key])
    return rows if limit is None else rows[:limit]
