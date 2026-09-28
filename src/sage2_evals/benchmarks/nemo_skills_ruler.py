"""RULER through NeMo-Skills (Sage2: RULER 128k / 64k completions).

ns's RULER setup, run locally: ``nemo_skills/dataset/ruler/prepare.py`` (pinned ns)
generates the 13 synthetic tasks with NVIDIA/RULER's generators (pinned commit,
baked into the image at ``/opt/ruler`` with its four source json files verified by
sha256) for the evaluated model's tokenizer, 100 samples per task at
``max_seq_length - 50`` tokens (ns's template allowance), seed 42. Each task then
runs as its own ns generation with the task's ns GENERATION_ARGS: the text
completions endpoint (ns applies the model's chat template with its tokenizer and
prefills RULER's answer prefix), per-task ``tokens_to_generate`` and ns's RULER
match (``all``, or ``part`` for QA). The headline is ns's ``ruler_score``: the
mean accuracy over the 13 tasks.

The data depend on the tokenizer, so they are generated in the job (a few minutes
at 128k) and cached under ``<output_dir>/ns-data`` keyed by the RULER commit,
length and tokenizer; ``--limit N`` takes the first N samples of *each* task.

Options: ``tokenizer`` (default: ``--model``), ``enable_thinking`` (default: the
chat template's default, i.e. on for Granite), ``tasks`` (comma-separated subset,
for debugging; the headline then averages only those), ``ruler_dir``, the
sampling options and ``ns.<key>=<value>`` of the NeMo-Skills family, and
``answers=gold`` (serves each sample's expected outputs; needs a tokenizer but no
GPU).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import httpx

from sage2_evals import data
from sage2_evals.benchmarks.nemo_skills import (
    GENERATE_MODULE,
    NS_COMMIT,
    NS_REPO,
    NemoSkillsBenchmark,
    _GoldServer,
    _hydra,
    _jsonable,
    _passthrough,
    _read_jsonl,
    _run_ns,
    _sha256,
    _status_line,
    _statuses,
    _merge_counts,
    _write_jsonl,
    log,
)
from sage2_evals.registry import register

RULER_REPO = "https://github.com/NVIDIA/RULER"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
RULER_DIR = "/opt/ruler"
"""The pinned RULER checkout with its source data (docker/extras/nemoskills.sh)."""

RULER_JSON_SHA256 = {
    "english_words.json": "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",
    "hotpotqa.json": "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    "PaulGrahamEssays.json": "8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd",
    "squad.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
}
"""RULER's generator inputs (scripts/data/synthetic/json): LFS word list, HotpotQA
dev distractor, the essays RULER's script scrapes, SQuAD v2.0 dev."""

TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
"""ns's RULER tasks (the 13 ``ruler_score`` averages)."""

NUM_SAMPLES = 100
TEMPLATE_TOKENS = 50


class RulerBenchmark(NemoSkillsBenchmark):
    """RULER at one context length; subclasses set ``id`` and ``max_seq_length``."""

    metric = "accuracy"
    default_repeats = 1
    ns_benchmark = "ruler"
    ns_metric = "accuracy"
    ns_metrics_type = "ruler"
    dataset = f"{RULER_REPO}/tree/{RULER_COMMIT}"
    dataset_revision = RULER_COMMIT
    harness_packages = (*NemoSkillsBenchmark.harness_packages, "wonderwords", "nltk", "transformers")
    max_seq_length: ClassVar[int]

    # -- config ------------------------------------------------------------

    def tokenizer(self) -> str:
        return self.opt("tokenizer", self.config.model)

    def tasks(self) -> list[str]:
        tasks = [t for t in str(self.opt("tasks", ",".join(TASKS))).split(",") if t]
        unknown = sorted(set(tasks) - set(TASKS))
        if unknown:
            raise SystemExit(f"{self.id}: unknown RULER tasks {unknown}; known: {', '.join(TASKS)}")
        return tasks

    def setup_name(self) -> str:
        return f"sage2_{self.max_seq_length}"

    def ruler_dir(self) -> Path:
        return Path(self.opt("ruler_dir", RULER_DIR))

    # -- 1. data -----------------------------------------------------------

    def check_ruler_source(self) -> dict[str, str]:
        """The RULER checkout is the pinned commit and its json inputs match."""
        root = self.ruler_dir()
        stamp = root / "SAGE2_COMMIT"
        if not stamp.exists() or stamp.read_text().strip() != RULER_COMMIT:
            raise SystemExit(f"{self.id}: {root} is not RULER {RULER_COMMIT} (see docker/extras/nemoskills.sh)")
        shas = {}
        for name, want in RULER_JSON_SHA256.items():
            got = _sha256(root / "scripts/data/synthetic/json" / name)
            if got != want:
                raise SystemExit(f"{self.id}: RULER {name} sha256 {got} != pinned {want}")
            shas[name] = got
        return shas

    def tokenizer_fingerprint(self) -> dict[str, str]:
        tok = Path(self.tokenizer())
        if not tok.is_dir():
            return {"tokenizer": self.tokenizer()}
        files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model")
        return {"tokenizer": str(tok), **{f: _sha256(tok / f) for f in files if (tok / f).exists()}}

    def prepare_data(self) -> tuple[Path, dict[str, Any]]:
        """ns's RULER prepare for this length and tokenizer: ``<setup>/<task>/test.jsonl``."""
        tasks = self.tasks()
        if self.config.dataset and Path(self.config.dataset).is_dir():
            setup_dir = Path(self.config.dataset)
            return setup_dir, {"source": "local", "path": str(setup_dir), **self._task_files(setup_dir, tasks)}
        if self.tokenizer() in ("", "none"):
            raise SystemExit(f"{self.id}: RULER data are built for a tokenizer: pass --model or --option tokenizer=")
        spec = {
            "ns_commit": NS_COMMIT,
            "ruler_commit": RULER_COMMIT,
            "max_seq_length": self.max_seq_length,
            "template_tokens": TEMPLATE_TOKENS,
            "num_samples": NUM_SAMPLES,
            "tasks": tasks,
            **self.tokenizer_fingerprint(),
        }
        spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        ddir = self.config.output_dir / "ns-data" / "ruler"
        setup_dir = ddir / self.setup_name()
        prov_path = setup_dir / "sage2-provenance.json"
        if prov_path.exists() and json.loads(prov_path.read_text()).get("spec_hash") == spec_hash:
            prov = json.loads(prov_path.read_text())
        else:
            json_shas = self.check_ruler_source()
            self._run_prepare(ddir, tasks)
            prov = {"spec_hash": spec_hash, "spec": spec, "ruler_json_sha256": json_shas}
            prov.update(self._task_files(setup_dir, tasks))
            prov_path.write_text(json.dumps(prov, indent=2))
        return setup_dir, {
            "source": "ns prepare_data ruler",
            "ns_commit": NS_COMMIT,
            "ruler": {"repo": RULER_REPO, "commit": RULER_COMMIT},
            "setup": self.setup_name(),
            **{k: v for k, v in prov.items() if k != "spec_hash"},
        }

    def _task_files(self, setup_dir: Path, tasks: list[str]) -> dict[str, Any]:
        files = {}
        for t in tasks:
            path = setup_dir / t / "test.jsonl"
            rows = len(_read_jsonl(path))
            # RULER's generator scripts can fail without failing prepare.py.
            if rows != NUM_SAMPLES:
                raise RuntimeError(f"{self.id}: {path} has {rows} samples, expected {NUM_SAMPLES} (see ns-data/ruler/prepare.log)")
            files[t] = {"rows": rows, "sha256": _sha256(path)}
        return {"task_files": files}

    def _run_prepare(self, ddir: Path, tasks: list[str]) -> None:
        import nemo_skills.dataset.ruler as ns_ruler

        if ddir.exists():
            shutil.rmtree(ddir)
        shutil.copytree(Path(ns_ruler.__file__).parent, ddir, ignore=shutil.ignore_patterns("__pycache__"))
        # RULER's prepare writes next to its scripts: work on a copy of the checkout.
        tmp = ddir / "tmp"
        shutil.copytree(self.ruler_dir(), tmp / "RULER", ignore=shutil.ignore_patterns(".git", "__pycache__"))
        # ns's prepare pip-installs RULER's deps; they are in the nemoskills extra
        # already, and the image env must not change at run time.
        shim = ddir / "bin"
        shim.mkdir()
        (shim / "pip").write_text("#!/bin/sh\necho \"sage2: pip disabled during RULER prepare: $*\" >&2\n")
        (shim / "pip").chmod(0o755)
        argv = [
            "--setup", self.setup_name(),
            "--max_seq_length", str(self.max_seq_length),
            "--template_tokens", str(TEMPLATE_TOKENS),
            "--tmp_data_dir", str(tmp),
            "--tasks", *tasks,
            # passed through to RULER's prepare.py
            "--tokenizer_path", self.tokenizer(),
            "--num_samples", str(NUM_SAMPLES),
        ]  # fmt: skip
        spec = {"dir": str(ddir), "argv": argv, "hf_revisions": {}, "redirects": {}, "pinned_urls": {}}
        (ddir / "sage2-prepare.json").write_text(json.dumps(spec, indent=2))
        env = dict(os.environ, PATH=f"{shim}:{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = [sys.executable, "-c", "from sage2_evals.benchmarks.nemo_skills import _prepare_main; _prepare_main()"]
        log.info("%s: ns prepare_data ruler %s", self.id, shlex.join(argv))
        with (ddir / "prepare.log").open("ab") as logf:
            subprocess.run([*cmd, str(ddir / "sage2-prepare.json")], check=True, env=env, cwd=ddir, stdout=logf, stderr=subprocess.STDOUT)

    # -- 2. generation -------------------------------------------------------

    def task_generation_args(self, setup_dir: Path, task: str) -> list[str]:
        """The task's ns GENERATION_ARGS (written by prepare), plus sampling."""
        module = runpy.run_path(str(setup_dir / task / "__init__.py"))
        args = shlex.split(module.get("GENERATION_ARGS", ""))
        s = self.sampling()
        args += [
            f"++tokenizer={self.tokenizer()}",
            f"++inference.temperature={_hydra(s['temperature'])}",
            f"++inference.top_p={_hydra(s['top_p'])}",
            f"++inference.top_k={s['top_k']}",
        ]
        if s["max_tokens"] is not None:  # the task's tokens_to_generate otherwise
            args.append(f"++inference.tokens_to_generate={s['max_tokens']}")
        if "enable_thinking" in self.config.options:
            args.append(f"++chat_template_kwargs.enable_thinking={str(self.opt('enable_thinking', True)).lower()}")
        return args + _passthrough(self.config.options, "ns.")

    def check_context(self, base_url: str, served: str) -> int | None:
        """The served max_model_len must hold a full-length sample."""
        with contextlib.suppress(httpx.HTTPError, ValueError, KeyError):
            models = httpx.get(base_url.rstrip("/") + "/models", timeout=30).json()["data"]
            lens = [m.get("max_model_len") for m in models if m.get("id") == served] or [m.get("max_model_len") for m in models]
            n = next((x for x in lens if x), None)
            if n is not None and n < self.max_seq_length:
                raise SystemExit(
                    f"{self.id}: {served} is served with max_model_len={n} < {self.max_seq_length}; "
                    f"serve with --max-model-len {self.max_seq_length}"
                )
            return n
        return None

    # -- entry point ---------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        setup_dir, provenance = self.prepare_data()
        tasks = self.tasks()
        out = self.config.output_dir / "ruler"
        inputs = {}
        for t in tasks:
            rows = data.take(_read_jsonl(setup_dir / t / "test.jsonl"), self.config.limit, key="index")
            inputs[t] = out / t / "input.jsonl"
            _write_jsonl(inputs[t], rows)
            _reset_task_if_input_changed(out / t)
        n = sum(len(_read_jsonl(p)) for p in inputs.values())
        log.info("%s: %d tasks, %d samples x %d repeats", self.id, len(tasks), n, self.repeats)

        max_model_len = None
        with contextlib.ExitStack() as stack:
            if self.gold:
                all_rows = [r for p in inputs.values() for r in _read_jsonl(p)]
                base_url = stack.enter_context(RulerGoldServer(all_rows)).base_url
                served_model_name = "gold"
            else:
                max_model_len = self.check_context(base_url, served_model_name)
            for t in tasks:
                for k in range(self.repeats):
                    self._generate_task(setup_dir, t, inputs[t], base_url, served_model_name, k)

        agg = self.aggregation()
        per_task, per_task_metrics = {}, {}
        for t in tasks:
            files = [out / t / f"output-rs{k}.jsonl" for k in range(self.repeats)]
            m = self.compute_metrics(files)
            per_task_metrics[f"{self.setup_name()}.{t}"] = m["_all_"]
            per_task[t] = {
                "accuracy": m["_all_"][agg]["accuracy"] * self.value_scale,
                "n": len(_read_jsonl(inputs[t])),
                "statuses": _merge_counts(_statuses(f) for f in files),
            }
            log.info("%s %s: accuracy=%.4f", self.id, t, per_task[t]["accuracy"])

        if set(tasks) == set(TASKS):  # ns's own group score
            from nemo_skills.dataset.ruler.ruler_score import compute_score

            raw = compute_score(dict(per_task_metrics))[self.setup_name()][agg]["accuracy"]
        else:
            raw = sum(m[agg]["accuracy"] for m in per_task_metrics.values()) / len(tasks)
        return {
            "value": raw * self.value_scale,
            "n": n,
            "dataset": self.dataset_source()[0],
            "dataset_revision": self.dataset_source()[1],
            "harness": {"name": "nemo-skills", "repo": NS_REPO, "commit": NS_COMMIT},
            "ns_benchmark": f"ruler.{self.setup_name()}",
            "ns_metric": {"aggregation": agg, "key": "accuracy", "raw": raw, "scale": self.value_scale, "score": "ruler_score"},
            "max_seq_length": self.max_seq_length,
            "served_max_model_len": max_model_len,
            "tasks": tasks,
            "partial_task_set": set(tasks) != set(TASKS),
            "per_task": per_task,
            "statuses": _merge_counts(v["statuses"] for v in per_task.values()),
            "data_provenance": provenance,
            "answers": "gold" if self.gold else "model",
            "sampling": self.sampling(),
            "generation_args": {t: self.task_generation_args(setup_dir, t) for t in tasks},
            "ns_metrics": _jsonable(per_task_metrics),
        }

    def _generate_task(self, setup_dir: Path, task: str, input_file: Path, base_url: str, served: str, k: int) -> None:
        output = input_file.parent / f"output-rs{k}.jsonl"
        if output.exists():
            return
        cmd = [
            sys.executable, "-m", GENERATE_MODULE,
            f"++input_file={input_file}",
            f"++output_file={output}",
            "++server.server_type=vllm",
            f"++server.base_url={base_url}",
            f"++server.model={served}",
            "++server.enable_soft_fail=True",
            "++skip_filled=True",
            f"++max_concurrent_requests={self.config.workers}",
            f"++inference.random_seed={self.config.seed + k}",
            *self.task_generation_args(setup_dir, task),
        ]  # fmt: skip
        _run_ns(cmd, output, log_path=output.parent / f"output-rs{k}.log", what=f"{self.id} {task} rs{k}")
        log.info("%s %s repeat %d: generated %s", self.id, task, k, _status_line(output))


def _reset_task_if_input_changed(task_dir: Path) -> None:
    stamp = task_dir / "input.sha256"
    digest = _sha256(task_dir / "input.jsonl")
    if stamp.exists() and stamp.read_text().strip() != digest:
        log.warning("%s: input changed since the last run; discarding old generations", task_dir.name)
        for f in task_dir.glob("output-rs*"):
            f.unlink()
    stamp.write_text(digest + "\n")


class RulerGoldServer(_GoldServer):
    """Answers each sample with its expected outputs. A sample is found by the
    tail of its (up to 128k-token) context, which ends the prompt text."""

    TAIL = 2000

    def __init__(self, rows: list[dict]):
        super().__init__(rows, answer=lambda r: " ".join(map(str, r["expected_answer"])))
        self.tails = [r["question"][-self.TAIL :] for r in rows]

    def lookup(self, prompt: str) -> str:
        window = prompt[-(self.TAIL + 20000) :]
        for row, tail in zip(self.rows, self.tails):
            if tail and tail in window:
                return self.answer(row)
        return ""


@register
class Ruler128k(RulerBenchmark):
    id = "ruler-128k"
    max_seq_length = 131072


@register
class Ruler64k(RulerBenchmark):
    id = "ruler-64k"
    max_seq_length = 65536
