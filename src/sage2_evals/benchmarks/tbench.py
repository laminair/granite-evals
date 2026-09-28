"""Terminal-Bench 2.1 (Sage2: agentic coding).

Metric: pass@1[avg-of-8] resolve rate, i.e. the fraction of tasks whose tests
pass (harbor reward 1), per independent repeat, averaged over 8 repeats.

The harness is Harbor, Terminal-Bench 2.x's official one, with its reference
agent Terminus 2. Each (task, repeat) is one harbor ``Trial``: harbor's agent
loop, prompts, per-task agent/verifier timeouts and test verification run
unchanged. The one substitution is the container backend: harbor's Docker
environment becomes :class:`sage2_evals.sandbox.harbor_env.SandboxEnvironment`
(enroot on BlueVela), started from each task's prebuilt ``docker_image``.

Tasks are read from the pinned HF mirror of the dataset (``registry.json`` +
``tasks/``). At the pinned revision, the harbor content digests of all 89
tasks are checked against the digests of the published dataset manifest, so a
score always names the exact task bytes.

``--option agent=oracle`` runs each task's reference ``solution/solve.sh``
instead of a model (no GPU / server needed): every runnable task should pass.

Trials are written to ``<output_dir>/repeat-<k>/<task>/`` (harbor's own trial
layout plus ``sage2.json``); a finished trial is skipped on restart.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from sage2_evals import data
from sage2_evals.registry import Benchmark, register

log = logging.getLogger(__name__)

N_TASKS = 89
# sha256 of the comma-joined sorted harbor content digests of the 89 tasks,
# equal to the digests in harbor-framework/terminal-bench-2-1@7131e43
# tasks/dataset.toml (which the HF revision below mirrors).
TASKS_DIGEST = "0fd3dbc55e6d7a6207dc0d93c5e933684b3ec87ac12941de0420b73c08627ee5"

# Tasks that can't run on the sandbox, with the reason; excluded from n and
# recorded in results.json. Established by an oracle run on BlueVela.
EXCLUDED: dict[str, str] = {}

# Tasks whose services listen on fixed ports. The enroot sandbox shares the
# host network, so these run one at a time (per sage2-evals process).
HOST_PORT_TASKS = frozenset(
    {
        "configure-git-webserver",
        "git-multibranch",
        "headless-terminal",
        "hf-model-inference",
        "install-windows-3.11",
        "kv-store-grpc",
        "nginx-request-logging",
        "pypi-server",
        "qemu-alpine-ssh",
        "qemu-startup",
    }
)

# Exception types harbor itself doesn't retry (RetryConfig default): outcomes
# of the agent or the tests, not of the infrastructure.
FINAL_EXCEPTIONS = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
        "ApiUsageLimitError",
        "AgentSafetyRefusalError",
        "AgentAuthenticationError",
        "ModelNotFoundError",
        "ContextLengthExceededError",
        "OutputLengthExceededError",
    }
)
AGENTS = ("terminus-2", "oracle")
IMAGE_IMPORT_WORKERS = 4


def task_digests(tasks_root: Path, names: list[str]) -> str:
    """Aggregate harbor content digest of ``names`` under ``tasks_root``."""
    from harbor.publisher.packager import Packager

    digests = sorted(Packager.compute_content_hash(tasks_root / n)[0] for n in names)
    return hashlib.sha256(",".join(digests).encode()).hexdigest()


def load_tasks(source: str, revision: str | None) -> tuple[Path, list[dict]]:
    """(dataset root, registry rows ``{name, path}``) of ``source``: an HF
    dataset repo in harbor registry layout, or a local directory of one."""
    if Path(source).exists():
        root = Path(source)
    else:
        from huggingface_hub import snapshot_download

        log.info("downloading %s@%s", source, revision or "main")
        root = Path(snapshot_download(source, repo_type="dataset", revision=revision))
    registry = json.loads((root / "registry.json").read_text())
    rows = [dict(t) for entry in registry for t in entry["tasks"]]
    return root, rows


def _task_config(task_dir: Path) -> dict:
    import tomllib

    return tomllib.loads((task_dir / "task.toml").read_text())


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


@register
class TerminalBench21(Benchmark):
    id = "terminal-bench-2.1"
    metric = "pass@1[avg-of-8] resolve rate"
    default_repeats = 8
    extra = "tbench"
    harness_packages = ("harbor",)
    dataset = "harborframework/terminal-bench-2.1"
    dataset_revision = "3e235dff6880252a587fa479c09bcb1e16edf2eb"

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like ``default``."""
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def agent(self) -> str:
        agent = self.opt("agent", "terminus-2")
        if agent not in AGENTS:
            raise SystemExit(f"{self.id}: agent must be one of {', '.join(AGENTS)}, not {agent!r}")
        return agent

    def needs_server(self) -> bool:
        return self.agent != "oracle"

    def sampling(self) -> dict[str, Any]:
        """Sampling options given on the CLI; unset ones fall back to the
        checkpoint's generation_config, which vLLM applies by default."""
        out = {}
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                out[key] = cast(self.config.options[key])
        return out

    def excluded(self) -> dict[str, str]:
        """``--option exclude=a,b`` adds to (``exclude=none`` clears) EXCLUDED."""
        spec = self.opt("exclude", "")
        if spec == "none":
            return {}
        extra = {n: "excluded by --option exclude" for n in spec.split(",") if n}
        return {**EXCLUDED, **extra}

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        source, revision = self.dataset_source()
        root, rows = load_tasks(source, revision)
        names = [r["name"] for r in rows]
        digest = None
        if source == self.dataset and revision == self.dataset_revision:
            digest = task_digests(root, names)
            if len(names) != N_TASKS or digest != TASKS_DIGEST:
                raise SystemExit(f"{self.id}: task digests of {source}@{revision} don't match the pinned dataset")

        excluded = {n: why for n, why in self.excluded().items() if n in names}
        rows = [r for r in rows if r["name"] not in excluded]
        if pattern := self.opt("tasks", ""):
            rows = [r for r in rows if re.search(pattern, r["name"])]
        tasks = data.take(rows, self.config.limit, key="name")
        for t in tasks:
            t["dir"] = root / t["path"]
            t["image"] = _task_config(t["dir"]).get("environment", {}).get("docker_image", "")
        log.info("%s: %d tasks (%d excluded) x %d repeats, agent %s", self.id, len(tasks), len(excluded), self.repeats, self.agent)

        agent = self._agent_config(base_url, served_model_name)
        image_errors = self._prefetch_images(tasks)
        reports = asyncio.run(self._run_all(tasks, agent, image_errors))

        per_repeat = []
        for k in range(self.repeats):
            rs = [r for r in reports if r["repeat"] == k]
            resolved = sum(r["resolved"] for r in rs)
            per_repeat.append(
                {
                    "repeat": k,
                    "resolved": resolved,
                    "n": len(rs),
                    "resolve_rate": resolved / len(rs) if rs else 0.0,
                    "statuses": _count(r["status"] for r in rs),
                }
            )
        per_task = {t["name"]: sum(r["resolved"] for r in reports if r["task"] == t["name"]) for t in tasks}
        return {
            "value": sum(r["resolve_rate"] for r in per_repeat) / len(per_repeat) if per_repeat else 0.0,
            "n": len(tasks),
            "n_total": len(names),
            "excluded": excluded,
            "dataset": source,
            "dataset_revision": revision,
            "tasks_digest": digest,
            "agent": self.agent,
            "sampling": self.sampling(),
            "per_repeat": per_repeat,
            "per_task_resolved": per_task,
            "tasks": [t["name"] for t in tasks],
        }

    # -- setup -------------------------------------------------------------

    def _agent_config(self, base_url: str, served: str) -> dict:
        """Kwargs for harbor's ``AgentConfig``."""
        if self.agent == "oracle":
            return {"name": "oracle"}
        sampling = self.sampling()
        max_len = self._max_model_len(base_url, served)
        call_kwargs: dict[str, Any] = {"api_key": "EMPTY"}
        call_kwargs.update({k: v for k, v in sampling.items() if k != "temperature"})
        kwargs: dict[str, Any] = {
            "api_base": base_url,
            "temperature": sampling.get("temperature"),  # None: not sent
            "llm_call_kwargs": call_kwargs,
            # LiteLLM knows nothing about a locally served model.
            "model_info": {
                "max_input_tokens": max_len,
                "max_output_tokens": sampling.get("max_tokens", max_len),
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        }
        if max_turns := self.opt("max_turns", 0):
            kwargs["max_turns"] = max_turns
        return {"name": "terminus-2", "model_name": f"hosted_vllm/{served}", "kwargs": kwargs}

    def _max_model_len(self, base_url: str, served: str) -> int:
        import httpx

        try:
            models = httpx.get(f"{base_url.rstrip('/')}/models", timeout=30).json()["data"]
            for m in models:
                if m.get("id") == served and m.get("max_model_len"):
                    return int(m["max_model_len"])
        except Exception:
            log.warning("%s: could not read max_model_len from %s", self.id, base_url, exc_info=True)
        return self.opt("max_model_len", 32768)

    def _backend(self) -> str:
        import os

        return self.opt("sandbox", "") or os.environ.get("SAGE2_SANDBOX", "enroot")

    def _prefetch_images(self, tasks: list[dict]) -> dict[str, str]:
        """Import every task image into the enroot cache before any trial, so a
        slow import doesn't eat harbor's environment start timeout. Returns
        ``{image: error}`` for images that failed."""
        if self._backend() != "enroot":
            return {}
        from sage2_evals.sandbox.enroot import ensure_squashfs

        images = sorted({t["image"] for t in tasks if t["image"]})
        errors: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=IMAGE_IMPORT_WORKERS) as pool:
            futures = {pool.submit(ensure_squashfs, image): image for image in images}
            for f in concurrent.futures.as_completed(futures):
                if e := f.exception():
                    log.error("%s: importing %s failed: %s", self.id, futures[f], e)
                    errors[futures[f]] = type(e).__name__
        return errors

    # -- trials ------------------------------------------------------------

    async def _run_all(self, tasks: list[dict], agent: dict, image_errors: dict[str, str]) -> list[dict]:
        sem = asyncio.Semaphore(self.config.workers)
        port_lock = asyncio.Lock()

        async def one(task: dict, k: int) -> dict:
            async with sem:
                lock = port_lock if task["name"] in HOST_PORT_TASKS else contextlib.nullcontext()
                async with lock:
                    return await self._trial(task, k, agent, image_errors)

        # Repeat-major, so a partial run has whole repeats done first.
        jobs = [one(t, k) for k in range(self.repeats) for t in tasks]
        return list(await asyncio.gather(*jobs))

    async def _trial(self, task: dict, k: int, agent: dict, image_errors: dict[str, str]) -> dict:
        name = task["name"]
        repeat_dir = self.config.output_dir / f"repeat-{k}"
        tdir = repeat_dir / name
        summary = tdir / "sage2.json"
        if summary.exists():
            return json.loads(summary.read_text())
        base = {"task": name, "repeat": k, "resolved": False, "reward": 0.0}
        if task["image"] in image_errors:
            return {**base, "status": f"error:image:{image_errors[task['image']]}"}

        attempts = self.opt("max_retries", 3) + 1
        report: dict = {}
        for attempt in range(attempts):
            if tdir.exists():
                old = repeat_dir / ".attempts" / f"{name}-{int(time.time())}-{attempt}"
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tdir, old)
            try:
                result = await self._run_trial(task, k, agent, repeat_dir)
            except Exception as e:  # one broken task must not sink the run
                log.exception("%s: %s repeat %d attempt %d failed", self.id, name, k, attempt)
                report = {**base, "status": f"error:{type(e).__name__}"}
                continue
            report = self._report(result, base)
            exc = result.exception_info
            if exc is None or exc.exception_type in FINAL_EXCEPTIONS:
                break
            log.warning("%s: %s repeat %d attempt %d: %s", self.id, name, k, attempt, exc.exception_type)
        report["attempts"] = attempt + 1
        log.info("%s repeat %d: %s %s reward=%s", self.id, k, name, report["status"], report["reward"])
        if not report["status"].startswith("error:"):
            # Infrastructure errors aren't persisted: a resumed run retries them.
            tdir.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps(report, indent=2))
        return report

    async def _run_trial(self, task: dict, k: int, agent: dict, repeat_dir: Path):
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial

        config = TrialConfig(
            task=TaskConfig(path=task["dir"]),
            trial_name=task["name"],
            trials_dir=repeat_dir,
            timeout_multiplier=self.opt("timeout_multiplier", 1.0),
            agent=AgentConfig(**agent),
            environment=EnvironmentConfig(
                import_path="sage2_evals.sandbox.harbor_env:SandboxEnvironment",
                kwargs={"backend": self.opt("sandbox", "")},
                delete=True,
            ),
        )
        if agent["name"] == "terminus-2":
            config.agent.kwargs = {
                **config.agent.kwargs,
                "llm_call_kwargs": {**agent["kwargs"]["llm_call_kwargs"], "seed": self.config.seed + k},
            }
        trial = await Trial.create(config)
        return await trial.run()

    @staticmethod
    def _report(result, base: dict) -> dict:
        rewards = (result.verifier_result.rewards or {}) if result.verifier_result else {}
        reward = float(rewards.get("reward", 0.0) or 0.0)
        exc = result.exception_info
        status = "graded" if exc is None else exc.exception_type
        if exc is not None and exc.exception_type not in FINAL_EXCEPTIONS:
            status = f"error:{exc.exception_type}"
        ar = result.agent_result
        return {
            **base,
            "resolved": reward >= 1.0,
            "reward": reward,
            "status": status,
            "n_input_tokens": ar.n_input_tokens if ar else None,
            "n_output_tokens": ar.n_output_tokens if ar else None,
            "exception": exc.exception_message[:2000] if exc else None,
        }
