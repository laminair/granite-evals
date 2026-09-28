"""SWE-bench family (Sage2: SWE Bench Verified / Pro / Multilingual).

Metric: pass@1[avg-of-3] resolve rate, i.e. the resolve rate of each of
``repeats`` independent agent runs, averaged.

Two phases per instance and repeat, both inside a per-instance sandbox built
from the instance's SWE-bench image:

1. generate: mini-swe-agent (upstream loop, prompts and swebench.yaml config)
   drives the served model until it submits a patch.
2. grade: a *fresh* sandbox applies the patch, runs the dataset's eval_script
   and scores the log with swebench's own ``get_eval_report``.

Everything is written under ``<output_dir>/repeat-<k>/<instance_id>/`` and a
finished instance is skipped on restart, so granite.build retries resume.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

from sage2_evals import data
from sage2_evals.registry import Benchmark, register
from sage2_evals.sandbox import make_sandbox

log = logging.getLogger(__name__)

PATCH_FILE = "/tmp/patch.diff"
EVAL_TIMEOUT_S = 1800


class SWEBench(Benchmark):
    metric = "pass@1[avg-of-3] resolve rate"
    default_repeats = 3
    extra = "swebench"
    upstream_dataset: ClassVar[str]
    split: ClassVar[str] = "test"

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like ``default``."""
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def gold(self) -> bool:
        """``--option patch=gold`` grades the reference patch instead of the
        model's: checks images, sandbox and grading on a node without a model
        (every instance should resolve)."""
        return self.opt("patch", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        rows, source = data.load_split(
            self.id, dataset=self.config.dataset, revision=self.config.dataset_revision, split=self.split
        )
        if pattern := self.opt("instances", ""):
            rows = [r for r in rows if re.search(pattern, r["instance_id"])]
        instances = data.take(rows, self.config.limit, key="instance_id")
        log.info("%s: %d instances x %d repeats", self.id, len(instances), self.repeats)

        per_repeat = []
        for k in range(self.repeats):
            repeat_dir = self.config.output_dir / f"repeat-{k}"
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                reports = list(
                    pool.map(
                        lambda inst: self._instance(inst, repeat_dir, base_url, served_model_name, k),
                        instances,
                    )
                )
            resolved = sum(r["resolved"] for r in reports)
            per_repeat.append(
                {
                    "repeat": k,
                    "resolved": resolved,
                    "n": len(reports),
                    "resolve_rate": resolved / len(reports) if reports else 0.0,
                    "statuses": _count(r["status"] for r in reports),
                }
            )
            log.info("%s repeat %d: %d/%d resolved", self.id, k, resolved, len(reports))

        return {
            "value": sum(r["resolve_rate"] for r in per_repeat) / len(per_repeat),
            "n": len(instances),
            "dataset": source,
            "per_repeat": per_repeat,
            "instances": [i["instance_id"] for i in instances],
        }

    # -- one instance ------------------------------------------------------

    def _instance(self, instance: dict, repeat_dir: Path, base_url: str, served: str, k: int) -> dict:
        iid = instance["instance_id"]
        idir = repeat_dir / iid
        idir.mkdir(parents=True, exist_ok=True)
        report_path = idir / "report.json"
        if report_path.exists():
            return json.loads(report_path.read_text())

        try:
            patch_path = idir / "patch.diff"
            if not patch_path.exists():
                if self.gold:
                    patch_path.write_text(instance["patch"])
                else:
                    patch_path.write_text(self._generate(instance, idir, base_url, served, k))
            report = self._grade(instance, patch_path.read_text(), idir)
        except Exception as e:  # one broken instance must not sink the run
            log.exception("%s: %s failed", self.id, iid)
            # Not persisted: an infrastructure error should be retried on resume.
            return {"instance_id": iid, "resolved": False, "status": f"error:{type(e).__name__}"}
        report_path.write_text(json.dumps(report, indent=2))
        log.info("%s repeat %d: %s %s resolved=%s", self.id, k, iid, report["status"], report["resolved"])
        return report

    def _agent_config(self, base_url: str, served: str, k: int) -> dict:
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.utils.serialize import recursive_merge

        base = get_config_from_spec(builtin_config_dir / "benchmarks" / "swebench.yaml")
        model_kwargs = {"api_base": base_url, "api_key": "EMPTY", "seed": self.config.seed + k}
        # Unset sampling options fall back to the checkpoint's generation_config,
        # which vLLM applies by default.
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                model_kwargs[key] = cast(self.config.options[key])
        return recursive_merge(
            base,
            {
                "agent": {
                    "step_limit": self.opt("step_limit", 250),
                    "cost_limit": 0,  # local model: no cost, bound by step_limit
                },
                "model": {
                    "model_name": f"hosted_vllm/{served}",
                    "cost_tracking": "ignore_errors",
                    "model_kwargs": model_kwargs,
                },
            },
        )

    def _generate(self, instance: dict, idir: Path, base_url: str, served: str, k: int) -> str:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.models import get_model

        from sage2_evals.sandbox.minisweagent_env import SandboxEnvironment

        config = self._agent_config(base_url, served, k)
        env_config = {key: v for key, v in config["environment"].items() if key != "environment_class"}
        env = SandboxEnvironment(image=instance["image"], backend=self.opt("sandbox", ""), **env_config)
        try:
            agent = DefaultAgent(get_model(config=config["model"]), env, **config["agent"])
            exit_status, submission = "unknown", ""
            try:
                info = agent.run(instance["problem_statement"])
                exit_status, submission = info.get("exit_status"), info.get("submission") or ""
            finally:
                agent.save(idir / "traj.json", {"info": {"exit_status": exit_status}, "instance_id": instance["instance_id"]})
            return submission
        finally:
            env.cleanup()

    def _grade(self, instance: dict, patch: str, idir: Path) -> dict:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import GIT_APPLY_CMDS
        from swebench.harness.utils import make_test_spec

        iid = instance["instance_id"]
        base = {"instance_id": iid, "resolved": False}
        if not patch.strip():
            return {**base, "status": "empty_patch"}
        if instance.get("image_assets"):
            raise NotImplementedError(f"{iid}: image_assets staging is not supported yet")

        spec = make_test_spec(instance)
        with make_sandbox(instance["image"], backend=self.opt("sandbox", "")) as sb:
            sb.write_file(PATCH_FILE, patch)
            attempts = []
            for i, cmd in enumerate(GIT_APPLY_CMDS):
                if i:
                    sb.execute("git checkout -- . ; git clean -fd", cwd="/testbed")
                r = sb.execute(f"{cmd} {PATCH_FILE}", cwd="/testbed")
                attempts.append(f"$ {cmd} {PATCH_FILE}  (exit {r.returncode})\n{r.output}")
                if r.returncode == 0:
                    break
            (idir / "patch_apply.txt").write_text("\n".join(attempts))
            if r.returncode != 0:
                return {**base, "status": "patch_failed"}

            sb.write_file("/eval.sh", spec.eval_script)
            result = sb.execute("/bin/bash /eval.sh", cwd="/testbed", timeout=self.opt("eval_timeout", EVAL_TIMEOUT_S))
            log_path = idir / "test_output.txt"
            log_path.write_text(result.output)
            if result.timed_out:
                return {**base, "status": "eval_timeout"}

        report = get_eval_report(
            test_spec=spec,
            prediction={"instance_id": iid, "model_patch": patch, "model_name_or_path": self.config.model},
            test_log_path=str(log_path),
            include_tests_status=True,
        )[iid]
        return {**base, "resolved": bool(report["resolved"]), "status": "graded", "report": report}


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


@register
class SWEBenchVerified(SWEBench):
    id = "swebench-verified"
    upstream_dataset = "SWE-bench/SWE-bench_Verified"
