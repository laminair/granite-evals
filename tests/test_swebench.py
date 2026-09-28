"""SWE-bench adapter tests with a fake sandbox (no containers, no model)."""

import pytest

pytest.importorskip("swebench")
pytest.importorskip("minisweagent")

from sage2_evals.benchmarks import swebench as sb_mod  # noqa: E402
from sage2_evals.registry import RunConfig  # noqa: E402
from sage2_evals.sandbox import ExecResult  # noqa: E402

INSTANCE = {
    "instance_id": "org__repo-1",
    "image": "swebench/sweb.eval.x86_64.org_1776_repo-1:latest",
    "repo": "org/repo",
    "version": "1.0",
    "problem_statement": "fix it",
    "FAIL_TO_PASS": '["test_a.py::test_fixed"]',
    "PASS_TO_PASS": '["test_a.py::test_kept"]',
    "log_parser": "parse_log_pytest",
    "eval_type": "pass_and_fail",
    "eval_script": "cd /testbed\npytest test_a.py",
}

PYTEST_LOG = """>>>>> Start Test Output
PASSED test_a.py::test_fixed
PASSED test_a.py::test_kept
>>>>> End Test Output
"""


class FakeSandbox:
    def __init__(self, apply_ok=True, log=PYTEST_LOG):
        self.apply_ok, self.log, self.commands, self.files = apply_ok, log, [], {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def write_file(self, path, content):
        self.files[path] = content

    def execute(self, command, *, cwd="/", timeout=None, env=None, stdin=None):
        self.commands.append(command)
        if command.startswith(("git apply", "patch ")):
            return ExecResult("", 0 if self.apply_ok else 1)
        if command == "/bin/bash /eval.sh":
            return ExecResult(self.log, 0)
        return ExecResult("", 0)


@pytest.fixture
def bench(tmp_path):
    return sb_mod.SWEBenchVerified(RunConfig(model="m", output_dir=tmp_path))


def test_empty_patch_is_unresolved_without_a_sandbox(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("no sandbox for empty patch"))
    assert bench._grade(INSTANCE, "  \n", tmp_path)["status"] == "empty_patch"


def test_resolved_when_tests_pass(bench, tmp_path, monkeypatch):
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    report = bench._grade(INSTANCE, "diff --git a/x b/x\n", tmp_path)
    assert report["status"] == "graded" and report["resolved"] is True
    assert sb_mod.PATCH_FILE in fake.files and "/eval.sh" in fake.files
    assert (tmp_path / "test_output.txt").read_text() == PYTEST_LOG


def test_unresolved_when_fail_to_pass_fails(bench, tmp_path, monkeypatch):
    log = PYTEST_LOG.replace("PASSED test_a.py::test_fixed", "FAILED test_a.py::test_fixed")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox(log=log))
    assert bench._grade(INSTANCE, "diff\n", tmp_path)["resolved"] is False


def test_patch_that_never_applies(bench, tmp_path, monkeypatch):
    fake = FakeSandbox(apply_ok=False)
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    assert bench._grade(INSTANCE, "diff\n", tmp_path)["status"] == "patch_failed"
    assert "/bin/bash /eval.sh" not in fake.commands


def test_run_aggregates_repeats_and_resumes(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchVerified(RunConfig(model="m", output_dir=tmp_path, repeats=2, limit=1, workers=1))
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: ([INSTANCE, {**INSTANCE, "instance_id": "z"}], "src"))
    calls = []
    monkeypatch.setattr(bench, "_generate", lambda inst, idir, url, served, k: calls.append(k) or ("diff\n" if k == 0 else ""))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox())
    out = bench.run("http://x/v1", "m")
    assert out["n"] == 1 and out["instances"] == ["org__repo-1"]
    assert [r["resolved"] for r in out["per_repeat"]] == [1, 0]
    assert out["value"] == 0.5
    bench.run("http://x/v1", "m")  # resume: nothing regenerated
    assert calls == [0, 1]


def test_agent_config_points_at_served_model(bench):
    config = bench._agent_config("http://h:1/v1", "granite", 2)
    assert config["model"]["model_name"] == "hosted_vllm/granite"
    assert config["model"]["model_kwargs"]["api_base"] == "http://h:1/v1"
    assert config["model"]["model_kwargs"]["seed"] == 2
    assert config["environment"]["cwd"] == "/testbed"


def test_gold_mode_grades_reference_patch_without_a_model(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchVerified(
        RunConfig(model="m", output_dir=tmp_path, repeats=1, workers=1, options={"patch": "gold"})
    )
    assert bench.needs_server() is False
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: ([{**INSTANCE, "patch": "gold-diff\n"}], "src"))
    monkeypatch.setattr(bench, "_generate", lambda *a: pytest.fail("gold mode must not call the agent"))
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    assert bench.run("", "m")["value"] == 1.0
    assert fake.files[sb_mod.PATCH_FILE] == "gold-diff\n"
