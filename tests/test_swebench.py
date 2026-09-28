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
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [INSTANCE, {**INSTANCE, "instance_id": "z"}])
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
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [{**INSTANCE, "patch": "gold-diff\n"}])
    monkeypatch.setattr(bench, "_generate", lambda *a: pytest.fail("gold mode must not call the agent"))
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    assert bench.run("", "m")["value"] == 1.0
    assert fake.files[sb_mod.PATCH_FILE] == "gold-diff\n"


# -- SWE-bench Multilingual ----------------------------------------------------

ML_INSTANCE = {
    **INSTANCE,
    "instance_id": "tokio-rs__tokio-1",
    "image": "swebench/sweb.eval.x86_64.tokio-rs_1776_tokio-1:latest",
    "repo": "tokio-rs/tokio",
    "FAIL_TO_PASS": '["time::test_fixed"]',
    "PASS_TO_PASS": '["time::test_kept"]',
    "log_parser": "parse_log_cargo",
    "eval_script": "cd /testbed\ncargo test",
    "patch": "gold-diff\n",
}

CARGO_LOG = """>>>>> Start Test Output
test time::test_fixed ... ok
test time::test_kept ... ok
>>>>> End Test Output
"""


def test_multilingual_is_registered_like_verified(tmp_path):
    from sage2_evals import registry

    cls = registry.get("swebench-multilingual")
    assert cls is sb_mod.SWEBenchMultilingual
    assert cls.metric == "pass@1[avg-of-3] resolve rate" and cls.default_repeats == 3
    assert cls.dataset == "SWE-bench/SWE-bench_Multilingual" and len(cls.dataset_revision) == 40
    bench = cls(RunConfig(model="m", output_dir=tmp_path))
    assert bench.dataset_config() is None
    assert bench._agent_config("http://h/v1", "g", 0)["environment"]["cwd"] == "/testbed"


def test_multilingual_grades_with_the_instance_log_parser(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchMultilingual(RunConfig(model="m", output_dir=tmp_path))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox(log=CARGO_LOG))
    assert bench._grade(ML_INSTANCE, "diff\n", tmp_path)["resolved"] is True
    failing = CARGO_LOG.replace("test_fixed ... ok", "test_fixed ... FAILED")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox(log=failing))
    assert bench._grade(ML_INSTANCE, "diff\n", tmp_path)["resolved"] is False


# -- SWE-bench Pro -------------------------------------------------------------

import hashlib  # noqa: E402

PRO_IID = "instance_org__repo-abc-vnan"
PRO_INSTANCE = {
    "instance_id": PRO_IID,
    "docker_image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{PRO_IID}",
    "patch": "hf-gold\n",
    "problem_statement": "not what the agent sees",
}
PRO_CONFIG_YAML = """agent:
  system_template: 'sys'
  instance_template: 'Please solve this issue: {{task}} on {{system}} {{machine}}'
  step_limit: 0
  cost_limit: 3.0
environment:
  env:
    PAGER: cat
model:
  model_kwargs:
    drop_params: true
    reasoning_effort: high
  model_class: litellm
"""


def pro_files(iid=PRO_IID):
    t = f"tasks/{iid}/"
    return {
        t + "instruction.md": b"A code repository is available in the `/app` directory. Do X.",
        t + "solution/gold_patch.diff": b"gold-diff\n",
        t + "tests/test.sh": b"#!/bin/bash\necho verifier\n",
        t + "tests/run_script.sh": b"#!/bin/bash\n",
        t + "tests/parser.py": b"print(1)\n",
        t + "tests/config.json": b'{"fail_to_pass": "[]"}',
        t + "tests/test_patch.patch": b"diff --git a/t b/t\n",
        sb_mod.PRO_AGENT_CONFIG: PRO_CONFIG_YAML.encode(),
    }


def make_fetch(files, calls=None):
    sums = "".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k, v in files.items()).encode()

    def fetch(path):
        if calls is not None:
            calls.append(path)
        return sums if path == "SHA256SUMS" else files[path]

    return fetch


@pytest.fixture
def pro(tmp_path):
    bench = sb_mod.SWEBenchPro(RunConfig(model="m", output_dir=tmp_path, repeats=1, workers=1))
    # A different commit skips the pinned SHA256SUMS digest, so fake sums verify.
    bench.tasks = sb_mod.ProTasks(tmp_path / "tasks", commit="test", fetch=make_fetch(pro_files()))
    return bench


class ProSandbox:
    """Fake of a V2 image: /app exists, test.sh writes the given reward."""

    def __init__(self, reward="1", apply_rc=0, patch=""):
        self.reward, self.apply_rc, self.patch = reward, apply_rc, patch
        self.commands, self.files = [], {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def start(self):
        pass

    def close(self):
        pass

    def write_file(self, path, content):
        self.files[path] = content

    def execute(self, command, *, cwd="/", timeout=None, env=None, stdin=None):
        self.commands.append((command, cwd))
        if command == sb_mod.PRO_WORKDIR:
            return ExecResult("/app\n", 0)
        if command == sb_mod.PRO_APPLY:
            return ExecResult("Applied patch", self.apply_rc)
        if command.startswith("/tests/test.sh"):
            return ExecResult("", 0 if self.reward == "1" else 1)
        if command == "cat /logs/verifier/reward.txt":
            return ExecResult(f"{self.reward}\n", 0) if self.reward is not None else ExecResult("No such file", 1)
        if command == "cat /logs/verifier/test-stdout.txt":
            return ExecResult("RESULT: PASSED\n", 0)
        if command == "cat /logs/verifier/output.json":
            return ExecResult('{"tests": []}', 0)
        if command.startswith("uname"):
            return ExecResult("Linux\n5.14\n#1 SMP\nx86_64\n", 0)
        if command == "cat /tmp/model.patch":
            return ExecResult(self.patch, 0)
        return ExecResult("", 0)


def test_pro_is_registered_with_its_pins():
    from sage2_evals import registry

    cls = registry.get("swebench-pro")
    assert cls.metric == "pass@1[avg-of-3] resolve rate" and cls.default_repeats == 3
    assert cls.dataset == "ScaleAI/SWE-bench_Pro" and len(cls.dataset_revision) == 40
    assert len(sb_mod.PRO_COMMIT) == 40 and len(sb_mod.PRO_SHA256SUMS_SHA256) == 64


def test_pro_subset_selects_the_hf_config(tmp_path):
    cfg = lambda **o: sb_mod.SWEBenchPro(RunConfig(model="m", output_dir=tmp_path, options=o))  # noqa: E731
    assert cfg().dataset_config() == "default"
    assert cfg(subset="hard").dataset_config() == "hard"
    with pytest.raises(SystemExit):
        cfg(subset="v1").dataset_config()


def test_pro_task_files_are_verified_and_cached(tmp_path):
    files, calls = pro_files(), []
    tasks = sb_mod.ProTasks(tmp_path, commit="test", fetch=make_fetch(files, calls))
    assert set(tasks.tests(PRO_IID)) == {"test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch"}
    n = len(calls)
    again = sb_mod.ProTasks(tmp_path, commit="test", fetch=make_fetch(files, calls))
    assert again.tests(PRO_IID)["test.sh"].startswith("#!/bin/bash")
    assert len(calls) == n  # all served from the cache
    cached = tmp_path / "test" / f"tasks/{PRO_IID}/tests/test.sh"
    cached.write_text("corrupted")
    fresh = sb_mod.ProTasks(tmp_path, commit="test", fetch=make_fetch(files, calls))
    assert fresh.tests(PRO_IID)["test.sh"].startswith("#!/bin/bash")
    assert calls[n:] == [f"tasks/{PRO_IID}/tests/test.sh"]  # a bad cached file is fetched again

    sums_fetch = make_fetch(files)
    bad = sb_mod.ProTasks(tmp_path / "bad", commit="test", fetch=lambda p: sums_fetch(p) if p == "SHA256SUMS" else b"evil")
    with pytest.raises(RuntimeError, match="sha256"):
        bad.task_file(PRO_IID, "instruction.md")
    with pytest.raises(KeyError):
        sb_mod.ProTasks(tmp_path / "x", commit="test", fetch=make_fetch(files)).tests("instance_missing")


def test_pinned_sha256sums_digest_is_enforced(tmp_path):
    tasks = sb_mod.ProTasks(tmp_path, fetch=lambda p: b"not the pinned file")
    with pytest.raises(RuntimeError, match="sha256"):
        tasks.sums()


def test_pro_grade_runs_the_task_verifier_in_a_fresh_sandbox(pro, tmp_path, monkeypatch):
    fake = ProSandbox(reward="1")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda image, **k: fake if image == PRO_INSTANCE["docker_image"] else None)
    report = pro._grade(PRO_INSTANCE, "diff --git a/x b/x\n", tmp_path)
    assert report["status"] == "graded" and report["resolved"] is True and report["reward"] == 1.0
    assert fake.files["/tmp/replay.patch"] == "diff --git a/x b/x\n"
    assert set(fake.files) >= {f"/tests/{n}" for n in ("test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch")}
    commands = [c for c, _ in fake.commands]
    assert commands.index(sb_mod.PRO_APPLY) < commands.index("/tests/test.sh > /logs/verifier/test-stdout.txt 2>&1")
    assert ("/tests/test.sh > /logs/verifier/test-stdout.txt 2>&1", "/app") in fake.commands
    assert (tmp_path / "test_output.txt").read_text() == "RESULT: PASSED\n"
    assert (tmp_path / "output.json").exists()


def test_pro_grade_unresolved_on_zero_reward_or_missing_reward(pro, tmp_path, monkeypatch):
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: ProSandbox(reward="0"))
    assert pro._grade(PRO_INSTANCE, "diff\n", tmp_path) == {
        "instance_id": PRO_IID, "resolved": False, "status": "graded", "reward": 0.0, "apply_rc": 0, "verifier_rc": 1,
    }
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: ProSandbox(reward=None))
    assert pro._grade(PRO_INSTANCE, "diff\n", tmp_path)["status"] == "no_reward"


def test_pro_grade_still_runs_the_verifier_when_the_patch_does_not_apply(pro, tmp_path, monkeypatch):
    # patch_replay grades whatever applied; the verifier decides.
    fake = ProSandbox(reward="0", apply_rc=1)
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    report = pro._grade(PRO_INSTANCE, "diff\n", tmp_path)
    assert report["apply_rc"] == 1 and report["status"] == "graded" and not report["resolved"]
    assert any(c.startswith("/tests/test.sh") for c, _ in fake.commands)


def test_pro_empty_patch_needs_no_sandbox(pro, tmp_path, monkeypatch):
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("no sandbox for an empty patch"))
    assert pro._grade(PRO_INSTANCE, "\n", tmp_path)["status"] == "empty_patch"


def test_pro_agent_config_layers_the_protocol_file_over_mini(pro):
    config = pro._agent_config("http://h:1/v1", "granite", 1)
    assert config["agent"]["instance_template"].startswith("Please solve this issue: {{task}}")
    assert config["agent"]["step_limit"] == 0 and config["agent"]["cost_limit"] == 0
    assert config["agent"]["wall_time_limit_seconds"] == sb_mod.PRO_AGENT_BUDGET_S
    assert config["model"]["model_name"] == "hosted_vllm/granite"
    assert config["model"]["model_kwargs"]["reasoning_effort"] == "high"
    assert config["model"]["model_kwargs"]["api_base"] == "http://h:1/v1"
    assert config["model"]["model_kwargs"]["seed"] == 1
    assert config["environment"]["env"]["PAGER"] == "cat"


def test_pro_generate_gives_the_instruction_and_captures_the_diff(pro, tmp_path, monkeypatch):
    fake = ProSandbox(patch="diff --git a/f b/f\n+fix\n")
    from sage2_evals.sandbox import minisweagent_env

    monkeypatch.setattr(minisweagent_env, "make_sandbox", lambda *a, **k: fake)
    seen = {}

    class FakeAgent:
        def __init__(self, model, env, **kwargs):
            seen["agent"], seen["cwd"] = kwargs, env.config.cwd

        def run(self, task, **kwargs):
            seen["task"], seen["vars"] = task, kwargs
            return {"exit_status": "TimeExceeded", "submission": ""}  # no submit: the diff still counts

        def save(self, path, *extra):
            path.write_text("{}")

    import minisweagent.agents.default as default_mod
    import minisweagent.models as models_mod

    monkeypatch.setattr(default_mod, "DefaultAgent", FakeAgent)
    monkeypatch.setattr(models_mod, "get_model", lambda config: object())
    patch = pro._generate(PRO_INSTANCE, tmp_path, "http://h/v1", "granite", 0)
    assert patch == "diff --git a/f b/f\n+fix\n"
    assert seen["task"].startswith("A code repository is available in the `/app` directory")
    assert seen["cwd"] == "/app" and seen["vars"] == {"system": "Linux", "release": "5.14", "version": "#1 SMP", "machine": "x86_64"}
    assert (sb_mod.PRO_CAPTURE, "/app") in fake.commands
    assert (tmp_path / "traj.json").exists()


def test_pro_gold_run_uses_the_task_reference_patch(pro, tmp_path, monkeypatch):
    pro.config.options["patch"] = "gold"
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [PRO_INSTANCE])
    monkeypatch.setattr(pro, "_generate", lambda *a: pytest.fail("gold mode must not call the agent"))
    fake = ProSandbox(reward="1")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    out = pro.run("", "m")
    assert out["value"] == 1.0 and out["subset"] == "default"
    assert out["verifier"]["commit"] == sb_mod.PRO_COMMIT
    assert fake.files["/tmp/replay.patch"] == "gold-diff\n"


# -- check=data ----------------------------------------------------------------


def test_parse_image_ref():
    assert sb_mod.parse_image_ref("swebench/sweb.eval.x86_64.a_1776_b-1:latest") == (
        "registry-1.docker.io",
        "swebench/sweb.eval.x86_64.a_1776_b-1",
        "latest",
    )
    assert sb_mod.parse_image_ref(f"ghcr.io/scaleapi/swe-bench_pro-v2:{PRO_IID}") == (
        "ghcr.io",
        "scaleapi/swe-bench_pro-v2",
        PRO_IID,
    )
    assert sb_mod.parse_image_ref("ubuntu") == ("registry-1.docker.io", "library/ubuntu", "latest")


def test_probe_image_gets_an_anonymous_token():
    import httpx

    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), request.headers.get("authorization")))
        if request.url.path == "/token":
            return httpx.Response(200, json={"token": "t0k"})
        if request.headers.get("authorization") != "Bearer t0k":
            return httpx.Response(401, headers={"www-authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io"'})
        if request.url.path.endswith("/missing"):
            return httpx.Response(404)
        return httpx.Response(200, headers={"docker-content-digest": "sha256:abc"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert sb_mod.probe_image("ghcr.io/o/r:tag", client=client) == {"status": 200, "digest": "sha256:abc"}
    assert seen[1] == ("GET", "https://ghcr.io/token?service=ghcr.io&scope=repository%3Ao%2Fr%3Apull", None)
    assert seen[2][0] == "HEAD" and seen[2][2] == "Bearer t0k"
    assert sb_mod.probe_image("ghcr.io/o/r:missing", client=client)["status"] == 404


def test_check_mode_needs_no_model_and_reports_failures(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchMultilingual(
        RunConfig(model="none", output_dir=tmp_path, workers=2, options={"check": "data"})
    )
    assert bench.needs_server() is False
    bad = {**ML_INSTANCE, "instance_id": "bad-2", "log_parser": "parse_log_nope", "image": "x/missing:latest"}
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [ML_INSTANCE, bad])
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("check mode runs no sandbox"))
    monkeypatch.setattr(sb_mod, "probe_image", lambda ref: {"status": 404 if "missing" in ref else 200, "digest": None})
    out = bench.run("", "none")
    assert out["mode"] == "check=data" and out["n"] == 2 and out["ok"] == 1 and out["value"] == 0.5
    (failure,) = out["failures"]
    assert failure["instance_id"] == "bad-2"
    assert any("parse_log_nope" in p for p in failure["problems"])
    assert "image manifest: HTTP 404" in failure["problems"]
    assert (tmp_path / "check.json").exists()


def test_pro_check_mode_verifies_every_task_file(pro, tmp_path, monkeypatch):
    pro.config.options["check"] = "data"
    missing = {**PRO_INSTANCE, "instance_id": "instance_gone"}
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [PRO_INSTANCE, missing])
    monkeypatch.setattr(sb_mod, "probe_image", lambda ref: {"status": 200, "digest": "d"})
    out = pro.run("", "none")
    assert out["ok"] == 1 and out["subset"] == "default"
    assert out["failures"][0]["instance_id"] == "instance_gone"
    assert "KeyError" in out["failures"][0]["problems"][0]
