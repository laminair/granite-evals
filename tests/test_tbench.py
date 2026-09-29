"""Terminal-Bench adapter tests with fake trials and a host-shell sandbox
(no containers, no model)."""

import asyncio
import hashlib
import json
import logging
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor.environments.base")

from sage2_evals.benchmarks import tbench as tb  # noqa: E402
from sage2_evals.registry import RunConfig  # noqa: E402
from sage2_evals.sandbox import Sandbox  # noqa: E402
from sage2_evals.sandbox import harbor_env  # noqa: E402

TASK_TOML = """version = "1.0"
[agent]
timeout_sec = 900.0
[verifier]
timeout_sec = 900.0
[environment]
docker_image = "alexgshaw/{name}:20251031"
"""


def make_dataset(root: Path, names: list[str]) -> Path:
    for n in names:
        d = root / "tasks" / n
        (d / "environment").mkdir(parents=True)
        (d / "tests").mkdir()
        (d / "solution").mkdir()
        (d / "task.toml").write_text(TASK_TOML.format(name=n))
        (d / "instruction.md").write_text(f"do {n}\n")
        (d / "environment" / "Dockerfile").write_text("FROM ubuntu\nWORKDIR /app\n")
        (d / "tests" / "test.sh").write_text("exit 0\n")
        (d / "solution" / "solve.sh").write_text("true\n")
    registry = [{"name": "terminal-bench-2.1", "version": "2.1.0", "tasks": [{"name": n, "path": f"tasks/{n}"} for n in names]}]
    (root / "registry.json").write_text(json.dumps(registry))
    return root


def fake_result(reward=1.0, exc=None):
    return SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": reward}) if reward is not None else None,
        exception_info=SimpleNamespace(exception_type=exc, exception_message="boom") if exc else None,
        agent_result=SimpleNamespace(n_input_tokens=10, n_output_tokens=5),
    )


def bench(tmp_path, names, **options):
    ds = make_dataset(tmp_path / "ds", names)
    cfg = RunConfig(model="org/m", output_dir=tmp_path / "out", dataset=str(ds), options={"sandbox": "podman", **options})
    return tb.TerminalBench21(cfg)


def test_metadata():
    cls = tb.TerminalBench21
    assert cls.metric == "pass@1[avg-of-8] resolve rate" and cls.default_repeats == 8
    assert len(cls.dataset_revision) == 40
    assert tb.HOST_PORT_TASKS.isdisjoint(tb.EXCLUDED)
    assert len(tb.TASK_DIGESTS) == tb.N_TASKS
    assert hashlib.sha256(",".join(sorted(tb.TASK_DIGESTS.values())).encode()).hexdigest() == tb.TASKS_DIGEST


def test_oracle_needs_no_server(tmp_path):
    assert not bench(tmp_path, ["a"], agent="oracle").needs_server()
    assert bench(tmp_path / "x", ["a"]).needs_server()
    with pytest.raises(SystemExit):
        bench(tmp_path / "y", ["a"], agent="claude-code").needs_server()


def test_run_scores_resumes_and_records(tmp_path, monkeypatch):
    b = bench(tmp_path, ["c-task", "a-task", "b-task"], agent="oracle")
    b.repeats = 2
    calls = []

    async def run_trial(self, task, k, agent, repeat_dir):
        calls.append((task["name"], k))
        assert agent == {"name": "oracle"}
        (repeat_dir / task["name"]).mkdir(parents=True, exist_ok=True)
        if task["name"] == "b-task":
            return fake_result(0.0, "AgentTimeoutError")
        return fake_result(1.0)

    monkeypatch.setattr(tb.TerminalBench21, "_run_trial", run_trial)
    out = b.run("", "")
    assert out["n"] == 3 and out["n_total"] == 3 and out["excluded"] == {}
    assert out["value"] == pytest.approx(2 / 3)
    assert out["per_repeat"][0]["statuses"] == {"graded": 2, "AgentTimeoutError": 1}
    assert out["per_task_resolved"] == {"a-task": 2, "b-task": 0, "c-task": 2}
    assert out["tasks_digest"] is None  # dataset override: no pin to check
    assert len(calls) == 6
    assert json.loads((tmp_path / "out" / "repeat-1" / "a-task" / "sage2.json").read_text())["resolved"]

    calls.clear()
    assert b.run("", "")["value"] == pytest.approx(2 / 3)
    assert calls == []  # everything resumed


def test_limit_exclusions_and_task_filter(tmp_path, monkeypatch):
    b = bench(tmp_path, ["d", "c", "b", "a"], agent="oracle", exclude="a", tasks="^[a-c]$")
    b.config.limit = 1
    b.repeats = 1

    async def run_trial(self, task, k, agent, repeat_dir):
        return fake_result(1.0)

    monkeypatch.setattr(tb.TerminalBench21, "_run_trial", run_trial)
    out = b.run("", "")
    assert out["tasks"] == ["b"] and out["excluded"] == {"a": "excluded by --option exclude"}


def test_infra_errors_retry_and_are_not_persisted(tmp_path, monkeypatch, caplog):
    b = bench(tmp_path, ["a", "b"], agent="oracle", max_retries="2")
    b.repeats = 1
    attempts = {"a": 0, "b": 0}

    async def run_trial(self, task, k, agent, repeat_dir):
        attempts[task["name"]] += 1
        (repeat_dir / task["name"]).mkdir(parents=True, exist_ok=True)
        if task["name"] == "a":
            raise RuntimeError("sandbox died")
        if attempts["b"] == 1:
            return fake_result(None, "DownloadVerifierDirError")
        return fake_result(1.0)

    monkeypatch.setattr(tb.TerminalBench21, "_run_trial", run_trial)
    with caplog.at_level(logging.CRITICAL):
        out = b.run("", "")
    assert attempts == {"a": 3, "b": 2}
    assert out["per_repeat"][0]["statuses"] == {"error:RuntimeError": 1, "graded": 1}
    assert out["value"] == 0.5
    assert not (tmp_path / "out" / "repeat-0" / "a" / "sage2.json").exists()
    assert (tmp_path / "out" / "repeat-0" / ".attempts").is_dir()  # earlier attempts kept


def test_pinned_dataset_digest_is_checked(tmp_path, monkeypatch):
    root = make_dataset(tmp_path / "ds", ["a"])
    monkeypatch.setattr(tb, "load_tasks", lambda s, r, d: (root, [{"name": "a", "path": "tasks/a"}]))
    b = tb.TerminalBench21(RunConfig(model="m", output_dir=tmp_path / "out", options={"agent": "oracle"}))
    with pytest.raises(SystemExit, match="digests"):
        b.run("", "")


def test_task_digests_are_harbor_content_hashes(tmp_path):
    root = make_dataset(tmp_path / "ds", ["a", "b"])
    dirs = {n: root / "tasks" / n for n in ("a", "b")}
    d1 = tb.task_digests(dirs)
    (root / "tasks" / "a" / "instruction.md").write_text("changed\n")
    assert tb.task_digests(dirs) != d1
    assert "" not in tb.task_digest_map(dirs).values()


def test_pinned_check_hashes_registry_paths(tmp_path, monkeypatch):
    """The check hashes the tasks where registry.json puts them (tasks/<name>)."""
    root = make_dataset(tmp_path / "ds", ["a", "b"])
    dirs = {n: root / "tasks" / n for n in ("a", "b")}
    got = tb.task_digest_map(dirs)
    monkeypatch.setattr(tb, "TASK_DIGESTS", got)
    monkeypatch.setattr(tb, "TASKS_DIGEST", tb.task_digests(dirs))
    monkeypatch.setattr(tb, "load_tasks", lambda s, r, d: (root, [{"name": n, "path": f"tasks/{n}"} for n in ("a", "b")]))
    monkeypatch.setattr(tb.TerminalBench21, "_run_trial", lambda *a: None)
    b = tb.TerminalBench21(RunConfig(model="m", output_dir=tmp_path / "out", options={"agent": "oracle", "sandbox": "podman"}))
    b.repeats = 0
    assert b.run("", "")["tasks_digest"] == tb.TASKS_DIGEST


def test_terminus_config(tmp_path, monkeypatch):
    b = bench(tmp_path, ["a"], temperature="0.6", max_tokens="512")
    monkeypatch.setattr(tb.TerminalBench21, "_max_model_len", lambda self, url, served: 131072)
    agent = b._agent_config("http://h:8000/v1", "granite-4.2-3b")
    assert agent["name"] == "terminus-2" and agent["model_name"] == "hosted_vllm/granite-4.2-3b"
    kw = agent["kwargs"]
    assert kw["temperature"] == 0.6 and kw["api_base"] == "http://h:8000/v1"
    assert kw["llm_call_kwargs"] == {"api_key": "EMPTY", "max_tokens": 512}
    assert kw["model_info"]["max_input_tokens"] == 131072
    assert b.sampling() == {"temperature": 0.6, "max_tokens": 512}
    # Unset sampling options are not sent (generation_config defaults apply).
    plain = bench(tmp_path / "p", ["a"])
    monkeypatch.setattr(tb.TerminalBench21, "_max_model_len", lambda self, url, served: 4096)
    assert plain._agent_config("u", "s")["kwargs"]["temperature"] is None


# -- harbor environment -------------------------------------------------------


class HostSandbox(Sandbox):
    """Runs commands on the host shell: exercises exec/timeout/file transfer."""

    def start(self):
        pass

    def _exec_argv(self, command, cwd, env):
        exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env.items())
        return ["bash", "-c", f"{exports}cd {shlex.quote(cwd)} && {command}"]

    def close(self):
        pass


@pytest.fixture
def env(tmp_path):
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    edir = tmp_path / "environment"
    edir.mkdir()
    (edir / "Dockerfile").write_text("FROM x\nWORKDIR /somewhere\nWORKDIR /app\n")
    e = harbor_env.SandboxEnvironment(
        environment_dir=edir,
        environment_name="t",
        session_id="t__env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="alexgshaw/t:1", cpus=1, memory_mb=2048),
        backend="podman",
    )
    e._sandbox = HostSandbox("x")
    e._scratch = tmp_path / "scratch"
    e._scratch.mkdir()
    return e


def test_env_workdir_from_dockerfile(env):
    assert env._workdir == "/app"


def test_env_exec_output_env_and_timeout(env, tmp_path):
    r = asyncio.run(env.exec("echo out; echo err >&2; echo $FOO; exit 3", cwd=str(tmp_path), env={"FOO": "bar"}))
    assert (r.return_code, r.stdout, r.stderr) == (3, "out\nbar\n", "err\n")
    r = asyncio.run(env.exec("echo started; sleep 30", cwd="/", timeout_sec=1))
    assert r.return_code == harbor_env.TIMEOUT_RC and r.stdout == "started\n"


def test_env_exec_returns_while_background_process_holds_output(env):
    r = asyncio.run(asyncio.wait_for(env.exec("(sleep 20 &) ; echo done", cwd="/"), 10))
    assert r.return_code == 0 and r.stdout == "done\n"


def test_env_non_root_user_goes_through_su(env, monkeypatch):
    seen = []

    async def run(command, **kw):
        seen.append(command)
        return 0, "", ""

    monkeypatch.setattr(env, "_run", run)
    asyncio.run(env.exec("whoami", user="agent"))
    asyncio.run(env.exec("whoami", user="root"))
    assert seen == ["su agent -s /bin/bash -c whoami", "whoami"]


def test_env_file_roundtrip(env, tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("A")
    (src / "sub" / "b.sh").write_text("#!/bin/sh\n")
    (tmp_path / "blob").write_text("B")
    (src / "linked.txt").symlink_to(tmp_path / "blob")  # HF snapshot layout
    box = tmp_path / "box"
    asyncio.run(env.upload_dir(src, str(box / "tests")))
    assert (box / "tests" / "sub" / "b.sh").read_text() == "#!/bin/sh\n"
    assert not (box / "tests" / "linked.txt").is_symlink()
    assert (box / "tests" / "linked.txt").read_text() == "B"
    asyncio.run(env.upload_file(src / "a.txt", str(box / "deep" / "x.txt")))
    assert (box / "deep" / "x.txt").read_text() == "A"

    back = tmp_path / "back"
    asyncio.run(env.download_dir(str(box / "tests"), back))
    assert (back / "a.txt").read_text() == "A" and (back / "sub" / "b.sh").exists()
    asyncio.run(env.download_file(str(box / "deep" / "x.txt"), back / "f" / "x.txt"))
    assert (back / "f" / "x.txt").read_text() == "A"
    with pytest.raises(RuntimeError):
        asyncio.run(env.download_file(str(box / "missing"), back / "m.txt"))
    assert not (back / "m.txt").exists()


def test_env_enroot_passes_no_harness_secrets(env, monkeypatch):
    monkeypatch.setenv("SAGE2_JUDGE_API_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("ENROOT_DATA_PATH", "/scratch/data")
    got = env._enroot_env()
    assert "SAGE2_JUDGE_API_KEY" not in got and "HF_TOKEN" not in got
    assert got["ENROOT_DATA_PATH"] == "/scratch/data" and got["NVIDIA_VISIBLE_DEVICES"] == "void"


def test_leftover_daemons_are_killed_by_root(monkeypatch):
    """Processes that drop the marker (nginx workers) are found by their root."""
    roots = {"10": "/scratch/data/sage2-x", "11": "/", "12": "/scratch/data/sage2-x"}
    killed = []

    def readlink(path):
        pid = path.split("/")[2]
        if pid not in roots:
            raise OSError(path)
        return roots[pid]

    monkeypatch.setattr(harbor_env.os, "listdir", lambda p: [*roots, "self", "99999"])
    monkeypatch.setattr(harbor_env.os, "readlink", readlink)
    monkeypatch.setattr(harbor_env.os, "kill", lambda pid, sig: killed.append(pid))
    assert harbor_env._kill_rooted("/scratch/data/sage2-x") == 2 and killed == [10, 12]
    monkeypatch.setenv("ENROOT_DATA_PATH", "/scratch/data")
    assert harbor_env.enroot_rootfs("sage2-x") == "/scratch/data/sage2-x"


def test_no_paid_endpoints(tmp_path, monkeypatch):
    """Terminus 2 talks only to the served model: nothing to meter."""
    b = bench(tmp_path, ["a"])
    monkeypatch.setattr(tb.TerminalBench21, "_max_model_len", lambda self, url, served: 4096)
    kw = b._agent_config("http://127.0.0.1:8000/v1", "m")["kwargs"]
    assert kw["api_base"] == "http://127.0.0.1:8000/v1" and kw["llm_call_kwargs"]["api_key"] == "EMPTY"


def test_symlinked_dataset_is_materialized(tmp_path):
    """HF cache snapshots are symlinks into a blob store, which harbor refuses."""
    real = make_dataset(tmp_path / "blobs", ["a"])
    snap = tmp_path / "snap"
    for f in real.rglob("*"):
        if f.is_file():
            link = snap / f.relative_to(real)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(f)
    root, rows = tb.load_tasks(str(snap), None, tmp_path / "out" / "dataset")
    assert root == tmp_path / "out" / "dataset" and rows == [{"name": "a", "path": "tasks/a"}]
    assert not tb._has_symlinks(root) and (root / "tasks" / "a" / "tests" / "test.sh").read_text() == "exit 0\n"
    from harbor.models.task.task import Task

    with pytest.raises(ValueError, match="stay within the task"):
        Task(snap / "tasks" / "a")
    Task(root / "tasks" / "a")  # harbor accepts the copy
    # A plain local dataset is used in place.
    assert tb.load_tasks(str(real), None, tmp_path / "unused")[0] == real
