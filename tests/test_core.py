import json

import pytest

from sage2_evals import cli, data, registry, suites
from sage2_evals.registry import Benchmark, RunConfig
from sage2_evals.results import write_results
from sage2_evals.sandbox import enroot
from sage2_evals.sandbox.enroot import enroot_uri
from sage2_evals.serving import ServerConfig, VLLMServer


@pytest.mark.parametrize(
    "image,uri",
    [
        ("swebench/sweb.eval.x86_64.a_1776_b-1:latest", "docker://swebench/sweb.eval.x86_64.a_1776_b-1:latest"),
        ("docker.io/swebench/x:latest", "docker://swebench/x:latest"),
        ("us.icr.io/ns/img:1.0", "docker://us.icr.io#ns/img:1.0"),
        ("localhost:5000/img", "docker://localhost:5000#img"),
    ],
)
def test_enroot_uri(image, uri):
    assert enroot_uri(image) == uri


def test_enroot_sandbox_keeps_tmp_between_commands(tmp_path, monkeypatch):
    # enroot mounts a fresh tmpfs on /tmp per `start`; the sandbox binds its own dir.
    monkeypatch.setenv("ENROOT_TEMP_PATH", str(tmp_path))
    monkeypatch.setattr(enroot, "ensure_squashfs", lambda image: tmp_path / "x.sqsh")
    monkeypatch.setattr(enroot.subprocess, "run", lambda *a, **k: None)
    sb = enroot.EnrootSandbox("img:latest")
    sb.start()
    argv = sb._exec_argv("true", "/testbed", {})
    assert argv[argv.index("--mount") + 1] == f"{sb.tmp}:/tmp"
    assert sb.tmp.parent == tmp_path and sb.tmp.is_dir()
    assert sb._run_env()["NVIDIA_VISIBLE_DEVICES"] == "void"
    sb.close()
    assert not sb.tmp.exists()


def test_take_is_stable_and_limited():
    rows = [{"id": "b"}, {"id": "c"}, {"id": "a"}]
    assert [r["id"] for r in data.take(rows, 2, key="id")] == ["a", "b"]
    assert len(data.take(rows, None, key="id")) == 3


def test_mirror_repo_needs_org(monkeypatch):
    monkeypatch.delenv(data.HF_ORG_ENV, raising=False)
    with pytest.raises(SystemExit):
        data.mirror_repo("swebench-verified")
    monkeypatch.setenv(data.HF_ORG_ENV, "some-org")
    assert data.mirror_repo("swebench-verified") == "some-org/sage2-swebench-verified"


def test_suites_reference_unique_ids():
    for name in suites.names():
        ids = [b["id"] for b in suites.load(name)["benchmarks"]]
        assert len(ids) == len(set(ids)), name


def test_every_implemented_benchmark_is_in_a_suite():
    listed = {b["id"] for name in suites.names() for b in suites.load(name)["benchmarks"]}
    assert set(registry.all_benchmarks()) <= listed


class _Dummy(Benchmark):
    id = "dummy"
    metric = "accuracy"

    def run(self, base_url, served_model_name):
        return {"value": 0.5, "n": 2, "extra": 1}


def test_results_schema(tmp_path):
    b = _Dummy(RunConfig(model="m", output_dir=tmp_path, limit=2))
    path = write_results(b, b.run("", "m"), started=0.0, served_model_name="m")
    record = json.loads(path.read_text())
    assert record["benchmark"] == "dummy"
    assert record["value"] == 0.5 and record["n"] == 2
    assert record["smoke"] is True and record["details"] == {"extra": 1}


def test_vllm_command_has_granite_parsers(tmp_path):
    cmd = VLLMServer(ServerConfig(model="/m", served_model_name="g"), tmp_path / "log").command()
    assert cmd[:3] == ["vllm", "serve", "/m"]
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "auto"


def test_cli_list(capsys):
    assert cli.main(["list", "--suite", "granite42"]) == 0
    out = capsys.readouterr().out
    assert "[x] swebench-verified" in out
    assert "[ ] birdbench" in out
