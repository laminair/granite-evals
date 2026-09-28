import json

import pytest

from sage2_evals import cli, data, registry, suites
from sage2_evals.registry import Benchmark, RunConfig
from sage2_evals.results import write_results
from sage2_evals.sandbox import enroot
from sage2_evals.sandbox.enroot import enroot_uri
from sage2_evals.serving import ServerConfig, VLLMServer, server_env


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


def test_dataset_source_pin_and_override(tmp_path):
    cls = registry.get("swebench-verified")
    pinned = cls(RunConfig(model="m", output_dir=tmp_path))
    assert pinned.dataset_source() == (cls.dataset, cls.dataset_revision)
    assert len(cls.dataset_revision) == 40
    other = cls(RunConfig(model="m", output_dir=tmp_path, dataset="/local/ds"))
    assert other.dataset_source() == ("/local/ds", None)


def test_every_benchmark_pins_its_dataset():
    for bid, cls in registry.all_benchmarks().items():
        assert cls.dataset, bid
        assert cls.dataset_revision or cls.dataset.startswith(("http", "git+")), bid


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


def _granite42(d):
    # Excerpts of ibm-granite/granite-4.2-3b's chat_template.jinja and plugin.
    d.mkdir()
    (d / "chat_template.jinja").write_text("<tool_call>\n<function={{ tool_call.name }}>\n<parameter=")
    (d / "granite_thinking_parser.py").write_text(
        '@ReasoningParserManager.register_module("granite_thinking_parser")\nclass GraniteThinkingParser: ...'
    )
    return d


def test_vllm_command_resolves_granite42_parsers(tmp_path):
    m = _granite42(tmp_path / "m")
    cmd = VLLMServer(ServerConfig(model=str(m), served_model_name="g"), tmp_path / "log").command()
    assert cmd[:3] == ["vllm", "serve", str(m)]
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "granite_thinking_parser"
    assert cmd[cmd.index("--reasoning-parser-plugin") + 1] == str(m / "granite_thinking_parser.py")


def test_vllm_command_parsers_explicit_and_undetected(tmp_path):
    m = _granite42(tmp_path / "m")
    cfg = ServerConfig(model=str(m), served_model_name="g", tool_call_parser="hermes", reasoning_parser="")
    cmd = VLLMServer(cfg, tmp_path / "log").command()
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert "--reasoning-parser" not in cmd
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }}"}')
    cmd = VLLMServer(ServerConfig(model=str(tmp_path / "plain"), served_model_name="g"), tmp_path / "log").command()
    assert "--tool-call-parser" not in cmd and "--reasoning-parser" not in cmd


def test_vllm_env_avoids_jit_sampler(monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    assert server_env()["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    assert server_env()["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_cli_list(capsys):
    assert cli.main(["list", "--suite", "granite42"]) == 0
    out = capsys.readouterr().out
    assert "[x] swebench-verified" in out
    assert "[ ] birdbench" in out
