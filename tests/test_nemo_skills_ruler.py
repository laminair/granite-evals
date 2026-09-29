"""RULER through NeMo-Skills: config, source pinning, per-task args (thinking on:
chat format and budgets; off: ns's text/prefill), gold server, context check, the
think-tag guard, and the ns metrics/ruler_score path (with the nemoskills extra)."""

import hashlib
import json
from pathlib import Path

import pytest

from sage2_evals import registry
from sage2_evals.benchmarks import nemo_skills as nsb
from sage2_evals.benchmarks import nemo_skills_ruler as nsr
from sage2_evals.registry import RunConfig

IDS = {"ruler-128k": 131072, "ruler-64k": 65536}

ARGS = (  # ns prepare --data_format default (thinking off)
    '"++prompt_config=generic/default ++eval_type=ruler ++eval_config.match_type=all '
    '++inference.tokens_to_generate=128 ++start_assistant_response_key=generation ++inference.endpoint_type=text "'
)
CHAT_ARGS = '"++prompt_config=generic/default ++eval_type=ruler ++eval_config.match_type=all "'  # --data_format chat
OFF = {"enable_thinking": "false"}


def bench(bid, tmp_path, **kw):
    kw.setdefault("model", "/models/granite")
    return registry.get(bid)(RunConfig(output_dir=tmp_path, **kw))


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid):
    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark)
    assert (cls.metric, cls.default_repeats, cls.max_seq_length) == ("accuracy", 1, IDS[bid])
    assert cls.extra == "nemoskills" and cls.dataset_revision == nsr.RULER_COMMIT
    assert cls.dataset.startswith("https://github.com/NVIDIA/RULER/tree/")


def test_no_paid_api(tmp_path, monkeypatch):
    from sage2_evals import meter

    monkeypatch.setattr(meter, "metered", lambda *a, **k: pytest.fail("paid API used"))
    b = bench("ruler-128k", tmp_path)
    assert b.judge is None and not b.judge_args


def test_tasks_option(tmp_path):
    assert bench("ruler-64k", tmp_path).tasks() == list(nsr.TASKS) and len(nsr.TASKS) == 13
    assert bench("ruler-64k", tmp_path, options={"tasks": "vt,qa_1"}).tasks() == ["vt", "qa_1"]
    with pytest.raises(SystemExit, match="unknown RULER tasks"):
        bench("ruler-64k", tmp_path, options={"tasks": "vt,nope"}).tasks()


def test_needs_a_tokenizer(tmp_path):
    with pytest.raises(SystemExit, match="tokenizer"):
        bench("ruler-64k", tmp_path, model="none").prepare_data()


def _ruler_src(tmp_path, commit=nsr.RULER_COMMIT):
    root = tmp_path / "ruler"
    j = root / "scripts/data/synthetic/json"
    j.mkdir(parents=True)
    for name in nsr.RULER_JSON_SHA256:
        (j / name).write_text(name)
    (root / "SAGE2_COMMIT").write_text(commit + "\n")
    return root


def test_ruler_source_checked(tmp_path, monkeypatch):
    root = _ruler_src(tmp_path)
    b = bench("ruler-128k", tmp_path, options={"ruler_dir": str(root)})
    with pytest.raises(SystemExit, match="sha256"):
        b.check_ruler_source()
    monkeypatch.setattr(nsr, "RULER_JSON_SHA256", {n: hashlib.sha256(n.encode()).hexdigest() for n in nsr.RULER_JSON_SHA256})
    assert set(b.check_ruler_source()) == set(nsr.RULER_JSON_SHA256)
    (root / "SAGE2_COMMIT").write_text("0" * 40)
    with pytest.raises(SystemExit, match="is not RULER"):
        b.check_ruler_source()


def _setup(tmp_path, tasks=nsr.TASKS, n=nsr.NUM_SAMPLES, args=CHAT_ARGS):
    d = tmp_path / "setup"
    for t in tasks:
        (d / t).mkdir(parents=True)
        (d / t / "__init__.py").write_text(f"METRICS_TYPE = 'ruler'\nGENERATION_ARGS = ({args})\n")
        rows = [
            {"index": i, "question": f"{t} context {i} ... what is the value {i}?", "expected_answer": [f"v{i}", "w"], "length": 10}
            for i in range(n)
        ]
        (d / t / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_task_generation_args_thinking_off_is_ns(tmp_path):
    d = _setup(tmp_path, tasks=["vt"], args=ARGS)
    b = bench("ruler-128k", tmp_path, options=OFF)
    assert (b.data_format(), b.thinking_budget(), b.setup_name()) == ("default", 0, "sage2_131072")
    args = b.task_generation_args(d, "vt")
    assert args[:6] == [
        "++prompt_config=generic/default",
        "++eval_type=ruler",
        "++eval_config.match_type=all",
        "++inference.tokens_to_generate=128",
        "++start_assistant_response_key=generation",
        "++inference.endpoint_type=text",
    ]
    assert "++tokenizer=/models/granite" in args and "++inference.temperature=null" in args
    assert sum(a.startswith("++inference.tokens_to_generate=") for a in args) == 1  # the task's own
    assert args[-1] == "++chat_template_kwargs.enable_thinking=false"
    assert b.required_context() == 131072 and b.departures() == []
    b = bench("ruler-128k", tmp_path, options={"max_tokens": "64", **OFF, "tokenizer": "/tok"})
    args = b.task_generation_args(d, "vt")
    assert args[-2:] == ["++inference.tokens_to_generate=64", "++chat_template_kwargs.enable_thinking=false"]
    assert "++tokenizer=/tok" in args


def test_task_generation_args_thinking_on(tmp_path):
    d = _setup(tmp_path, tasks=["vt", "niah_single_1", "qa_2"])
    b = bench("ruler-64k", tmp_path)  # enable_thinking unset: the chat template's default (on)
    assert (b.thinking(), b.data_format(), b.setup_name()) == (True, "chat", "sage2_65536_chat")
    args = b.task_generation_args(d, "vt")
    assert not any(a.startswith("++start_assistant_response_key") for a in args)  # no answer-prefix prefill
    assert not any("enable_thinking" in a for a in args)
    assert args[-2:] == ["++inference.endpoint_type=chat", f"++inference.tokens_to_generate={nsr.DEFAULT_THINKING_BUDGET + 30}"]
    budgets = {t: b.tokens_to_generate(t) for t in ("vt", "niah_single_1", "qa_2")}
    assert budgets == {t: nsr.DEFAULT_THINKING_BUDGET + n for t, n in (("vt", 30), ("niah_single_1", 128), ("qa_2", 32))}
    b = bench("ruler-64k", tmp_path, options={"thinking_budget": "1000", "enable_thinking": "true"})
    assert b.task_generation_args(d, "niah_single_1")[-2:] == [
        "++inference.tokens_to_generate=1128",
        "++chat_template_kwargs.enable_thinking=true",
    ]
    assert b.required_context() == 65536 + 1000
    assert len(b.departures()) == 1 and "thinking_budget 1000" in b.departures()[0]
    # max_tokens still replaces every task's budget
    b = bench("ruler-64k", tmp_path, options={"max_tokens": "4096"})
    assert b.tokens_to_generate("qa_2") == 4096 and b.required_context() == 65536 + 4096 - 30  # vt: the smallest answer budget


def test_data_format_must_match_thinking(tmp_path):
    text, chat = _setup(tmp_path / "t", tasks=["vt"], args=ARGS), _setup(tmp_path / "c", tasks=["vt"])
    with pytest.raises(SystemExit, match="not ns's 'chat' RULER format"):
        bench("ruler-64k", tmp_path).task_generation_args(text, "vt")
    with pytest.raises(SystemExit, match="not ns's 'default' RULER format"):
        bench("ruler-64k", tmp_path, options=OFF).task_generation_args(chat, "vt")


def test_sample_length(tmp_path):
    b = bench("ruler-128k", tmp_path, options={"sample_length": "114688"})
    assert (b.sample_length(), b.setup_name(), b.required_context()) == (114688, "sage2_114688_chat", 114688 + nsr.DEFAULT_THINKING_BUDGET)
    assert any("114688 tokens, not 131072" in d for d in b.departures())
    with pytest.raises(SystemExit, match="sample_length"):
        bench("ruler-64k", tmp_path, options={"sample_length": "70000"}).sample_length()


def test_prepare_passes_format_and_length(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(nsr.RulerBenchmark, "check_ruler_source", lambda self: {})
    monkeypatch.setattr(nsr.RulerBenchmark, "_task_files", lambda self, d, tasks: {"task_files": {}})
    monkeypatch.setattr(nsr.RulerBenchmark, "_run_prepare", lambda self, ddir, tasks: calls.append(self.setup_name()) or (ddir / self.setup_name()).mkdir(parents=True))
    for opts in ({}, OFF, {"sample_length": "60000"}):
        _, prov = bench("ruler-64k", tmp_path, options={"tasks": "vt", **opts}).prepare_data()
        assert prov["spec"]["data_format"] == ("default" if opts == OFF else "chat")
        assert prov["spec"]["max_seq_length"] == int(opts.get("sample_length", 65536))
    assert calls == ["sage2_65536_chat", "sage2_65536", "sage2_60000_chat"]


def test_task_files_must_be_complete(tmp_path):
    d = _setup(tmp_path, tasks=["vt"], n=3)
    with pytest.raises(RuntimeError, match="expected 100"):
        bench("ruler-128k", tmp_path)._task_files(d, ["vt"])


def test_gold_server_matches_by_context_tail():
    rows = [
        {"question": "x" * 5000 + " needle A?", "expected_answer": ["1234"]},
        {"question": "x" * 5000 + " needle B?", "expected_answer": ["alpha", "beta"]},
    ]
    s = nsr.RulerGoldServer(rows)
    prompt = "<|start|>" + rows[1]["question"] + "<|end|><|assistant|><think>\nAnswer:"
    assert s.lookup(prompt) == "alpha beta"
    assert s.lookup("unrelated") == ""


class _Models(nsb._Server):
    def __init__(self, n):
        self.n = n

    def handle(self, method, path, headers, body):
        return nsb._json_response({"data": [{"id": "served", "max_model_len": self.n}]})


def test_context_check(tmp_path):
    b = bench("ruler-128k", tmp_path, options=OFF)
    with _Models(131072) as s:
        assert b.check_context(s.base_url, "served") == 131072
    with _Models(32768) as s, pytest.raises(SystemExit, match="--max-model-len 131072$"):
        b.check_context(s.base_url, "served")
    assert b.check_context("http://127.0.0.1:9/v1", "served") is None  # unknown: not fatal


def test_context_check_thinking(tmp_path):
    """Thinking needs the sample plus its budget; at the model's limit the message
    names the options instead of picking one."""
    b = bench("ruler-128k", tmp_path)
    need = 131072 + nsr.DEFAULT_THINKING_BUDGET
    with _Models(131072) as s, pytest.raises(SystemExit) as e:
        b.check_context(s.base_url, "served")
    msg = str(e.value)
    assert f"< {need}" in msg and f"--max-model-len {need}" in msg
    assert f"sample_length={131072 - nsr.DEFAULT_THINKING_BUDGET}" in msg and "enable_thinking=false" in msg
    with _Models(131072) as s:
        assert bench("ruler-64k", tmp_path).check_context(s.base_url, "served") == 131072
        small = {"sample_length": str(131072 - nsr.DEFAULT_THINKING_BUDGET)}
        assert bench("ruler-128k", tmp_path, options=small).check_context(s.base_url, "served") == 131072


def test_context_checked_before_data(tmp_path, monkeypatch):
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: pytest.fail("built data first"))
    with _Models(65536) as s, pytest.raises(SystemExit, match="max_model_len=65536"):
        bench("ruler-64k", tmp_path).run(s.base_url, "served")


def test_think_tags_in_generation_fail(tmp_path):
    b = bench("ruler-64k", tmp_path)
    f = tmp_path / "out.jsonl"
    f.write_text(json.dumps({"index": 0, "generation": "The value is 12.", "reasoning_content": "needle 12"}) + "\n")
    b.check_generations(f)
    for gen in ("<think>needle 12</think>The value is 12.", "needle 12</think>12"):
        f.write_text(json.dumps({"index": 3, "generation": gen}) + "\n")
        with pytest.raises(RuntimeError, match=r"think tags .* samples \[3\]"):
            b.check_generations(f)


def test_reset_if_input_changed(tmp_path):
    d = tmp_path / "vt"
    d.mkdir()
    (d / "input.jsonl").write_text("a\n")
    nsr._reset_task_if_input_changed(d)
    (d / "output-rs0.jsonl").write_text("{}\n")
    nsr._reset_task_if_input_changed(d)
    assert (d / "output-rs0.jsonl").exists()
    (d / "input.jsonl").write_text("b\n")
    nsr._reset_task_if_input_changed(d)
    assert not (d / "output-rs0.jsonl").exists()


# -- with the harness ---------------------------------------------------------


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


def _fake_generate(correct_every, cut_every=0):
    """Rows ``index % cut_every == 1`` stop at the length cap inside their thinking."""

    def gen(self, setup_dir, task, input_file, base_url, served, k):
        rows = [json.loads(line) for line in Path(input_file).read_text().splitlines()]
        out = Path(input_file).parent / f"output-rs{k}.jsonl"
        lines = []
        for r in rows:
            cut = bool(cut_every) and r["index"] % cut_every == 1
            g = {"generation": "", "reasoning_content": "v1 ...", "finish_reason": "length", "num_generated_tokens": 100}
            if not cut:
                g = {"generation": "g", "reasoning_content": "r", "finish_reason": "stop", "num_generated_tokens": 10 + r["index"]}
            lines.append(json.dumps({**r, **g, "is_correct": not cut and r["index"] % correct_every == 0}) + "\n")
        out.write_text("".join(lines))

    return gen


def test_run_scores_with_ns_ruler_score(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path)
    b = bench("ruler-128k", tmp_path / "out", limit=4, options={"answers": "gold"})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(2))
    r = b.run("", "")
    assert r["n"] == 13 * 4 and r["value"] == pytest.approx(0.5)  # indices 0..3: 0 and 2 correct
    assert r["per_task"]["vt"] == {"accuracy": pytest.approx(0.5), "n": 4, "statuses": {"stop": 4}}
    assert not r["partial_task_set"] and r["ns_metric"]["score"] == "ruler_score"
    assert r["statuses"] == {"stop": 52}
    th = r["thinking"]
    assert (th["enabled"], th["budget"], th["data_format"], th["scoring_source"]) == (
        True,
        nsr.DEFAULT_THINKING_BUDGET,
        "chat",
        nsr.SCORING_SOURCE,
    )
    assert th["per_task"]["vt"]["tokens_to_generate"] == nsr.DEFAULT_THINKING_BUDGET + 30
    assert th["per_task"]["vt"]["generated_tokens"] == {"p50": 12, "p95": 13, "max": 13}
    assert r["sample_length"] == 131072 and r["required_context"] == 131072 + nsr.DEFAULT_THINKING_BUDGET
    assert len(r["departures"]) == 1 and r["generation_args"]["vt"][-1].startswith("++inference.tokens_to_generate=")


def test_run_cut_off_thinking_scores_zero(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path, tasks=["vt"])
    b = bench("ruler-64k", tmp_path / "out", limit=4, options={"answers": "gold", "tasks": "vt"})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(1, cut_every=2))
    r = b.run("", "")
    assert r["value"] == pytest.approx(0.5)  # indices 1 and 3 cut off inside the thinking
    vt = r["thinking"]["per_task"]["vt"]
    assert (vt["length_no_content"], vt["length_with_content"], vt["with_reasoning"]) == (2, 0, 4)


def test_run_thinking_off_record(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path, tasks=["vt"], args=ARGS)
    b = bench("ruler-64k", tmp_path / "out", limit=2, options={"answers": "gold", "tasks": "vt", **OFF})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(1))
    r = b.run("", "")
    assert r["thinking"]["enabled"] is False and r["thinking"]["budget"] == 0 and r["departures"] == []
    assert r["thinking"]["per_task"]["vt"]["tokens_to_generate"] == 30


def test_run_task_subset(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path, tasks=["vt", "cwe"])
    b = bench("ruler-64k", tmp_path / "out", limit=3, options={"answers": "gold", "tasks": "vt,cwe"})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(1))
    r = b.run("", "")
    assert r["value"] == 1.0 and r["partial_task_set"] and r["tasks"] == ["vt", "cwe"]


def test_ns_ruler_evaluator_accepts_gold_answers(ns, tmp_path):
    """The gold server's answer passes ns's own RULER match (all / part)."""
    from nemo_skills.evaluation.evaluator import evaluate

    rows = [{"generation": "alpha beta", "expected_answer": ["alpha", "beta"]}, {"generation": "x", "expected_answer": ["alpha"]}]
    f = tmp_path / "out.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    evaluate("ruler", {"input_file": str(f), "match_type": "all"})
    got = [json.loads(line)["is_correct"] for line in f.read_text().splitlines()]
    assert got == [1.0, 0.0]


# Imported only inside functions that the hf tokenizer path never calls.
_RULER_LAZY = {"nemo", "google", "tiktoken", "manifest_utils", "constants", "template", "tokenizer"}


@pytest.mark.skipif(not Path(nsr.RULER_DIR, "scripts", "data").is_dir(), reason="no image RULER checkout")
def test_image_has_ruler_generator_imports():
    """In the image: every module RULER's generators import at the top is installed
    (the job's prepare runs them with this interpreter)."""
    import ast
    import importlib.util
    import sys

    root = Path(nsr.RULER_DIR, "scripts", "data")
    local = {p.stem for p in root.rglob("*.py")}
    missing = {}
    for p in root.rglob("*.py"):
        if p.name.startswith("download_"):  # build-time only (docker/extras/nemoskills.sh)
            continue
        for node in ast.parse(p.read_text()).body:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top in local or top in _RULER_LAZY or top in sys.stdlib_module_names:
                    continue
                if importlib.util.find_spec(top) is None:
                    missing.setdefault(top, []).append(str(p.relative_to(root)))
    assert not missing, missing
