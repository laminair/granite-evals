"""NeMo-Skills adapter: config, pinning, judge metering (always), and ns's own
evaluate/metrics plus a gold end-to-end run (with the nemoskills extra)."""

import json

import httpx
import pytest

from sage2_evals import registry
from sage2_evals.benchmarks import nemo_skills as nsb
from sage2_evals.registry import RunConfig

IDS = {
    "aime25": ("pass@1[avg-of-4] symbolic correct", 4),
    "hmmt-feb25": ("pass@1[avg-of-4] symbolic correct", 4),
    "gpqa": ("pass@1[avg-of-2] symbolic correct", 2),
    "mmlu-pro": ("symbolic correct", 1),
    "arena-hard-v2": ("win rate", 1),
}


def bench(bid, tmp_path, **kw):
    return registry.get(bid)(RunConfig(model="m", output_dir=tmp_path, **kw))


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid):
    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark)
    assert (cls.metric, cls.default_repeats) == IDS[bid]
    assert cls.extra == "nemoskills" and "nemo-skills" in cls.harness_packages
    assert len(cls.dataset_revision) == 40


def test_pins_include_primary_dataset(tmp_path):
    b = bench("gpqa", tmp_path)
    assert b.pins() == {"Idavidrein/gpqa": "83022cefff930aea54f654c0b282e74b9eeda5c6"}
    assert bench("aime25", tmp_path).pins() == {}  # bundled in the pinned ns commit
    urls = registry.get("arena-hard-v2").pinned_urls
    assert len(urls) == 3
    for orig, (pinned, sha) in urls.items():
        assert "/main/" in orig and nsb._ARENA_HARD_COMMIT in pinned and len(sha) == 64


def test_sampling_defaults_to_generation_config(tmp_path):
    assert bench("aime25", tmp_path).sampling() == {"temperature": None, "top_p": None, "top_k": -1, "max_tokens": None}
    b = bench("aime25", tmp_path, options={"temperature": "0.6", "max_tokens": "1000"})
    assert b.sampling()["temperature"] == 0.6 and b.sampling()["max_tokens"] == 1000
    assert nsb._hydra(None) == "null"


def test_passthrough_options():
    opts = {"ns.chat_template_kwargs.enable_thinking": "false", "judge.x": "1", "temperature": "1"}
    assert nsb._passthrough(opts, "ns.") == ["++chat_template_kwargs.enable_thinking=false"]
    assert nsb._passthrough(opts, "judge.") == ["++x=1"]


def test_aggregation_follows_repeats(tmp_path):
    assert bench("aime25", tmp_path).aggregation() == "pass@1[avg-of-4]"
    assert bench("aime25", tmp_path, repeats=1).aggregation() == "pass@1"


def test_gold_disables_server(tmp_path):
    assert bench("aime25", tmp_path).needs_server()
    assert not bench("aime25", tmp_path, options={"answers": "gold"}).needs_server()


def test_gold_generation_formats(tmp_path):
    row = {"expected_answer": "C", "problem": "p"}
    assert bench("gpqa", tmp_path).gold_generation(row) == "Answer: C"
    assert bench("mmlu-pro", tmp_path).gold_generation(row) == "Answer: C"
    assert bench("aime25", tmp_path).gold_generation({"expected_answer": "70"}) == "\\boxed{70}"


def test_gold_server_matches_longest_problem():
    rows = [{"problem": "Find x.", "a": "1"}, {"problem": "Find x. Then find y.", "a": "2"}]
    with nsb._GoldServer(rows, lambda r: r["a"]) as s:
        r = httpx.post(
            s.base_url + "/chat/completions",
            json={"model": "gold", "messages": [{"role": "user", "content": "Solve: Find x. Then find y. Box it."}]},
        )
        assert r.json()["choices"][0]["message"]["content"] == "2"
        assert httpx.get(s.base_url + "/models").json()["data"][0]["id"] == "gold"


def test_usage_fields_openai_and_anthropic_style():
    assert nsb.usage_fields(
        {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}
    ) == {"prompt_tokens": 10, "completion_tokens": 3, "cached_tokens": 4, "cache_creation_tokens": 0}
    assert nsb.usage_fields(
        {"input_tokens": 7, "output_tokens": 2, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 1}
    ) == {"prompt_tokens": 7, "completion_tokens": 2, "cached_tokens": 5, "cache_creation_tokens": 1}


class _Upstream(nsb._Server):
    def __init__(self):
        self.auth = []

    def handle(self, method, path, headers, body):
        self.auth.append((path, {k.lower(): v for k, v in headers.items()}.get("authorization")))
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 60}}
        return nsb._json_response({"choices": [{"message": {"content": "[[A>B]]"}}], "usage": usage, "model": "j"})


def test_metering_proxy_injects_key_and_totals_usage(tmp_path):
    log = tmp_path / "judge" / "usage.jsonl"
    with _Upstream() as up, nsb._MeteringProxy(up.base_url, "sekret", log) as proxy:
        for _ in range(3):
            r = httpx.post(proxy.base_url + "/chat/completions", json={"model": "j"}, headers={"Authorization": "x"})
            assert r.status_code == 200
    assert up.auth == [("/v1/chat/completions", "Bearer sekret")] * 3
    assert "sekret" not in log.read_text()
    u = nsb.judge_usage(log, n_examples=3)
    assert (u["requests"], u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"]) == (3, 300, 60, 180)
    assert u["per_example"]["prompt_tokens"] == 100 and u["failed_requests"] == 0


def test_judge_endpoint_defaults_and_self(tmp_path):
    ep = bench("arena-hard-v2", tmp_path).judge_endpoint("http://v/v1", "g")
    assert ep == {
        "base_url": nsb.JUDGE_BASE_URL,
        "model": "aws/claude-sonnet-5",
        "api_key_env": "SAGE2_JUDGE_API_KEY",
        "is_self": False,
    }
    ep = bench("arena-hard-v2", tmp_path, options={"judge_model": "self"}).judge_endpoint("http://v/v1", "g")
    assert ep["base_url"] == "http://v/v1" and ep["model"] == "g" and ep["is_self"]


def test_statuses_count_soft_failures(tmp_path):
    f = tmp_path / "o.jsonl"
    f.write_text('{"finish_reason": "stop"}\n{"finish_reason": "error"}\n{"error": "x"}\n')
    assert nsb._statuses(f) == {"stop": 1, "error": 2}


# -- with the harness ---------------------------------------------------------


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


def test_module_generation_args(ns, tmp_path):
    assert bench("aime25", tmp_path).generation_args([])[:2] == ["++prompt_config=generic/math", "++eval_type=math"]
    gpqa = bench("gpqa", tmp_path)
    assert gpqa.split() == "diamond" and gpqa.metrics_type() == "multichoice"
    assert "++prompt_config=eval/aai/mcq-4choices" in gpqa.generation_args([])
    assert bench("mmlu-pro", tmp_path).split() == "test"
    arena = bench("arena-hard-v2", tmp_path)
    assert arena.uses_judge() and not bench("aime25", tmp_path).uses_judge()
    assert arena.judge_module() == "nemo_skills.inference.eval.arena_judge"
    assert "++inference.top_p=null" in arena.judge_overrides()


def _math_rows(tmp_path, name, correct):
    f = tmp_path / name
    rows = [
        {"problem": f"p{i}", "expected_answer": "1", "predicted_answer": "1", "symbolic_correct": c, "num_generated_tokens": 5}
        for i, c in enumerate(correct)
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


def test_ns_math_metrics_map_to_value(ns, tmp_path):
    b = bench("aime25", tmp_path, repeats=2)
    files = [_math_rows(tmp_path, "output-rs0.jsonl", [True, False]), _math_rows(tmp_path, "output-rs1.jsonl", [True, True])]
    m = b.compute_metrics(files)
    assert m["_all_"][b.aggregation()]["symbolic_correct"] == pytest.approx(75.0)
    assert m["_all_"]["pass@2"]["symbolic_correct"] == pytest.approx(100.0)


def _arena_file(tmp_path, verdicts):
    f = tmp_path / "judged.jsonl"
    rows = [
        {"judgement-gen-base": f"[[{a}]]", "judgement-base-gen": f"[[{b}]]", "category": "hard_prompt", "generation": "x"}
        for a, b in verdicts
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


def test_arena_one_sided_sample_falls_back(ns, tmp_path):
    b = bench("arena-hard-v2", tmp_path)
    m = b.compute_metrics([_arena_file(tmp_path, [("A>B", "B>A")] * 3)])  # candidate wins every game
    assert m["_all_"]["pass@1"]["score"] == 100.0
    assert "one-sided" in m["win_rate_method"]


def test_arena_mixed_sample_uses_ns_fit(ns, tmp_path):
    b = bench("arena-hard-v2", tmp_path)
    verdicts = [("A>B", "B>A"), ("B>A", "A>B"), ("A=B", "A=B"), ("A>>B", "B>A"), ("B>>A", "A>B")]
    m = b.compute_metrics([_arena_file(tmp_path, verdicts)])
    assert 0 < m["_all_"]["pass@1"]["score"] < 100
    assert m["win_rate_method"] == "bootstrapped Bradley-Terry (ns)"


def test_aime25_gold_end_to_end_and_resume(ns, tmp_path):
    """Reference answers through ns's real prompt -> generate -> math grading."""
    b = bench("aime25", tmp_path, limit=3, repeats=2, workers=3, options={"answers": "gold"})
    r = b.run("", "")
    assert (r["value"], r["n"]) == (1.0, 3)
    assert r["data_provenance"]["sha256"] == nsb.AIME25.prepared_sha256
    assert r["statuses"] == {"stop": 6}
    gen = tmp_path / "generation" / "output-rs0.jsonl"
    first = [json.loads(line) for line in gen.read_text().splitlines()]
    assert [row["id"] for row in first] == ["aime25-0", "aime25-1", "aime25-2"]
    assert all(row["symbolic_correct"] for row in first)
    mtime = gen.stat().st_mtime_ns
    assert b.run("", "")["value"] == 1.0 and gen.stat().st_mtime_ns == mtime  # resumed, not regenerated
    b2 = bench("aime25", tmp_path, limit=2, repeats=1, options={"answers": "gold"})
    assert b2.run("", "")["n"] == 2 and len(gen.read_text().splitlines()) == 2  # new input: regenerated


def test_paid_judge_goes_through_meter(tmp_path, monkeypatch):
    from sage2_evals import meter

    calls = []
    monkeypatch.setattr(meter, "metered", lambda url, role: calls.append((url, role)) or "http://127.0.0.1:9/metered")
    b = bench("arena-hard-v2", tmp_path)
    assert b.judge_upstream(b.judge_endpoint("http://v/v1", "g")) == "http://127.0.0.1:9/metered"
    assert calls == [(nsb.JUDGE_BASE_URL, "judge")]
    # the CLI has metered a judge_base_url option already; self-judging is unpaid
    opt = bench("arena-hard-v2", tmp_path, options={"judge_base_url": "http://127.0.0.1:8/v1"})
    assert opt.judge_upstream(opt.judge_endpoint("", "")) == "http://127.0.0.1:8/v1"
    me = bench("arena-hard-v2", tmp_path, options={"judge_model": "self"})
    assert me.judge_upstream(me.judge_endpoint("http://v/v1", "g")) == "http://v/v1"
    assert len(calls) == 1


def test_judge_calls_reach_the_meter_proxy(tmp_path, monkeypatch):
    """End to end through the real sage2 meter: judge proxy -> meter -> upstream."""
    from sage2_evals import meter

    monkeypatch.setenv("SAGE2_SPEND_LEDGER", str(tmp_path / "ledger.jsonl"))
    with _Upstream() as up, meter.Meters({}, {"benchmark": "arena-hard-v2"}) as meters:
        monkeypatch.setattr(nsb, "JUDGE_BASE_URL", up.base_url)
        b = bench("arena-hard-v2", tmp_path)
        upstream = b.judge_upstream(b.judge_endpoint("", ""))
        assert upstream.startswith("http://127.0.0.1:") and upstream != up.base_url
        with nsb._MeteringProxy(upstream, "k", tmp_path / "usage.jsonl") as proxy:
            httpx.post(proxy.base_url + "/chat/completions", json={"model": "aws/claude-sonnet-5"})
        assert meters.summary()["calls"] == 1
    assert up.auth == [("/v1/chat/completions", "Bearer k")]
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1


class _Broke(nsb._Server):
    def handle(self, method, path, headers, body):
        return nsb._json_response({"error": {"type": "budget_exhausted"}}, 402)


def test_proxy_flags_budget_exhausted(tmp_path):
    with _Broke() as up, nsb._MeteringProxy(up.base_url, "", tmp_path / "u.jsonl") as proxy:
        assert httpx.post(proxy.base_url + "/chat/completions", json={}).status_code == 402
        assert proxy.budget_exhausted
