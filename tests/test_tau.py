"""τ³-bench: scoring, aggregation, endpoints and metering, resume, limit.

The end-to-end tests run the real harness (tau2) on its pinned task data against
a local fake OpenAI-compatible server that plays agent, user simulator and
judge (no GPU, no network). They need the `tau` extra and the data (baked into
the image at /opt/tau2-data, or SAGE2_TAU2_TEST_DATA / the fetch cache).
"""

import importlib.util
import json
import os
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sage2_evals import meter, registry
from sage2_evals.benchmarks import tau
from sage2_evals.registry import RunConfig

SUITE = {
    "tau3-bench": "pass@1 (avg of 3)",
    "tau3-airline": "pass@1",
    "tau3-retail": "pass@1",
    "tau3-telecom": "pass@1",
    "tau3-banking-knowledge": "pass@1",
}


def _bench(bid, tmp_path, **kw):
    options = kw.pop("options", {})
    return registry.get(bid)(RunConfig(model="m", output_dir=tmp_path, options=options, **kw))


# -- pure logic ----------------------------------------------------------------


def test_ids_metrics_and_pins():
    for bid, metric in SUITE.items():
        cls = registry.get(bid)
        assert cls.metric == metric
        assert cls.dataset == tau.TAU2_REPO and cls.dataset_revision == tau.TAU2_COMMIT
        assert cls.extra == "tau" and cls.default_repeats == 4
    assert registry.get("tau3-bench").domains == ("airline", "retail", "telecom")
    assert registry.get("tau3-banking-knowledge").domains == ("banking_knowledge",)


def test_trial_seeds_match_harness():
    random.seed(300)
    expected = [random.randint(0, 1000000) for _ in range(4)]
    assert tau.trial_seeds(300, 4) == expected


def test_pass_hat_k_and_summary():
    assert tau.pass_hat_k(4, 2, 1) == 0.5
    assert tau.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    recs = [
        {"task_id": "a", "trial": 0, "reward": 1.0, "status": "success"},
        {"task_id": "a", "trial": 1, "reward": 0.0, "status": "fail"},
        {"task_id": "b", "trial": 0, "reward": 1.0, "status": "success"},
        {"task_id": "b", "trial": 1, "reward": 1.0, "status": "success"},
    ]
    s = tau.domain_summary(recs, 2)
    assert s["pass_hat_1"] == 0.75 and s["pass_hat_2"] == 0.5
    assert s["per_trial_pass_1"] == [1.0, 0.5]
    assert s["statuses"] == {"fail": 1, "success": 3}


def test_usage_counts_cached_tokens():
    class M:
        usage = {"prompt_tokens": 100, "completion_tokens": 10}
        raw_data = {"usage": {"prompt_tokens_details": {"cached_tokens": 60}, "cache_creation_input_tokens": 5}}

    acc = tau.empty_usage()
    tau.add_usage(acc, M())
    tau.add_usage(acc, M())
    assert acc == {"calls": 2, "prompt_tokens": 200, "completion_tokens": 20, "cached_tokens": 120,
                   "cache_creation_tokens": 10}


def test_task_filename_is_safe_and_unique():
    a = tau.task_filename("[mms_issue]a|b[PERSONA:Hard]")
    b = tau.task_filename("[mms_issue]a|b[PERSONA:None]")
    assert "/" not in a and "|" not in a and a != b


class _T:
    evaluation_criteria = None

    def __init__(self, i):
        self.id = str(i)


def test_aggregate_is_mean_of_core_domains_and_limit(tmp_path, monkeypatch):
    b = _bench("tau3-bench", tmp_path, limit=2, repeats=2, options={"user_model": "self"})
    seen = []

    def load_tasks():
        return {d: [_T(i) for i in range(2)] for d in b.domains}, tau.TAU2_REPO, tau.TAU2_COMMIT

    rewards = {"airline": 1.0, "retail": 0.0, "telecom": 1.0}

    def sim(domain, task, trial, seed, agent, user, judge):
        seen.append((domain, task.id, trial, seed))
        r = rewards[domain] if task.id == "0" else 0.0
        return {"domain": domain, "task_id": task.id, "trial": trial, "reward": r,
                "status": "success" if r else "fail", "usage": tau._zero_usage()}

    monkeypatch.setattr(b, "load_tasks", load_tasks)
    monkeypatch.setattr(b, "_simulation", sim)
    monkeypatch.setattr(tau, "install_hooks", lambda: None)
    out = b.run("http://vllm/v1", "served")
    # per domain pass^1: airline .5, retail 0, telecom .5 -> mean 1/3
    assert out["value"] == pytest.approx(1 / 3)
    assert set(out["domains"]) == {"airline", "retail", "telecom"}
    assert out["n"] == 6 and len(seen) == 12
    assert {s[3] for s in seen} == set(tau.trial_seeds(300, 2))


# -- endpoints and metering --------------------------------------------------


def test_self_endpoints_use_the_served_model(tmp_path):
    b = _bench("tau3-retail", tmp_path, options={"user_model": "self", "judge_model": "self"})
    u = b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)
    assert u.model == "hosted_vllm/granite" and u.kwargs["api_base"] == "http://vllm/v1" and u.is_self


def test_paid_endpoint_needs_its_key(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE2_USER_API_KEY", raising=False)
    b = _bench("tau3-airline", tmp_path)
    with pytest.raises(SystemExit, match="SAGE2_USER_API_KEY"):
        b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)


def test_default_gateway_goes_through_the_meter(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGE2_USER_API_KEY", "k")
    monkeypatch.setenv("SAGE2_JUDGE_API_KEY", "k")
    b = _bench("tau3-retail", tmp_path)
    with meter.Meters(b.config.options, {}) as meters:
        u = b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)
        j = b.llm_endpoint("judge", "http://vllm/v1", "granite", temperature=0.0)
        proxy = meter.metered(tau.DEFAULT_GATEWAY, role="user")
        assert u.kwargs["api_base"] == proxy == j.kwargs["api_base"]
        assert proxy.startswith("http://127.0.0.1:") and len(meters._meters) == 1
    assert u.model == "openai/aws/claude-sonnet-5" and u.record()["base_url"] == tau.DEFAULT_GATEWAY


def test_given_base_url_is_metered_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGE2_USER_API_KEY", "k")
    b = _bench("tau3-airline", tmp_path, options={"user_base_url": "https://gw.example/v1"})
    with meter.Meters(b.config.options, {}) as meters:
        u = b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)
        assert u.kwargs["api_base"].startswith("http://127.0.0.1:")
        assert list(meters._meters) == ["https://gw.example/v1"]


# -- end to end with the harness -----------------------------------------------


def _data_dir():
    cands = [os.environ.get("SAGE2_TAU2_TEST_DATA", ""), tau.BAKED_DATA_ROOT / tau.TAU2_COMMIT / "data",
             Path(os.environ.get("SAGE2_TAU2_DATA_CACHE", Path.home() / ".cache/sage2/tau2-data"))
             / tau.TAU2_COMMIT / "data"]
    for c in cands:
        if c and (Path(c) / "tau2" / "domains").is_dir():
            return Path(c)
    return None


harness = pytest.mark.skipif(
    importlib.util.find_spec("tau2") is None or _data_dir() is None,
    reason="needs the tau extra and the pinned tau2-bench data",
)


class _FakeLLM(BaseHTTPRequestHandler):
    """Agent (served model), user simulator and judge in one OpenAI-compatible
    server. The agent looks up a user once, then answers; the user asks once,
    then stops; the judge answers in a markdown fence."""

    protocol_version = "HTTP/1.1"
    calls: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeLLM.calls.append({"path": self.path, "model": body["model"], "auth": self.headers.get("Authorization")})
        msgs = body["messages"]
        model = body["model"]
        msg = {"role": "assistant", "content": "ok"}
        if model == "granite":
            if msgs[-1]["role"] == "user" and not any(m["role"] == "tool" for m in msgs):
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "get_user_details", "arguments": json.dumps({"user_id": "nobody"})}}]}
            else:
                msg["content"] = "I could not find you. Anything else?"
        elif model == "judge-model":
            msg["content"] = '```json\n{"results": []}\n```'
        else:  # user simulator: flipped roles, so its own turns are "assistant"
            turns = sum(m["role"] == "assistant" for m in msgs)
            msg["content"] = "Hi, please change my flight." if turns == 0 else "###STOP###"
        out = json.dumps({
            "id": "x", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                      "prompt_tokens_details": {"cached_tokens": 40}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-litellm-response-cost", "0.001")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def fake_llm():
    _FakeLLM.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@harness
def test_airline_end_to_end_metered_and_resumable(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("SAGE2_USER_API_KEY", "user-key")
    monkeypatch.setenv("SAGE2_SPEND_LEDGER", str(tmp_path / "ledger.jsonl"))
    opts = {"user_base_url": fake_llm, "user_model": "aws/claude-sonnet-5", "max_retries": "0"}
    b = _bench("tau3-airline", tmp_path / "run", limit=1, repeats=1, dataset=str(_data_dir()), options=dict(opts))
    with meter.Meters(b.config.options, {"benchmark": b.id}) as meters:
        out = b.run(fake_llm, "granite")
        spend = meters.summary()
    rec = out["domains"]["airline"]
    # graded by the harness on the final DB; the fake agent changes nothing
    assert rec["n_simulations"] == 1 and sum(rec["statuses"].values()) == 1
    assert set(rec["statuses"]) <= {"success", "fail"}
    assert rec["termination_reasons"] == {"user_stop": 1}
    # the user simulator went through the meter, with its key; the agent did not
    user_calls = [c for c in _FakeLLM.calls if c["model"] == "aws/claude-sonnet-5"]
    assert len(user_calls) == 2 and all(c["auth"] == "Bearer user-key" for c in user_calls)
    assert spend["calls"] == 2 and spend["usd"] == pytest.approx(0.002)
    assert out["user_sim_usage"] == {"calls": 2, "prompt_tokens": 200, "completion_tokens": 20,
                                     "cached_tokens": 80, "cache_creation_tokens": 0}
    assert out["agent_usage"]["calls"] == 2 and out["judge"] is None
    saved = next((tmp_path / "run" / "airline" / "trial-0").glob("*.json"))
    sim = json.loads(saved.read_text())["simulation"]
    assert any(tc["name"] == "get_user_details" for m in sim["messages"] for tc in (m.get("tool_calls") or []))

    # resume: nothing is simulated again
    n = len(_FakeLLM.calls)
    b2 = _bench("tau3-airline", tmp_path / "run", limit=1, repeats=1, dataset=str(_data_dir()), options=dict(opts))
    assert b2.run(fake_llm, "granite")["domains"]["airline"]["pass_hat_1"] == rec["pass_hat_1"]
    assert len(_FakeLLM.calls) == n


@harness
def test_retail_nl_assertions_use_the_configured_judge(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("SAGE2_JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("SAGE2_USER_API_KEY", "user-key")
    opts = {"user_base_url": fake_llm, "user_model": "user-model", "judge_base_url": fake_llm,
            "judge_model": "judge-model", "max_retries": "0"}
    # the judge runs only for tasks graded on NL assertions: pick the first one
    everything = _bench("tau3-retail", tmp_path, dataset=str(_data_dir()), options=dict(opts)).load_tasks()[0]
    task = next(t for t in everything["retail"]
                if "NL_ASSERTION" in tau._basis(t) and t.evaluation_criteria.nl_assertions)
    opts["tasks"] = "^" + re.escape(task.id) + "$"
    b = _bench("tau3-retail", tmp_path, limit=1, repeats=1, dataset=str(_data_dir()), options=opts)
    out = b.run(fake_llm, "granite")
    assert out["tasks"] == {"retail": [task.id]}
    judge_calls = [c for c in _FakeLLM.calls if c["model"] == "judge-model"]
    assert len(judge_calls) == 1 and judge_calls[0]["auth"] == "Bearer judge-key"
    assert out["judge_usage"]["calls"] == 1 and not out["judge_is_self"]
    assert out["judge"]["model"] == "judge-model"
    saved = json.loads(next((tmp_path / "retail" / "trial-0").glob("*.json")).read_text())
    assert saved["reward_info"]["nl_assertions"] == []  # fenced JSON parsed


@harness
def test_budget_exhausted_stops_paid_calls(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("SAGE2_USER_API_KEY", "k")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"cost_usd": 100.0}) + "\n")
    monkeypatch.setenv("SAGE2_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("SAGE2_SPEND_BUDGET_USD", "50")
    opts = {"user_base_url": fake_llm, "max_retries": "0"}
    b = _bench("tau3-airline", tmp_path / "run", limit=2, repeats=1, dataset=str(_data_dir()), options=opts)
    with meter.Meters(b.config.options, {}):
        out = b.run(fake_llm, "granite")
    assert out["budget_exhausted"]
    assert out["domains"]["airline"]["statuses"] == {"error:budget_exhausted": 2}
    assert not [c for c in _FakeLLM.calls if c["model"] == "aws/claude-sonnet-5"]
    assert not list((tmp_path / "run").rglob("*.json"))  # nothing saved: rerunnable


@harness
def test_gold_scores_every_task(tmp_path):
    for bid in ("tau3-airline", "tau3-telecom", "tau3-banking-knowledge"):
        b = _bench(bid, tmp_path / bid, limit=3, repeats=1, dataset=str(_data_dir()), options={"agent": "gold"})
        assert b.run("", "")["value"] == 1.0
