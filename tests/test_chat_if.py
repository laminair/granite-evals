"""chat-if family tests: fakes only (no GPU, no network, no server)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sage2_evals.benchmarks import chat_if
from sage2_evals.registry import RunConfig

LANGS = ("en", "de")
CATEGORIES = ("biology", "math")
LETTERS = "ABCDEFGHIJ"
ALL_CATEGORIES = list(chat_if.SUBJECTS.values())


def _row(lang, qid, category, answer_index, split):
    return {
        "question_id": qid,
        "question": f"[{lang}] question {qid} about {category}?",
        **{f"option_{i}": f"option {LETTERS[i]}" for i in range(10)},
        "answer": LETTERS[answer_index],
        "answer_index": answer_index,
        "cot_content": "A: Let's think step by step. Something. The answer is (A).",
        "category": category,
        "src": "test",
        "question_id_src": qid,
    }


@pytest.fixture
def dataset_dir(tmp_path, monkeypatch):
    """A local MMLU-ProX-Lite look-alike: one config per language, one test
    row per category (interleaved), 5 validation rows per category for the
    5-shot prompts. A private datasets cache: datasets keys a local
    folder's cache by its name, so tests would read each other's rows."""
    datasets = pytest.importorskip("datasets")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    root = tmp_path / "mmlu-prox-lite"
    configs = []
    for lang in LANGS:
        (root / lang).mkdir(parents=True)
        test = [_row(lang, q, ALL_CATEGORIES[q], q % 10, "test") for q in range(len(ALL_CATEGORIES))]
        val = [_row(lang, 100 + i, c, 0, "validation") for c in ALL_CATEGORIES for i in range(5)]
        for split, rows in (("test", test), ("validation", val)):
            (root / lang / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        configs.append(
            f"- config_name: {lang}\n  data_files:\n"
            f"  - split: test\n    path: {lang}/test.jsonl\n"
            f"  - split: validation\n    path: {lang}/validation.jsonl\n"
        )
    (root / "README.md").write_text("---\nconfigs:\n" + "".join(configs) + "---\n")
    return root


def test_registered_with_suite_metadata():
    from sage2_evals.registry import get

    cls = get("mmlu-prox-lite")
    assert cls is chat_if.MMLUProXLite
    assert cls.metric == "exact match (custom-extract)"
    assert cls.default_repeats == 1 and cls.extra == "lmeval"
    assert len(cls.dataset_revision) == 40


def test_task_names_round_trip():
    for lang in ("en", "zh"):
        for subject in chat_if.SUBJECTS:
            assert chat_if.parse_task(chat_if.task_name(lang, subject)) == (lang, subject)


def test_languages_option(tmp_path):
    def langs(**options):
        return chat_if.MMLUProXLite(RunConfig(model="m", output_dir=tmp_path, options=options)).languages()

    assert langs() == list(chat_if.IBM_LANGUAGES)
    assert "nl" not in langs() and len(langs()) == 11
    assert langs(languages="all") == list(chat_if.ALL_LANGUAGES)
    assert langs(languages="ja, en") == ["ja", "en"]
    with pytest.raises(SystemExit):
        langs(languages="en,nl")


def test_select_examples_interleaves_languages_and_maps_to_subject_indices():
    rows = {
        lang: [{"question_id": q, "category": CATEGORIES[q % 2]} for q in (4, 1, 2, 3, 0)]
        for lang in LANGS
    }
    # Order is (question_id, language): q0 en, q0 de, q1 en, q1 de, q2 en.
    # In each language the biology docs (even ids) are, in dataset order,
    # q4, q2, q0 -> q0 is biology index 2, q2 is index 1; q1 is math index 0.
    assert chat_if.select_examples(rows, 5) == {
        "mmlu_prox_lite_de_biology": [2],
        "mmlu_prox_lite_de_math": [0],
        "mmlu_prox_lite_en_biology": [1, 2],
        "mmlu_prox_lite_en_math": [0],
    }
    full = chat_if.select_examples(rows, None)
    assert sum(len(v) for v in full.values()) == 10


def test_sample_status():
    assert chat_if.sample_status(chat_if.ERROR_MARKER, "[invalid]") == "error"
    assert chat_if.sample_status("", "[invalid]") == "no_content"
    assert chat_if.sample_status("no letter here", "[invalid]") == "no_answer"
    assert chat_if.sample_status("the answer is (B)", "B") == "answered"


def test_summarize_macro_over_languages_and_ignores_other_filters():
    def s(score, resp="the answer is (A)", extracted="A", flt=chat_if.FILTER):
        return {"filter": flt, "exact_match": score, "resps": [[resp]], "filtered_resps": [extracted]}

    out = chat_if.summarize(
        {
            "mmlu_prox_lite_en_math": [s(1.0), s(0.0, "", "[invalid]"), s(1.0, flt="other")],
            "mmlu_prox_lite_de_law": [s(1.0), s(1.0)],
            "mmlu_prox_lite_de_math": [s(0.0, chat_if.ERROR_MARKER, "[invalid]"), s(1.0)],
        }
    )
    assert out["per_language"]["en"] == {"exact_match": 0.5, "n": 2}
    assert out["per_language"]["de"] == {"exact_match": 0.75, "n": 4}
    assert out["value"] == pytest.approx(0.625) and out["n"] == 6
    assert out["per_subject"]["math"]["n"] == 4
    assert out["statuses"] == {"answered": 4, "no_content": 1, "error": 1}


def _kw(tmp_path, **options):
    return chat_if.MMLUProXLite(RunConfig(model="m", output_dir=tmp_path, options=options)).gen_kwargs()


def test_gen_kwargs_default_to_the_granite_card_and_the_task_stops(tmp_path):
    # Granite 4.2 card, thinking mode: temperature 1.0, top_p 0.95, 8192 tokens;
    # no until (the task's own stops), no chat_template_kwargs (thinking on).
    # No do_sample: lm-eval's cache skips do_sample=True requests (resume).
    assert _kw(tmp_path) == {"temperature": 1.0, "top_p": 0.95, "max_gen_toks": 8192}
    assert _kw(tmp_path, temperature="0.6", top_p="0.9", max_tokens="16384", thinking="off") == {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_gen_toks": 16384,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # NeMo Evaluator's lm-eval chat protocol, spelled as options.
    assert _kw(tmp_path, temperature="0.0000001", top_p="0.9999999", max_tokens="2048") == {
        "temperature": 1e-7, "top_p": 0.9999999, "max_gen_toks": 2048,
    }  # fmt: skip


@pytest.mark.parametrize(
    ("spec", "until"),
    [
        ("task", None),
        ("none", []),
        ("%3C/s%3E", ["</s>"]),
        ("%3C/s%3E,Q:,%E9%97%AE%E9%A2%98%EF%BC%9A", ["</s>", "Q:", "问题："]),
        ("a%2Cb,x%20y", ["a,b", "x y"]),
    ],
)
def test_stop_option(tmp_path, spec, until):
    kw = _kw(tmp_path, stop=spec)
    if until is None:
        assert "until" not in kw
    else:
        assert kw["until"] == until


@pytest.mark.parametrize("spec", ["a,,b", "a,b,c,d,e", ""])
def test_stop_option_rejects_empty_and_too_many(tmp_path, spec):
    with pytest.raises(SystemExit):
        _kw(tmp_path, stop=spec)


def test_bv_smoke_options_carry_a_stop_list_unchanged():
    """bv-smoke splits OPTIONS on spaces and re-parses them in ``bash -c``:
    a percent-encoded stop list has nothing either would touch."""
    import shlex

    options = "stop=%3C/s%3E,Q:,%3C%7Cim_end%7C%3E max_tokens=8192"
    opts = "".join(f" --option {kv}" for kv in options.split())
    assert shlex.split(opts) == ["--option", "stop=%3C/s%3E,Q:,%3C%7Cim_end%7C%3E", "--option", "max_tokens=8192"]


def test_chat_payload_sends_the_card_sampling_and_stops(tmp_path):
    """What lm-eval's local-chat-completions puts on the wire for the
    task's generation_kwargs after our overrides."""
    pytest.importorskip("lm_eval")
    from lm_eval.models.openai_completions import LocalChatCompletion

    task = {"until": ["</s>", "Q:", "问题：", "<|im_end|>"], "do_sample": False, "temperature": 0.0, "max_gen_toks": 2048}
    msgs = [{"role": "user", "content": "q"}]

    def payload(**options):
        gk = {**task, **_kw(tmp_path, **options)}
        return LocalChatCompletion._create_payload(SimpleNamespace(_max_gen_toks=256, model="m"), msgs, gen_kwargs=gk, seed=1, eos="<|end_of_text|>")

    p = payload()
    assert (p["temperature"], p["top_p"], p["max_tokens"]) == (1.0, 0.95, 8192)
    assert p["stop"] == ["</s>", "Q:", "问题：", "<|im_end|>"]
    assert "chat_template_kwargs" not in p and "do_sample" not in p
    assert payload(stop="none")["stop"] == ["<|end_of_text|>"]
    assert payload(stop="%3C/s%3E")["stop"] == ["</s>", "<|end_of_text|>"]
    assert payload(thinking="off")["chat_template_kwargs"] == {"enable_thinking": False}


def test_task_spec_pins_revision_per_subject_task(tmp_path):
    b = chat_if.MMLUProXLite(RunConfig(model="m", output_dir=tmp_path))
    spec = b.task_spec(["mmlu_prox_lite_en_math"], b.dataset, b.dataset_revision)
    assert spec["task"] == [{"task": "mmlu_prox_lite_en_math", "dataset_kwargs": {"revision": b.dataset_revision}}]
    local = b.task_spec(["mmlu_prox_lite_en_math"], "/data/x", None)
    assert local["task"] == [{"task": "mmlu_prox_lite_en_math", "dataset_kwargs": {}, "dataset_path": "/data/x"}]


# --- end to end through lm-eval (needs the lmeval extra) ---------------------


def _bench(tmp_path, dataset_dir, limit=None, **options):
    options = {"languages": ",".join(LANGS), **options}
    cfg = RunConfig(model="m", output_dir=tmp_path / "out", limit=limit, dataset=str(dataset_dir), options=options)
    return chat_if.MMLUProXLite(cfg)


def test_gold_scores_one_end_to_end(tmp_path, dataset_dir):
    pytest.importorskip("lm_eval")
    b = _bench(tmp_path, dataset_dir, answers="gold")
    assert not b.needs_server()
    out = b.run("", "")
    assert out["value"] == 1.0 and out["n"] == 28
    rep = out["per_repeat"][0]
    assert rep["statuses"] == {"answered": 28}
    assert set(rep["per_language"]) == set(LANGS)
    rows = [json.loads(x) for x in (tmp_path / "out/repeat-0/samples/mmlu_prox_lite_de_computer_science.jsonl").read_text().splitlines()]
    assert rows and all(r["filter"] == "custom-extract" for r in rows)
    assert "Die Antwort ist" in rows[0]["resps"][0][0]


class FakeChatLM:
    """Stands in for local-chat-completions: answers A, fails on one prompt."""

    calls: list = []
    gen_kwargs: list = []

    @staticmethod
    def make(fail_on=None):
        from lm_eval.api.model import LM

        class _LM(LM):
            def generate_until(self, requests, disable_tqdm=False):
                out = []
                for req in requests:
                    ctx = req.args[0]
                    FakeChatLM.calls.append(ctx)
                    FakeChatLM.gen_kwargs.append(req.args[1])
                    if fail_on and fail_on in ctx:
                        out.append(chat_if.ERROR_MARKER)
                    else:
                        lang = "de" if "[de]" in ctx else "en"
                        out.append("Reasoning... " + chat_if.gold_response(lang, "A"))
                    self.cache_hook.add_partial("generate_until", req.args, out[-1])
                return out

            def loglikelihood(self, requests, disable_tqdm=False):
                raise NotImplementedError

            def loglikelihood_rolling(self, requests, disable_tqdm=False):
                raise NotImplementedError

            def apply_chat_template(self, chat_history, add_generation_prompt=True):
                return json.dumps(chat_history)

            @property
            def tokenizer_name(self):
                return "fake"

        return _LM()


def test_limit_errors_and_resume(tmp_path, dataset_dir, monkeypatch):
    pytest.importorskip("lm_eval")
    FakeChatLM.calls = []
    # q3 is the only failing question, in English only.
    monkeypatch.setattr(chat_if.MMLUProXLite, "make_lm", lambda self, *a, **k: FakeChatLM.make("[en] question 3 "))
    b = _bench(tmp_path, dataset_dir, limit=8)
    out = b.run("http://x/v1", "m")
    rep = out["per_repeat"][0]
    assert out["n"] == 8 and rep["statuses"] == {"answered": 7, "error": 1}
    assert len(FakeChatLM.calls) == 8
    # The harness's 5-shot CoT chat: the description as system message, then
    # the shots as user turns (the task's fewshot target is "", the worked
    # answer is part of each shot's text, so lm-eval emits no assistant turns),
    # then the question ending in the language's "let's think step by step".
    first = json.loads(FakeChatLM.calls[0])
    assert [m["role"] for m in first] == ["system"] + ["user"] * 6
    assert "The answer is (A)." in first[1]["content"] and "question 0 " in first[-1]["content"]
    # Answer A is right for q0 only (answer_index = q % 10); q0..q3 x en,de.
    assert rep["per_language"]["en"]["exact_match"] == 0.25
    assert out["value"] == 0.25

    # Resume: everything answered comes from the cache; the error is retried.
    FakeChatLM.calls = []
    monkeypatch.setattr(chat_if.MMLUProXLite, "make_lm", lambda self, *a, **k: FakeChatLM.make())
    out2 = _bench(tmp_path, dataset_dir, limit=8).run("http://x/v1", "m")
    assert len(FakeChatLM.calls) == 1 and "[en] question 3 " in FakeChatLM.calls[0]
    assert out2["per_repeat"][0]["statuses"] == {"answered": 8}


@pytest.mark.parametrize(("stop", "until"), [(None, ["</s>", "Q:", "Frage:", "<|im_end|>"]), ("none", [])])
def test_requests_carry_the_generation_kwargs_and_results_record_them(tmp_path, dataset_dir, monkeypatch, stop, until):
    pytest.importorskip("lm_eval")
    FakeChatLM.calls, FakeChatLM.gen_kwargs = [], []
    monkeypatch.setattr(chat_if.MMLUProXLite, "make_lm", lambda self, *a, **k: FakeChatLM.make())
    out = _bench(tmp_path, dataset_dir, limit=4, **({"stop": stop} if stop else {})).run("http://x/v1", "m")
    de = [gk for ctx, gk in zip(FakeChatLM.calls, FakeChatLM.gen_kwargs) if "[de]" in ctx]
    assert de and all(gk["until"] == until for gk in de)
    assert all((gk["temperature"], gk["top_p"], gk["max_gen_toks"]) == (1.0, 0.95, 8192) for gk in FakeChatLM.gen_kwargs)
    gk = out["generation_kwargs"]
    assert set(gk) == set(LANGS)
    assert gk["de"]["until"] == until and gk["en"]["until"] == (until and ["</s>", "Q:", "Question:", "<|im_end|>"])
    assert gk["de"]["max_gen_toks"] == 8192 and gk["de"]["top_p"] == 0.95
    assert out["stop"] == ("task" if stop is None else [])
    assert json.dumps(out)  # results.json-serialisable
    assert all(not gk.get("do_sample") for gk in FakeChatLM.gen_kwargs)  # cacheable, so resumable


def _completion(question: str) -> dict:
    """A vLLM-shaped chat completion, by question: q0 stops on a stop string,
    q1 runs into max_tokens inside the thought (content None), q2 fails,
    others end on EOS."""
    if "question 0 " in question:
        return {"choices": [{"index": 0, "finish_reason": "stop", "stop_reason": "Q:",
                             "message": {"content": "", "reasoning_content": "think Q"}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 300}}  # fmt: skip
    if "question 1 " in question:
        return {"choices": [{"index": 0, "finish_reason": "length", "stop_reason": None,
                             "message": {"content": None, "reasoning_content": "x" * 50}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 8192}}  # fmt: skip
    if "question 2 " in question:
        raise RuntimeError("server down")
    return {"choices": [{"index": 0, "finish_reason": "stop", "stop_reason": None,
                         "message": {"content": "the answer is (A)", "reasoning_content": "hm"}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 1000}}  # fmt: skip


def _fake_http(monkeypatch, sent: list, respond=_completion):
    """Replace only the HTTP round trip of lm-eval's TemplateAPI.amodel_call:
    the payload is built and the response parsed by the real code."""
    from lm_eval.models.api_models import TemplateAPI

    async def call(self, session, sem, messages, *, generate=True, cache_keys=None, ctxlens=None, gen_kwargs=None, **kw):
        import copy

        payload = self._create_payload(self.create_message(messages), generate=True, gen_kwargs=copy.deepcopy(gen_kwargs), seed=self._seed)
        sent.append(payload)
        answers = self.parse_generations(outputs=respond(payload["messages"][-1]["content"]))
        answers = [a if a is not None else "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER" for a in answers]
        for res, key in zip(answers, cache_keys or [], strict=False):
            self.cache_hook.add_partial("generate_until", key, res)
        return answers

    monkeypatch.setattr(TemplateAPI, "amodel_call", call)
    monkeypatch.setattr(chat_if, "RETRY_BACKOFF_S", 0.0)


def test_samples_and_results_record_finish_reason_and_tokens(tmp_path, dataset_dir, monkeypatch):
    pytest.importorskip("lm_eval")
    sent: list = []
    _fake_http(monkeypatch, sent)
    b = _bench(tmp_path, dataset_dir, limit=8, max_retries="2")  # q0..q3 x en,de
    b.config.workers = 4
    out = b.run("http://x/v1", "m")
    assert len(sent) == 8 + 2  # q2 x 2 languages retried once each
    req = out["per_repeat"][0]["requests"]
    assert req["finish_reason"] == {"stop": 4, "length": 2, "error": 2}
    assert req["stop_reason"] == {"Q:": 2, "eos": 2}
    assert req["no_content_finish_reason"] == {"stop": 2, "length": 2}
    assert req["completion_tokens"] == {"n": 6, "mean": round((2 * 300 + 2 * 8192 + 2 * 1000) / 6, 1), "max": 8192}
    # q3's English "the answer is" does not match the German regex.
    assert out["per_repeat"][0]["statuses"] == {"no_content": 4, "error": 2, "answered": 1, "no_answer": 1}

    def rows():
        sdir = tmp_path / "out/repeat-0/samples"
        return [json.loads(x) for f in sorted(sdir.glob("*.jsonl")) for x in f.read_text().splitlines()]

    by_q = {(r["doc"]["question"], r["filter"]): r["sage2_request"] for r in rows()}
    q0 = by_q[("[en] question 0 about biology?", "custom-extract")]
    assert q0 == {"finish_reason": "stop", "stop_reason": "Q:", "completion_tokens": 300, "prompt_tokens": 900,
                  "reasoning_chars": 7, "content_chars": 0}  # fmt: skip
    assert by_q[("[de] question 1 about business?", "custom-extract")]["finish_reason"] == "length"
    assert by_q[("[de] question 2 about chemistry?", "custom-extract")]["finish_reason"] == "error"

    # Resume: the cached answers keep their records; the failed ones are
    # retried and their new records replace the error ones.
    sent.clear()
    ok = _completion("question 9 ")
    _fake_http(monkeypatch, sent, respond=lambda q: ok)
    out2 = _bench(tmp_path, dataset_dir, limit=8).run("http://x/v1", "m")
    assert len(sent) == 2
    req2 = out2["per_repeat"][0]["requests"]
    assert req2["finish_reason"] == {"stop": 6, "length": 2}
    assert req2["completion_tokens"]["n"] == 8
    assert all(r["sage2_request"] for r in rows())


def test_gold_has_no_request_records(tmp_path, dataset_dir):
    pytest.importorskip("lm_eval")
    out = _bench(tmp_path, dataset_dir, answers="gold").run("", "")
    assert "requests" not in out["per_repeat"][0]
    rows = (tmp_path / "out/repeat-0/samples/mmlu_prox_lite_en_math.jsonl").read_text().splitlines()
    assert "sage2_request" not in json.loads(rows[0])


def test_response_meta_and_request_key():
    pytest.importorskip("lm_eval")
    from lm_eval.api.model import hash_args

    assert chat_if.response_meta({"choices": [{"finish_reason": "stop", "message": {"content": "ab", "reasoning": "xyz"}}]}) == [
        {"finish_reason": "stop", "stop_reason": None, "completion_tokens": None, "prompt_tokens": None,
         "reasoning_chars": 3, "content_chars": 2},
    ]  # fmt: skip
    args = ("ctx", {"until": ["Q:"], "max_gen_toks": 8192})
    assert chat_if.request_key(args) == hash_args("generate_until", args)


def test_chat_lm_turns_exhausted_retries_into_an_error_marker(monkeypatch):
    pytest.importorskip("lm_eval")
    import asyncio

    from lm_eval.models.openai_completions import LocalChatCompletion

    calls = []

    async def boom(self, *a, **k):
        calls.append(1)
        raise RuntimeError("server down")

    monkeypatch.setattr(LocalChatCompletion, "amodel_call", boom)
    monkeypatch.setattr(chat_if, "RETRY_BACKOFF_S", 0.0)
    b = chat_if.MMLUProXLite(RunConfig(model="m", output_dir=".", options={"max_retries": 3}))
    lm = b.make_lm("http://x/v1", "m", seed=1)
    assert lm.base_url == "http://x/v1/chat/completions" and lm.model == "m"
    out = asyncio.run(lm.amodel_call(None, None, [{"role": "user", "content": "hi"}], generate=True))
    assert out == [chat_if.ERROR_MARKER] and len(calls) == 3 and lm.errors == 1


def test_drop_errors_from_cache(tmp_path):
    sqlitedict = pytest.importorskip("sqlitedict")
    with sqlitedict.SqliteDict(str(tmp_path / "cache_rank0.db"), autocommit=True) as d:
        d["a"], d["b"] = "the answer is (A)", chat_if.ERROR_MARKER
    assert chat_if.drop_errors_from_cache(tmp_path) == 1
    with sqlitedict.SqliteDict(str(tmp_path / "cache_rank0.db")) as d:
        assert dict(d) == {"a": "the answer is (A)"}


# --- IFBench (NeMo-Skills) ------------------------------------------------------


def test_ifbench_registered_and_pinned(tmp_path):
    from sage2_evals.registry import get

    cls = get("ifbench")
    assert cls is chat_if.IFBench
    assert cls.metric == "pass@1[avg-of-2] loose accuracy" and cls.default_repeats == 2
    assert cls.extra == "ifbench" and cls.ns_metric == "prompt_loose_accuracy"
    b = cls(RunConfig(model="m", output_dir=tmp_path))
    assert b.repeats == 2 and b.aggregation() == "pass@1[avg-of-2]"
    assert b.pins() == {}  # a GitHub file, pinned by URL + sha256, not an HF repo
    pinned, sha = cls.pinned_urls[chat_if.IFBENCH_TEST_URL]
    assert chat_if.IFBENCH_DATA_COMMIT in pinned and len(sha) == 64
    assert len(cls.prepared_sha256) == 64  # the prepared test.jsonl is checked too


def test_ifbench_has_no_gold_mode(tmp_path):
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path, options={"answers": "gold"}))
    with pytest.raises(SystemExit, match="no answers=gold"):
        b.run("", "")


def test_ifbench_pin_covers_what_ns_prepare_reads(tmp_path):
    pytest.importorskip("nemo_skills")
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path))
    prepare = (Path(b.ns_module().__file__).parent / "prepare.py").read_text()
    assert f'URL = "{chat_if.IFBENCH_TEST_URL}"' in prepare
    args = b.generation_args([])
    assert {"++prompt_config=generic/default", "++eval_type=ifbench", "++generation_key=response"} <= set(args)
    assert b.metrics_type() == "if"


def _if_rows(path, loose, strict):
    """ns generation rows after ns's ifbench evaluator merged IFBench's results."""
    rows = []
    for i, (lo, st) in enumerate(zip(loose, strict, strict=True)):
        rows.append(
            {
                "key": str(i),
                "prompt": f"p{i}",
                "response": "r",
                "instruction_id_list": ["a", "b"],
                "num_generated_tokens": 3,
                "loose_eval": {"follow_all_instructions": all(lo), "follow_instruction_list": lo},
                "strict_eval": {"follow_all_instructions": all(st), "follow_instruction_list": st},
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_ifbench_value_is_prompt_loose_accuracy_avg_of_2(tmp_path):
    pytest.importorskip("nemo_skills")
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path))
    files = [
        _if_rows(tmp_path / "output-rs0.jsonl", [[True, True], [True, False]], [[True, False], [False, False]]),
        _if_rows(tmp_path / "output-rs1.jsonl", [[True, True], [True, True]], [[True, True], [False, False]]),
    ]
    m = b.compute_metrics(files)["_all_"][b.aggregation()]
    assert m["prompt_loose_accuracy"] == pytest.approx(75.0)  # (1/2 + 2/2) / 2
    assert m["prompt_strict_accuracy"] == pytest.approx(25.0)
    assert m["instruction_loose_accuracy"] == pytest.approx(87.5)


def test_ifbench_puts_the_pinned_nltk_data_first(tmp_path, monkeypatch):
    """IFBench's verifiers nltk.download() at build time; ~/nltk_data would win over
    the image's pinned copy unless NLTK_DATA names it (inherited by run_eval)."""
    from sage2_evals.benchmarks.nemo_skills import NemoSkillsBenchmark

    seen = {}
    monkeypatch.setattr(NemoSkillsBenchmark, "run", lambda self, *a: seen.setdefault("NLTK_DATA", __import__("os").environ["NLTK_DATA"]))
    monkeypatch.setenv("NLTK_DATA", f"/elsewhere:{chat_if.IFBENCH_NLTK_DATA}")
    chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path)).run("", "")
    assert seen["NLTK_DATA"].split(":") == [chat_if.IFBENCH_NLTK_DATA, "/elsewhere"]


def _ifbench_phase(tmp_path, monkeypatch, phase, events):
    """IFBench on 2 prompts with ns's generation and evaluator faked: the evaluator
    grades each marked file (prompt 0 followed, prompt 1 not)."""
    from sage2_evals.benchmarks import nemo_skills as nsb

    prepared = tmp_path / "prepared.jsonl"
    nsb._write_jsonl(prepared, [{"key": str(i), "prompt": f"p{i}", "instruction_id_list": ["a"]} for i in range(2)])

    def fake_generate(cmd, out, *, log_path, what):
        events.append(("generate", cmd[-1]))
        rows = nsb._read_jsonl(prepared)
        nsb._write_jsonl(out, [{**r, "response": "r", "num_generated_tokens": 3} for r in rows])

    def fake_evaluate(self, output, args, what):
        if not output.exists():
            raise self.not_generated(what)
        if nsb._unscored(output).exists():
            events.append(("evaluate", nsb._eval_overrides(args), os.environ["NLTK_DATA"].split(os.pathsep)[0]))
            ev = [{"follow_all_instructions": i == 0, "follow_instruction_list": [i == 0]} for i in range(2)]
            nsb._write_jsonl(output, [{**r, "loose_eval": e, "strict_eval": e} for r, e in zip(nsb._read_jsonl(output), ev)])
            nsb._unscored(output).unlink()

    monkeypatch.setattr(nsb, "_run_ns", fake_generate)
    monkeypatch.setattr(nsb.NemoSkillsBenchmark, "_evaluate", fake_evaluate)
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path, phase=phase))
    monkeypatch.setattr(b, "prepare_data", lambda: (prepared, {"sha256": "x"}))
    return b.run("http://127.0.0.1:9/v1" if phase == "generate" else "", "m")


def test_ifbench_generate_then_score(tmp_path, monkeypatch):
    pytest.importorskip("nemo_skills")
    monkeypatch.setenv("NLTK_DATA", "/elsewhere")
    g = _ifbench_phase(tmp_path, monkeypatch, "generate", events := [])
    assert g["n"] == 2 and "value" not in g and events == [("generate", "++eval_type=null")] * 2
    assert os.environ["NLTK_DATA"] == "/elsewhere"  # only scoring needs the pinned nltk data
    s = _ifbench_phase(tmp_path, monkeypatch, "score", events := [])
    graded = ("evaluate", ["++eval_type=ifbench"], chat_if.IFBENCH_NLTK_DATA)
    assert events == [graded, graded] and (s["value"], s["n"]) == (pytest.approx(0.5), 2)
    assert _ifbench_phase(tmp_path, monkeypatch, "score", events := [])["value"] == pytest.approx(0.5)
    assert events == []  # re-scored from the graded files
    (tmp_path / "generation" / "output-rs1.jsonl").unlink()
    with pytest.raises(RuntimeError, match="no generation for repeat 1"):
        _ifbench_phase(tmp_path, monkeypatch, "score", [])


def test_ifbench_is_splittable_mmlu_prox_is_not():
    assert chat_if.IFBench.splittable and not chat_if.MMLUProXLite.splittable


# Every instruction of the pinned IFBench test data (newer than the IFBench checkout
# ns pins, whose own data/ has 294 prompts) through its verifier, as ns's evaluator
# runs them (cwd = the IFBench checkout, the job's python), with the exceptions
# ns's patch would swallow (scored "not followed") reported instead. nltk.download is
# a no-op here: the pinned data must be enough, and a build-time test must not
# change it.
IFBENCH_VERIFY = r"""
import json, sys
import nltk
nltk.download = lambda *a, **k: True
import instructions_registry as reg
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
response = (
    "Birds sing at dawn. Do you hear them? I walked to the river with Maria and John, and the water was cold.\n\n"
    "* Peter met them in Paris on 3 May 2024.\n* They counted 42 boats, 7 bridges and 1 dog!\n\n"
    "Run, because the rain is coming; however, nobody ran. P.S. The end."
)
seen, failed = set(), {}
for row in rows:
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        seen.add(iid)
        try:
            inst = reg.INSTRUCTION_DICT[iid](iid)
            inst.build_description(**{k: v for k, v in kw.items() if v is not None})
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=row["prompt"])
            inst.check_following(response)
        except Exception as e:
            failed.setdefault(iid, repr(e)[:300])
found = {p: str(nltk.data.find(p)) for p in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords", "taggers/averaged_perceptron_tagger_eng")}
print(json.dumps({"rows": len(rows), "instructions": sorted(seen), "failed": failed, "nltk": found}))
"""


def test_ifbench_verifiers_run_in_the_image(tmp_path):
    import os
    import subprocess
    import sys

    import hashlib
    import urllib.request

    if not (chat_if.IFBENCH_DIR / "run_eval.py").is_file():
        pytest.skip("no IFBench checkout (built by docker/extras/ifbench.sh)")
    url, sha = chat_if.IFBench.pinned_urls[chat_if.IFBENCH_TEST_URL]
    data = tmp_path / "IFBench_test.jsonl"
    with urllib.request.urlopen(url, timeout=60) as f:
        data.write_bytes(f.read())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == sha
    env = dict(os.environ, NLTK_DATA=chat_if.IFBENCH_NLTK_DATA, HOME=str(tmp_path))
    r = subprocess.run(
        [sys.executable, "-c", IFBENCH_VERIFY, str(data)], cwd=chat_if.IFBENCH_DIR, env=env,
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-3000:]
    report = json.loads(r.stdout.strip().splitlines()[-1])
    assert report["rows"] == 300 and len(report["instructions"]) > 50
    assert report["failed"] == {}
    assert all(p.startswith(chat_if.IFBENCH_NLTK_DATA + "/") for p in report["nltk"].values()), report["nltk"]
