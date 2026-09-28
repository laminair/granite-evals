"""chat-if family tests: fakes only (no GPU, no network, no server)."""

import json
from pathlib import Path

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


def test_gen_kwargs_default_to_the_task_and_record_overrides(tmp_path):
    def kw(**options):
        return chat_if.MMLUProXLite(RunConfig(model="m", output_dir=tmp_path, options=options)).gen_kwargs()

    assert kw() == {}
    assert kw(temperature="0.6", top_p="0.95", max_tokens="8192", thinking="off") == {
        "temperature": 0.6,
        "do_sample": True,
        "top_p": 0.95,
        "max_gen_toks": 8192,
        "chat_template_kwargs": {"enable_thinking": False},
    }


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

    @staticmethod
    def make(fail_on=None):
        from lm_eval.api.model import LM

        class _LM(LM):
            def generate_until(self, requests, disable_tqdm=False):
                out = []
                for req in requests:
                    ctx = req.args[0]
                    FakeChatLM.calls.append(ctx)
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
