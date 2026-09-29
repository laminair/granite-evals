"""Chat & instruction-following family: MMLU-ProX lite (IBM), IFBench.

MMLU-ProX lite (IBM): exact match (custom-extract)
---------------------------------------------------
lm-evaluation-harness's own ``mmlu_prox_lite_{lang}_{subject}`` tasks (5-shot
CoT prompts, the ``custom-extract`` regex filter, ``exact_match``), sent to the
served model in lm-eval's chat mode, as NeMo Evaluator's lm-eval container runs
its ``mmlu_prox_*_chat`` tasks and as Granite 4.2's card says its numbers were
produced ("an evaluation framework based on NeMo Evaluator SDK"):
``local-chat-completions --apply_chat_template --fewshot_as_multiturn``.
The task's few-shot target is empty (each shot's text carries its worked
answer), so lm-eval sends the description as the system turn, the 5 shots as 5
user turns and the question as the last user turn; that is the harness's
behaviour, kept as is.

"(IBM)" is read as IBM's language subset: the Granite 4.x supported languages
that MMLU-ProX covers (the card lists en, de, es, fr, ja, pt, ar, cs, it, ko, nl,
zh; MMLU-ProX has no Dutch), i.e. 11 languages x 588 test questions. Neither the
card nor the blog defines it, so this is the most defensible reading, not a
confirmed one; ``--option languages=all`` (or a comma list) changes it.

The headline value is the mean over languages of each language's exact match
(lm-eval's ``mmlu_prox_lite_{lang}`` group: size-weighted over subjects). All
languages have the same 588 questions, so this is also the micro average.

Generation follows the task configs (greedy, ``max_gen_toks: 2048``, the task's
stop strings); ``--option temperature=/top_p=/max_tokens=`` override them and
``--option thinking=off`` sends ``chat_template_kwargs.enable_thinking=false``.
The served model's reasoning parser keeps the thinking out of ``content``, which
is what the regex sees. A thought cut off at ``max_tokens`` leaves no content
and scores as wrong, like in the reference protocol.

Resume: lm-eval's response cache (``repeat-<k>/lm_cache``) keeps every
finished request, so a restarted run only sends the missing ones. A request
that keeps failing is answered with an error marker, scored wrong, counted in
``statuses`` and dropped from the cache, so the next run retries it.
``--option answers=gold`` answers every question with its reference letter in
the language's answer phrase: it checks data, prompts, extraction and scoring
without a model and must score 1.0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

from sage2_evals import data
from sage2_evals.benchmarks.nemo_skills import NemoSkillsBenchmark
from sage2_evals.registry import Benchmark, register

log = logging.getLogger(__name__)

# Granite 4.x supported languages that MMLU-ProX has (no nl), in suite order.
IBM_LANGUAGES = ("en", "de", "es", "fr", "ja", "pt", "ar", "cs", "it", "ko", "zh")
ALL_LANGUAGES = (
    "af", "ar", "bn", "cs", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "mr",
    "ne", "pt", "ru", "sr", "sw", "te", "th", "uk", "ur", "vi", "wo", "yo", "zh", "zu",
)  # fmt: skip
# lm-eval task suffix -> the dataset's ``category`` value (lm_eval/tasks/mmlu_prox/*/utils.py).
SUBJECTS = {
    "biology": "biology",
    "business": "business",
    "chemistry": "chemistry",
    "computer_science": "computer science",
    "economics": "economics",
    "engineering": "engineering",
    "health": "health",
    "history": "history",
    "law": "law",
    "math": "math",
    "other": "other",
    "philosophy": "philosophy",
    "physics": "physics",
    "psychology": "psychology",
}
FILTER = "custom-extract"
METRIC = "exact_match"
ERROR_MARKER = "[sage2-evals: request failed]"
PROGRESS_EVERY = 100
RETRY_BACKOFF_S = 1.0  # doubles per attempt, capped at 30 s


def task_name(lang: str, subject: str) -> str:
    return f"mmlu_prox_lite_{lang}_{subject}"


def parse_task(name: str) -> tuple[str, str]:
    """``mmlu_prox_lite_de_computer_science`` -> ("de", "computer_science")."""
    rest = name.removeprefix("mmlu_prox_lite_")
    lang, _, subject = rest.partition("_")
    return lang, subject


def select_examples(rows_by_lang: dict[str, list[dict]], limit: int | None) -> dict[str, list[int]]:
    """lm-eval ``samples``: per task, the doc indices of the first ``limit``
    examples, ordered by (question_id, language order). An example is one
    question in one language, so a small limit still covers every language.
    Indices are positions in the task's docs: the language's test split
    filtered to the subject, order kept (lm-eval's ``process_docs``)."""
    langs = list(rows_by_lang)
    examples = []
    for lang, rows in rows_by_lang.items():
        seen: dict[str, int] = {}
        for row in rows:
            subject = next(s for s, c in SUBJECTS.items() if c == row["category"])
            idx = seen.get(subject, 0)
            seen[subject] = idx + 1
            examples.append({"order": (row["question_id"], langs.index(lang)), "task": task_name(lang, subject), "idx": idx})
    samples: dict[str, list[int]] = {}
    for ex in data.take(examples, limit, key="order"):
        samples.setdefault(ex["task"], []).append(ex["idx"])
    return {t: sorted(v) for t, v in sorted(samples.items())}


def sample_status(response: str, extracted: str) -> str:
    if response == ERROR_MARKER:
        return "error"
    if not response or response.startswith("LMEVAL_MODEL_NONE_ANSWER"):
        return "no_content"  # e.g. the thought ran into max_tokens
    if extracted == "[invalid]":
        return "no_answer"
    return "answered"


def summarize(samples: dict[str, list[dict]]) -> dict[str, Any]:
    """Per-language and headline exact match from lm-eval's logged samples
    (``custom-extract`` filter only)."""
    per_lang: dict[str, list[float]] = {}
    per_subject: dict[str, list[float]] = {}
    statuses: dict[str, int] = {}
    for task, rows in samples.items():
        lang, subject = parse_task(task)
        for row in rows:
            if row.get("filter") != FILTER:
                continue
            score = float(row[METRIC])
            per_lang.setdefault(lang, []).append(score)
            per_subject.setdefault(subject, []).append(score)
            resp = row["resps"][0][0] if row.get("resps") else ""
            status = sample_status(resp, row["filtered_resps"][0] if row.get("filtered_resps") else "")
            statuses[status] = statuses.get(status, 0) + 1
    by_lang = {lang: {"exact_match": sum(v) / len(v), "n": len(v)} for lang, v in per_lang.items()}
    return {
        "value": sum(x["exact_match"] for x in by_lang.values()) / len(by_lang) if by_lang else 0.0,
        "n": sum(x["n"] for x in by_lang.values()),
        "per_language": by_lang,
        "per_subject": {s: {"exact_match": sum(v) / len(v), "n": len(v)} for s, v in sorted(per_subject.items())},
        "statuses": statuses,
    }


def gold_response(lang: str, answer: str) -> str:
    from lm_eval.tasks.mmlu_prox.lang_libs import LANG_LIBS

    return LANG_LIBS[lang][5].format(answer)


def _chat_lm_class():
    """lm-eval's local-chat-completions, made robust: bounded retries per
    request inside the call, then an error marker instead of an exception (one
    broken request must not sink ~7k), plus progress log lines."""
    from lm_eval.models.openai_completions import LocalChatCompletion

    class Sage2ChatCompletions(LocalChatCompletion):
        def __init__(self, *args, benchmark_id: str = "", **kwargs):
            super().__init__(*args, **kwargs)
            self.benchmark_id = benchmark_id
            self.done = self.errors = 0

        def _progress(self) -> None:
            if self.done % PROGRESS_EVERY == 0:
                log.info("%s: %d requests done, %d errors", self.benchmark_id, self.done, self.errors)

        async def amodel_call(self, session, sem, messages, *, generate=True, cache_keys=None, ctxlens=None, gen_kwargs=None, **kwargs):
            last: BaseException | None = None
            attempts = max(1, self.max_retries)
            for attempt in range(attempts):
                try:
                    out = await super().amodel_call(
                        session, sem, messages, generate=generate, cache_keys=cache_keys,
                        ctxlens=ctxlens, gen_kwargs=gen_kwargs, **kwargs,
                    )  # fmt: skip
                    self.done += 1
                    self._progress()
                    return out
                except Exception as e:  # noqa: BLE001 - retried, then scored as failed
                    last = e
                    if attempt + 1 < attempts:
                        await asyncio.sleep(min(RETRY_BACKOFF_S * 2**attempt, 30))
            log.warning("%s: request failed after %d attempts: %r", self.benchmark_id, self.max_retries, last)
            self.done += 1
            self.errors += 1
            self._progress()
            return [ERROR_MARKER]

    return Sage2ChatCompletions


def _gold_lm_class():
    """Answers each request with its reference letter (``answers=gold``)."""
    from lm_eval.api.model import LM

    class GoldLM(LM):
        def generate_until(self, requests, disable_tqdm: bool = False):
            out = []
            for req in requests:
                lang, _ = parse_task(req.task_name)
                out.append(gold_response(lang, req.doc["answer"]))
            return out

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError

        def apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
            return json.dumps(chat_history, ensure_ascii=False)

        @property
        def tokenizer_name(self) -> str:
            return "gold"

    return GoldLM


def drop_errors_from_cache(cache_dir: Path) -> int:
    """Delete cached error markers so a restart retries those requests."""
    from sqlitedict import SqliteDict

    dropped = 0
    for db in cache_dir.glob("*.db"):
        with SqliteDict(str(db), autocommit=True) as d:
            for key in [k for k, v in d.items() if v == ERROR_MARKER]:
                del d[key]
                dropped += 1
    return dropped


@register
class MMLUProXLite(Benchmark):
    id = "mmlu-prox-lite"
    metric = "exact match (custom-extract)"
    default_repeats = 1
    extra = "lmeval"
    harness_packages = ("lm-eval",)
    dataset = "li-lab/MMLU-ProX-Lite"
    dataset_revision = "e82aafb9460529687d3c7e51b401d8dd1dd309dd"
    split: ClassVar[str] = "test"

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def gold(self) -> bool:
        return self.opt("answers", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    def languages(self) -> list[str]:
        spec = self.opt("languages", "ibm")
        if spec == "ibm":
            return list(IBM_LANGUAGES)
        if spec == "all":
            return list(ALL_LANGUAGES)
        langs = [x.strip() for x in spec.split(",") if x.strip()]
        unknown = sorted(set(langs) - set(ALL_LANGUAGES))
        if unknown:
            raise SystemExit(f"{self.id}: unknown languages {unknown}; MMLU-ProX has {', '.join(ALL_LANGUAGES)}")
        return langs

    def task_spec(self, tasks: list[str], source: str, revision: str | None) -> dict:
        """One inline lm-eval group over the upstream subject tasks, each
        pinned to ``source``@``revision`` (lm-eval passes ``dataset_kwargs`` to
        ``datasets.load_dataset``)."""
        override: dict[str, Any] = {"dataset_kwargs": {"revision": revision} if revision else {}}
        if source != self.dataset:
            override["dataset_path"] = source
        return {"group": "sage2_mmlu_prox_lite", "task": [{"task": t, **override} for t in tasks]}

    def gen_kwargs(self) -> dict[str, Any]:
        """Overrides of the task's generation_kwargs; empty = the task's own."""
        kw: dict[str, Any] = {}
        if "temperature" in self.config.options:
            kw["temperature"] = float(self.config.options["temperature"])
            kw["do_sample"] = kw["temperature"] > 0
        if "top_p" in self.config.options:
            kw["top_p"] = float(self.config.options["top_p"])
        if "max_tokens" in self.config.options:
            kw["max_gen_toks"] = int(self.config.options["max_tokens"])
        if self.opt("thinking", "default") == "off":
            kw["chat_template_kwargs"] = {"enable_thinking": False}
        return kw

    def make_lm(self, base_url: str, served: str, seed: int):
        if self.gold:
            return _gold_lm_class()()
        return _chat_lm_class()(
            base_url=f"{base_url.rstrip('/')}/chat/completions",
            model=served,
            num_concurrent=self.config.workers,
            max_retries=self.opt("max_retries", 5),
            timeout=self.opt("request_timeout", 3600),
            tokenized_requests=False,
            tokenizer_backend=None,
            seed=seed,
            benchmark_id=self.id,
        )

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        source, revision = self.dataset_source()
        langs = self.languages()
        samples = None
        if self.config.limit is not None:
            rows = {lang: data.load_split(source, revision=revision, split=self.split, name=lang) for lang in langs}
            samples = select_examples(rows, self.config.limit)
            tasks = list(samples)
        else:
            tasks = [task_name(lang, s) for lang in langs for s in SUBJECTS]
        log.info("%s: %d languages, %d tasks, limit=%s", self.id, len(langs), len(tasks), self.config.limit)

        per_repeat = []
        for k in range(self.repeats):
            summary = self._repeat(k, tasks, samples, source, revision, base_url, served_model_name)
            per_repeat.append({"repeat": k, **summary})
            log.info("%s repeat %d: exact_match %.4f over %d (%s)", self.id, k, summary["value"], summary["n"], summary["statuses"])

        return {
            "value": sum(r["value"] for r in per_repeat) / len(per_repeat),
            "n": per_repeat[0]["n"],
            "dataset": source,
            "dataset_revision": revision,
            "languages": langs,
            "mode": "gold" if self.gold else "local-chat-completions, apply_chat_template, fewshot_as_multiturn",
            "gen_kwargs_overrides": self.gen_kwargs(),
            "per_repeat": per_repeat,
        }

    def _repeat(self, k, tasks, samples, source, revision, base_url, served) -> dict[str, Any]:
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager

        repeat_dir = self.config.output_dir / f"repeat-{k}"
        cache_dir = repeat_dir / "lm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        results = simple_evaluate(
            model=self.make_lm(base_url, served, seed=self.config.seed + 1234 + k),
            tasks=[self.task_spec(tasks, source, revision)],
            task_manager=TaskManager(),
            samples=samples,
            use_cache=None if self.gold else str(cache_dir / "cache"),
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            gen_kwargs=self.gen_kwargs() or None,
            log_samples=True,
            bootstrap_iters=0,
            random_seed=self.config.seed,
        )
        if not self.gold:
            if dropped := drop_errors_from_cache(cache_dir):
                log.warning("%s repeat %d: %d failed requests dropped from the cache for retry", self.id, k, dropped)
        self._write(repeat_dir, results)
        summary = summarize(results["samples"])
        summary["duration_s"] = round(time.time() - started, 1)
        return summary

    def _write(self, repeat_dir: Path, results: dict) -> None:
        """Per-example artifacts: ``samples/<task>.jsonl`` and lm-eval's
        per-task/group results as ``lm_eval_results.json``."""
        sdir = repeat_dir / "samples"
        sdir.mkdir(exist_ok=True)
        for task, rows in results["samples"].items():
            with (sdir / f"{task}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        keep = {key: results.get(key) for key in ("results", "groups", "configs", "versions", "n-shot", "n-samples", "config")}
        (repeat_dir / "lm_eval_results.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False, default=str))


# -- IFBench -------------------------------------------------------------------

IFBENCH_DATA_COMMIT = "fcd289db21d43aaa96c6d9291d32561cd6e19305"
"""allenai/IFBench main when the pinned ns commit was made (2026-09-11): the
last change to data/IFBench_test.jsonl (2026-08-30). ns's prepare.py reads the
file from ``refs/heads/main``; this is that read, pinned."""
IFBENCH_TEST_URL = "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl"
IFBENCH_TEST_SHA256 = "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a"


@register
class IFBench(NemoSkillsBenchmark):
    """IFBench (AllenAI: 300 prompts with out-of-distribution verifiable output
    constraints) as NeMo-Skills' ``ifbench`` benchmark: ns's generic/default
    prompt (the prompt is the user turn), ns's ifbench evaluator (IFBench's own
    ``run_eval`` strict and loose verifiers, at the IFBench commit ns's image
    pins, see docker/extras/ifbench.sh) and ns's ``if`` metrics. The value is
    the prompt-level loose accuracy (a prompt counts if all its instructions
    are followed by the response or one of IFBench's loose variants of it:
    first/last line or markdown ``*`` removed), mean over 2 generations.

    No gold mode: IFBench has no reference responses."""

    id = "ifbench"
    metric = "pass@1[avg-of-2] loose accuracy"
    default_repeats = 2
    extra = "ifbench"
    harness_packages = ("nemo-skills", "litellm", "spacy", "nltk", "syllapy", "emoji", "langdetect")
    ns_benchmark = "ifbench"
    ns_metric = "prompt_loose_accuracy"
    dataset = f"https://github.com/allenai/IFBench/blob/{IFBENCH_DATA_COMMIT}/data/IFBench_test.jsonl"
    dataset_revision = IFBENCH_DATA_COMMIT
    # ns prepare's test.jsonl from the pinned data (BV job 1956210, image ifbench:952082c).
    prepared_sha256 = "4dcc770a51d3d56d26c3b84410a734582587ce95a7cb7a5a0e58575f483308e3"
    pinned_urls = {
        IFBENCH_TEST_URL: (
            f"https://raw.githubusercontent.com/allenai/IFBench/{IFBENCH_DATA_COMMIT}/data/IFBench_test.jsonl",
            IFBENCH_TEST_SHA256,
        )
    }

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.gold:
            raise SystemExit(f"{self.id}: IFBench has no reference responses, so there is no answers=gold mode")
        return super().run(base_url, served_model_name)
