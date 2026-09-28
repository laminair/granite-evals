"""BFCL v4 (Berkeley Function Calling Leaderboard), Sage2 metric "overall_accuracy".

The pinned ``bfcl-eval`` package does everything that defines the number: test
data (shipped inside the package, so the package pin is the dataset pin), tool
schemas, the multi-turn / memory / web-search execution environments, the AST
and state checkers, and the leaderboard formula. The headline value is BFCL's
own "Overall Acc" (score/data_overall.csv): a 10/10/10/30/40 weighted mix of
non-live AST, live AST, irrelevance, multi-turn and agentic (web search +
memory) over all 22 scoring categories.

Handler: function-calling (FC) mode through BFCL's ``OpenAICompletionsHandler``
against the vLLM endpoint, i.e. the checkpoint's own chat template and tool
parser do the tool-call formatting, as for every "(FC)" leaderboard entry. (BFCL's
prompt-mode Granite handler hardcodes the Granite 3/4.0 prompt format, which is
not Granite 4.2's.) The thin subclass here only changes what is sent: sampling
parameters are sent only when set as options, so the checkpoint's
generation_config applies by default (BFCL itself sends temperature=0.001).

Web search: BFCL's ``WebSearchAPI.search_engine_query`` calls SerpAPI's DuckDuckGo
engine. Here it is patched, in the BFCL child process only, to call IBM's
google-search MCP server (tool ``google_pse_search``) and map its results onto
the same ``{title, href, body}`` shape; ``fetch_url_content`` is BFCL's own. This
is a documented deviation: search results (and so web-search scores) are not
identical to SerpAPI/DuckDuckGo's.

BFCL runs in a child process (``python -m sage2_evals.benchmarks.bfcl``) because
it reads ``BFCL_PROJECT_ROOT`` at import time and keeps state in module globals.
Generation resumes natively (finished test ids are skipped); inference errors
that look like infrastructure failures are dropped on restart so they are retried.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sage2_evals.registry import Benchmark, register

log = logging.getLogger(__name__)

HARNESS = "bfcl-eval"
HARNESS_VERSION = "2026.3.23"
REGISTRY_NAME = "sage2-fc"  # BFCL model-registry key; no "_" (BFCL maps "_" <-> "/")
DEFAULT_SEARCH_MCP_URL = "https://mcp.ete-server.vpc-int.res.ibm.com/mcp"
SEARCH_TOOL = "google_pse_search"
CHILD_CONFIG = "sage2_bfcl_config.json"
SEARCH_STATS = "web_search_stats.json"

# An inference error with one of these in its message is an infrastructure
# failure, not a model failure: drop it on restart so the entry is regenerated.
INFRA_ERROR = re.compile(
    r"Connection error|APIConnectionError|APITimeoutError|Request timed out|InternalServerError"
    r"|Error code: 5\d\d|ServiceUnavailable|RemoteProtocolError",
)


@register
class BFCLv4(Benchmark):
    id = "bfcl-v4"
    metric = "overall_accuracy accuracy"
    default_repeats = 1
    extra = "bfcl"
    harness_packages = (HARNESS,)
    # The data ships inside the pinned package (bfcl_eval/data), so the dataset
    # is the exact package release, pinned by its wheel's sha256.
    dataset = f"https://pypi.org/project/{HARNESS}/{HARNESS_VERSION}/"
    dataset_revision = "sha256:3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    def sampling(self) -> dict[str, Any]:
        """Sampling sent with every request; unset ones come from the checkpoint's
        generation_config (vLLM applies it)."""
        out: dict[str, Any] = {}
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                out[key] = cast(self.config.options[key])
        return out

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.repeats != 1:
            raise SystemExit("bfcl-v4 is a single-run metric; use --repeats 1")
        root = self.config.output_dir / "bfcl"
        root.mkdir(parents=True, exist_ok=True)
        categories = [c for c in self.opt("categories", "all_scoring").split(",") if c]
        child = {
            "served_model_name": served_model_name,
            "categories": categories,
            "limit": self.config.limit,
            "num_threads": self.config.workers,
            "sampling": self.sampling(),
            "search_mcp_url": self.opt("search_mcp_url", DEFAULT_SEARCH_MCP_URL),
            "include_input_log": self.opt("include_input_log", "false").lower() == "true",
        }
        (root / CHILD_CONFIG).write_text(json.dumps(child, indent=2))
        env = {
            **os.environ,
            "BFCL_PROJECT_ROOT": str(root),
            "OPENAI_BASE_URL": base_url,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "EMPTY",
            "TQDM_DISABLE": "1",
        }
        log.info("%s: categories=%s limit=%s workers=%d sampling=%s", self.id, ",".join(categories),
                 self.config.limit, self.config.workers, child["sampling"] or "generation_config")
        cmd = [sys.executable, "-m", "sage2_evals.benchmarks.bfcl", str(root / CHILD_CONFIG)]
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"bfcl child process exited with {proc.returncode}")

        per_category = read_category_scores(root / "score" / REGISTRY_NAME)
        overall = read_overall(root / "score" / "data_overall.csv")
        stats_path = root / SEARCH_STATS
        search_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        for cat, s in per_category.items():
            log.info("%s: %s accuracy=%.4f (%d/%d)", self.id, cat, s["accuracy"], s["correct_count"], s["total_count"])
        log.info("%s: overall_accuracy=%.4f", self.id, overall["overall"])
        return {
            "value": overall["overall"],
            "n": sum(s["total_count"] for s in per_category.values()),
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "harness": f"{HARNESS}=={HARNESS_VERSION}",
            "handler": "OpenAICompletionsHandler (FC) via vLLM tool parser",
            "sampling": child["sampling"] or "checkpoint generation_config",
            "categories": categories,
            "partial": self.config.limit is not None,
            "groups": overall["groups"],
            "per_category": per_category,
            "web_search_backend": {
                "type": f"mcp:{SEARCH_TOOL}",
                "url": child["search_mcp_url"],
                "replaces": "SerpAPI duckduckgo engine (BFCL default)",
                **search_stats,
            },
        }


# -- result parsing (parent side, no harness import) ------------------------


def _pct(text: str) -> float | None:
    text = text.strip()
    if not text or text == "N/A":
        return None
    return float(text.rstrip("%")) / 100


OVERALL_GROUPS = {
    "non_live": "Non-Live AST Acc",
    "live": "Live Acc",
    "multi_turn": "Multi Turn Acc",
    "web_search": "Web Search Acc",
    "memory": "Memory Acc",
    "relevance": "Relevance Detection",
    "irrelevance": "Irrelevance Detection",
}


def read_overall(path: Path) -> dict[str, Any]:
    """BFCL's own "Overall Acc" and group scores from data_overall.csv, as fractions."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one model row, found {len(rows)}")
    row = rows[0]
    return {
        "overall": _pct(row["Overall Acc"]) or 0.0,
        "groups": {k: _pct(row[col]) for k, col in OVERALL_GROUPS.items()},
    }


def read_category_scores(score_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-category header ({accuracy, correct_count, total_count}) of each score file."""
    out = {}
    for path in sorted(score_dir.rglob("BFCL_v4_*_score.json")):
        cat = path.name[len("BFCL_v4_"):-len("_score.json")]
        with path.open() as f:
            header = json.loads(f.readline())
        out[cat] = {k: header[k] for k in ("accuracy", "correct_count", "total_count")}
    return out


# -- test selection (--limit) -----------------------------------------------


def select_ids(entries_by_category: dict[str, list[dict]], limit: int | None) -> dict[str, list[str]]:
    """First ``limit`` scored entries of every category, in the given (harness)
    order. Memory entries bring the prerequisite conversations they depend on."""
    out = {}
    for cat, entries in entries_by_category.items():
        scored = [e for e in entries if "prereq" not in e["id"]]
        chosen = scored if limit is None else scored[:limit]
        ids = {e["id"] for e in chosen}
        for e in chosen:
            ids.update(e.get("depends_on") or [])
        out[cat] = [e["id"] for e in entries if e["id"] in ids]
    return out


def prune_infra_errors(result_file: Path) -> int:
    """Drop results that failed for infrastructure reasons, so resume retries them."""
    if not result_file.exists():
        return 0
    keep, dropped = [], 0
    for line in result_file.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        res = rec.get("result")
        if isinstance(res, str) and res.startswith("Error during inference") and INFRA_ERROR.search(res):
            dropped += 1
            continue
        keep.append(line)
    if dropped:
        result_file.write_text("".join(line + "\n" for line in keep))
    return dropped


# -- web search through the MCP server ---------------------------------------


_RESULT_HEAD = re.compile(r"^\s*\d+\.\s+(.*)$")


def parse_search_text(text: str) -> list[dict[str, str]]:
    """Parse google_pse_search's text result ("Found N results for: q", then
    numbered blocks: title / URL / snippet) into BFCL's {title, href, body}."""
    results = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        if len(lines) < 2:
            continue
        m = _RESULT_HEAD.match(lines[0])
        if not m or not lines[1].startswith(("http://", "https://")):
            continue
        results.append({"title": m.group(1).strip(), "href": lines[1], "body": " ".join(lines[2:]).strip()})
    return results


def _jsonrpc_payload(response) -> dict:
    """A JSON-RPC message from a streamable-HTTP response (plain JSON or SSE)."""
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    msg = json.loads(data)
                    if "result" in msg or "error" in msg:
                        return msg
        raise RuntimeError("no JSON-RPC result in event stream")
    return response.json()


class MCPSearch:
    """Minimal streamable-HTTP MCP client for one tool (thread-safe, re-initializes
    the session on expiry, retries throttling and server errors with backoff)."""

    def __init__(self, url: str, *, client=None, max_attempts: int = 8, backoff_s: float = 2.0):
        import httpx

        self.url = url
        self.client = client or httpx.Client(timeout=60.0)
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.session: str | None = None
        self._lock = threading.RLock()
        self._id = 0
        self.stats = {"calls": 0, "errors": 0, "retries": 0, "throttled": 0, "empty": 0}

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            h["mcp-session-id"] = self.session
        return h

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _initialize(self) -> None:
        self.session = None
        r = self.client.post(self.url, headers=self._headers(), json={
            "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "sage2-evals-bfcl", "version": "1"}},
        })
        r.raise_for_status()
        self.session = r.headers.get("mcp-session-id")
        self.client.post(self.url, headers=self._headers(),
                         json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    def search(self, query: str, num_results: int) -> list[dict[str, str]]:
        self.stats["calls"] += 1
        body = {"jsonrpc": "2.0", "method": "tools/call",
                "params": {"name": SEARCH_TOOL, "arguments": {"query": query, "num_results": num_results}}}
        backoff = self.backoff_s
        last = ""
        for attempt in range(self.max_attempts):
            try:
                with self._lock:
                    if self.session is None:
                        self._initialize()
                r = self.client.post(self.url, headers=self._headers(), json=body | {"id": self._next_id()})
                if r.status_code in (400, 404) and "session" in r.text.lower():
                    with self._lock:
                        self.session = None
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    self.stats["throttled" if r.status_code == 429 else "retries"] += 1
                    last = f"HTTP {r.status_code}"
                    raise _Retry()
                r.raise_for_status()
                msg = _jsonrpc_payload(r)
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                result = msg["result"]
                text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
                if result.get("isError"):
                    if "429" in text or "rate" in text.lower():
                        self.stats["throttled"] += 1
                        last = text[:200]
                        raise _Retry()
                    raise RuntimeError(text[:500])
                hits = parse_search_text(text)
                if not hits:
                    self.stats["empty"] += 1
                return hits
            except _Retry:
                pass
            except Exception as e:  # network errors: retry; the last one is reported
                last = f"{type(e).__name__}: {e}"
                if isinstance(e, RuntimeError) and str(e).startswith("MCP error"):
                    break
            self.stats["retries"] += 1
            time.sleep(backoff + random.uniform(0, backoff))
            backoff = min(backoff * 2, 60)
        self.stats["errors"] += 1
        raise RuntimeError(f"web search failed after {self.max_attempts} attempts: {last}")


class _Retry(Exception):
    pass


def make_search_engine_query(search: MCPSearch):
    """A replacement for WebSearchAPI.search_engine_query with BFCL's signature
    and return shape ({title, href, body}, body dropped without snippets)."""

    def search_engine_query(self, keywords: str, max_results: int | None = 10, region: str | None = "wt-wt"):
        # region: the Google PSE tool has no region parameter; ignored.
        n = max(1, min(int(max_results or 10), 10))
        try:
            hits = search.search(keywords, n)
        except Exception as e:
            print(f"[WebSearchAPI/MCP] {e}", flush=True)
            return {"error": str(e)}
        if not hits:
            return {"error": "Failed to retrieve the search results from server. Please try again later."}
        if self.show_snippet:
            return hits[:n]
        return [{"title": h["title"], "href": h["href"]} for h in hits[:n]]

    return search_engine_query


# -- the BFCL child process ----------------------------------------------------


def _make_handler(sampling: dict[str, Any], progress: dict[str, int]):
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    class Sage2FCHandler(OpenAICompletionsHandler):
        """BFCL's OpenAI-compatible FC handler; sends sampling only when configured
        and keeps the (vLLM) reasoning trace out of the chat history but in the log."""

        def _query_FC(self, inference_data: dict):
            message = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
            kwargs = {"messages": message, "model": self.model_name, **sampling}
            if tools:
                kwargs["tools"] = tools
            return self.generate_with_backoff(**kwargs)

        def _parse_query_response_FC(self, api_response):
            data = super()._parse_query_response_FC(api_response)
            msg = api_response.choices[0].message
            history: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                history["tool_calls"] = [
                    {"id": tc.id, "type": tc.type,
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            data["model_responses_message_for_chat_history"] = history
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            if reasoning:
                data["reasoning_content"] = reasoning
            return data

        def write(self, result, result_dir, update_mode=False):
            super().write(result, result_dir, update_mode=update_mode)
            items = result if isinstance(result, list) else [result]
            for item in items:
                progress["done"] += 1
                print(f"bfcl-v4: generated {progress['done']}/{progress['total']} {item['id']}", flush=True)

    return Sage2FCHandler


def child_main(config_path: str) -> None:
    from types import SimpleNamespace

    cfg = json.loads(Path(config_path).read_text())
    root = Path(os.environ["BFCL_PROJECT_ROOT"])

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.constants.eval_config import TEST_IDS_TO_GENERATE_PATH
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search
    from bfcl_eval.utils import (
        get_directory_structure_by_category,
        get_file_name_by_category,
        is_memory,
        load_dataset_entry,
        parse_test_category_argument,
        sort_key,
    )

    categories = parse_test_category_argument(cfg["categories"])
    entries = {c: sorted(load_dataset_entry(c), key=sort_key) for c in categories}
    ids = select_ids(entries, cfg["limit"])
    Path(TEST_IDS_TO_GENERATE_PATH).write_text(json.dumps(ids, indent=2))
    progress = {"done": 0, "total": sum(len(v) for v in ids.values())}
    print(f"bfcl-v4: {progress['total']} entries in {len(ids)} categories", flush=True)

    MODEL_CONFIG_MAPPING[REGISTRY_NAME] = ModelConfig(
        model_name=cfg["served_model_name"],
        display_name=f"{cfg['served_model_name']} (FC, sage2)",
        url="",
        org="",
        license="",
        model_handler=_make_handler(cfg["sampling"], progress),
        is_fc_model=True,
        underscore_to_dot=True,  # OpenAI-style tool names: "." is sent as "_"
    )

    search = MCPSearch(cfg["search_mcp_url"])
    web_search.WebSearchAPI.search_engine_query = make_search_engine_query(search)

    result_root = root / "result" / REGISTRY_NAME
    for cat in categories:
        if is_memory(cat):
            continue  # memory prereq chains carry state; never partially regenerate
        f = result_root / get_directory_structure_by_category(cat) / get_file_name_by_category(cat, is_result_file=True)
        if n := prune_infra_errors(f):
            print(f"bfcl-v4: {cat}: retrying {n} entries that hit infrastructure errors", flush=True)

    generation_main(SimpleNamespace(
        model=[REGISTRY_NAME], test_category=cfg["categories"], temperature=sampling_temperature(cfg),
        include_input_log=cfg["include_input_log"], exclude_state_log=False, num_gpus=1,
        num_threads=cfg["num_threads"], gpu_memory_utilization=0.9, backend="vllm",
        skip_server_setup=True, local_model_path=None, result_dir=None, allow_overwrite=False,
        run_ids=True, enable_lora=False, max_lora_rank=None, lora_modules=None,
    ))
    (root / SEARCH_STATS).write_text(json.dumps(search.stats, indent=2))
    evaluation_main([REGISTRY_NAME], cfg["categories"], None, None, cfg["limit"] is not None)


def sampling_temperature(cfg: dict) -> float:
    # Only used for BFCL's own bookkeeping; Sage2FCHandler sends cfg["sampling"].
    return float(cfg["sampling"].get("temperature", 0.0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    child_main(sys.argv[1])
