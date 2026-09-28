"""BIRD (birdbench) tests: execution-match scoring, data extraction and the run
loop (gold and model mode) on a synthetic dev set, with fakes (no network)."""

import hashlib
import io
import json
import sqlite3
import zipfile
from types import SimpleNamespace

import pytest

from sage2_evals.benchmarks import birdbench
from sage2_evals.registry import RunConfig, get


def make_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score REAL);
        INSERT INTO t VALUES (1, 'a', 1.0), (2, 'b', 2.0), (3, 'c', 3.0);
    """)
    conn.commit()
    conn.close()
    return path


QUESTIONS = [
    {"question_id": 1, "db_id": "toy", "question": "How many rows?", "evidence": "rows are in t",
     "SQL": "SELECT COUNT(*) FROM t", "difficulty": "simple"},
    {"question_id": 0, "db_id": "toy", "question": "Names?", "evidence": "",
     "SQL": "SELECT name FROM t ORDER BY id", "difficulty": "moderate"},
    {"question_id": 2, "db_id": "toy", "question": "Max score?", "evidence": "",
     "SQL": "SELECT MAX(score) FROM t", "difficulty": "challenging"},
]


@pytest.fixture
def dev(tmp_path):
    d = tmp_path / "data" / birdbench.DEV_DIR
    make_db(d / "dev_databases" / "toy" / "toy.sqlite")
    (d / "dev.json").write_text(json.dumps(QUESTIONS))
    return d


def test_registered_with_suite_metric():
    cls = get("birdbench")
    assert cls is birdbench.BirdBench
    assert cls.metric == "pass@1 execution match" and cls.extra == "bird"
    assert cls.dataset_revision == f"sha256:{birdbench.DEV_ZIP_SHA256}"


def test_execution_match_statuses(tmp_path):
    db = make_db(tmp_path / "toy.sqlite")
    em = birdbench.execution_match
    # Set semantics: row order and duplicates don't matter (BIRD's comparison).
    assert em("SELECT name FROM t ORDER BY id DESC", "SELECT name FROM t", db, 5)["status"] == "correct"
    assert em("SELECT name FROM t UNION ALL SELECT 'a'", "SELECT name FROM t", db, 5)["correct"]
    assert em("SELECT name FROM t WHERE id < 3", "SELECT name FROM t", db, 5)["status"] == "wrong"
    assert em("SELEC nonsense", "SELECT 1", db, 5)["status"] == "pred_error"
    assert em("SELECT 1", "SELECT nope FROM t", db, 5)["status"] == "gold_error"


def test_execution_is_read_only(tmp_path):
    db = make_db(tmp_path / "toy.sqlite")
    out = birdbench.execution_match("DROP TABLE t", "SELECT 1", db, 5)
    assert out["status"] == "pred_error" and "readonly" in out["error"]
    assert birdbench.execution_match("SELECT COUNT(*) FROM t", "SELECT 3", db, 5)["correct"]


def test_execution_timeout_aborts_query(tmp_path):
    db = make_db(tmp_path / "toy.sqlite")
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    out = birdbench.execution_match(slow, "SELECT 1", db, 0.3)
    assert out["status"] == "timeout" and out["exec_s"] < 5
    # The budget is shared: a slow gold after the prediction also times out.
    assert birdbench.execution_match("SELECT 1", slow, db, 0.3)["status"] == "timeout"


def test_question_text_and_summarize():
    q = QUESTIONS[0]
    assert birdbench.question_text(q, False) == "How many rows?"
    assert birdbench.question_text(q, True) == "rows are in t\nHow many rows?"
    assert birdbench.question_text(QUESTIONS[1], True) == "Names?"
    s = birdbench.summarize([
        {"difficulty": "simple", "correct": True, "status": "correct"},
        {"difficulty": "simple", "correct": False, "status": "timeout"},
        {"difficulty": "moderate", "correct": False, "status": "wrong"},
    ])
    assert s["accuracy"] == pytest.approx(1 / 3)
    assert s["by_difficulty"]["simple"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert "challenging" not in s["by_difficulty"]
    assert s["statuses"] == {"correct": 1, "timeout": 1, "wrong": 1}


def test_extract_dev_nested_zip_skips_macos_junk(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("dev_databases/toy/toy.sqlite", b"db")
        z.writestr("dev_databases/.DS_Store", b"x")
        z.writestr("__MACOSX/dev_databases/._toy", b"x")
    outer = tmp_path / "dev.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr(f"{birdbench.DEV_DIR}/dev.json", "[]")
        z.writestr(f"{birdbench.DEV_DIR}/dev_databases.zip", inner.getvalue())
        z.writestr(f"__MACOSX/{birdbench.DEV_DIR}/._dev.json", b"x")
    cache = tmp_path / "cache"
    birdbench.extract_dev(outer, cache)
    dev = cache / birdbench.DEV_DIR
    assert (dev / "dev_databases" / "toy" / "toy.sqlite").read_bytes() == b"db"
    assert not (dev / "dev_databases.zip").exists()
    assert not (dev / "dev_databases" / ".DS_Store").exists()
    assert not (cache / "__MACOSX").exists()


def test_download_rejects_sha_mismatch(tmp_path, monkeypatch):
    import contextlib

    import httpx

    payload = b"not the pinned zip"

    @contextlib.contextmanager
    def fake_stream(method, url, **kw):
        yield httpx.Response(200, content=payload, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    dest = tmp_path / "dev.zip"
    with pytest.raises(RuntimeError, match="sha256"):
        birdbench.download("https://example.invalid/dev.zip", dest, "0" * 64)
    assert not dest.exists() and not dest.with_suffix(".partial").exists()
    birdbench.download("https://example.invalid/dev.zip", dest, hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload


def test_gold_mode_scores_everything_and_resumes(tmp_path, dev):
    cfg = RunConfig(model="none", output_dir=tmp_path / "out", dataset=str(dev), limit=2,
                    workers=2, options={"sql": "gold"})
    b = birdbench.BirdBench(cfg)
    assert not b.needs_server()
    out = b.run("", "")
    assert out["value"] == 1.0 and out["n"] == 2 and out["mode"] == "gold"
    # --limit takes the first questions by question_id.
    assert sorted(p.name for p in (tmp_path / "out" / "repeat-0").iterdir()) == ["0.json", "1.json"]
    rec = json.loads((tmp_path / "out" / "repeat-0" / "1.json").read_text())
    assert rec["pred_sql"] == rec["gold_sql"] and rec["status"] == "correct"
    # A persisted result is reused on restart, not re-executed.
    rec["correct"], rec["status"] = False, "wrong"
    (tmp_path / "out" / "repeat-0" / "1.json").write_text(json.dumps(rec))
    assert birdbench.BirdBench(cfg).run("", "")["value"] == 0.5


def test_model_mode_with_fakes(tmp_path, dev, monkeypatch):
    sent = []

    class Completions:
        def create(self, **kw):
            sent.append(kw)
            user = kw["messages"][-1]["content"]
            sql = {"How many rows?": "SELECT COUNT(*) FROM t", "Names?": "SELECT id FROM t"}.get(user, "SELEC")
            msg = SimpleNamespace(content=f"thinking\n```sql\n{sql}\n```", reasoning="r")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

    class Prompt:
        def fill(self, values):
            assert values["sql_context"] == "CREATE TABLE t (...)"
            return [{"role": "system", "content": "sys"}, {"role": "user", "content": values["question"]}]

    def extract(text):
        return text.split("```sql")[-1].split("```")[0].strip()

    monkeypatch.setattr(birdbench, "_client", lambda url: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(birdbench, "_prompt", lambda: Prompt())
    monkeypatch.setattr(birdbench, "_extractor", lambda dev: extract)
    monkeypatch.setattr(birdbench.BirdBench, "schemas", lambda self, dev: {"toy": "CREATE TABLE t (...)"})

    cfg = RunConfig(model="m", output_dir=tmp_path / "out", dataset=str(dev), workers=1, repeats=2, seed=7,
                    options={"temperature": "0.6", "max_tokens": "100"})
    out = birdbench.BirdBench(cfg).run("http://srv/v1", "served")
    assert out["value"] == pytest.approx(1 / 3) and out["n"] == 3
    assert out["per_repeat"][0]["statuses"] == {"correct": 1, "pred_error": 1, "wrong": 1}
    assert out["sampling"] == {"temperature": 0.6, "max_tokens": 100}
    assert {kw["seed"] for kw in sent} == {7, 8} and all(kw["model"] == "served" for kw in sent)
    assert all("top_p" not in kw for kw in sent)  # unset: the checkpoint's generation_config
    rec = json.loads((tmp_path / "out" / "repeat-1" / "1.json").read_text())
    assert rec["pred_sql"] == "SELECT COUNT(*) FROM t" and rec["reasoning"] == "r"


def test_nemo_skills_prompt_and_extraction(tmp_path):
    pytest.importorskip("nemo_skills")
    prompt = birdbench._prompt()
    messages = prompt.fill({"question": "How many rows?", "sql_context": "CREATE TABLE t (id INT);"})
    assert messages[0]["role"] == "system" and "```sql" in messages[0]["content"]
    assert "How many rows?" in messages[-1]["content"] and "CREATE TABLE t" in messages[-1]["content"]
    extract = birdbench._extractor(tmp_path)
    assert extract("x\n```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```").strip() == "SELECT 2"


def test_nemo_skills_schema_context(tmp_path, dev):
    pytest.importorskip("nemo_skills")
    cfg = RunConfig(model="m", output_dir=tmp_path / "out", dataset=str(dev))
    schemas = birdbench.BirdBench(cfg).schemas(dev)
    assert "CREATE TABLE t" in schemas["toy"]
    assert (dev.parent / "sql_context.json").exists()
