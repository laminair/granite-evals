"""NeMo-Skills code benchmarks (LiveCodeBench v6, SciCode): config, data handling,
gold servers, sandbox checks, and ns's metrics/args (with the nemoskills extra)."""

import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

from sage2_evals import registry
from sage2_evals.benchmarks import nemo_skills as nsb
from sage2_evals.benchmarks import nemo_skills_code as nsc
from sage2_evals.registry import RunConfig

IDS = {
    "livecodebench-v6": ("pass@1[avg-of-2] accuracy", 2, "accuracy"),
    "scicode": ("pass@1[avg-of-2] subtask accuracy", 2, "subtask_accuracy"),
}


def bench(bid, tmp_path, **kw):
    return registry.get(bid)(RunConfig(model="m", output_dir=tmp_path, **kw))


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid):
    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark)
    assert (cls.metric, cls.default_repeats, cls.ns_metric) == IDS[bid]
    assert cls.extra == "nemoskills" and "nemo-skills" in cls.harness_packages
    assert len(cls.dataset_revision) == 40


@pytest.mark.parametrize("bid", IDS)
def test_no_paid_api(bid, tmp_path, monkeypatch):
    """No judge: nothing here may reach a paid endpoint or the meter."""
    from sage2_evals import meter

    monkeypatch.setattr(meter, "metered", lambda *a, **k: pytest.fail("paid API used"))
    b = bench(bid, tmp_path)
    assert b.judge is None and not b.judge_args and not b.judge_generation_module
    assert "judge_base_url" not in b.config.options


def test_pins(tmp_path):
    assert bench("livecodebench-v6", tmp_path).pins() == {nsc.LCB_DATASET: nsc.LCB_REVISION}
    assert bench("scicode", tmp_path).pins() == {nsc.SCICODE_DATASET: nsc.SCICODE_REVISION}


# -- LiveCodeBench -------------------------------------------------------------


def _lcb_rows():
    return [
        {
            "task_id": tid,
            "question_content": f"problem {tid}",
            "starter_code": "",
            "public_test_cases": "[big]",
            "private_test_cases": "big",
            "metadata": "{}",
            "release_version": "v6",
        }
        for tid in ("abc365_a", "zzz999_a", "3533")
    ]


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_lcb_tests_file_and_slim_input(tmp_path):
    b = bench("livecodebench-v6", tmp_path, limit=2)
    rows = b.load_rows(_write(tmp_path / "in.jsonl", _lcb_rows()))
    assert [r["task_id"] for r in rows] == ["abc365_a", "zzz999_a"]  # --limit: first N
    assert all(not set(nsc.LCB_TEST_COLUMNS) & set(r) for r in rows)
    full = [json.loads(line) for line in b.tests_file().read_text().splitlines()]
    assert [r["task_id"] for r in full] == ["abc365_a", "zzz999_a"] and full[0]["private_test_cases"] == "big"
    assert "question" not in rows[0]


def test_lcb_gold_uses_hand_solutions(tmp_path):
    b = bench("livecodebench-v6", tmp_path, options={"answers": "gold"})
    rows = b.load_rows(_write(tmp_path / "in.jsonl", _lcb_rows()))
    assert sorted(r["task_id"] for r in rows) == ["3533", "abc365_a"]
    server = nsb._GoldServer(rows, b.gold_generation)
    text = server.lookup("system...\nproblem 3533\n\nformat")
    assert text.startswith("```python\n") and "finalPositionOfSnake" in text and text.endswith("```")


def test_lcb_hand_solutions_run():
    """The reference programs themselves, on the problems' sample tests."""
    import subprocess

    samples = {
        "abc365_a": [("2023\n", "365"), ("1992\n", "366"), ("1800\n", "365"), ("1600\n", "366")],
        "abc365_b": [("4\n8 2 5 1\n", "3"), ("8\n1 2 3 4 5 10 9 11\n", "6")],
        "abc366_b": [("3\nabc\nde\nfghi\n", "fda\ngeb\nh*c\ni")],
    }
    for tid, cases in samples.items():
        for stdin, want in cases:
            out = subprocess.run([sys.executable, "-c", nsc.LCB_GOLD[tid]], input=stdin, capture_output=True, text=True)
            assert out.stdout.strip() == want, (tid, stdin, out.stderr)
    ns: dict = {}
    exec(nsc.LCB_GOLD["3533"], ns)
    assert ns["Solution"]().finalPositionOfSnake(2, ["RIGHT", "DOWN"]) == 3
    assert ns["Solution"]().finalPositionOfSnake(3, ["DOWN", "RIGHT", "UP"]) == 1
    exec(nsc.LCB_GOLD["3519"], ns)
    assert ns["Solution"]().winningPlayerCount(4, [[0, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 0]]) == 2
    assert ns["Solution"]().winningPlayerCount(5, [[1, 1], [1, 2], [1, 3], [1, 4]]) == 0


# -- SciCode -------------------------------------------------------------------


def _step(desc, header, code):
    return {"step_description_prompt": desc, "function_header": header, "ground_truth_code": code, "return_line": "return x"}


def test_scicode_gold_server_answers_the_next_step():
    rows = [
        {"problem_id": "1", "sub_steps": [_step("Compute A.", "def a(x):", "def a(x):\n    return 1"), _step("Compute B.", "def b(x):", "def b(x):\n    return 2")]},
        {"problem_id": "2", "sub_steps": [_step("Compute C.", "def c(x):", "def c(x):\n    return 3")]},
    ]
    s = nsc.SciCodeGoldServer(rows)
    prompt = f"steps:\nCompute A.\ndef a(x):\n    return 1\n\n{nsc.NEXT_STEP_MARKER}\n\nCompute B.\n\ndef b(x):\n\nreturn x\nDEPENDENCIES"
    assert s.lookup(prompt) == "```python\ndef b(x):\n    return 2\n```"
    assert s.lookup(f"{nsc.NEXT_STEP_MARKER}\nCompute C.\ndef c(x):") == "```python\ndef c(x):\n    return 3\n```"
    assert s.lookup(f"{nsc.NEXT_STEP_MARKER}\nsomething else") == ""


def test_scicode_gold_uses_dev_split(tmp_path, ns):
    assert bench("scicode", tmp_path).split() == "test"
    assert bench("scicode", tmp_path, options={"answers": "gold"}).split() == "dev"


def test_scicode_requires_sandbox_env(tmp_path):
    b = bench("scicode", tmp_path, options={"sandbox_env": str(tmp_path / "nope")})
    with pytest.raises(SystemExit, match="no sandbox env"):
        b.sandbox_python()


def _fake_env(tmp_path):
    """A 'sandbox env' that is this interpreter."""
    env = tmp_path / "env" / "bin"
    env.mkdir(parents=True)
    (env / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (env / "python").chmod(0o755)
    (env / "pip").write_text("#!/bin/sh\necho shim\n")
    (env / "pip").chmod(0o755)
    return str(env.parent)


def test_scicode_sandbox_report(tmp_path, monkeypatch):
    h5 = tmp_path / "test_data.h5"
    h5.write_bytes(b"targets")
    monkeypatch.setattr(nsc, "SCICODE_H5", str(h5))
    monkeypatch.setattr(nsc, "SCICODE_H5_SHA256", hashlib.sha256(b"targets").hexdigest())
    b = bench("scicode", tmp_path, options={"sandbox_env": _fake_env(tmp_path)})
    rows = [{"required_dependencies": "import json\nimport math"}, {"required_dependencies": "import no_such_module_x"}]
    r = b.sandbox_report(rows)
    assert r["h5_sha256_ok"] and r["dependency_blocks"] == 2
    assert list(r["import_failures"]) == ["import no_such_module_x"]
    monkeypatch.setattr(nsc, "SCICODE_H5_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="sha256"):
        b.sandbox_report(rows)


# -- with the harness ---------------------------------------------------------


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


def test_module_args(ns, tmp_path):
    lcb = bench("livecodebench-v6", tmp_path)
    args = lcb.generation_args([])
    assert args[:2] == ["++prompt_config=eval/livecodebench/default_reasoning", "++eval_type=livecodebench"]
    assert args[-1] == f"++eval_config.test_file={tmp_path / 'lcb-tests.jsonl'}"
    assert lcb.split() == "test_v6_2408_2505" and lcb.metrics_type() == "livecodebench"
    sc = bench("scicode", tmp_path)
    assert sc.generation_module() == "nemo_skills.inference.eval.scicode"
    assert sc.uses_sandbox() and sc.metrics_type() == "scicode"
    assert "++prompt_config=eval/scicode/background" in sc.generation_args([])
    for b in (lcb, sc):
        assert not b.uses_judge()


def test_lcb_gold_code_survives_ns_extraction(ns):
    from nemo_skills.evaluation.evaluator.code import preprocess_code

    for tid, code in nsc.LCB_GOLD.items():
        out = preprocess_code({"generation": f"```python\n{code}```", "task_id": tid}, "python")
        assert out["completion"].strip() == code.strip(), tid


def test_scicode_gold_code_survives_ns_extraction(ns):
    from nemo_skills.inference.eval.scicode_utils import extract_python_script

    code = "import numpy as np\ndef f(x):\n    return np.sum(x)\n"
    assert extract_python_script(f"```python\n{code}\n```").strip() == "def f(x):\n    return np.sum(x)"


def test_scicode_metrics_map_to_value(ns, tmp_path):
    b = bench("scicode", tmp_path, repeats=2)

    def rows(name, statuses):
        return _write(
            tmp_path / name,
            [{"eval_status": [{"process_status": s} for s in st], "num_generated_tokens": 1} for st in statuses],
        )

    files = [
        rows("output-rs0.jsonl", [["completed", "error"], ["completed", "completed", "completed"]]),
        rows("output-rs1.jsonl", [["completed", "completed"], ["error", "error", "error"]]),
    ]
    m = b.compute_metrics(files)
    # repeat 0: 4/5 subtasks, repeat 1: 2/5 -> 60%
    assert m["_all_"][b.aggregation()]["subtask_accuracy"] == pytest.approx(60.0)


def test_scicode_sandbox_server_runs(ns, tmp_path):
    """ns's server under the sandbox-env hook (here: this interpreter, if it has flask)."""
    pytest.importorskip("flask")
    pytest.importorskip("IPython")
    if sys.platform != "linux":
        pytest.skip("ns's sandbox server sets RLIMIT_AS, which only Linux allows")
    b = bench("scicode", tmp_path, options={"sandbox_env": _fake_env(tmp_path)})
    with b.ns_sandbox() as args:
        port = args[-1].split("=")[1]
        r = httpx.post(
            f"http://127.0.0.1:{port}/execute",
            json={"generated_code": "echo hi; pip install scipy==1.10.1", "language": "shell", "timeout": 10},
            timeout=60,
        )
        # the env's bin comes first on the sandbox's PATH: its pip shim answers
        assert r.status_code == 200 and r.json()["stdout"].split() == ["hi", "shim"]
