import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from lafc.metrics.cost import hit_rate
from lafc.policies.lru import LRUPolicy
from lafc.predictors.offline_from_trace import extract_actual_next_arrivals
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace

ROOT = Path(__file__).resolve().parents[1]


def test_relocated_dataset_cli_and_legacy_aliases(tmp_path):
    examples = tmp_path / "example inputs"
    examples.mkdir()
    sample = examples / "twemcache_sample.jsonl"
    sample.write_text('{"item_id":"a","cost":2}\n{"item_id":"b","cost":3}\n', encoding="utf-8")
    launcher = [["-m", "lafc.datasets.prepare"]]
    launcher.extend(
        [str(ROOT / "scripts" / "datasets" / script)]
        for script in ("prepare_all.py", "prepare_dataset.py")
    )
    outputs = []
    for index, command in enumerate(launcher):
        output = tmp_path / str(index)
        subprocess.run(
            [
                sys.executable,
                *command,
                "--dataset",
                "twemcache",
                "--sample-only",
                "--limit",
                "1",
                "--raw-dir",
                str(tmp_path / "no-raw-data"),
                "--example-dir",
                str(examples),
                "--output-dir",
                str(output),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        outputs.append({path.name: path.read_bytes() for path in (output / "twemcache").iterdir()})
    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0]["trace.jsonl"])["item_id"] == "a"
    assert outputs[0]["requests_only.txt"] == b"a\n"


def test_missing_inputs_report_dataset_without_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "lafc.datasets.prepare", "--dataset", "wiki2018"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Could not prepare wiki2018:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("suffix", ["csv", "json"])
def test_trace_next_arrivals_lru_victims_and_metrics(tmp_path, suffix):
    trace = tmp_path / f"trace.{suffix}"
    ids = list("ABACBA")
    contents = (
        "t,page_id\n" + "".join(f"{t},{page}\n" for t, page in enumerate(ids))
        if suffix == "csv"
        else json.dumps({"requests": ids})
    )
    trace.write_text(contents, encoding="utf-8")
    requests, pages = load_trace(str(trace))
    next_arrivals = [2, 4, 5, math.inf, math.inf, math.inf]
    assert extract_actual_next_arrivals(ids) == next_arrivals
    assert [request.actual_next for request in requests] == next_arrivals
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    assert (result.total_hits, result.total_misses, result.total_cost) == (1, 5, 5)
    assert [event.evicted for event in result.events] == [None, None, None, "B", "A", "C"]
    assert hit_rate(result.events) == pytest.approx(1 / 6)
