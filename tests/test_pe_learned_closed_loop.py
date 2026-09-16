import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pe_learned_closed_loop import MRUPolicy, RandomPolicy, MODEL_SHA, policy
from lafc.policies.lfu import LFUPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists

ROOT = Path(__file__).parents[1]
CAMPAIGN_DIR = ROOT / "analysis" / "pe_publication_learned_closed_loop_20260916"


def test_model_identity_and_policy_names():
    assert len(MODEL_SHA) == 64
    assert MRUPolicy.name == "mru"
    assert RandomPolicy.name == "random"


def test_lfu_resolves_via_runner_dispatch():
    resolved = policy("lfu", model_path=Path("unused"), seed=0)
    assert isinstance(resolved, LFUPolicy)
    assert resolved.name == "lfu"


def test_lfu_closed_loop_smoke():
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "A", "C", "B"])
    result = run_policy(policy("lfu", model_path=Path("unused"), seed=0), requests, pages, capacity=2)
    assert result.total_hits == 1
    assert result.total_misses == 4
    assert result.events[3].evicted == "B"
    assert result.events[4].evicted == "C"


def test_family_naming_cloudphysics_maps_to_alibaba_block():
    protocol = json.loads((CAMPAIGN_DIR / "CLOSED_LOOP_PROTOCOL.json").read_text())
    manifest = json.loads((CAMPAIGN_DIR / "campaign_manifest.json").read_text())

    assert protocol["family_internal_ids"]["alibaba-block"] == "cloudphysics"
    assert "alibaba-block" in protocol["families"]
    assert "cloudphysics" not in protocol["families"]

    cells = manifest["cells_by_key"]
    families = {c["family"] for c in cells.values()}
    assert families == {"alibaba-block", "metacdn", "metakv", "twemcache", "wiki2018"}
    assert len(families) == 5

    mapped_cell = cells["alibaba-block__cap32"]
    assert mapped_cell["family"] == "alibaba-block"
    assert mapped_cell["internal_family"] == "cloudphysics"
