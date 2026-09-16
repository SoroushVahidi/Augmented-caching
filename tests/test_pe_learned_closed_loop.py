from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from pe_learned_closed_loop import MRUPolicy, RandomPolicy, MODEL_SHA

def test_model_identity_and_policy_names():
    assert len(MODEL_SHA) == 64
    assert MRUPolicy.name == "mru"
    assert RandomPolicy.name == "random"
