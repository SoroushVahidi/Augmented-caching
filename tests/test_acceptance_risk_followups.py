from pathlib import Path
import json
from scripts.experiments.run_tie_aware_exact_target_oracle import choose, tie_stats

def test_tie_policies_only_choose_minima():
    rows=[{'candidate_page_id':'z','eviction_loss_label':1},{'candidate_page_id':'a','eviction_loss_label':1},{'candidate_page_id':'m','eviction_loss_label':2}]
    assert choose('CURRENT_DETERMINISTIC')(rows)=='a'
    assert choose('LRU_WITHIN_MINIMA')(rows)=='z'
    assert choose('MRU_WITHIN_MINIMA')(rows)=='a'

def test_configs_are_frozen_and_row_counts():
    root=Path(__file__).parents[1]
    assert json.loads((root/'configs/common_model_objective_control_v1.json').read_text())['expected_evaluation_rows']==84
    assert json.loads((root/'configs/tie_aware_exact_target_oracle_v1.json').read_text())['expected_rows']==189
