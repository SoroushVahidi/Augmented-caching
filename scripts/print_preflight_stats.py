import csv
from pathlib import Path
from collections import defaultdict

def main():
    shards_dir = Path("~/lafc-work/Augmented-caching/data/derived/evict_value_v1_wulver_preflight_20260915/shards").expanduser()
    shards = list(shards_dir.glob("**/*.csv"))
    print(f"Processing {len(shards)} CSV shards...")

    decisions = defaultdict(lambda: defaultdict(list))
    row_counts = defaultdict(int)

    for shard in shards:
        family = shard.name.split("__cap")[0].replace("data__processed__", "").replace("__trace.jsonl", "")
        with shard.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                h = int(row["horizon"])
                did = row["decision_id"]
                y_loss = float(row["y_loss"])
                decisions[(family, h)][did].append(y_loss)
                row_counts[(family, h)] += 1

    print("\n--- HORIZON PREFLIGHT SCIENTIFIC REPORT ---")
    for family_h in sorted(decisions.keys()):
        family, h = family_h
        h_decs = decisions[family_h]
        total_decs = len(h_decs)
        tied_decs = 0
        candidate_rows = row_counts[family_h]
        
        for did, losses in h_decs.items():
            if len(losses) > 1 and min(losses) == max(losses):
                tied_decs += 1
                
        tie_rate = tied_decs / total_decs if total_decs > 0 else 0.0
        print(f"Family: {family:<15} H={h:<3} | Decisions: {total_decs:<5} Rows: {candidate_rows:<6} Tie Rate: {tie_rate:.4f}")

if __name__ == "__main__":
    main()
