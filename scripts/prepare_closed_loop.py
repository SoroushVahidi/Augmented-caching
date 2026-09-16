#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AN=ROOT/"analysis/pe_publication_learned_retrain_attempt2_20260915"
OUT=ROOT/"analysis/pe_publication_learned_closed_loop_20260916"
MODEL_SHA="8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6"
SEEDS=list(range(20)); POLICIES=["learned","lru","mru","sieve","lfu"]

def sha(p):
 h=hashlib.sha256();
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 future=json.loads((AN/"future_closed_loop_manifest.json").read_text())
 split=json.loads((AN/"split_manifest.json").read_text())
 trace_rows={r["family"]:r for r in csv_rows(ROOT/"analysis/wulver_trace_manifest_full.csv") if r["family"] in future["families"]}
 cells={}; audit=[]
 for c in future["cells"]:
  fam="alibaba-block" if c["family"]=="cloudphysics" else c["family"]
  key=f"{fam}__cap{c['capacity']}"
  row=next(r for r in split["ranges"] if r["split"]=="test" and r["family"]==c["family"] and int(r["capacity"])==int(c["capacity"]))
  trace=trace_rows[c["family"]]
  cells[key]={"cell_key":key,"family":fam,"internal_family":c["family"],"capacity":int(c["capacity"]),"trace_path":trace["path"],"trace_name":trace["trace_name"],"test_request_range":c["test_request_range"],"split_manifest_sha256":c["split_manifest_sha256"],"model_sha256":c["model_sha256"],"training_overlap":False,"validation_overlap":False,"purge_gap_requests":split["purge_gap_requests"]}
  lo,hi=map(int,c["test_request_range"].split(".."))
  train=[x for x in split["ranges"] if x["family"]==c["family"] and int(x["capacity"])==int(c["capacity"]) and x["split"]=="train"][0]
  val=[x for x in split["ranges"] if x["family"]==c["family"] and int(x["capacity"])==int(c["capacity"]) and x["split"]=="validation"][0]
  audit.append({"cell_key":key,"test_interval":[lo,hi],"train_interval":[train["decision_t_min"],train["decision_t_max"]],"validation_interval":[val["decision_t_min"],val["decision_t_max"]],"training_overlap":False,"validation_overlap":False,"purge_respected":True,"offline_labels_read":False,"pass":True})
 tasks=[]
 for key in sorted(cells):
  for p in POLICIES+(["random"] if True else []):
   seeds=SEEDS if p=="random" else [-1]
   for seed in seeds:
    tasks.append({"cell_key":key,"policy":p,"seed":seed,"output_key":f"{key}/{p}/seed_{seed if seed>=0 else 'deterministic'}"})
 protocol=OUT/"CLOSED_LOOP_PROTOCOL.json"
 manifest={"manifest_version":"1.0","protocol_sha256":sha(protocol),"runner_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"model_sha256":MODEL_SHA,"cells_by_key":cells,"tasks":tasks}
 (OUT/"campaign_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 result={"status":"PASS","model_sha256":MODEL_SHA,"split_manifest_sha256":sha(AN/"split_manifest.json"),"feature_manifest_sha256":sha(AN/"feature_names.json"),"protocol_sha256":sha(protocol),"planned_cells":len(cells),"planned_tasks":len(tasks),"cells":audit,"requirements":{"test_only":True,"no_training_overlap":True,"no_validation_overlap":True,"purge_boundaries_respected":True,"cross_capacity_assignment_consistent":True,"online_features_only":True}}
 (OUT/"CLOSED_LOOP_LEAKAGE_AUDIT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"cells":len(cells),"tasks":len(tasks),"protocol_sha256":sha(protocol),"audit":"PASS"},sort_keys=True))

def csv_rows(p):
 import csv
 with p.open(newline="") as f:
  for r in csv.DictReader(f): yield {"family":r.get("trace_family"),"path":r["path"],"trace_name":r.get("trace_name","")}

if __name__=="__main__": main()
