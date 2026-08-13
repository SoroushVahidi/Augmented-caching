"""Tie-aware exact H4 oracle replay; current condition is regression-gated."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,tempfile
from pathlib import Path
import random
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import SCORE_START,SCORE_END,score_window
from lafc.oracle_diagnostics import _run_replay,select_exact_target_candidate
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy
from lafc.supervision_objective_ablation import ObjectiveAblationConfig

ROOT=Path(__file__).resolve().parents[2]; DATA_ROOT=ROOT.parent/'Augmented-caching'; FOLDS=ROOT/'configs/fair_cross_family_v1/folds'; FAMILIES=['brightkite','citibike','cloudphysics','metacdn','metakv','twemcache','wiki2018']; CAPS=[32,64,128]
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False) as f: json.dump(x,f,indent=2,sort_keys=True);f.write('\n');q=Path(f.name)
 os.replace(q,p)
def fold(f):return json.loads((FOLDS/f'{f}.json').read_text())
def choose(policy,rng=None):
 def fn(rows):
  values={str(r['candidate_page_id']):float(r['eviction_loss_label']) for r in rows}; best=min(values.values()); mins=[str(r['candidate_page_id']) for r in rows if values[str(r['candidate_page_id'])]==best]
  if policy=='CURRENT_DETERMINISTIC': return min(mins)
  if policy=='LRU_WITHIN_MINIMA': return mins[0]
  if policy=='MRU_WITHIN_MINIMA': return mins[-1]
  return rng.choice(mins)
 return fn
def tie_stats(decisions):
 if not decisions:return {'fraction_tied_decisions':0,'fraction_all_tied':0,'mean_optimal_set_fraction':0}
 all_t=sum(len(d.optimal_candidates)==len(d.candidate_values) for d in decisions)
 return {'fraction_tied_decisions':sum(len(d.optimal_candidates)>1 for d in decisions)/len(decisions),'fraction_all_tied':all_t/len(decisions),'mean_optimal_set_fraction':sum(len(d.optimal_candidates)/len(d.candidate_values) for d in decisions)/len(decisions)}
def bool_score(hit_sequence):
 window=hit_sequence[SCORE_START:SCORE_END]; misses=sum(1 for h in window if not h)
 return misses, misses/len(window)
def main(args):
 cfg=json.loads(args.config.read_text()); out=args.out; out.mkdir(parents=True,exist_ok=True); atomic(out/'config_snapshot.json',cfg)
 old={}
 oldp=ROOT/'analysis/exact_target_oracle_replication_v1/policy_comparison.csv'
 with oldp.open() as f:
  for r in csv.DictReader(f):
   if r['policy']=='exact_finite_horizon_eviction_loss_oracle':old[(r['family'],int(r['capacity']))]=int(r['misses'])
 rows=[]; manifest={'status':'RUNNING','expected_rows':189,'completed_units':0,'units':{},'source_head':os.popen('git rev-parse HEAD').read().strip()};atomic(out/'completion_manifest.json',manifest)
 for fam in FAMILIES:
  fd=fold(fam); rel=Path(fd['test_trace_path']); path=rel if rel.is_absolute() else ROOT/rel
  if not path.exists(): path=DATA_ROOT/rel
  req,pages,_=load_trace_from_any(str(path));
  if sha(path)!=fd['test_trace_sha256']:raise RuntimeError('trace hash mismatch '+fam)
  for cap in CAPS:
   unit=out/'units'/f'{fam}_cap{cap}'; unit.mkdir(parents=True,exist_ok=True)
   if (unit/'summary.json').exists(): u=json.loads((unit/'summary.json').read_text());rows+=u['rows'];continue
   unit_rows=[]; ocfg=ObjectiveAblationConfig(horizon=4)
   lru=run_policy(LRUPolicy(),req,pages,cap); ls=score_window(lru.events,SCORE_START,SCORE_END); unit_rows.append({'family':fam,'capacity':cap,'tie_policy':'LRU_REFERENCE','seed':'','misses':ls.misses,'miss_ratio':ls.miss_ratio,'delta_vs_LRU':0,'delta_vs_current_exact':None,'trace_sha256':sha(path)})
   conditions=[('CURRENT_DETERMINISTIC',None),('LRU_WITHIN_MINIMA',None),('MRU_WITHIN_MINIMA',None)]+[('RANDOM_WITHIN_MINIMA',s) for s in cfg['random_seeds']]
   for pol,seed in conditions:
    rng=random.Random(seed) if seed is not None else None
    rep=_run_replay(requests=req,capacity=cap,trace_name=str(fd['test_trace_name']),trace_family=fam,cfg=ocfg,objective='eviction_loss',policy_name=pol,choose_candidate=choose(pol,rng))
    misses,ratio=bool_score(rep.hit_sequence); st=tie_stats([d for d in rep.decisions if SCORE_START<=d.request_t<SCORE_END]); key=(fam,cap)
    if pol=='CURRENT_DETERMINISTIC' and key in old and misses!=old[key]: raise RuntimeError(f'TIE_ORACLE_REGRESSION_FAILED {fam} cap{cap}: {misses}!={old[key]}')
    row={'family':fam,'capacity':cap,'tie_policy':pol,'seed':seed if seed is not None else '','misses':misses,'miss_ratio':ratio,'delta_vs_LRU':misses-ls.misses,'delta_vs_current_exact':None,'trace_sha256':sha(path),**st};unit_rows.append(row)
   current=next(r for r in unit_rows if r['tie_policy']=='CURRENT_DETERMINISTIC');
   for r in unit_rows:r['delta_vs_current_exact']=r['misses']-current['misses']
   atomic(unit/'summary.json',{'status':'COMPLETE','rows':unit_rows});rows+=unit_rows;manifest['units'][f'{fam}_cap{cap}']={'status':'COMPLETE'};manifest['completed_units']=len(manifest['units']);atomic(out/'completion_manifest.json',manifest);print(json.dumps({'unit':f'{fam}_cap{cap}','completed_units':manifest['completed_units']}),flush=True)
 with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 manifest['status']='COMPLETE';atomic(out/'completion_manifest.json',manifest);atomic(out/'integrity_audit.json',{'status':'PASS','rows':len(rows),'expected_rows':189,'unique_keys':len({(r['family'],r['capacity'],r['tie_policy'],str(r['seed'])) for r in rows})})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/tie_aware_exact_target_oracle_v1.json');ap.add_argument('--out',type=Path,default=ROOT/'analysis/tie_aware_exact_target_oracle_v1');main(ap.parse_args())
