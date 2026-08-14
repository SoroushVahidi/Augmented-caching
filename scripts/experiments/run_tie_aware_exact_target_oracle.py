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
CORE_CSV_FIELDS=['family','capacity','tie_policy','seed','misses','miss_ratio','delta_vs_LRU','delta_vs_current_exact','trace_sha256']
DIAGNOSTIC_CSV_FIELDS=['fraction_tied_decisions','fraction_all_tied','mean_optimal_set_fraction']
EXPECTED_TIE_KEYS=[('LRU_REFERENCE',''),('CURRENT_DETERMINISTIC',''),('LRU_WITHIN_MINIMA',''),('MRU_WITHIN_MINIMA','')]+[('RANDOM_WITHIN_MINIMA',s) for s in range(5)]
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
def _norm_seed(seed):
 if seed in (None,''): return ''
 return int(seed)
def summary_csv_fieldnames(rows):
 present=set()
 for r in rows: present.update(r)
 ordered=[f for f in CORE_CSV_FIELDS+DIAGNOSTIC_CSV_FIELDS if f in present]
 extras=sorted(present-set(ordered))
 return ordered+extras
def write_summary_csv(path,rows):
 fieldnames=summary_csv_fieldnames(rows)
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fieldnames,restval=''); w.writeheader(); w.writerows(rows); q=Path(f.name)
 os.replace(q,path)
 return fieldnames
def write_integrity_audit(path,rows,extra=None):
 keys={(r['family'],int(r['capacity']),r['tie_policy'],str(_norm_seed(r.get('seed')))) for r in rows}
 old={}
 oldp=ROOT/'analysis/exact_target_oracle_replication_v1/policy_comparison.csv'
 if oldp.exists():
  with oldp.open() as f:
   for r in csv.DictReader(f):
    if r['policy']=='exact_finite_horizon_eviction_loss_oracle': old[(r['family'],int(r['capacity']))]=int(r['misses'])
 mismatches=[]
 for r in rows:
  if r.get('tie_policy')=='CURRENT_DETERMINISTIC':
   key=(r['family'],int(r['capacity']))
   if key in old and int(r['misses'])!=old[key]: mismatches.append({'family':r['family'],'capacity':int(r['capacity']),'got':int(r['misses']),'expected':old[key]})
 payload={'status':'PASS' if not mismatches else 'FAIL','rows':len(rows),'expected_rows':189,'unique_keys':len(keys),'families':FAMILIES,'capacities':CAPS,'csv_fieldnames':summary_csv_fieldnames(rows),'current_deterministic_matches_prior_exact_oracle':len(mismatches)==0,'current_deterministic_mismatches':mismatches}
 if extra: payload.update(extra)
 if mismatches: payload['status']='FAIL'
 atomic(path,payload); return payload
def aggregate_completed_units(out):
 rows=[]; units={}; seen=set()
 for fam in FAMILIES:
  for cap in CAPS:
   name=f'{fam}_cap{cap}'; p=out/'units'/name/'summary.json'
   if not p.exists(): raise FileNotFoundError(f'missing unit summary: {p}')
   data=json.loads(p.read_text());
   if data.get('status')!='COMPLETE': raise ValueError(f'unit not complete: {p}')
   unit_rows=data.get('rows')
   if not isinstance(unit_rows,list) or len(unit_rows)!=9: raise ValueError(f'expected 9 rows in {p}')
   keys=[]
   hashes=set()
   for r in unit_rows:
    if r.get('family')!=fam or int(r.get('capacity'))!=cap: raise ValueError(f'row family/capacity mismatch in {p}')
    seed=_norm_seed(r.get('seed')); key=(r.get('tie_policy'),seed); keys.append(key)
    full=(fam,cap)+key
    if full in seen: raise ValueError(f'duplicate primary key {full} in {p}')
    seen.add(full); hashes.add(r.get('trace_sha256'))
   if sorted(keys)!=sorted(EXPECTED_TIE_KEYS): raise ValueError(f'policy/seed mismatch in {p}: {keys}')
   if len(hashes)!=1 or None in hashes or '' in hashes: raise ValueError(f'inconsistent trace hash in {p}')
   units[name]={'status':'COMPLETE'}; rows.extend(unit_rows)
 if len(rows)!=189: raise ValueError(f'expected 189 rows, found {len(rows)}')
 return rows,units
def write_campaign_aggregates(out,rows,units,source_head,recovery=False):
 write_summary_csv(out/'summary.csv',rows)
 manifest={'status':'COMPLETE','expected_rows':189,'completed_units':len(units),'units':units,'source_head':source_head}
 if recovery: manifest['recovered_from_units']=True; manifest['recovery_id']='tie_aware_exact_oracle_recovery_20260814'
 atomic(out/'completion_manifest.json',manifest)
 extra={'recovery':recovery}
 if recovery: extra['recovery_id']='tie_aware_exact_oracle_recovery_20260814'
 write_integrity_audit(out/'integrity_audit.json',rows,extra=extra)
def main(args):
 out=args.out
 if getattr(args,'aggregate_only',False):
  rows,units=aggregate_completed_units(out)
  oldp=out/'completion_manifest.json'; source_head=None
  if oldp.exists():
   try: source_head=json.loads(oldp.read_text()).get('source_head')
   except Exception: source_head=None
  if not source_head: source_head=os.popen('git rev-parse HEAD').read().strip()
  write_campaign_aggregates(out,rows,units,source_head,recovery=True); print(json.dumps({'recovered_rows':len(rows),'completed_units':len(units),'status':'COMPLETE'}),flush=True); return
 cfg=json.loads(args.config.read_text()); out.mkdir(parents=True,exist_ok=True); atomic(out/'config_snapshot.json',cfg)
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
 write_campaign_aggregates(out,rows,manifest['units'],manifest['source_head'],recovery=False)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/tie_aware_exact_target_oracle_v1.json');ap.add_argument('--out',type=Path,default=ROOT/'analysis/tie_aware_exact_target_oracle_v1');ap.add_argument('--aggregate-only',action='store_true');main(ap.parse_args())
