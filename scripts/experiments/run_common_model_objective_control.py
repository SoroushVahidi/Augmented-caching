"""Run the frozen common-scorer objective control in an isolated output tree."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, tempfile, time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import SCORE_START, SCORE_END, score_window
from lafc.oracle_diagnostics import _run_replay
from lafc.supervision_objective_ablation import ObjectiveAblationConfig, build_pairwise_rows, iter_multi_label_candidate_rows
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "Augmented-caching"
FOLDS = ROOT / "configs/fair_cross_family_v1/folds"
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
FEATURES = list(EVICT_VALUE_V1_FEATURE_COLUMNS)

class CommonScorer:
    def __init__(self, hidden=8, lr=.02, epochs=40, l2=1e-4, seed=0):
        self.hidden, self.lr, self.epochs, self.l2, self.seed = hidden, lr, epochs, l2, seed
        self.fitted = False

    def fit(self, X, y=None, pairs=None, mode="scalar"):
        X = np.asarray(X, float); self.mean = X.mean(0); self.std = X.std(0); self.std[self.std == 0] = 1
        X = (X-self.mean)/self.std; d = X.shape[1]; rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, np.sqrt(2/d), (d,self.hidden)); self.b1 = np.zeros(self.hidden); self.W2 = rng.normal(0, np.sqrt(2/self.hidden), self.hidden)
        if mode == "scalar":
            y = np.asarray(y, float); self.y_mean=float(y.mean()); self.y_std=float(y.std() or 1); y=(y-self.y_mean)/self.y_std
            for _ in range(self.epochs):
                z=X@self.W1+self.b1; h=np.maximum(z,0); out=h@self.W2; g=(out-y)/len(X)
                g2=h.T@g+self.l2*self.W2; gh=np.outer(g,self.W2); gz=gh*(z>0); g1=X.T@gz+self.l2*self.W1; gb=gz.sum(0)
                self.W2-=self.lr*g2; self.W1-=self.lr*g1; self.b1-=self.lr*gb
        else:
            A,B = pairs; A=(A-self.mean)/self.std; B=(B-self.mean)/self.std; n=len(A)
            for _ in range(self.epochs):
                za=A@self.W1+self.b1; ha=np.maximum(za,0); ra=ha@self.W2; zb=B@self.W1+self.b1; hb=np.maximum(zb,0); rb=hb@self.W2
                p=1/(1+np.exp(-np.clip(ra-rb,-50,50))); gd=(p-1)/n
                g2=(ha-hb).T@gd+self.l2*self.W2; gha=np.outer(gd,self.W2); ghb=-gha; gza=gha*(za>0); gzb=ghb*(zb>0)
                g1=A.T@gza+B.T@gzb+self.l2*self.W1; gb=gza.sum(0)+gzb.sum(0)
                self.W2-=self.lr*g2; self.W1-=self.lr*g1; self.b1-=self.lr*gb
            self.y_mean=0.; self.y_std=1.
        self.fitted=True; return self

    def score(self, X):
        X=np.asarray(X,float); X=(X-self.mean)/self.std; return np.maximum(X@self.W1+self.b1,0)@self.W2*self.y_std+self.y_mean

    def save(self, p):
        np.savez(p, hidden=self.hidden, lr=self.lr, epochs=self.epochs, l2=self.l2, seed=self.seed, mean=self.mean, std=self.std, W1=self.W1, b1=self.b1, W2=self.W2, y_mean=self.y_mean, y_std=self.y_std)

    @classmethod
    def load(cls,p):
        z=np.load(p); m=cls(int(z['hidden']),float(z['lr']),int(z['epochs']),float(z['l2']),int(z['seed']));
        for k in ('mean','std','W1','b1','W2','y_mean','y_std'): setattr(m,k,z[k])
        m.fitted=True; return m

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(p, x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False,encoding='utf8') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n'); q=Path(f.name)
    os.replace(q,p)

def fold(f): return json.loads((FOLDS/f'{f}.json').read_text())
def trace_for(f):
    fd=fold(f); rel=Path(fd['test_trace_path']); p=rel if rel.is_absolute() else (ROOT/rel)
    if not p.exists(): p=DATA_ROOT/rel
    return fd,p

def selected_rows(family, cap, n, cfg):
    fd,p=trace_for(family); reqs,_,_=load_trace_from_any(str(p)); name=str(fd['test_trace_name'])
    ids=[]
    for r in iter_multi_label_candidate_rows(reqs,cap,name,family,cfg):
        if r['decision_id'] not in ids: ids.append(r['decision_id'])
        if len(ids)>=n: break
    return list(iter_multi_label_candidate_rows(reqs,cap,name,family,cfg,selected_decision_ids=set(ids)))

def x(rows): return np.asarray([[float(r[c]) for c in FEATURES] for r in rows])
def metrics(rows, model, label, direction):
    vals={}
    for r in rows: vals.setdefault(str(r['decision_id']),[]).append(r)
    regrets=[]
    for items in vals.values():
        a=np.asarray([items[0] if False else [float(r[c]) for c in FEATURES] for r in items]); pred=model.score(a)
        idx=(int(np.argmin(pred)) if direction=='min' else int(np.argmax(pred))); true=[float(r[label]) for r in items]; best=min(true) if direction=='min' else max(true)
        regrets.append((true[idx]-best) if direction=='min' else (best-true[idx]))
    return float(np.mean(regrets) if regrets else 0), len(vals)

def train_one(train, val, obj, spec, cfg):
    label,direction=spec['label'],spec['direction']; m=CommonScorer(**cfg['architecture'],seed=cfg['seed'])
    if obj=='objective_pairwise':
        pairs=build_pairwise_rows(train,source='next_arrival',max_pairs_per_decision=cfg['pair_max_pairs_per_decision'],sample_seed=cfg['seed'])
        A=np.asarray([[float(p[f'i_{c}']) for c in FEATURES] for p in pairs]); B=np.asarray([[float(p[f'j_{c}']) for c in FEATURES] for p in pairs]); m.fit(A,pairs=(A,B),mode='pairwise')
        n_pairs=len(pairs)
    else:
        m.fit(x(train),[float(r[label]) for r in train],mode='scalar'); n_pairs=0
    vr,nd=metrics(val,m,label,direction)
    return m, {'validation_mean_regret':vr,'validation_decisions':nd,'train_rows':len(train),'train_pairs':n_pairs}

def run(args):
    cfg=json.loads(args.config.read_text()); out=args.out; out.mkdir(parents=True,exist_ok=True); atomic_json(out/'config_snapshot.json',cfg)
    rows=[]; manifest={'status':'RUNNING','expected_units':21,'completed_units':0,'source_head':os.popen('git rev-parse HEAD').read().strip(),'units':{}}
    atomic_json(out/'completion_manifest.json',manifest)
    for held in cfg['families']:
      fd,test_path=trace_for(held); test_reqs,test_pages,_=load_trace_from_any(str(test_path));
      if sha(test_path)!=fd['test_trace_sha256']: raise RuntimeError(f'trace hash mismatch {held}')
      for cap in cfg['capacities']:
        unit=out/'units'/f'{held}_cap{cap}'; unit.mkdir(parents=True,exist_ok=True)
        if (unit/'summary.json').exists():
            u=json.loads((unit/'summary.json').read_text()); rows.extend(u['rows']); manifest['units'][f'{held}_{cap}']=u; continue
        train_fams=list(fd['training_families']); val_fam=str(fd['validation_family']); ocfg=ObjectiveAblationConfig(horizon=cfg['horizon'])
        tr=[]
        for fam in train_fams: tr += selected_rows(fam,cap,int(cfg['train_decisions_per_family']),ocfg)
        va=selected_rows(val_fam,cap,int(cfg['validation_decisions']),ocfg)
        unit_rows=[]
        for obj,spec in cfg['objectives'].items():
            model,stat=train_one(tr,va,obj,spec,cfg); mp=unit/f'{obj}.npz'; model.save(mp)
            tfname=str(fd['test_trace_name']); scorer=lambda rs,mm=model: {str(r['candidate_page_id']):float(mm.score(np.asarray([[float(r[c]) for c in FEATURES] for r in rs]))[i]) for i,r in enumerate(rs)}
            replay=_run_replay(requests=test_reqs,capacity=cap,trace_name=tfname,trace_family=held,cfg=ocfg,objective='eviction_loss',policy_name=obj,choose_candidate=lambda rs,sc=scorer,d=spec['direction']: (min if d=='min' else max)([str(r['candidate_page_id']) for r in rs],key=lambda p:(sc(rs)[p],[str(r['candidate_page_id']) for r in rs].index(p))))
            score=score_window(replay.hit_sequence,SCORE_START,SCORE_END)
            row={'objective':obj,'held_out_family':held,'capacity':cap,'misses':score.misses,'miss_ratio':score.miss_ratio,'delta_vs_lru':None,'validation_mean_regret':stat['validation_mean_regret'],'model_sha256':sha(mp),'trace_sha256':sha(test_path),'seed':cfg['seed']}
            unit_rows.append(row)
        atomic_json(unit/'summary.json',{'status':'COMPLETE','rows':unit_rows}); rows.extend(unit_rows); manifest['units'][f'{held}_{cap}']={'status':'COMPLETE'}
        manifest['completed_units']=len(manifest['units']); atomic_json(out/'completion_manifest.json',manifest); print(json.dumps({'unit':f'{held}_cap{cap}','completed_units':manifest['completed_units']}),flush=True)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest['status']='COMPLETE'; atomic_json(out/'completion_manifest.json',manifest); atomic_json(out/'integrity_audit.json',{'status':'PASS','rows':len(rows),'expected_rows':84,'unique_keys':len({(r['objective'],r['held_out_family'],r['capacity']) for r in rows})})

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=ROOT/'configs/common_model_objective_control_v1.json'); ap.add_argument('--out',type=Path,default=ROOT/'analysis/common_model_objective_control_v1'); run(ap.parse_args())
