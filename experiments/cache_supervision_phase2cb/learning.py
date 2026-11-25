"""Matched small estimators, packed inference and validation-only calibration."""
from dataclasses import asdict, dataclass
from time import perf_counter, process_time, process_time_ns

from numba import njit
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SEED=20261125
GRID=(0.,.005,.01,.025,.05,.1,.2,.4,None)
EPSILONS=(0.,.01,.025,.05,.10)
RISK_GRID=(.25,.5,.75,1.)
ARCHS=("ridge","hgb")


@dataclass(frozen=True)
class Spec:
    stage: str
    name: str
    arch: str
    budget: int
    target: str
    features: str = "F0"
    shortlist: str = "low_frequency"
    k: int = 8


def clean(x):
    active=[]
    for j in range(x.shape[1]):
        if np.ptp(x[:,j])>0 and not any(np.array_equal(x[:,j],x[:,q]) for q in active):
            active.append(j)
    return np.asarray(active,dtype=np.intp)


@njit(cache=True, nogil=True)
def packed_trees(x, roots, f, cutoff, left, right, leaf_value, bias, mean, scale):
    result=np.empty(len(x),dtype=np.float64)
    for i in range(len(x)):
        total=bias
        for root in roots:
            node=root
            while left[node]>=0:
                j=f[node]
                v=(x[i,j]-mean[j])/scale[j]
                node=left[node] if v<=cutoff[node] else right[node]
            total+=leaf_value[node]
        result[i]=total
    return result


def pack_tree(estimator,scaler):
    roots=[]
    parts=[[] for _ in range(5)]
    end=0
    for (tree,) in estimator._predictors:
        n=tree.nodes
        roots.append(end)
        for p,v in zip(parts,(n["feature_idx"].astype(np.int64),n["num_threshold"],
                              np.where(n["is_leaf"],-1,n["left"].astype(np.int64)+end),
                              n["right"].astype(np.int64)+end,n["value"])):
            p.append(v)
        end+=len(n)
    return (np.asarray(roots,np.int64),*[np.concatenate(a) for a in parts],
            float(estimator._baseline_prediction.item()),scaler.mean_,scaler.scale_)


def fit(spec, train):
    cpu,wall=process_time(),perf_counter()
    n,k,d=train["x"].shape
    x=train["x"].reshape(-1,d).astype(np.float64,copy=False)
    columns=clean(x) if spec.features=="F1" else np.arange(d,dtype=np.intp)
    x=x[:,columns]
    scaler=StandardScaler().fit(x)
    normalized=scaler.transform(x)
    if spec.target=="two_part":
        targets=(train["prevented"],train["additional"])
    else:
        targets=(train["y"],)
    fit_models=[]
    packed=[]
    for y in targets:
        if spec.arch=="ridge":
            estimator=Ridge(alpha=10.,solver="cholesky")
        else:
            estimator=HistGradientBoostingRegressor(max_iter=30,max_depth=3,min_samples_leaf=32,
                learning_rate=.10,l2_regularization=10.,early_stopping=False,random_state=SEED)
        # One total group weight. Fixed-K matched fits use the same effective penalty.
        estimator.fit(normalized,y.ravel(),sample_weight=np.full(n*k,8/k))
        fit_models.append(estimator)
        if spec.arch=="ridge":
            w=estimator.coef_ / scaler.scale_
            packed.append((w,float(estimator.intercept_-scaler.mean_ @ w)))
        else:
            packed.append(pack_tree(estimator,scaler))
    return dict(spec=asdict(spec),columns=columns,scaler=scaler,models=fit_models,packed=packed,
                train_groups=n,candidate_rows=n*k,train_resolved_positive=int((train["resolved"].max(axis=1)>0).sum()),
                train_target_positive=int((train["y"].max(axis=1)>0).sum()),
                training_cpu_seconds=process_time()-cpu,training_wall_seconds=perf_counter()-wall,
                storage_shape=x.shape)


class Scorer:
    def __init__(self, artifact, reference=False):
        self.artifact=artifact
        self.spec=Spec(**artifact["spec"])
        self.reference=reference
        self.columns=artifact["columns"]
        self.complete=np.array_equal(self.columns,np.arange(artifact["storage_shape"][1])) and self.spec.features!="F1"

    def __call__(self, x, profile=None):
        a=self.artifact
        x=np.asarray(x,dtype=np.float64)
        if not self.complete:
            x=x[:,self.columns]
        start=process_time_ns() if profile is not None else 0
        if self.reference:
            scaled=a["scaler"].transform(x)
            normalized=process_time_ns() if profile is not None else 0
            raw=[m.predict(scaled) for m in a["models"]]
            if profile is not None:
                profile["normalization"]+=normalized-start
                profile["model_api"]+=process_time_ns()-normalized
        elif self.spec.arch=="ridge":
            raw=[x @ w + bias for w,bias in a["packed"]]
            if profile is not None:
                profile["linear_dot"]+=process_time_ns()-start
        else:
            raw=[packed_trees(x,*case) for case in a["packed"]]
            if profile is not None:
                profile["tree_traversal_with_normalization"]+=process_time_ns()-start
        if len(raw)==1:
            return raw[0],None
        benefit=np.maximum(0.,raw[0])
        risk=np.clip(raw[1],0.,1.)
        return benefit-risk,risk


def derived(score, rank, risk, rule, value, risk_cap=1.):
    if rule=="raw":
        return int(score.argmax())
    if rule=="oldest_tie":
        eligible=score==score.max()
        return int(np.where(eligible,rank,32767).argmin())
    if rule=="approx":
        eligible=score>=score.max()-value
        return int(np.where(eligible,rank,32767).argmin())
    if value is None:
        return 0
    if rule=="risk":
        allowed=(risk<=risk_cap).copy()
        allowed[0]=True
        score=np.where(allowed,score,-np.inf)
    best=int(score.argmax())
    return best if score[best]-score[0]>value else 0


def wrapped(scorer, valid):
    x=valid["x"]
    flat=x.reshape(-1,x.shape[-1])
    scores,risk=scorer(flat)
    shape=x.shape[:2]
    return scores.reshape(shape),risk.reshape(shape) if risk is not None else None


def decisions(scores,risk,rank,rule,value,risk_cap=1.):
    best=scores.argmax(axis=1)
    if rule=="raw":
        return best
    if rule in ("approx","oldest_tie"):
        eps=value if rule=="approx" else 0.
        allowed=scores>=scores.max(axis=1,keepdims=True)-eps
        return np.where(allowed,rank,32767).argmin(axis=1)
    if value is None:
        return np.zeros(len(scores),dtype=np.int64)
    if rule=="risk":
        allowed=(risk<=risk_cap)
        allowed[:,0]=True
        scores=np.where(allowed,scores,-np.inf)
        best=scores.argmax(axis=1)
    gain=scores[np.arange(len(best)),best]-scores[:,0]
    return np.where(gain>value,best,0)


def summarize_decision(y,chosen,w):
    observed=y[np.arange(len(y)),chosen]
    best=y.max(axis=1)
    overrides=chosen!=0
    benefit=(observed>0)&overrides
    harm=(observed<0)&overrides
    neutral=(observed==0)&overrides
    return dict(groups=len(y),positive_groups=int((best>0).sum()),overrides=int(overrides.sum()),
                beneficial=int(benefit.sum()),harmful=int(harm.sum()),neutral=int(neutral.sum()),
                weighted_net_advantage=float(w@observed),weighted_overrides=float(w@overrides),
                expected_regret=float((w@(best-observed))/w.sum()),
                beneficial_precision=float((w@benefit)/max(1.,w@overrides)),
                beneficial_recall=float((w@benefit)/max(1.,w@(best>0))),
                harmful_intervention_fraction=float((w@harm)/max(1.,w@overrides)))



def reliability(prediction, observed, weight, bounds):
    result=[]
    for i,lo in enumerate(bounds):
        hi=bounds[i+1] if i+1<len(bounds) else np.inf
        region=(prediction>=lo)&(prediction<hi)
        mass=weight*region
        total=float(mass.sum())
        if total:
            result.append(dict(lower=lo,upper=None if not np.isfinite(hi) else hi,rows=int(region.sum()),
                               mean_prediction=float((mass*prediction).sum()/total),
                               mean_observed=float((mass*observed).sum()/total)))
    return result

def validate(artifact,valid,only_rule=None):
    opt=Scorer(artifact)
    scores,risk=wrapped(opt,valid)
    # Verify both raw scores and induced actions between reference and packed paths.
    n=min(256,len(scores))
    small={k:v[:n] for k,v in valid.items()}
    ref,ref_risk=wrapped(Scorer(artifact,True),small)
    np.testing.assert_allclose(scores[:n],ref,rtol=1e-10,atol=1e-10)
    if risk is not None:
        np.testing.assert_allclose(risk[:n],ref_risk,rtol=1e-10,atol=1e-10)
    parity=dict(rows=n,maximum_score_error=float(np.max(np.abs(scores[:n]-ref))),decision_checks=0)
    truth,w,rank=valid["resolved"],valid["weight"],valid["rank"]
    relative=scores-scores[:,:1]
    error=np.abs(relative-valid["y"])
    nz=valid["y"]!=0
    rules=(only_rule,) if only_rule else ("gate",)
    configurations=[]
    selected={}
    predicted_benefit=None
    if risk is not None and "additional" in valid:
        predicted_benefit=scores+risk
        risk_reliability=reliability(risk,valid["additional"],w[:,None],(0.,.05,.10,.25,.50,.75))
        benefit_reliability=reliability(predicted_benefit,valid["prevented"],w[:,None],(0.,.01,.05,.1,.25,.5,1.))
    for rule in rules:
        if rule in ("gate","preliminary"):
            options=[(x,1.) for x in GRID]
        elif rule=="risk":
            options=[(x,r) for x in GRID for r in RISK_GRID]
        elif rule=="approx":
            options=[(x,1.) for x in EPSILONS]
        else:
            options=[(0.,1.)]
        for value,cap in options:
            chosen=decisions(scores,risk,rank,rule,value,cap)
            reference=decisions(ref,ref_risk,rank[:n],rule,value,cap)
            np.testing.assert_array_equal(chosen[:n],reference)
            parity["decision_checks"]+=n
            row=summarize_decision(truth,chosen,w)
            row.update(rule=rule,threshold=value,risk_cap=cap,
                       target_mae=float(np.average(error.mean(axis=1),weights=w)),
                       target_nonzero_mae=float(error[nz].mean()) if nz.any() else 0.,
                       cost_weighted_ranking_error=float((w@(truth.max(axis=1)-truth[np.arange(len(truth)),scores.argmax(axis=1)]))/w.sum()))
            if predicted_benefit is not None:
                row["risk_brier"]=float(np.average(((risk-valid["additional"])**2).mean(axis=1),weights=w))
                row["benefit_mae"]=float(np.average(np.abs(predicted_benefit-valid["prevented"]).mean(axis=1),weights=w))
                row["risk_calibration_bins"]=risk_reliability
                row["benefit_calibration_bins"]=benefit_reliability
            configurations.append(row)
        win=max(configurations,key=lambda v:(v["weighted_net_advantage"],-v["weighted_overrides"],
                                            float("inf") if v["threshold"] is None else v["threshold"],-v["risk_cap"]))
        win["selected"]=True
        selected[rule]={"rule":rule,"value":win["threshold"],"risk_cap":win["risk_cap"]}
    return selected,configurations,parity
