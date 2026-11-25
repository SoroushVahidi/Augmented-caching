"""Deterministic training, atomic epoch checkpoints, and held-out inference."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from features import GLOBAL_NAMES, STATIC_NAMES, GHOST_LIMIT
from network import Architecture, VictimNetwork
from sequences import collate
from supervision import OfflineMetrics, objective_terms


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def evaluate(net, blocks, device, objective, horizon, batch_size=8):
    net.eval()
    by_cell = {}
    with torch.inference_mode():
        for key, cell in blocks.items():
            if not cell:
                continue
            metrics = OfflineMetrics()
            for first in range(0, len(cell), batch_size):
                g,t,x,m,n,o,d,v = collate(cell[first:first+batch_size], device, horizon, net.config.order)
                pred = net(g,t,x,m,burn_in=len(g[0])//2)
                metrics.add(pred,n,o,d,v,m,objective,horizon)
            by_cell[key] = metrics.report()
    return by_cell


def fit(dest, train, val, config, provenance, *, epochs, device, seed=7, objective="T03",
        horizon=16, batch_size=32, lr=0.001, initial=None):
    dest.mkdir(parents=True, exist_ok=True)
    settings = dict(architecture=asdict(config), objective=objective, horizon=horizon, seed=seed,
                    lr=lr,batch_size=batch_size,epochs=epochs,patience=5,clip=1.,weight_decay=.0001,
                    burn_in=128,unroll=128,features=dict(candidate=STATIC_NAMES,request=GLOBAL_NAMES,ghost_limit=GHOST_LIMIT),
                    initial=checksum(initial) if initial else None, provenance=provenance)
    completed = dest / "FIT_COMPLETE.json"
    if completed.exists():
        done = json.loads(completed.read_text())
        if checksum(dest / "selected.pt") != done["selected_sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        saved = torch.load(dest/"selected.pt", weights_only=True, map_location="cpu")
        if saved["settings"] != settings:
            raise ValueError("Completed fit settings differ")
        return done
    set_seed(seed)
    model = VictimNetwork(config).to(device)
    if initial:
        model.load_state_dict(torch.load(initial, weights_only=True, map_location=device)["model"])
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    best = None
    rng = np.random.default_rng(seed)
    epoch_start, selected, curve = 1, 0, []
    last = dest/"latest.pt"
    if last.exists():
        state = torch.load(last, weights_only=True)
        if state["settings"] != settings:
            raise ValueError("Cannot resume a different experiment")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["sampler"]
        random.setstate(state["python"])
        nr = state["numpy"]
        np.random.set_state((nr[0], nr[1].numpy(), nr[2], nr[3], nr[4]))
        torch.set_rng_state(state["torch"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
        epoch_start, selected, best, curve = state["epoch"]+1, state["selected"], state["best"], state["curve"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epoch_start, epochs+1):
        if epoch - selected > 5 and selected:
            break
        start = perf_counter()
        model.train()
        sums = np.zeros(5)
        gradient_norms = None
        order = rng.permutation(len(train))
        steps = 0
        for first in range(0, len(train), batch_size):
            chosen = [train[int(i)] for i in order[first:first+batch_size]]
            g,t,x,m,n,o,d,v = collate(chosen, device, horizon, config.order)
            optim.zero_grad(set_to_none=True)
            pred = model(g,t,x,m,burn_in=len(g[0])//2)
            loss, terms = objective_terms(pred,n,o,d,v,m,objective,horizon)
            if not torch.isfinite(loss):
                raise FloatingPointError((dest,epoch))
            if first == 0:
                gradient_norms = []
                relevant = {"T01":(1.,0.,1.),"T02":(1.,0.,0.),"T03":(1.,.5,.25)}[objective]
                for term, weight in zip(terms, relevant):
                    if weight:
                        grad, = torch.autograd.grad(weight*term, pred, retain_graph=True)
                        gradient_norms.append(float(grad.norm()))
                    else:
                        gradient_norms.append(0.)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optim.step()
            sums += [float(loss.detach()),*(float(q.detach()) for q in terms),float(norm)]
            steps += 1
        schedule.step()
        scores = evaluate(model,val,device,objective,horizon)
        valid = [r for r in scores.values() if r["ranking_loss"] is not None]
        if valid:
            key = (float(np.mean([r["ranking_loss"] for r in valid])),
                   float(np.mean([r["mean_cf_regret"] for r in valid])))
        else:
            key = (0.,float(np.mean([r["mean_cf_regret"] for r in scores.values()])))
        improved = best is None or key < best
        if improved:
            best, selected = key, epoch
        curve.append(dict(epoch=epoch,seconds=perf_counter()-start,train_loss=float(sums[0]/steps),
                          train_rank=float(sums[1]/steps),train_reuse=float(sums[2]/steps),train_value=float(sums[3]/steps),
                          gradient_norm=float(sums[4]/steps),weighted_head_gradient=gradient_norms,
                          validation=scores))
        nr = np.random.get_state()
        payload = dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=schedule.state_dict(),
                       sampler=rng.bit_generator.state,python=random.getstate(),
                       numpy=(nr[0],torch.from_numpy(nr[1]),int(nr[2]),int(nr[3]),float(nr[4])),
                       torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device=="cuda" else [],
                       epoch=epoch,selected=selected,best=best,curve=curve,settings=settings)
        if improved:
            checkpoint(dest/"selected.pt",payload)
        checkpoint(last,payload)
    summary = dict(state="COMPLETED_SUCCESS", epochs=curve[-1]["epoch"], selected_epoch=selected,
                   training_seconds=sum(row["seconds"] for row in curve),
                   selected_sha256=checksum(dest/"selected.pt"), latest_sha256=checksum(last),
                   parameters=sum(p.numel() for p in model.parameters()),
                   peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else 0,curve=curve)
    atomic_json(completed,summary)
    return summary


def load(path, device="cpu"):
    payload = torch.load(path, weights_only=True, map_location=device)
    model = VictimNetwork(Architecture(**payload["settings"]["architecture"])).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()
