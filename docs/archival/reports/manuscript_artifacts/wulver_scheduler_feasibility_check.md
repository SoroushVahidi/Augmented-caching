# Wulver scheduler feasibility check — canonical `heavy_r1` eval (jobs 909870, 910352)

**Date (cluster):** 2026-04-12 (EDT).  
**Scope:** Whether pending canonical eval jobs can be made runnable by changing Slurm options **without** blind resubmission and **without** breaking the manuscript-safe `heavy_r1` path.

---

## 1. Current blocker (evidence)

### 1.1 Queue state

```text
$ squeue -u sv96 -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R'
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            909870   general evictv1-     sv96 PD       0:00      1 (ReqNodeNotAvail, Reserved for maintenance)
            910352   general evictv1-     sv96 PD       0:00      1 (ReqNodeNotAvail, Reserved for maintenance)
```

Both jobs use partition `general`, QOS `standard`, walltime **3-00:00:00** (72h), **16 CPUs**, **64G** RAM (from `scontrol show job`).

### 1.2 Slurm’s stated reason (per-job)

Job **909870** (910352 is identical on the fields that matter):

```text
JobState=PENDING Reason=ReqNodeNotAvail,_Reserved_for_maintenance
TimeLimit=3-00:00:00
Partition=general
ReqTRES=cpu=16,mem=64G,node=1,billing=16
```

### 1.3 No estimated start

```text
$ squeue --start -j 909870,910352
             JOBID PARTITION     NAME     USER ST          START_TIME  NODES SCHEDNODES           NODELIST(REASON)
            909870   general evictv1-     sv96 PD                 N/A      1 (null)               (ReqNodeNotAvail, Reserved for maintenance)
            910352   general evictv1-     sv96 PD                 N/A      1 (null)               (ReqNodeNotAvail, Reserved for maintenance)
```

### 1.4 Active maintenance reservation timeline (all affected nodes)

```text
$ sinfo -T
RESV_NAME               STATE           START_TIME             END_TIME     DURATION  NODELIST
Apr_14_Sched_Maint3  INACTIVE  2026-04-14T09:00:00  2026-04-16T21:00:00   2-12:00:00  n[0001-0127,0700-0724,0751-0766,0781-0788,0801-0819,0852,1438,1440,1541,1543,1545,1547]
```

```text
$ scontrol show reservation Apr_14_Sched_Maint3
ReservationName=Apr_14_Sched_Maint3 StartTime=2026-04-14T09:00:00 EndTime=2026-04-16T21:00:00 Duration=2-12:00:00
   Nodes=n[0001-0127,0700-0724,0751-0766,0781-0788,0801-0819,0852,1438,1440,1541,1543,1545,1547] ... Flags=MAINT,IGNORE_JOBS,SPEC_NODES,ALL_NODES
   State=INACTIVE
```

**Interpretation (conservative):** `State=INACTIVE` here means the reservation is **not currently in effect**; it is **scheduled** for 2026-04-14 09:00 through 2026-04-16 21:00. Slurm backfilling will not start a job whose **requested walltime** cannot complete **before** that window on nodes covered by the reservation. At check time (`2026-04-12T10:17:35 EDT`), the time until `2026-04-14T09:00:00` is **well under 72 hours**, so a **72h** job cannot be placed on those nodes until after the reservation ends.

This matches the pending reason **`ReqNodeNotAvail, Reserved for maintenance`**: it is **not** evidence that only `general` is “down for maintenance” today; it is evidence that **the requested runtime cannot fit** before the **upcoming** maintenance reservation on the node pool the scheduler is considering.

---

## 2. Slurm options inspected (commands / objects)

| Inspection | Command / object | Purpose |
|------------|-------------------|---------|
| User queue | `squeue -u sv96` | Confirm IDs, partition, pending reason |
| Job detail | `scontrol show job 909870` / `910352` | Partition, QOS, tres, timelimit, reason |
| Start estimate | `squeue --start -j 909870,910352` | Whether Slurm predicts a start |
| Reservations | `sinfo -T`, `scontrol show reservation Apr_14_Sched_Maint3` | Upcoming maint window and node coverage |
| Node reasons | `sinfo -R` | Drains / admin holds (orthogonal to maint window) |
| Partition policy | `scontrol show partition` for `general`, `gpu`, `bigmem`, `debug`, `course`, `course_gpu` | Max wall, allowed accounts/QOS, node lists |
| User association | `sacctmgr -p show assoc user=sv96 format=User,Account,Partition,QOS,DefaultQOS` | Which QOS names exist on the account |
| QOS catalog | `sacctmgr -n show qos format=Name,MaxWall,...` | Wall limits per QOS (informational) |
| Cluster time | `date` | Anchor “hours until maint” |

**User association (evidence):**

```text
$ sacctmgr -p show assoc user=sv96 format=User,Account,Partition,QOS,DefaultQOS
User|Account|Partition|QOS|Def QOS|
sv96|ikoutis||debug,low,standard||
```

**Partition constraints relevant to “try another partition” (abridged `scontrol show partition` output):**

- **`general`:** `DenyQos=debug,test,course`; nodes `n[0006-0023,...]` (subset of `n0001-0127` reservation range).
- **`gpu`:** same `DenyQos`; GPU-capable nodes, still within reservation nodelist (`n0001-0005`, `1541`, etc.).
- **`bigmem`:** nodes `n[0125-0126]` — inside `n0001-0127` reservation range.
- **`debug`:** `MaxTime=08:00:00`, `AllowQos=debug` — **cannot** host the documented **72h** eval.
- **`course` / `course_gpu`:** `AllowAccounts=courses`, `AllowQos=course` — **not** available to account `ikoutis` / this workflow as documented (`general` + `standard`).

---

## 3. Repo / canonical constraints (evidence)

### 3.1 Documented Slurm convention for this experiment

From `docs/wulver_heavy_evict_value_experiment.md`:

- “Slurm partition/qos: **`general` + `standard`**”
- “CPU-only jobs (no GPU flags)”
- Eval requests **72 hours**; “A **24 hour** eval was observed to **TIMEOUT** before writing the canonical `..._policy_comparison_heavy_r1.{csv,md}` files; **do not shorten this without measuring**.”

### 3.2 Canonical driver and resources

From `slurm/evict_value_v1_wulver_heavy_eval.sbatch`:

- `#SBATCH --partition=general`
- `#SBATCH --qos=standard`
- `#SBATCH --cpus-per-task=16`
- `#SBATCH --mem=64G`
- `#SBATCH --time=72:00:00`

### 3.3 Manuscript-safe artifact definition (unchanged by partition if outputs match)

Canonical outputs are fixed by filename/tag (`EXP_TAG=heavy_r1`) and the designated driver (`docs/evict_value_v1_kbs_canonical_artifacts.md`, `docs/kbs_manuscript_workflow.md`). **In principle**, the same Python command line on the same inputs/outputs could remain manuscript-safe **if** cluster policy allows the job and it **completes** without timeout. **This report does not claim that any alternate partition is policy-correct for this job** (e.g., submitting CPU-only work to `gpu` without cluster guidance).

---

## 4. Would a different Slurm configuration help **soon**, safely?

| Change | Likely helps before maint? | Manuscript-safe / repo-aligned? |
|--------|----------------------------|----------------------------------|
| **Different partition (`gpu`, `bigmem`, …)** | **No** for a **72h** job: candidate node sets are still inside **`Apr_14_Sched_Maint3`**. Also **`course`** is account/QOS-gated away from this user’s documented path. | **Uncertain / not documented** for `gpu`/`bigmem`; **not** aligned with documented `general` + `standard`. |
| **Different QOS (`low`, etc.)** | **No:** QOS does not remove the maintenance reservation or the **72h-vs-time-to-maint** inequality. | Repo documents **`standard`**. |
| **More CPU / more RAM** | **No** for the **scheduling deadlock**: walltime is still **72h**; nodes are still in the same pre-maint window. Might shorten runtime **if** the app scaled perfectly—**not assumed** here. | **Risky** if it encourages compensating by **shortening** walltime. |
| **Shorter walltime** | **Possibly** could allow Slurm to **start** a job before maint (backfill), but repo explicitly warns **24h timed out** and says **not** to shorten **72h** without measurement. | **Not conservative** for canonical claims while `..._heavy_r1.csv` is missing. |
| **“Reservation-aware” sbatch flag** | No Slurm knob was identified that overrides **`MAINT,IGNORE_JOBS`** on the scheduled reservation for normal batch jobs. | N/A |

**Conclusion:** The pending state is explained by **scheduled cluster maintenance** intersecting the **72h** canonical eval walltime on the **shared** CPU node pools. **Partition/QOS tweaks that stay within the documented canonical resource envelope do not create a node pool with a clear 72h runway before 2026-04-14 09:00.**

---

## 5. Resubmission guidance (no jobs submitted in this check)

- **Do not blindly resubmit** duplicate eval jobs: **909870** and **910352** are redundant `evictv1-heavy-eval` submissions; only one should ultimately produce the canonical CSV/MD pair.
- **Immediate resubmit with the same `general` / `standard` / 72h / 16c / 64G** configuration **does not** address the maint backfill constraint; it would likely return to the same pending reason until after **`2026-04-16T21:00:00`** (reservation end) or until walltime is reduced (not manuscript-safe per repo docs).

---

## 6. Final verdict (exactly one)

**WAIT_MAINTENANCE_BLOCKS_ALL_REASONABLE_CONFIGS**

**Rationale:** Evidence from `scontrol show job`, `sinfo -T`, and `scontrol show reservation Apr_14_Sched_Maint3` shows a **cluster-wide** upcoming maintenance window covering the node families used by **`general`** (and other plausible CPU partitions), combined with a **72h** walltime that **does not fit** in the remaining time before that window. Repo documentation requires **`general` + `standard`**, **72h** eval walltime, and **explicitly discourages** shortening walltime without measurement—so **RESUBMIT_WITH_SAFE_CONFIG** is not supported for “start sooner,” and **CANONICAL_PATH_TOO_RISKY_TO_CHANGE** is not the best single label because the issue is **scheduler/maintenance feasibility**, not ambiguity about whether a documented alternate exists.
