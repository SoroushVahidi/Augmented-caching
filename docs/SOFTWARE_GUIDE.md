# Software and reproducibility

## Repository map

| Location | Use |
| --- | --- |
| [src/lafc](../src/lafc/) | Maintained Python simulator, policies, metrics, trace readers and dataset preparation |
| [scripts/datasets](../scripts/datasets/) | Compatibility entry points and source-specific preparation tools |
| [configs](../configs/) | Environment constraints and campaign configurations |
| [tests](../tests/) | Deterministic software checks and small fixtures |
| [experiments](../experiments/), [analysis](../analysis/) and [slurm](../slurm/) | Campaign drivers, job scripts and evidence tied to their recorded revisions |
| [historical](../historical/), [reports](../reports/) and [post-acceptance-research](../post-acceptance-research/) | Published and historical evidence, pointers and manifests |

Keep generated runs in a dedicated output directory. The runner defaults to the ignored `output/` directory. The tracked `outputs/` tree contains preserved evidence. Use the [experiment index](research/README.md) and [reviewer matrix](reviewer/REPRODUCTION_MATRIX.md) to select the exact archived code, trace and comparator contract.

## Install and test

Use a supported Python interpreter and a virtual environment. The optional Linux x86-64 [Python 3.12 constraints](../configs/environments/python312-test.txt) capture the tested software environment. The public workflow tests Python 3.10 and 3.12. On a workstation or Wulver, select the interpreter first and use that interpreter to create the environment.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c configs/environments/python312-test.txt -e '.[dev]'
python -m pip check
OMP_NUM_THREADS=1 python -m pytest -q
sha256sum --strict --quiet -c post-acceptance-research/MANIFEST.sha256
```

The core install is `python -m pip install -e .`. For LRB without developer tools, install `python -m pip install -e '.[lrb]'`. Native engines and historical training and replay drivers have separate dependencies and environment pins in their own runbooks. Keep their compiler, baseline and contract revisions with the corresponding result.

## Run and prepare inputs

Use `python -m lafc.runner.run_policy --help` for supported policies and arguments. Online modes include LRU, SIEVE, FIFO-Reinsertion, learned eviction policies and prediction-augmented paging. `offline_belady` is the unit-request offline reference. Weighted general caching uses the [separate offline solver](../scripts/run_offline_general_caching_approx.py). Policy names across native and Python engines do not identify identical timing boundaries. See the [baseline documentation](baselines.md) for their contracts.

```sh
python -m lafc.runner.run_policy --policy lru --trace data/example_unweighted.json --capacity 3 --output-dir output/smoke
lafc-prepare-dataset --help
```

The simulator accepts JSON and CSV request traces. The [dataset preparation guide](datasets.md) describes canonical JSONL and CSV exports. Pass `--raw-dir`, `--output-dir`, and optionally `--example-dir` to the preparation CLI to run from a different working directory. Defaults are relative to the working directory. Use `--dataset` to select an input family and consult its terms before using trace payloads. Download released supervision from the [immutable LAFC-Evict v1.0 revision](https://huggingface.co/datasets/SoroushVahidi/lafc-evict/tree/37173bc96de2a615455bf9713bb90d846156af29).

If a module is missing, check the interpreter and installation with `python -m pip check`. If preparation cannot find a file, check the selected family's expected input layout and the explicit root paths. On Wulver use scripts and explicit roots from the allocation. Submit cluster jobs with a minimal environment, set inputs explicitly in the batch script, and omit credentials when jobs use public or local files.

See [contributing](../CONTRIBUTING.md), [software licensing](../LICENSE), and the [article citation](../CITATION.cff).
