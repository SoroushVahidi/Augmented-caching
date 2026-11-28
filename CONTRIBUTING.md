# Contributing

Run the base suite and the tiny LRU example before opening a pull request.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m lafc.runner.run_policy --policy lru --trace data/example_unweighted.json --capacity 3
```

Use the [software guide](docs/SOFTWARE_GUIDE.md) for the qualified Python environment and portable inputs.

Add a request-level regression case when changing cache behavior. Record the trace identity, capacity, cache initialization, score window, tie rule and CPU timing boundary for benchmark changes. Keep archived experiment commands attached to their original revisions.

The source code is [MIT licensed](LICENSE). Consult the [data and reproduction guide](docs/research/METHODOLOGY.md) before using trace inputs. Store generated runs, model weights, private traces and scheduler logs outside the checkout. Commit small, intentionally reviewed fixtures and aggregate evidence with provenance. Cite the published paper using [CITATION.cff](CITATION.cff).
