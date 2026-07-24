# autolab
 
**A framework for autonomous ML experimentation: an agent proposes hypotheses, runs experiments against any model or task, and analyzes the results — with versioned runs, full reproducibility, and a plugin interface for custom objectives.**
 
[![CI](https://github.com/a0merr/autolab/actions/workflows/ci.yml/badge.svg)](https://github.com/a0merr/autolab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- Record the demo (see docs/RECORDING.md), save as docs/demo.gif, then
     uncomment the line below to put it above the fold:
![autolab demo](docs/demo.gif)
-->

> **Status: working, early.** The core loop, versioned run store, replay, CLI, analysis, and both agent backends are implemented and tested (see the [roadmap](#roadmap) for what's done vs. planned). autolab grew out of an earlier project that auto-tuned a small GPT; this is the generalization of that idea into a model- and task-agnostic framework.

## Try it in 30 seconds

No API key, no training, no GPU — the demo drives a real search with the built-in random agent:

```bash
git clone https://github.com/a0merr/autolab.git
cd autolab && pip install -e .
python quickstart.py
```

You'll watch the agent run an experiment loop, print a report with the best run and the search tree, and **replay the best run to the exact same metric** — that reproducibility is the whole point. Swap in the LLM agent and your own `Task` to point it at a real model.

---
 
## What this is
 
Most ML progress is a loop: form a hypothesis, run an experiment, read the results, decide what to try next. `autolab` puts an LLM agent inside that loop and gives it the infrastructure to run the loop *responsibly* — every run versioned, every result reproducible, every experiment comparable to the last.
 
The agent isn't the hard part. The hard part is the plumbing that makes its experiments trustworthy: seeds, configs, logged metrics, and a record you can audit. autolab is that plumbing, with an agent driving it.
 
---
 
## Why autolab

There are great tools for parts of this loop. autolab's bet is that an **LLM agent reasoning over your run history** picks better next experiments than a fixed sweep — and that this is only trustworthy if every run is **versioned and replayable**. It pairs the two.

| | Search strategy | Reproducibility | Extension surface |
|---|---|---|---|
| **autolab** | LLM agent reasons over run history (swappable; random fallback) | versioned run store, exact replay built in | one `Task` class (2 methods) |
| Optuna / Hyperopt | samplers (TPE, random, CMA-ES) | study storage; replay is on you | objective function + distributions |
| Ray Tune | schedulers + search algos, distributed | checkpoints; framework-coupled | trainable + config space |
| Weights & Biases | sweeps (grid/random/bayes) + tracking | excellent logging; not an optimizer driver | agent + sweep YAML |

When to reach for autolab: you want the *proposer itself* to be smart and pluggable, you care about auditing exactly why the search went where it did, and you want to point one small interface at anything — a model, a prompt, a pipeline, a feature subset. When a classic sampler or a managed dashboard already fits, use those; autolab is happy to wrap them behind the `Agent` or `Task` interface.

> Honest status: autolab is young. The above is about *design intent and fit*, not feature parity with mature tools. Distributed execution and a sampler zoo are on the [roadmap](#roadmap), not in the box yet.

---
 
## How it works
 
```
   ┌──────────────────────────────────────────────────────┐
   │                                                        │
   ▼                                                        │
 PROPOSE ──▶ RUN ──▶ ANALYZE ──▶ agent reads results, decides next
 (agent picks   (experiment    (metrics logged,         │
  a hypothesis   runs under      compared to prior runs) │
  + config)      a fixed seed)                           │
   ▲                                                        │
   └──────────────── versioned run store ◀─────────────────┘
```
 
1. **Propose.** The agent inspects the history of past runs and proposes the next experiment — a hypothesis plus a concrete config (hyperparameters, data, objective).
2. **Run.** autolab executes the experiment through a task plugin under a fixed seed, capturing the config, environment, and metrics.
3. **Analyze.** Results are logged, versioned, and compared against prior runs. The agent reads the analysis and the loop repeats.
Because every run is captured as a versioned artifact, the whole search is reproducible after the fact — you can replay any run or trace exactly why the agent went where it did.
 
---
 
## Features
 
- **Agent-driven search** — an LLM proposes hypotheses and configs from the run history, instead of a fixed grid or random sweep.
- **Model- and task-agnostic** — anything that fits the `Task` interface can be optimized: a model, a prompt, a pipeline, a hyperparameter set.
- **Versioned runs** — every experiment stores its config, seed, environment, and metrics as an immutable record.
- **Reproducibility by default** — fixed seeds and captured configs mean any run can be replayed exactly.
- **Plugin interface** — add a new objective or task by implementing one small class; no changes to the core loop.
- **Pluggable agent backend** — works with the Anthropic API out of the box; the agent interface is swappable.
- **Safe to leave running** — schema-constrained proposals, sanitized configs, and circuit breakers that stop a diverging or crash-looping search before it burns the night.
---

## Running unattended

An overnight search is only useful if it stops itself when things go wrong. Three layers do that.

**Proposals are constrained, not trusted.** `AnthropicAgent` derives a JSON Schema from your parameter space and passes it as a structured output, so a response is guaranteed to be valid JSON containing every parameter with the right type — a categorical can only be one of your listed choices. Bounds are then enforced by `space.coerce`, which clamps out-of-range numbers, snaps categoricals to the nearest choice, and resamples anything unusable (missing key, `NaN`, a string where a float belongs). A malformed proposal costs one resampled value, never a crashed loop.

> Note: `temperature` is not a knob on current Claude models — it is rejected with a 400. Reasoning depth and cost are controlled with `output_config.effort`, which `AnthropicAgent` defaults to `"low"`.

**Circuit breakers stop a runaway loop.** Every `Lab` gets one by default:

```python
from autolab import CircuitBreaker, Lab

lab = Lab(
    task=TuneClassifier(),
    objective="accuracy",
    budget=200,
    breaker=CircuitBreaker(
        max_consecutive_errors=3,      # crashing task / broken environment
        max_consecutive_nonfinite=3,   # loss diverged to NaN or inf
        max_agent_failures=3,          # agent silently degraded to random search
        max_cost_usd=5.00,             # estimated API spend cap
        max_seconds=8 * 3600,          # wall-clock cap
    ),
)
lab.run()   # raises BreakerTripped when a limit fires
```

Tripping is a stop, not a rollback: everything already recorded stays in the store, and `BreakerTripped` names the limit that fired. Pass `CircuitBreaker.off()` to disable.

**Failures stay visible.** A transient API failure degrades to a random sample so one bad response can't kill an eight-hour run — but it increments `agent.consecutive_failures`, so `max_agent_failures` catches a search that has quietly become random search. Configuration errors (bad key, unknown model, rejected schema) are raised immediately instead, since they would recur on every call. Estimated spend is available any time via `agent.cost_usd()`.

---
 
## Installation
 
```bash
git clone https://github.com/a0merr/autolab.git
cd autolab
pip install -e .
```
 
Requires Python 3.11+. Set your API key in the environment:
 
```bash
export ANTHROPIC_API_KEY=sk-...
```
 
---
 
## Quickstart
 
Define a task by implementing the `Task` interface — `propose_space` describes what the agent is allowed to vary, and `run` executes one experiment and returns metrics. Then hand it to the loop.
 
```python
from autolab import Lab, Task
 
class TuneClassifier(Task):
    def propose_space(self):
        # What the agent is allowed to change.
        return {
            "learning_rate": (1e-5, 1e-1),
            "hidden_dim":    [64, 128, 256, 512],
            "dropout":       (0.0, 0.5),
        }
 
    def run(self, config, seed):
        model = build_model(config, seed)
        acc = train_and_evaluate(model)
        return {"accuracy": acc}   # metrics the agent will optimize
 
lab = Lab(
    task=TuneClassifier(),
    objective="accuracy",     # maximize this
    budget=25,                # number of experiments
)
 
best = lab.run()
print(best.config, best.metrics)
 
lab.report()   # summary of the search + best runs
```
 
Every experiment the agent runs is written to the run store and can be inspected, compared, or replayed later.
 
---
 
## Plugin interface
 
A task is the only thing you have to write. The contract is small:
 
```python
class Task:
    def propose_space(self) -> dict:
        """Describe the parameters the agent may vary."""
 
    def run(self, config: dict, seed: int) -> dict:
        """Run one experiment with this config; return a dict of metrics."""
```
 
That's the whole extension surface. The agent, the run store, reproducibility, and analysis are all handled by the core — a new objective is one class, not a fork.
 
---
 
## Parallel execution

Run experiments in batched rounds across processes. The agent proposes a whole batch from the current history; results are written to the store in submission order, so **the run store is identical whether you run serial or parallel** — only wall-clock changes.

```python
from autolab import Lab, ProcessExecutor
from tasks.sklearn_tuning import GradientBoostingTuning

lab = Lab(
    GradientBoostingTuning(),
    objective="accuracy",
    budget=40,
    concurrency=8,                          # up to 8 experiments at once
    executor=ProcessExecutor(max_workers=8),
)
best = lab.run()
```

The default is `SerialExecutor` with `concurrency=1` — identical to the sequential loop. A job that crashes is captured as a failed run (with its error) rather than killing the batch, so the search always completes and stays auditable. With the LLM agent, keep `concurrency` modest (4–8): the bigger the batch, the less each proposal can learn from the others in the same round.
 
---
 
## Run store & reproducibility
 
Each run is persisted as a versioned record containing:
 
| Field | Purpose |
|---|---|
| `config` | The exact parameters used |
| `seed` | Fixed seed for deterministic replay |
| `metrics` | Everything the task reported |
| `env` | Library versions + hardware, for honest comparison |
| `parent` | The run this one was derived from, so the search is a traceable tree |
 
```bash
autolab runs list                 # every experiment, newest first
autolab runs show <run_id>        # full config + metrics for one run
autolab replay <run_id>           # re-execute it exactly
```
 
---
 
## Project structure
 
```
autolab/
├── autolab/
│   ├── lab.py           # the propose → run → analyze loop
│   ├── agent.py         # LLM agent: proposes hypotheses + configs
│   ├── task.py          # Task base class (the plugin interface)
│   ├── store.py         # versioned run store + replay
│   ├── analysis.py      # metric comparison across runs
│   └── cli.py           # `autolab runs / replay / report`
├── tasks/               # example task plugins
├── tests/
└── README.md
```
 
---
 
## Roadmap
 
- [x] Core propose → run → analyze loop
- [x] Versioned run store with replay
- [x] `Task` plugin interface
- [x] Anthropic agent backend
- [x] CLI: list / show / replay / report
- [x] Analysis: cross-run comparison + search-tree visualization
- [x] Swappable agent backends (`Agent` interface; `RandomAgent` + `AnthropicAgent` ship)
- [x] Example tasks: hyperparameter tuning, prompt/pipeline optimization, feature selection
- [x] Parallel experiment execution (batched rounds; `SerialExecutor` + `ProcessExecutor`)
- [x] Unattended-run safety: schema-constrained proposals, config sanitization, circuit breakers (error / divergence / cost / wall-clock)
---
 
## Testing
 
```bash
pytest
```
 
The loop, the run store, and replay are tested with deterministic stub tasks (no API or training required), so the core is verified without spending tokens or GPU time. Agent behavior is tested against a mocked backend with recorded responses.
 
---
 
## Disclaimer
 
autolab runs code and experiments that an LLM agent proposes. Review task plugins before running them, and run untrusted configurations in a sandbox. The agent optimizes the objective you give it — choose objectives carefully.
 
---
 
## License
 
Released under the MIT License — see [LICENSE](LICENSE).
 
## Contact
 
**Andrew Merritt** — [GitHub](https://github.com/a0merr) · [LinkedIn](https://www.linkedin.com/in/andrew-merritt-ab425537a) · a0merr05@louisville.edu
