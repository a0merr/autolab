# autolab
 
**A framework for autonomous ML experimentation: an agent proposes hypotheses, runs experiments against any model or task, and analyzes the results — with versioned runs, full reproducibility, and a plugin interface for custom objectives.**
 
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
 
> **Status: working, early.** The core loop, versioned run store, replay, CLI, analysis, and both agent backends are implemented and tested (see the [roadmap](#roadmap) for what's done vs. planned). autolab grew out of an earlier project that auto-tuned a small GPT; this is the generalization of that idea into a model- and task-agnostic framework. Try it in 30 seconds with `python quickstart.py` — no API key required.
 
---
 
## What this is
 
Most ML progress is a loop: form a hypothesis, run an experiment, read the results, decide what to try next. `autolab` puts an LLM agent inside that loop and gives it the infrastructure to run the loop *responsibly* — every run versioned, every result reproducible, every experiment comparable to the last.
 
The agent isn't the hard part. The hard part is the plumbing that makes its experiments trustworthy: seeds, configs, logged metrics, and a record you can audit. autolab is that plumbing, with an agent driving it.
 
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
- [x] Example tasks: hyperparameter tuning, prompt/pipeline optimization
- [ ] Example task: feature selection
- [ ] Parallel experiment execution
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
