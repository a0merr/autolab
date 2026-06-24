# Design: parallel experiment execution

Status: **Phase 1 implemented** in v0.2.0. Phases 2–3 still open.

This note is the original plan, the touch points, and the concurrency hazards.
Phase 1 (the `Executor` seam, `SerialExecutor`/`ProcessExecutor`, batched
rounds, `concurrency`, per-job error capture, determinism test) shipped as
described. The `AnthropicAgent.propose_batch` diverse-batch call also landed
early. Async fill-the-pool and per-worker env capture remain for later phases.

## The core tension

autolab's value is a *sequential* loop: the agent proposes from the full
history, observes, proposes again. True parallelism wants *independent* work.
These fight. The design must reconcile them, not pretend they don't conflict.

Resolution: **batch the proposals.** The agent proposes `K` configs from the
current history, all `K` run in parallel, results are collected, then the next
batch proposes from the now-larger history. Wall-clock drops from `budget`
serial runs to `ceil(budget / K)` rounds. The agent still learns — just at
batch granularity instead of per-run.

(An asynchronous "fill-the-pool" mode — propose one whenever a worker frees,
using pending-aware proposals — squeezes out more utilization but is
nondeterministic and needs the constant-liar trick from Bayesian opt. Deferred
to phase 3.)

## Invariant to preserve: determinism

Today's serial path stays **bit-identical**. For the parallel path, the store
contents must be independent of *completion order*: the parent writes runs in
**submission order**, not as-completed. Same agent + same seeds ⇒ identical
store, regardless of how the scheduler interleaves jobs. Only wall-clock
varies. This is a tested invariant, not a hope.

## Architecture: an `Executor` seam

New module `autolab/executors.py`:

```python
class Executor(ABC):
    def run_batch(self, task_name: str, jobs: list[tuple[dict, int]]) -> list[Result]:
        """Run (config, seed) jobs; return metrics (or error) in job order."""

class SerialExecutor(Executor):   # default — today's behavior, no deps
class ProcessExecutor(Executor):  # ProcessPoolExecutor, spawn context
class ThreadExecutor(Executor):   # for IO-bound / API tasks (phase 2)
```

Workers receive only `(task_name, config, seed)` — **never a pickled Task
instance**. The worker reconstructs the task by its `module:Class` path (reuse
`replay.load_task`), which sidesteps pickling closures, open files, or model
handles. The worker entrypoint is a module-level function (picklable):

```python
def _run_job(task_name, config, seed) -> Result:
    task = load_task(task_name)
    seed_everything(seed)        # MUST run in the worker, not the parent
    return Result(metrics=task.run(config, seed))
```

## Touch points

**`lab.py`** — `run()` goes from per-iteration to per-round:

```python
done = 0
while done < self.budget:
    k = min(self.concurrency, self.budget - done)
    history = self.store.list(newest_first=False)
    parent = self._parent_for(history)            # best-so-far at round start
    raw = self.agent.propose_batch(self._space, history, ..., k)
    configs = [space.clip(r, self._space) for r in raw]
    jobs = [(cfg, self.seed + done + j) for j, cfg in enumerate(configs)]
    results = self.executor.run_batch(self.task.qualified_name(), jobs)
    for (cfg, seed), res in zip(jobs, results):   # parent owns ALL writes
        self.store.add(..., config=cfg, seed=seed, metrics=res.metrics,
                       parent=parent, error=res.error)
    done += k
```

New `__init__` params: `concurrency: int = 1`, `executor: Executor | None = None`
(defaults to `SerialExecutor`). `concurrency=1` ⇒ today, byte-identical.

**`agent.py`** — add `propose_batch(space, history, objective, direction, k)`.
Base default: call `propose` `k` times (correct for `RandomAgent` — its RNG
advances, giving `k` independent draws). `AnthropicAgent` overrides with a
single API call asking for `k` *diverse* configs as a JSON list; falls back to
`k` random draws on any failure.

**`store.py`** — the main hazard (below). Decision: **workers compute, parent
writes.** The store stays single-writer, so `_next_index()` never races. Add an
optional `error: str | None` field to `Run` so failed jobs are recorded
honestly (budget count + audit stay truthful) instead of silently dropped.

**`env.py`** — captured once in the parent. Fine for a local pool. For a future
*distributed* executor, capture env per-worker and attach to each run. Deferred.

## Hazards

1. **Store id race** — two concurrent writers pick the same `_next_index()` and
   collide. *Solved* by keeping all writes in the single-threaded parent. (If
   workers must ever write directly — large artifacts — switch `run_id` to
   collision-free uuid and make `add` atomic with `open(path, "x")` + retry.)
2. **Seeding location** — `seed_everything` in the parent is useless across
   processes. It must run *inside the worker* before `task.run`.
3. **Pickling** — pass `task_name` + config + seed, reconstruct via import.
   Never pickle the Task instance.
4. **CUDA + fork** — fork after CUDA init corrupts the context. Use
   `mp_context="spawn"` in `ProcessExecutor`.
5. **Thread oversubscription** — a process pool × torch/MKL intra-op threads =
   thrash. Document setting `OMP_NUM_THREADS`/`torch.set_num_threads(1)` per
   worker.
6. **Agent value vs concurrency** — high `K` weakens within-batch learning
   (the `K` configs can't see each other's results). Guidance: small `K` (4–8)
   with `AnthropicAgent`; large `K` fine with `RandomAgent`.
7. **Within-batch duplicates** — agent may propose the same config twice.
   Mitigation: prompt for diversity; optional dedupe before submit.
8. **Per-job failure** — one crash must not kill the round. Capture per job,
   store an `error` record, continue.
9. **`parent` semantics** — all `K` in a round share the round-start best as
   parent; the search tree branches `K`-wide per round. Intended; document it.

## Rollout

- **Phase 1 (MVP, the bulk of the value):** `Executor` seam +
  `SerialExecutor`/`ProcessExecutor`; round loop + `concurrency` in `Lab`;
  default `propose_batch`; parent-owns-writes; `error` field. Determinism test:
  `SerialExecutor` and `ProcessExecutor` produce an identical store for
  `RandomAgent` at a fixed seed.
- **Phase 2:** `AnthropicAgent.propose_batch` (diverse `K` in one call);
  `ThreadExecutor`; richer error reporting.
- **Phase 3 (stretch):** async fill-the-pool executor with pending-aware
  proposals; per-worker env capture for distributed backends.

## Tests to add

- `SerialExecutor` vs `ProcessExecutor` ⇒ byte-identical store (determinism).
- `concurrency > 1` respects `budget` exactly (no over/under-run).
- a deliberately-crashing task ⇒ `error` record stored, loop completes.
- seed offsets correct across rounds (`seed + done + j`).
