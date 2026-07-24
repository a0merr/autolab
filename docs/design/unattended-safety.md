# Design: safe unattended runs

Status: **implemented** in v0.3.0.

An autonomous loop is only worth starting if it can be walked away from. This
note covers the three ways an overnight `Lab.run()` goes wrong on its own, and
the layer that catches each. The organizing principle: **the loop must not
crash on bad input, and must not keep going on bad output.**

## Threat model

| Failure | What it looks like at 3am | Caught by |
|---|---|---|
| Agent emits a malformed config | `SpaceError` kills the loop mid-search | `space.coerce` |
| Task or environment is broken | 200 identical stack traces, budget spent | `max_consecutive_errors` |
| Training diverges to NaN | "best" run is meaningless; search chases noise | `max_consecutive_nonfinite` + `is_scored` |
| API key wrong / model unknown | whole night of random search that *looks* fine | raise on 4xx + `max_agent_failures` |
| Loop burns credit producing nothing | surprise invoice | `max_cost_usd` |
| Anything unforeseen | still running at noon | `max_seconds` |

## Layer 1: constrain the proposal (`agent.py`)

Rather than validate an arbitrary LLM response after the fact, make the
invalid response unrepresentable. `_batch_schema` derives a JSON Schema from
the parameter space and passes it as `output_config.format`:

```python
{"lr": (1e-5, 1e-1), "dim": [64, 128]}
# becomes
{"lr": {"type": "number"}, "dim": {"enum": [64, 128]}}
# + required: [dim, lr], additionalProperties: false
```

That buys presence and type for free: no missing parameter, no string where a
float belongs, no invented optimizer name. It deliberately does **not** buy
bounds — structured outputs reject `minimum`/`maximum` — so range enforcement
stays in layer 2. This is why the two layers both exist; neither is redundant.

Two API facts drove this shape. `temperature`/`top_p`/`top_k` are removed on
current models (400 on Opus 5, Opus 4.8/4.7, Sonnet 5, Fable 5), so the
determinism knob is `output_config.effort`, defaulted to `"low"`. And
structured outputs are unavailable on some older models, hence the
`structured_output=False` escape hatch and the tolerant `_extract_configs`
parsing path behind it.

## Layer 2: repair, never raise (`space.coerce`)

`clip` stays strict — it is the right contract for a config a human wrote, and
`SpaceError` there is a useful bug report. `coerce` is the loop-facing variant:
per-parameter, best-effort, total.

```python
coerce({"lr": float("nan")}, {"lr": (0.0, 1.0), "dim": [64, 128]}, rng)
# -> {"lr": <resampled>, "dim": <resampled>}   never raises
```

Per-parameter, not all-or-nothing: one bad entry costs one resampled value
rather than discarding an otherwise-good proposal. The RNG is seeded off the
lab seed, so repairs are reproducible along with everything else.

`_as_number` now rejects non-finite values explicitly. `json.loads` parses
`NaN` and `Infinity` by default, and `max(lo, min(hi, nan))` returns `hi` —
so the old code would have accepted a NaN learning rate and silently reported
it as the upper bound.

## Layer 3: stop the loop (`guard.py`)

`CircuitBreaker` is checked at two points, chosen so each limit fires as early
as it can:

- `before_batch(agent)` — clock, cost, and agent-health limits, checked
  *before* spending another round's worth of API calls.
- `observe(run, objective)` — error and divergence limits, checked per run.
  The clock is re-checked here too, so one long experiment stops the search
  when it lands rather than a whole round later.

Nothing is checked mid-experiment, because a running job cannot be preempted.
So `max_seconds` overshoots by at most one experiment and `max_cost_usd` by at
most one round's worth of agent calls. Both bounds are documented on the class
rather than left to be discovered from a bill.

"At most one experiment" is only a bound if experiments end. A job wedged in a
stuck kernel or a socket with no timeout of its own makes it unbounded, and no
breaker can fire while it hangs — the loop is inside `run_batch`, not between
runs. `ProcessExecutor(timeout=…)` is what closes that: past the deadline the
workers are killed, the unfinished jobs are recorded as failed runs, and the
search continues. A hang then reaches `max_consecutive_errors` by the same path
a crash does.

Tripping raises `BreakerTripped` (a `RuntimeError`, so existing handlers still
catch it) carrying the reason and the run count. It is a **stop, not a
rollback** — the store is append-only and everything written before the trip
stays readable via `autolab runs list`. A batch is recorded in full before any
of its results are shown to the breaker: those experiments have already run and
already cost what they cost, so discarding a sibling's record because an
earlier result tripped would throw away paid-for work and leave the store
disagreeing with what actually executed.

Cost is estimated from a cached per-MTok price table plus cache-read/write
multipliers. An unrecognized model is priced at the highest known rate, so an
unknown model makes the cap fire early rather than never. List prices change
and negotiated rates differ, so `PRICING_USD_PER_MTOK` is public and mutable,
and both `TokenUsage.cost_usd` and `CircuitBreaker` accept a `pricing`
override — a stale table must not be the reason a cap silently stops working.

"Highest known rate" is a componentwise max over the table, not `max()` over
its `(input, output)` tuples. Tuple comparison ranks by input rate and only
consults output to break ties, so a table holding a cheap-in/expensive-out
model would have priced an *unknown* model below one already listed — the
opposite of erring high. Making the table public is what put that shape one
caller's edit away.

The breaker is duck-typed against the agent (`usage`, `model`,
`consecutive_failures`, `last_error`), so `RandomAgent` — which has none of
them — simply skips those checks instead of needing a null implementation.

## Layer 4: keep failures visible

A transient API failure degrades to a random sample, because one flaky
response must not kill an eight-hour run. But *silent* degradation is worse
than a crash: a search that quietly became random search still produces a
plausible-looking store. Two mitigations:

- `consecutive_failures` / `last_error` on the agent, so
  `max_agent_failures` catches sustained degradation.
- Configuration errors (400/401/403/404 — bad key, unknown model, rejected
  schema) are **re-raised**, not swallowed. They recur on every call, so
  falling back is never the right answer.

Failures are also made visible to the *agent*, which is a different audience.
A failed run reached the model as an empty metrics dict with no explanation, so
it could not distinguish "this config scored badly" from "this config crashed
the trainer" — and kept proposing into the region that crashes until
`max_consecutive_errors` stopped the search. History rows now carry `failed`
and a 200-character excerpt of the error: enough to recognize an OOM, not
enough for one stack trace to crowd out the rest of the history. An experiment
is only worth its cost if its outcome reaches the next proposal.

## Layer 5: say what happened, after the process is gone

A `BreakerTripped` raised at 3am in a cron job has nowhere to go. So the
outcome is **written to the store** as a `search` note: status, reason, run
count, start and stop stamps, and agent spend.

The note is written on *every* exit path, not only on a trip:

| status | meaning |
|---|---|
| `running` | written before the first experiment; still there ⇒ the process died |
| `completed` | budget spent, best run recorded |
| `tripped` | a breaker limit fired; `reason` names it |
| `crashed` | an exception escaped the loop, including "no successful runs" |

Writing it only on failure would have fixed "no record" while leaving "wrong
record": a search that trips on Monday and a clean one on Tuesday share a store
directory, and Tuesday morning cannot tell a stale reason from a current one.
Always overwriting makes the note describe exactly one search — the last.

Spend in the note is priced with the breaker's *own* pricing table, so the
recorded cost and the cap that fired on it can never disagree.

`autolab status` prints the note and exits non-zero unless the search
completed, which is enough for a cron wrapper to alert on without parsing
anything. `autolab report` prints a banner ahead of the summary for the same
reason: a truncated search otherwise reads as a small but complete one.

Notes live beside the runs but are matched by a distinct filename shape, so
they are excluded from iteration, `len()`, and run-id assignment. (That last
one was a latent bug: `_next_index` counted `*.json`, so *any* stray file in
the store directory would have shifted the next run id.)

## Related fixes: the id allocator

Fixing the glob stopped stray files shifting the next run id, but counting was
still the wrong operation. Delete one run from a five-run store and the count
points back at a record that is still there — which the next `add()` would have
silently overwritten, in a store whose whole contract is immutability. The
allocator is now `max(id) + 1`, and writes use exclusive create (`open("x")`),
so an id that somehow already exists raises instead of clobbering. That also
turns two processes sharing a store directory from silent data loss into an
error naming the collision.

## Related fixes: NaN in the store

Two problems, both found alongside the breaker work, both about a diverged
metric.

**NaN could win `best()`.** NaN compares false against everything, so
`max(runs, key=score)` returned whichever NaN run came first — a diverged run
could be reported as the best result. `store.is_scored` now gates ranking,
mean, standard deviation, and the search tree on a finite value. A NaN run is
still recorded faithfully; it just cannot win.

**NaN made the store non-standard JSON.** `json.dumps` emits a bare `NaN`
token by default. Python reads it back, so autolab never noticed; every other
parser rejects the file. Non-finite *metrics* are now stored as strings
(`"NaN"`, `"Infinity"`, `"-Infinity"`) and decoded on read — symmetric, and
faithful to a real experimental outcome.

Encoding is deliberately scoped to metrics rather than applied to the whole
record. A generic encoder would have to decode generically too, and a
categorical whose legitimate value is the string `"NaN"` would silently become
a float. Everywhere else, `allow_nan=False` raises instead: a non-finite
config cannot arise from `coerce`, so one indicates a bug worth surfacing
rather than round-tripping. That strictness also covers `extra`, which *is*
caller-supplied — documented on `add()` rather than quietly encoded, since the
same decode ambiguity would apply there.

The encoder substitutes only for non-finite values and returns everything else
untouched. Returning `float(value)` unconditionally — as it first did — would
have widened an integer metric to a float on the way to disk, changing a
recorded value in the one place whose job is to record values exactly.

## Deliberately not done

- **Pydantic / Instructor.** The API's own structured outputs cover this with
  no dependency and no second schema to keep in sync with `Space`.
- **Rewriting NaN metrics at the executor.** A NaN loss is a real experimental
  outcome and belongs in the record. Suppressing it at write time would hide
  the divergence the breaker exists to detect.
- **Retry/backoff in the agent.** The SDK already retries 429/5xx twice with
  exponential backoff. A second layer would multiply wall-clock on a genuine
  outage, which is exactly what `max_seconds` is for.
