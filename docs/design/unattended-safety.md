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

Tripping raises `BreakerTripped` (a `RuntimeError`, so existing handlers still
catch it) carrying the reason and the run count. It is a **stop, not a
rollback** — the store is append-only and everything written before the trip
stays readable via `autolab runs list`.

Cost is estimated from a cached per-MTok price table plus cache-read/write
multipliers. An unrecognized model is priced at the highest known rate, so an
unknown model makes the cap fire early rather than never.

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

## Related fix: NaN could win `best()`

Not a breaker concern, but found alongside it. NaN compares false against
everything, so `max(runs, key=score)` returned whichever NaN run came first —
a diverged run could be reported as the best result. `store.is_scored` now
gates ranking, mean, standard deviation, and the search tree on a finite
value. A NaN run is still recorded faithfully; it just cannot win.

## Deliberately not done

- **Pydantic / Instructor.** The API's own structured outputs cover this with
  no dependency and no second schema to keep in sync with `Space`.
- **Rewriting NaN metrics at the executor.** A NaN loss is a real experimental
  outcome and belongs in the record. Suppressing it at write time would hide
  the divergence the breaker exists to detect.
- **Retry/backoff in the agent.** The SDK already retries 429/5xx twice with
  exponential backoff. A second layer would multiply wall-clock on a genuine
  outage, which is exactly what `max_seconds` is for.
