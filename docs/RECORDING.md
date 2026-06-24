# Recording the README demo GIF

Goal: one short (~20–25s) loopable GIF that shows autolab *working* — agent
search → report with the derivation tree → exact replay. The whole thing is
scripted in [`scripts/demo.py`](../scripts/demo.py) so it's a single take with
no typing and no API key.

## One-take command

```bash
python scripts/demo.py
```

It prints captions in cyan, pauses ~1.6s between beats, and runs in three beats:

1. **Search** — the agent tunes a text-classification pipeline (30 runs).
2. **Report** — best run, mean accuracy, and the parent→child search tree.
3. **Replay** — the best run reproduces its metric exactly (green `reproduced`).

Tune pacing with the `PAUSE` constant in `scripts/demo.py` if beats feel fast
or slow on playback.

## Recording on Windows (recommended)

**ScreenToGif** (free, GUI, easiest for a clean crop):

1. Open a terminal — **Windows Terminal**, dark theme, font ~16pt, window ~100×30.
2. Pre-run once so Python/import is warm (avoids a cold-start stutter on tape):
   `python scripts/demo.py` then clear the screen.
3. Launch **ScreenToGif → Recorder**, frame the terminal, hit Record.
4. Run `python scripts/demo.py`, let all three beats finish, Stop.
5. Editor → trim dead frames at the ends → **Save as → GIF**. Target < 5 MB so
   GitHub renders it inline (reduce frames/quality or width if over).

## Recording on macOS / Linux / WSL (asciinema → GIF)

```bash
# record an asciinema cast
asciinema rec demo.cast -c "python scripts/demo.py"

# convert to GIF with agg (https://github.com/asciinema/agg)
agg --theme monokai --font-size 22 demo.cast docs/demo.gif
```

`agg` output is crisp and small — usually the best-looking option.

## Embedding in the README

Save the file as `docs/demo.gif`, then add it near the top of `README.md`,
right under the one-line description:

```markdown
![autolab demo](docs/demo.gif)
```

Keep it above the fold — it's the first thing that earns a star.

## Tips

- Dark theme + a high-contrast accent reads best when scaled down on GitHub.
- Don't show the install; show the *result*. The GIF answers "what do I get?",
  the README answers "how do I run it?".
- A clean loop (no cursor blink at the end) feels more polished — trim the last
  idle frames.
