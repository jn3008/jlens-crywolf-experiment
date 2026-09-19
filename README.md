# When Does J-Lens Cry Wolf?

A small experiment testing whether J-Lens readouts distinguish visible evaluation recognition from ordinary discussion of evaluations. The main collection uses 24 project-authored prompts: eight task types, each with an ordinary, semantic-control, and evaluation-cue version.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model]'
```

The default configuration is Qwen3-1.7B. `configs/experiment0.json` contains the larger target configuration. Model and lens revisions are pinned in those files.

## collect readouts

```bash
crywolf validate --prompts data/pilot-v2.jsonl
crywolf run --prompts data/pilot-v2.jsonl \
  --config configs/experiment0.json \
  --output outputs/collection
```

Each run stores the rendered prompt, exact tokenisation, generated continuation, stopping reason, per-layer J-Lens ranks and logits, top vocabulary tokens, configuration, and version information. Runs must use a new output directory.

## inspect and analyse

```bash
crywolf inspect --run-dir outputs/collection --example 0000
crywolf analyze --run-dir outputs/collection \
  --output outputs/collection/analysis.json
crywolf thresholds --run-dir outputs/collection \
  --output outputs/collection/thresholds.json
crywolf report --run-dir outputs/collection \
  --output outputs/collection/report.html
```

The analyses are descriptive. A low rank is a strong vocabulary readout; it is not evidence that the model selected or consciously represented that concept.

## figures and viewer

```bash
crywolf heatmap --run-dir outputs/collection \
  --example 0007 --group evaluation --group benchmark
crywolf view --run-dir outputs/collection
python -m http.server --directory outputs/collection/viewer 8000
```

Heatmaps are SVG files. The viewer is a directory of static assets: it must be served over HTTP and loads each example only when selected.

## visible-recognition review

```bash
crywolf probe --reviewer local-llama31-8b
crywolf review --run-dir outputs/collection \
  --reviewer local-llama31-8b \
  --output outputs/collection/review.jsonl
```

Reviewers see only the conversation and continuation. The labels describe visible text (`clear_recognition`, `no_observed_recognition`, or `ambiguous`). they dont measure hidden awareness. Reviewer calibration data is in `data/` and is optional for the J-Lens analysis.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -q
```