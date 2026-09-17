"""Write a compact HTML report for collected J-Lens readouts."""
import gzip
import html
import json
from collections import defaultdict
from pathlib import Path


def _load_example(run, directory):
    example_dir = run / "examples" / directory
    transcript = json.loads((example_dir / "transcript.json").read_text())
    if transcript.get("status") != "complete":
        raise ValueError(f"{directory}: transcript is not complete")
    return example_dir, transcript


def _event(group, position, layers, ranks, tokens, boundary, future_positions):
    values = [ranks[position][layer] for layer in layers]
    minimum = min(values)
    min_layers = [layer for layer, value in zip(layers, values) if value == minimum]
    first_layer = 0 if 0 in ranks[position] else layers[0]
    final_layer = layers[-1]
    later = next((p for p in future_positions if p > position), None)
    return {
        "group": group,
        "position": position,
        "token": tokens[position]["text"],
        "token_id": tokens[position]["token_id"],
        "region": "prompt" if position < boundary else "response",
        "min_rank": minimum,
        "min_layers": min_layers,
        "mean_rank": sum(values) / len(values),
        "first_layer": first_layer,
        "first_layer_rank": ranks[position][first_layer],
        "final_layer": final_layer,
        "final_layer_rank": ranks[position][final_layer],
        "later_position": later,
        "tokens_later": None if later is None else later - position,
        "later_token": None if later is None else tokens[later]["text"],
    }


def collect_example(run, directory, manifest):
    example_dir, transcript = _load_example(run, directory)
    layers = manifest["resolved_layers"]
    boundary = transcript["prompt_length"]
    tokens = transcript["tokens"]
    group_ids = {name: set(ids) for name, ids in manifest["group_token_ids"].items()}
    ranks = {name: defaultdict(dict) for name in group_ids}
    with gzip.open(example_dir / "readouts.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            cell = json.loads(line)
            layer, position = cell["layer"], cell["position"]
            for name in group_ids:
                ranks[name][position][layer] = int(cell["groups"][name]["rank"])
    response_group_positions = {
        name: [position for position in range(boundary, len(tokens))
               if tokens[position]["token_id"] in ids]
        for name, ids in group_ids.items()
    }
    output = {
        "id": transcript["id"],
        "condition": transcript["prompt_record"].get("condition", ""),
        "template_id": transcript["prompt_record"].get("template_id", ""),
        "pair_id": transcript["prompt_record"].get("pair_id", ""),
        "prompt": "\n".join(m["content"] for m in transcript["prompt_record"]["messages"]),
        "response": transcript["response"],
        "stop_reason": transcript.get("stop_reason"),
        "prompt_length": boundary,
        "response_tokens": len(transcript["generated_ids"]),
        "groups": {},
    }
    for name in group_ids:
        all_events = [
            _event(name, position, layers, ranks[name], tokens, boundary,
                   response_group_positions[name])
            for position in sorted(ranks[name])
        ]
        prompt_events = [e for e in all_events if e["region"] == "prompt"]
        response_events = [e for e in all_events if e["region"] == "response"]
        prompt_min = min((e["min_rank"] for e in prompt_events), default=None)
        response_min = min((e["min_rank"] for e in response_events), default=None)
        output["groups"][name] = {
            "prompt_min_rank": prompt_min,
            "response_min_rank": response_min,
            "min_events_prompt": [e for e in prompt_events if e["min_rank"] == prompt_min],
            "min_events_response": [e for e in response_events if e["min_rank"] == response_min],
            "top5_events_prompt": [e for e in prompt_events if e["min_rank"] <= 5],
            "top5_events_response": [e for e in response_events if e["min_rank"] <= 5],
        }
    return output


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def _event_table(events):
    if not events:
        return "<p class='muted'>None.</p>"
    rows = []
    for event in events:
        later = "—" if event["later_position"] is None else (
            f"{event['tokens_later']} tokens later: {event['later_token']!r} "
            f"(position {event['later_position']})")
        rows.append("<tr>" + "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in [
            event["position"], event["token"], event["min_rank"], _fmt(event["min_layers"]),
            event["mean_rank"], f"L{event['first_layer']}: {event['first_layer_rank']}",
            f"L{event['final_layer']}: {event['final_layer_rank']}", later]) + "</tr>")
    return "<table><thead><tr><th>Position</th><th>Consumed token</th><th>Min rank</th><th>Min layer(s)</th><th>Mean rank across layers</th><th>First layer</th><th>Final layer</th><th>Later output occurrence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _group_html(name, data):
    return f"""<section class='group'>
<h3>{html.escape(name)}</h3>
<table class='summary'><tr><th>Region</th><th>Lowest rank</th><th>Top-five events</th></tr>
<tr><td>Prompt</td><td>{_fmt(data['prompt_min_rank'])}</td><td>{len(data['top5_events_prompt'])}</td></tr>
<tr><td>Response</td><td>{_fmt(data['response_min_rank'])}</td><td>{len(data['top5_events_response'])}</td></tr></table>
<h4>Minimum-rank event(s) in prompt</h4>{_event_table(data['min_events_prompt'])}
<h4>Minimum-rank event(s) in response</h4>{_event_table(data['min_events_response'])}
<h4>Every response/prompt position whose best layer rank is ≤ 5</h4>{_event_table(data['top5_events_prompt'] + data['top5_events_response'])}
</section>"""


def write_report(run_dir, output):
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Report requires a schema-2 run directory")
    examples = []
    for directory in sorted((run / "examples").iterdir()):
        if directory.is_dir() and (directory / "transcript.json").exists():
            examples.append(collect_example(run, directory.name, manifest))
    blocks = []
    for example in examples:
        groups = "".join(_group_html(name, data) for name, data in example["groups"].items())
        blocks.append(f"""<article>
<h2>{html.escape(example['id'])}</h2>
<p class='meta'>Condition: {html.escape(example['condition'])} · template: {html.escape(example['template_id'])} · pair: {html.escape(example['pair_id'])} · stop: {html.escape(str(example['stop_reason']))}</p>
<h3>Prompt</h3><pre>{html.escape(example['prompt'])}</pre>
<h3>Response</h3><pre>{html.escape(example['response'])}</pre>
<h3>Human annotation</h3><textarea placeholder='Add the final human label, evidence, and notes here.'></textarea>
{groups}</article>""")
    template = """<!doctype html><meta charset='utf-8'><title>J-Lens result report</title>
<style>body{font:14px system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px;color:#172333;background:#f6f8fb}article,section{background:#fff;border:1px solid #d7e0eb;border-radius:8px;padding:18px;margin:18px 0}pre{white-space:pre-wrap;background:#f3f6fa;padding:12px;border-radius:5px;max-height:280px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 16px}th,td{border:1px solid #d7e0eb;padding:5px;text-align:left;vertical-align:top}th{background:#edf2f7}textarea{width:100%;height:80px;box-sizing:border-box}.meta,.muted{color:#536477}.group{margin-top:24px}</style>
<h1>J-Lens result report</h1><p class='meta'>Readout ranks only; reviewer classifications intentionally excluded. A lower rank means stronger full-vocabulary prominence. “Later output occurrence” uses exact single-token concept members.</p>
__BLOCKS__"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__BLOCKS__", "\n".join(blocks)), encoding="utf-8")
    return len(examples)
