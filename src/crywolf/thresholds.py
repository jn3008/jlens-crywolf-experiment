"""Threshold curves for intermediate ranks and final-layer rank one."""
import gzip
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_THRESHOLDS = (1, 2, 5, 10, 20, 50)


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def _read_cells(example_dir):
    with gzip.open(example_dir / "readouts.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _curve(rows, thresholds):
    """Return pooled and per-example conditional rates for one group."""
    by_example = defaultdict(list)
    for row in rows:
        by_example[row["id"]].append(row)
    totals = {threshold: {"eligible": 0, "hits": 0,
                          "intermediate_cells": 0, "total_intermediate_cells": 0}
              for threshold in thresholds}
    example_rows = []
    for example_id, positions in sorted(by_example.items()):
        token_count = len(positions)
        cell_count = sum(len(p["intermediate_ranks"]) for p in positions)
        per_threshold = {}
        for threshold in thresholds:
            eligible = sum(p["min_intermediate_rank"] <= threshold for p in positions)
            hits = sum(p["min_intermediate_rank"] <= threshold and p["final_rank"] == 1
                       for p in positions)
            intermediate_cells = sum(sum(rank <= threshold for rank in p["intermediate_ranks"])
                                     for p in positions)
            total_intermediate_cells = cell_count
            totals[threshold]["eligible"] += eligible
            totals[threshold]["hits"] += hits
            totals[threshold]["intermediate_cells"] += intermediate_cells
            totals[threshold]["total_intermediate_cells"] += total_intermediate_cells
            per_threshold[str(threshold)] = {
                "total_tokens": len(positions),
                "eligible_tokens": eligible,
                "final_rank1_tokens": hits,
                "eligibility_rate": _ratio(eligible, token_count),
                "unconditional_hit_rate": _ratio(hits, token_count),
                "intermediate_cells": intermediate_cells,
                "total_intermediate_cells": total_intermediate_cells,
                "intermediate_cell_rate": _ratio(intermediate_cells, total_intermediate_cells),
                "conditional_probability": _ratio(hits, eligible),
            }
        example_rows.append({"id": example_id, "thresholds": per_threshold, "tokens": token_count})
    total_tokens = sum(row["tokens"] for row in example_rows)
    threshold_rows = []
    for threshold in thresholds:
        total = totals[threshold]
        threshold_rows.append({
            "threshold": threshold,
            "total_tokens": total_tokens,
            "eligible_tokens": total["eligible"],
            "final_rank1_tokens": total["hits"],
            "eligibility_rate": _ratio(total["eligible"], total_tokens),
            "unconditional_hit_rate": _ratio(total["hits"], total_tokens),
            "intermediate_cells": total["intermediate_cells"],
            "total_intermediate_cells": total["total_intermediate_cells"],
            "intermediate_cell_rate": _ratio(total["intermediate_cells"], total["total_intermediate_cells"]),
            "pooled_conditional_probability": _ratio(total["hits"], total["eligible"]),
            "examples": len(example_rows),
            "mean_example_probability": round(sum(
                row["thresholds"][str(threshold)]["conditional_probability"]
                for row in example_rows
                if row["thresholds"][str(threshold)]["conditional_probability"] is not None
            ) / sum(
                row["thresholds"][str(threshold)]["conditional_probability"] is not None
                for row in example_rows
            ), 6) if any(
                row["thresholds"][str(threshold)]["conditional_probability"] is not None
                for row in example_rows
            ) else None,
        })
    return {"thresholds": threshold_rows, "examples": example_rows}


def threshold_analysis(run_dir, region="continuation", thresholds=DEFAULT_THRESHOLDS,
                       prompt_group_field="template_id", output=None):
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Threshold analysis requires a schema-2 run directory")
    prompts = {row["id"]: row for row in json.loads((run / "prompts.json").read_text())}
    layers = manifest["resolved_layers"]
    if len(layers) < 2:
        raise ValueError("Need at least two layers")
    thresholds = tuple(sorted(set(int(value) for value in thresholds)))
    if not thresholds or any(value < 1 for value in thresholds):
        raise ValueError("Thresholds must be positive integers")
    rows = defaultdict(list)
    example_summaries = []
    for directory in sorted((run / "examples").iterdir()):
        transcript_path = directory / "transcript.json"
        if not directory.is_dir() or not transcript_path.exists():
            continue
        transcript = json.loads(transcript_path.read_text())
        if transcript.get("status") != "complete":
            continue
        prompt = prompts.get(transcript["id"], {})
        prompt_group = prompt.get(prompt_group_field, "unknown")
        cells = defaultdict(dict)
        for cell in _read_cells(directory):
            if cell.get("region") != region:
                continue
            for group, value in cell.get("groups", {}).items():
                cells[(group, cell["position"])][cell["layer"]] = int(value["rank"])
        for (group, position), layer_ranks in cells.items():
            if any(layer not in layer_ranks for layer in layers):
                continue
            final_rank = layer_ranks[layers[-1]]
            minimum = min(layer_ranks[layer] for layer in layers[:-1])
            rows[(group, prompt_group, prompt.get("condition", "unknown"))].append({
                "id": transcript["id"], "position": position,
                "min_intermediate_rank": minimum, "final_rank": final_rank,
                "intermediate_ranks": [layer_ranks[layer] for layer in layers[:-1]],
            })
        example_summaries.append({"id": transcript["id"], "prompt_group": prompt_group,
                                 "condition": prompt.get("condition", "unknown")})
    results = {}
    for (group, prompt_group, condition), group_rows in sorted(rows.items()):
        results.setdefault(group, {}).setdefault(prompt_group, {})[condition] = _curve(
            group_rows, thresholds)
    paired_differences = {}
    for group, prompt_groups in results.items():
        for prompt_group, conditions in prompt_groups.items():
            cue = conditions.get("evaluation_cue")
            semantic = conditions.get("semantic_control")
            if not cue or not semantic:
                continue
            paired_differences.setdefault(group, {})[prompt_group] = []
            for cue_row, semantic_row in zip(cue["thresholds"], semantic["thresholds"]):
                paired_differences[group][prompt_group].append({
                    "threshold": cue_row["threshold"],
                    "evaluation_cue_minus_semantic_control": {
                        metric: round(cue_row[metric] - semantic_row[metric], 6)
                        if cue_row[metric] is not None and semantic_row[metric] is not None else None
                        for metric in ("eligibility_rate", "intermediate_cell_rate",
                                       "unconditional_hit_rate", "pooled_conditional_probability")
                    },
                })
    report = {
        "schema_version": 1,
        "run_dir": str(run.resolve()),
        "model": manifest.get("config", {}).get("model"),
        "region": region,
        "prompt_group_field": prompt_group_field,
        "thresholds": list(thresholds),
        "layers": layers,
        "intermediate_layers": layers[:-1],
        "final_layer": layers[-1],
        "definition": "A token is eligible when its minimum intermediate rank is at most X. A hit means rank 1 at the same token in the final layer. Pooled probability is hits divided by eligible tokens.",
        "results": results,
        "paired_differences": paired_differences,
        "examples": example_summaries,
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def print_report(report):
    print(f"Threshold analysis: {report['region']} region; final layer {report['final_layer']}")
    for group, prompt_groups in report["results"].items():
        thresholds = report.get("thresholds") or []
        # Fall back to the embedded rows for older JSON reports.
        if not thresholds:
            for conditions in prompt_groups.values():
                if conditions:
                    thresholds = [row["threshold"] for row in next(iter(conditions.values()))["thresholds"]]
                    break
        for metric, label in (("eligibility_rate", "eligibility rate"),
                              ("unconditional_hit_rate", "unconditional final-rank-1 rate"),
                              ("pooled_conditional_probability", "conditional final-rank-1 rate")):
            print(f"\n{group} — {label}")
            header = ["prompt group", "condition"] + [f"X<={value}" for value in thresholds]
            widths = [max(len(header[0]), max((len(str(name)) for name in prompt_groups), default=0)),
                      max(len(header[1]), max((len(str(condition)) for conditions in prompt_groups.values()
                                              for condition in conditions), default=0))]
            widths.extend(max(len(label_text), 8) for label_text in header[2:])
            line = "  " + "  ".join(f"{label_text:<{widths[index]}}"
                                     for index, label_text in enumerate(header))
            print(line)
            print("  " + "  ".join("-" * width for width in widths))
            for prompt_group, conditions in prompt_groups.items():
                for condition, result in conditions.items():
                    values = {row["threshold"]: row[metric] for row in result["thresholds"]}
                    cells = [str(prompt_group), str(condition)]
                    cells.extend("-" if values.get(value) is None else f"{values[value]:.3f}"
                                 for value in thresholds)
                    print("  " + "  ".join(f"{cell:<{widths[index]}}"
                                           for index, cell in enumerate(cells)))
                print("  " + "-" * sum(widths) + "-" * (2 * (len(widths) - 1)))
    if report.get("paired_differences"):
        print("\nPaired evaluation_cue minus semantic_control differences are in JSON under paired_differences.")
