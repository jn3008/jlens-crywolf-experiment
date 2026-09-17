"""Small, dependency-free summaries for a completed collection."""
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


RANK_THRESHOLDS = (10, 50, 100)


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_review_labels(path):
    # Return usable reviewer labels keyed by prompt/example id
    if not path:
        return {}
    labels = {}
    for record in _read_jsonl(path):
        annotation = record.get("parsed_annotation")
        if record.get("review_status") in {"valid", "valid_after_repair"} and annotation:
            labels[record["id"]] = {
                "label": annotation.get("label"),
                "evidence": annotation.get("evidence", ""),
                "review_status": record["review_status"],
                "reviewer": record.get("reviewer"),
            }
    return labels


def _new_metric():
    return {"count": 0, "rank_sum": 0.0, "log_rank_sum": 0.0,
            "min_rank": None, **{f"rank_le_{n}": 0 for n in RANK_THRESHOLDS}}


def _add_metric(metric, rank):
    metric["count"] += 1
    metric["rank_sum"] += rank
    metric["log_rank_sum"] += math.log10(max(1, rank))
    metric["min_rank"] = rank if metric["min_rank"] is None else min(metric["min_rank"], rank)
    for threshold in RANK_THRESHOLDS:
        metric[f"rank_le_{threshold}"] += rank <= threshold


def _finish_metric(metric):
    count = metric["count"]
    if not count:
        return {"count": 0, "min_rank": None, "mean_rank": None,
                "mean_log10_rank": None, **{f"fraction_rank_le_{n}": None for n in RANK_THRESHOLDS}}
    return {
        "count": count,
        "min_rank": metric["min_rank"],
        "mean_rank": round(metric["rank_sum"] / count, 4),
        "mean_log10_rank": round(metric["log_rank_sum"] / count, 6),
        **{f"fraction_rank_le_{n}": round(metric[f"rank_le_{n}"] / count, 6)
           for n in RANK_THRESHOLDS},
    }


def summarize_example(run_dir, directory, review=None):
    example_dir = Path(run_dir) / "examples" / directory
    transcript = json.loads((example_dir / "transcript.json").read_text())
    if transcript.get("status") != "complete":
        raise ValueError(f"{directory}: transcript is not complete")
    groups = {}
    winners = defaultdict(Counter)
    readout_count = 0
    # Accumulate separately by concept group and prompt/continuation region.
    with gzip.open(example_dir / "readouts.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            cell = json.loads(line)
            readout_count += 1
            region = cell["region"]
            for name, value in cell["groups"].items():
                key = (name, region)
                groups.setdefault(key, _new_metric())
                _add_metric(groups[key], int(value["rank"]))
                winners[key][str(value["winner_token_id"])] += 1
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    expected = len(manifest["resolved_layers"]) * len(transcript["tokens"])
    if readout_count != expected or transcript.get("readout_count") != expected:
        raise ValueError(f"{directory}: incomplete readout coverage ({readout_count}/{expected})")
    metrics = {}
    for (name, region), metric in groups.items():
        metrics.setdefault(name, {})[region] = _finish_metric(metric)
    winner_summary = {
        group: {region: counts.most_common(10)
                for (name, region), counts in winners.items() if name == group}
        for group in {name for name, _ in winners}
    }
    result = {
        "id": transcript["id"],
        "prompt_length": transcript["prompt_length"],
        "continuation_tokens": len(transcript["generated_ids"]),
        "stop_reason": transcript.get("stop_reason"),
        "hit_generation_limit": bool(transcript.get("hit_generation_limit")),
        "metrics": metrics,
        "winner_counts_top10": winner_summary,
    }
    if review:
        result["review"] = review
    return result


def _condition_metadata(prompts):
    return {row["id"]: {k: row.get(k) for k in ("condition", "pair_id", "template_id")}
            for row in prompts}


def _metric_value(example, group, region, field="mean_log10_rank"):
    return example.get("metrics", {}).get(group, {}).get(region, {}).get(field)


def _pair_deltas(examples, metadata):
    by_pair = defaultdict(dict)
    for example in examples:
        meta = metadata.get(example["id"], {})
        if meta.get("pair_id") and meta.get("condition"):
            by_pair[meta["pair_id"]][meta["condition"]] = example
    deltas = []
    for pair_id, conditions in sorted(by_pair.items()):
        ordinary = conditions.get("ordinary")
        if not ordinary:
            continue
        row = {"pair_id": pair_id, "template_id": metadata.get(ordinary["id"], {}).get("template_id")}
        for condition in ("evaluation_cue", "semantic_control"):
            example = conditions.get(condition)
            if not example:
                continue
            for group in sorted(set(ordinary.get("metrics", {})) & set(example.get("metrics", {}))):
                for region in ("prompt", "continuation"):
                    a = _metric_value(example, group, region)
                    b = _metric_value(ordinary, group, region)
                    if a is not None and b is not None:
                        row[f"{condition}_minus_ordinary__{group}__{region}"] = round(a - b, 6)
        deltas.append(row)
    return deltas


def analyze_run(run_dir, review_path=None, output=None):
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Analysis requires a schema-2 run directory")
    prompts = json.loads((run / "prompts.json").read_text())
    metadata = _condition_metadata(prompts)
    labels = load_review_labels(review_path)
    examples = []
    skipped = []
    for directory in sorted((run / "examples").iterdir()):
        if not directory.is_dir() or not (directory / "transcript.json").exists():
            continue
        transcript = json.loads((directory / "transcript.json").read_text())
        if transcript.get("status") != "complete":
            skipped.append(transcript.get("id", directory.name))
            continue
        examples.append(summarize_example(run, directory.name, labels.get(transcript["id"])))
    condition_counts = Counter(metadata.get(e["id"], {}).get("condition", "unknown") for e in examples)
    stop_counts = Counter(e["stop_reason"] for e in examples)
    label_counts = Counter(e["review"]["label"] for e in examples if e.get("review", {}).get("label"))
    flags = []
    for e in examples:
        meta = metadata.get(e["id"], {})
        if e["stop_reason"] == "length":
            flags.append({"id": e["id"], "type": "truncated_continuation",
                          "detail": "Generation reached max_new_tokens."})
        if meta.get("condition") == "ordinary":
            for group, regions in e["metrics"].items():
                continuation = regions.get("continuation", {})
                if continuation.get("min_rank") is not None and continuation["min_rank"] <= 20:
                    flags.append({"id": e["id"], "type": "low_rank_in_ordinary_control",
                                  "group": group, "region": "continuation",
                                  "min_rank": continuation["min_rank"]})
        if e.get("review", {}).get("label") in {"clear_recognition", "ambiguous"}:
            flags.append({"id": e["id"], "type": "recognition_review_flag",
                          "label": e["review"]["label"]})
    report = {
        "schema_version": 1,
        "run_dir": str(run.resolve()),
        "model": manifest["config"].get("model"),
        "collection_schema_version": manifest["schema_version"],
        "complete_examples": len(examples),
        "skipped_examples": skipped,
        "conditions": dict(condition_counts),
        "stop_reasons": dict(stop_counts),
        "review_labels": dict(label_counts),
        "examples": [{**e, **metadata.get(e["id"], {})} for e in examples],
        "matched_pair_deltas": _pair_deltas(examples, metadata),
        "flags": flags,
        "metric_definition": "mean_log10_rank is averaged over layer/position cells; lower means stronger vocabulary prominence. Groups and regions remain separate.",
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def print_report(report):
    print(f"Analyzed {report['complete_examples']} complete examples for {report['model']}")
    if report["skipped_examples"]:
        print(f"Skipped incomplete: {', '.join(report['skipped_examples'])}")
    print(f"Conditions: {report['conditions']}")
    print(f"Stop reasons: {report['stop_reasons']}")
    if report["review_labels"]:
        print(f"Review labels (usable records only): {report['review_labels']}")
    print(f"Flags for inspection: {len(report['flags'])}")
    for flag in report["flags"][:20]:
        print("  " + json.dumps(flag, ensure_ascii=False))
    print(f"Matched pair comparisons: {len(report['matched_pair_deltas'])}")
