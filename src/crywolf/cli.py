"""cli entry point. model dependencies load only for collection"""
import argparse
import json
from collections import Counter
from pathlib import Path


CONDITIONS = {"ordinary", "semantic_control", "evaluation_cue"}


def read_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("Every prompt needs a unique, nonempty string id")
        seen.add(row["id"])
        if row.get("condition") not in CONDITIONS:
            raise ValueError(f"{row['id']}: invalid prompt condition")
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError(f"{row['id']}: source is required")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{row['id']}: messages must be nonempty")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"{row['id']}: invalid message role")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"{row['id']}: empty message content")
        if "recognition" in row:
            raise ValueError("Recognition belongs in response annotations, never prompt labels")
    if not rows:
        raise ValueError("No prompts found")
    return rows


def concept_ids(tokenizer, concepts):
    # Find exact one-token vocabulary matches for each concept word
    result = {word: [] for word in concepts}
    special = set(tokenizer.all_special_ids)
    for token_id in range(len(tokenizer)):
        if token_id not in special:
            word = tokenizer.decode([token_id]).strip().casefold()
            if word in result:
                result[word].append(token_id)
    missing = [word for word, ids in result.items() if not ids]
    if missing:
        raise ValueError(f"Concepts without an exact single-token variant: {missing}")
    return result


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def run(args, rows, config):
    from crywolf.collect import collect_run
    collect_run(args, rows, config)


def review_command(args, parser):
    # Dispatch reviewer and calibration commands.
    from crywolf.review import (probe_run, read_demonstrations, read_probes, resolve_reviewer,
                                review_run)
    demonstrations = read_demonstrations(args.demonstrations) if args.demonstrations else None
    if args.command == "probe":
        if not args.reviewer:
            parser.error("probe needs --reviewer")
        probe_run(read_probes(args.probes)[:args.limit], resolve_reviewer(args.reviewers, args.reviewer),
                  timeout=args.timeout, output=args.output, demonstrations=demonstrations)
        return
    if not args.run_dir:
        parser.error(f"--run-dir is required for {args.command}")
    if args.command == "review":
        if not args.reviewer or not args.output:
            parser.error("review needs --reviewer and --output")
        spec = resolve_reviewer(args.reviewers, args.reviewer)
        counts = review_run(args.run_dir, spec, args.output, limit=args.limit,
                            timeout=args.timeout, resume=args.resume, demonstrations=demonstrations)
        print(json.dumps(counts, indent=2))
        print("Reviewer labels are an independent but imperfect measurement, not ground truth.")
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=['validate', 'run', 'inspect', 'view', 'review', 'probe', 'analyze', 'report', 'heatmap'])
    parser.add_argument("--run-dir", help="Saved schema-2 collection directory")
    parser.add_argument("--example", default="0000", help="Example directory number, e.g. 0000")
    parser.add_argument("--position", type=int, help="Absolute token position to inspect; defaults to last prompt token")
    parser.add_argument("--html", help="New HTML output file or directory for view")
    parser.add_argument("--prompts", default="data/pilot.jsonl")
    parser.add_argument("--probes", default="data/rubric-probes.jsonl", help="Hand-labelled rubric probes")
    parser.add_argument("--demonstrations", help="Adjudicated labelled probes to include as reviewer examples")
    parser.add_argument("--config", default="configs/local-qwen3-1.7b.json")
    parser.add_argument("--output", help="New output path; run defaults to outputs/collection")
    parser.add_argument("--reviewer", help="Reviewer name defined in --reviewers")
    parser.add_argument("--reviewers", default="configs/reviewers.json", help="Reviewer definitions")
    parser.add_argument("--review-file", help="Usable reviewer JSONL to include in analyze")
    parser.add_argument("--group", action="append", help="Concept group for heatmap; repeat to stack groups")
    parser.add_argument("--metric", choices=["rank", "logit"], default="rank")
    parser.add_argument("--width", type=int, default=900, help="Maximum SVG width in pixels")
    parser.add_argument("--cell-height", type=float, default=2, help="Heatmap cell height in pixels")
    parser.add_argument("--cell-width", type=float, help="Fixed token column width; default fits the requested width")
    parser.add_argument("--start-color", help="Light endpoint; defaults to the selected group's palette")
    parser.add_argument("--end-color", help="Dark endpoint; defaults to the selected group's palette")
    parser.add_argument("--gamma", type=float, default=1.0, help="Colour interpolation curve")
    parser.add_argument("--show-tokens", action="store_true")
    parser.add_argument("--mark-concept-tokens", action="store_true",
                        help="Add a red marker row for vocabulary tokens in the selected concept group")
    parser.add_argument("--token-angle", type=float, default=45)
    parser.add_argument("--token-every", type=int, default=10)
    parser.add_argument("--token-wrap", type=int, default=15,
                        help="Maximum vertical-offset steps before token labels wrap")
    parser.add_argument("--token-line-height", type=float, default=14,
                        help="Vertical spacing between token-label rows")
    parser.add_argument("--token-style", choices=["strip", "angled"], default="strip",
                        help="Token-label layout when --show-tokens is enabled")
    parser.add_argument("--layer-labels", type=int, default=5,
                        help="Number of layer labels, including first and final; minimum 2")
    parser.add_argument("--font-size", type=float, default=11,
                        help="Base SVG font size in pixels")
    parser.add_argument("--timeout", type=float, default=300, help="Reviewer request timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Append to an existing review file")
    parser.add_argument("--limit", type=int, help="Collect only the first N prompts")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.command in {"inspect", "view"}:
        from crywolf.inspect import load_example, print_inspection, write_directory_view
        if not args.run_dir:
            parser.error("--run-dir is required for inspect/view")
        try:
            if args.command == "inspect":
                print_inspection(load_example(args.run_dir, args.example), args.position)
            else:
                target = Path(args.html) if args.html else Path(args.run_dir) / "viewer"
                write_directory_view(args.run_dir, target, initial_example=args.example)
                print("Serve this directory with: python -m http.server --directory "
                      f"{target.resolve()} 8000")
        except (ValueError, OSError) as exc:
            parser.exit(1, f"{exc}\n")
        return
    if args.command == "analyze":
        if not args.run_dir:
            parser.error("--run-dir is required for analyze")
        from crywolf.analyze import analyze_run, print_report
        try:
            report = analyze_run(args.run_dir, review_path=args.review_file, output=args.output)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            parser.exit(1, f"{exc}\n")
        print_report(report)
        if args.output:
            print(f"Wrote analysis report: {Path(args.output).resolve()}")
        return
    if args.command == "report":
        if not args.run_dir or not args.output:
            parser.error("report needs --run-dir and --output")
        from crywolf.report import write_report
        try:
            count = write_report(args.run_dir, args.output)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Wrote report for {count} examples: {Path(args.output).resolve()}")
        return
    if args.command == "heatmap":
        if not args.run_dir or not args.group:
            parser.error("heatmap needs --run-dir and at least one --group")
        from crywolf.heatmap import write_heatmap
        try:
            if len(args.group) > 1 and args.show_tokens:
                parser.error("--show-tokens is incompatible with multiple --group values")
            output = args.output
            if not output:
                suffix = "_tokens" if args.show_tokens else ""
                filename = f"example{args.example}_{'_'.join(args.group)}{suffix}.svg"
                output = str(Path(args.run_dir) / filename)
            options = dict(metric=args.metric, width=args.width, cell_height=args.cell_height,
                           cell_width=args.cell_width, start_color=args.start_color,
                           end_color=args.end_color, gamma=args.gamma,
                           show_tokens=args.show_tokens, token_angle=args.token_angle,
                           token_every=args.token_every, token_wrap=args.token_wrap,
                           token_line_height=args.token_line_height, layer_labels=args.layer_labels,
                           font_size=args.font_size, token_style=args.token_style,
                           mark_concept_tokens=args.mark_concept_tokens)
            if len(args.group) == 1:
                width, height, tokens, layers = write_heatmap(
                    args.run_dir, args.example, args.group[0], output, **options)
            else:
                from crywolf.heatmap import write_heatmap_stack
                width, height, tokens, layers = write_heatmap_stack(
                    args.run_dir, args.example, args.group, output, **options)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Wrote {width}×{height}px SVG: {Path(output).resolve()} ({tokens} tokens × {layers} layers; {len(args.group)} group(s))")
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command in {"review", "probe"}:
        try:
            review_command(args, parser)
        except (ValueError, OSError, RuntimeError) as exc:
            parser.exit(1, f"{exc}\n")
        return
    rows = read_prompts(args.prompts)
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.command == "validate":
        print(json.dumps({"prompts": len(rows), "conditions": dict(Counter(r["condition"] for r in rows))}, indent=2))
    else:
        args.output = args.output or "outputs/collection"
        run(args, rows, json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
