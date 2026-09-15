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
    parser.add_argument("command", choices=['validate', 'run', 'inspect', 'view', 'review', 'probe'])
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
