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




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--prompts", default="data/pilot.jsonl")
    args = parser.parse_args()
    rows = read_prompts(args.prompts)
    print(json.dumps({"prompts": len(rows), "conditions": dict(Counter(r["condition"] for r in rows))}, indent=2))


if __name__ == "__main__":
    main()
