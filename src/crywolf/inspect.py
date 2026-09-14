"""Read and validate saved readouts without loading a model."""
import gzip
import json
import math
import shutil
from pathlib import Path


def load_example(run_dir, example="0000", validate=True):
    run = Path(run_dir)
    if len(example) != 4 or not example.isascii() or not example.isdigit():
        raise ValueError("--example must be four digits, e.g. 0000")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("This reader requires schema 2; older prompt-only runs are unsupported")
    directory = run / "examples" / example
    transcript = json.loads((directory / "transcript.json").read_text())
    if transcript.get("status") != "complete":
        raise ValueError("Example is incomplete; refusing to display partial measurements as a complete heatmap")
    vocabulary = json.loads((run / "vocabulary.json").read_text())
    layers = manifest["resolved_layers"]
    groups = manifest["group_token_ids"]
    ids = transcript["input_ids"] + transcript["generated_ids"]
    boundary = transcript["prompt_length"]
    if validate and (boundary != len(transcript["input_ids"]) or len(transcript["tokens"]) != len(ids)):
        raise ValueError("Transcript token lengths or prompt boundary disagree")
    for position, (token, token_id) in enumerate(zip(transcript["tokens"], ids)):
        if validate and (token["position"] != position or token["token_id"] != token_id
                or token["region"] != ("prompt" if position < boundary else "continuation")):
            raise ValueError(f"Token table mismatch at position {position}")
    cells = {}
    with gzip.open(directory / "readouts.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            cell = json.loads(line)
            layer, pos = cell["layer"], cell["position"]
            key = (layer, pos)
            if validate and (key in cells or layer not in layers or not 0 <= pos < len(ids)):
                raise ValueError(f"Duplicate or out-of-range readout: {key}")
            if validate and cell["region"] != transcript["tokens"][pos]["region"]:
                raise ValueError(f"Readout region mismatch: {key}")
            if validate and set(cell["groups"]) != set(groups):
                raise ValueError(f"Concept groups mismatch: {key}")
            for name, members in groups.items():
                if not validate:
                    continue
                winner = max(members, key=lambda t: cell["members"][str(t)]["logit"])
                group = cell["groups"][name]
                for token_id in members:
                    value = cell["members"][str(token_id)]
                    if not math.isfinite(value["logit"]) or not isinstance(value["rank"], int) or value["rank"] < 1:
                        raise ValueError(f"Invalid score/rank: {key}")
                if group != {"winner_token_id": winner, **cell["members"][str(winner)]}:
                    raise ValueError(f"Group maximum or winning token mismatch: {key}, {name}")
            cells[key] = cell
    expected = len(layers) * len(ids)
    if validate and (len(cells) != expected or transcript["readout_count"] != expected):
        raise ValueError(f"Incomplete coverage: expected {expected} layer/position cells, got {len(cells)}")
    return dict(manifest=manifest, transcript=transcript, vocabulary=vocabulary,
                layers=layers, groups=groups, cells=cells, run_dir=str(run.resolve()))


def print_inspection(example, position=None):
    t = example["transcript"]
    position = t["prompt_length"] - 1 if position is None else position
    if not 0 <= position < len(t["tokens"]):
        raise ValueError(f"Position must be between 0 and {len(t['tokens']) - 1}")
    print(f"{t['id']} | model: {example['manifest']['config']['model']}")
    print(f"Verified {len(example['cells'])} readouts across {len(example['layers'])} layers; "
          f"{t['prompt_length']} prompt + {len(t['generated_ids'])} continuation tokens; stop: {t['stop_reason']}")
    print(f"Run status: {example['manifest']['status']}; this example is complete")
    print("Token context (readout is AFTER consuming the indicated token):")
    for pos in range(max(0, position - 3), min(len(t['tokens']), position + 4)):
        token = t['tokens'][pos]
        print(f"{'>' if pos == position else ' '} {pos:4} {token['region']:12} {token['text']!r}")
    print("layer  group          logit       vocab rank  winner (token ID)")
    for layer in example["layers"]:
        for name, group in example["cells"][(layer, position)]["groups"].items():
            winner = group["winner_token_id"]
            word = example["vocabulary"].get(str(winner))
            print(f"{layer:5}  {name:13} {group['logit']:10.4f} {group['rank']:10}  {word!r} ({winner})")


def write_payload(payload, path):
    path = Path(path)
    template = Path(__file__).with_name("viewer.html").read_text()
    # prompt/response text is untrusted: prevent closing the JSON script element
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    html = template.replace("__DATA__", encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(html)


def write_directory_view(run_dir, directory, initial_example="0000"):
    # write the viewer and one compressed asset per example
    root = Path(directory)
    html_path = root / "viewer.html"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Viewer directory already exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "examples"
    assets.mkdir()
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("This viewer requires a schema-2 run")
    shutil.copyfile(run / "vocabulary.json", root / "vocabulary.json")
    entries, available = [], []
    for source in sorted((run / "examples").iterdir()):
        if not source.is_dir() or len(source.name) != 4 or not source.name.isascii() or not source.name.isdigit():
            continue
        transcript_path = source / "transcript.json"
        if not transcript_path.exists():
            entries.append(dict(key=source.name, id=source.name, status="missing transcript"))
            continue
        transcript = json.loads(transcript_path.read_text())
        entry = dict(key=source.name, id=transcript["id"], status=transcript.get("status", "unknown"),
                     condition=transcript.get("prompt_record", {}).get("condition", ""))
        if entry["status"] == "complete":
            meta = {"manifest": manifest, "transcript": transcript,
                    "layers": manifest["resolved_layers"], "groups": manifest["group_token_ids"],
                    "run_dir": str(run.resolve()), "vocabulary_url": "../vocabulary.json",
                    "readouts_url": f"{source.name}.readouts.jsonl.gz"}
            (assets / f"{source.name}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
            shutil.copyfile(source / "readouts.jsonl.gz", assets / meta["readouts_url"])
            entry["url"] = f"examples/{source.name}.meta.json"
            available.append(source.name)
            print(f"Wrote {source.name}: {entry['id']}", flush=True)
        entries.append(entry)
    if not available:
        raise ValueError("No complete examples available to visualize")
    write_payload(dict(examples=entries, initial_example=initial_example if initial_example in available else available[0]), html_path)
    print(f"Viewer written to {html_path.resolve()}", flush=True)
