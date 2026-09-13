"""Collect exact-token causal readouts without keeping full vocab logits."""
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path

from crywolf.cli import concept_ids, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_groups(tokenizer, groups):
    if not isinstance(groups, dict) or not groups:
        raise ValueError("concept_groups must be a nonempty mapping")
    for name, words in groups.items():
        if not name or not isinstance(words, list) or not words:
            raise ValueError("Each group needs a name and nonempty word list")
        if any(not isinstance(w, str) or not w or w != w.strip().casefold() for w in words):
            raise ValueError("Word forms must be nonempty, lowercase, and stripped")
    words = concept_ids(tokenizer, sorted({w for ws in groups.values() for w in ws}))
    return words, {name: sorted({t for w in ws for t in words[w]}) for name, ws in groups.items()}


def summarize_batch(scores, members, groups, top_k):
    # Compute exact ranks on the device and return only the useful summaries
    import torch
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite J-Lens readout")
    selected = scores[:, members].contiguous()
    ordered = scores.sort(dim=-1).values
    # Sorting lets us rank selected tokens without moving the whole vocab to CPU.
    # right=True makes ties share the same rank.
    ranks = scores.shape[-1] - torch.searchsorted(ordered, selected, right=True) + 1
    values, indices = scores.topk(min(top_k, scores.shape[-1]), dim=-1)
    selected, ranks, values, indices = (x.tolist() for x in (selected, ranks, values, indices))
    results = []
    for row_values, row_ranks, top_values, top_ids in zip(selected, ranks, values, indices):
        tokens = {str(t): {"logit": v, "rank": r} for t, v, r in zip(members, row_values, row_ranks)}
        maxima = {}
        for group, ids in groups.items():
            winner = max(ids, key=lambda t: tokens[str(t)]["logit"])
            maxima[group] = {"winner_token_id": winner, **tokens[str(winner)]}
        results.append({"members": tokens, "groups": maxima,
                        "top_tokens": [{"token_id": t, "logit": v} for t, v in zip(top_ids, top_values)]})
    return results


def summarize(scores, members, groups, top_k):
    return summarize_batch(scores.unsqueeze(0), members, groups, top_k)[0]


def replay_readouts(model, lens, input_ids, prompt_length, groups, layers, chunk_size, top_k, emit, progress=None):
    # Replay each layer's residual in small position chunks
    import torch
    members = sorted({token_id for ids in groups.values() for token_id in ids})
    count = 0
    started = last_update = time.monotonic()
    expected = len(layers) * input_ids.shape[1]
    def hook(layer):
        def capture(module, inputs, output):
            nonlocal count, last_update
            residual = output[0] if isinstance(output, tuple) else output
            for start in range(0, residual.shape[1], chunk_size):
                logits = model.unembed(lens.transport(residual[0, start:start + chunk_size].float(), layer)).float()
                summaries = summarize_batch(logits, members, groups, top_k)
                for offset, summary in enumerate(summaries):
                    position = start + offset
                    emit({"layer": layer, "position": position,
                          "region": "prompt" if position < prompt_length else "continuation",
                          **summary})
                    count += 1
                current = time.monotonic()
                if progress and (current - last_update >= 5 or start + chunk_size >= residual.shape[1]):
                    elapsed = current - started
                    remaining = elapsed * (expected - count) / count
                    progress(f"replay layer {layer}: {count}/{expected} readouts ({count / expected:.0%}), "
                             f"{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining")
                    last_update = current
        return capture
    handles = [model.layers[layer].register_forward_hook(hook(layer)) for layer in layers]
    try:
        with torch.no_grad():
            model.forward(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    expected = len(layers) * input_ids.shape[1]
    if count != expected:
        raise RuntimeError(f"Incomplete readouts: {count}, expected {expected}")
    return count


def collect_run(args, rows, config):
    import torch
    import transformers
    import jlens

    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    for key in ["max_prompt_tokens", "max_new_tokens", "readout_chunk_size", "top_k"]:
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU access may require execution outside the sandbox")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(config["seed"])
    out.mkdir(parents=True)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    manifest = {"schema_version": 2, "status": "running", "started_at": now(),
                "config": config, "device": args.device, "dtype": str(dtype),
                "python": platform.python_version(), "completed_ids": [],
                "prompt_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
                "selected_ids": [r["id"] for r in rows],
                "versions": {n: importlib.metadata.version(n) for n in
                             ["torch", "transformers", "jlens", "huggingface-hub"]},
                "measurement": "post-layer residual, exact-token causal replay; all sequence positions",
                "position_semantics": "state after consuming token at position; can influence next token",
                "rank_definition": "1 + count of ALL vocabulary logits strictly greater; ties share rank",
                "group_definition": "maximum member logit; ties choose lowest token ID; no cross-group aggregation",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")}}
    write_json(out / "manifest.json", manifest)
    write_json(out / "prompts.json", rows)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
        words, groups = resolve_groups(tokenizer, config["concept_groups"])
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["model_revision"], dtype=dtype).to(args.device).eval()
        model = jlens.from_hf(hf, tokenizer, force_bos=False)
        lens = jlens.JacobianLens.from_pretrained(config["lens_repo"], filename=config["lens_file"], revision=config["lens_revision"])
        layers = lens.source_layers if config["layers"] == "all" else config["layers"]
        if (lens.d_model != model.d_model or not layers or len(set(layers)) != len(layers)
                or not set(layers) <= set(lens.source_layers) or any(l < 0 or l >= model.n_layers for l in layers)):
            raise ValueError("Lens/model dimensions or selected layers are incompatible")
        generation = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
        generation.do_sample = False
        generation.num_beams = 1
        # Checkpoint sampling settings do not apply to this greedy run.
        generation.temperature = 1.0
        generation.top_p = 1.0
        generation.top_k = 50
        generation.max_new_tokens = config["max_new_tokens"]
        generation.pad_token_id = tokenizer.eos_token_id
        manifest.update(resolved_layers=layers, word_token_ids=words, group_token_ids=groups,
                        lens_n_prompts=lens.n_prompts, generation_config=generation.to_dict(),
                        chat_template=tokenizer.get_chat_template(),
                        tokenizer_vocab_size=len(tokenizer),
                        output_vocab_size=hf.get_output_embeddings().weight.shape[0])
        write_json(out / "manifest.json", manifest)
        # Shared by the viewer and the top-k readout summaries.
        write_json(out / "vocabulary.json", {
            str(i): tokenizer.decode([i], clean_up_tokenization_spaces=False) if i < len(tokenizer) else None
            for i in range(manifest["output_vocab_size"])})
        (out / "examples").mkdir()
        # Keep the CSV shape compatible with the reviewer tools.
        fields = ["id", "response", "recognition", "evidence", "evidence_start_char", "evidence_end_char",
                  "self_reference", "fictional_frame", "hypothetical_or_quoted",
                  "repeats_prompt_declaration", "annotator", "notes"]
        with (out / "annotations.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                example_dir = out / "examples" / f"{index:04d}"
                example_dir.mkdir()
                prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=config["enable_thinking"])
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) > config["max_prompt_tokens"]:
                    raise ValueError(f"{row['id']}: prompt exceeds limit; refusing truncation")
                if len(ids) + config["max_new_tokens"] > hf.config.get_text_config().max_position_embeddings:
                    raise ValueError("Requested sequence exceeds model context window")
                input_ids = torch.tensor([ids], device=args.device)
                print(f"{row['id']}: generating up to {config['max_new_tokens']} tokens...", flush=True)
                generation_started = time.monotonic()
                last_generation_update = [generation_started]
                class GenerationProgress(transformers.StoppingCriteria):
                    def __call__(self, sequence, scores, **kwargs):
                        current = time.monotonic()
                        if current - last_generation_update[0] >= 5:
                            print(f"{row['id']}: generated {sequence.shape[1] - len(ids)} tokens "
                                  f"in {current - generation_started:.1f}s", flush=True)
                            last_generation_update[0] = current
                        return False
                with torch.no_grad():
                    generated = hf.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                            generation_config=generation,
                                            stopping_criteria=transformers.StoppingCriteriaList([GenerationProgress()]))
                all_ids = generated[0].tolist()
                if all_ids[:len(ids)] != ids:
                    raise RuntimeError("Generation changed input prefix")
                continuation = all_ids[len(ids):]
                response = tokenizer.decode(continuation, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                eos = generation.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                stopped_eos = bool(continuation) and continuation[-1] in eos_ids
                record = {"id": row["id"], "prompt_record": row, "status": "generated",
                          "rendered_prompt": prompt, "input_ids": ids, "generated_ids": continuation,
                          "prompt_length": len(ids), "response": response,
                          "stop_reason": "eos" if stopped_eos else "length" if len(continuation) >= config["max_new_tokens"] else "other",
                          "hit_generation_limit": len(continuation) >= config["max_new_tokens"],
                          "tokens": [{"position": i, "token_id": t,
                                      "text": tokenizer.decode([t], clean_up_tokenization_spaces=False),
                                      "region": "prompt" if i < len(ids) else "continuation",
                                      "continuation_position": i - len(ids) if i >= len(ids) else None}
                                     for i, t in enumerate(all_ids)]}
                write_json(example_dir / "transcript.json", record)
                writer.writerow({"id": row["id"], "response": response})
                f.flush()
                print(f"{row['id']}: generation finished; collecting {len(all_ids)} tokens × {len(layers)} layers...", flush=True)
                with gzip.open(example_dir / "readouts.jsonl.gz", "wt", encoding="utf-8") as stream:
                    count = replay_readouts(model, lens, generated, len(ids), groups, layers,
                                            config["readout_chunk_size"], config["top_k"],
                                            lambda cell: stream.write(json.dumps(cell, allow_nan=False) + "\n"),
                                            progress=lambda message: print(f"{row['id']}: {message}", flush=True))
                record.update(status="complete", readout_count=count)
                write_json(example_dir / "transcript.json", record)
                manifest["completed_ids"].append(row["id"])
                write_json(out / "manifest.json", manifest)
                print(f"{row['id']}: {len(all_ids)} tokens × {len(layers)} layers = {count} readouts; {record['stop_reason']}", flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        (out / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_at"] = now()
        if args.device == "cuda":
            manifest["cuda_memory"] = {"device_name": torch.cuda.get_device_name(),
                                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        write_json(out / "manifest.json", manifest)
