import unittest

try:
    import torch
    import jlens
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from crywolf.collect import replay_readouts, summarize, summarize_batch


@unittest.skipIf(torch is None, "Install .[model] for collection integration tests")
class CollectionTests(unittest.TestCase):
    def test_batched_ranks_equal_direct_counts(self):
        torch.manual_seed(9)
        scores = torch.randint(-5, 6, (7, 97)).float()
        members = [0, 3, 26, 80]
        rows = summarize_batch(scores, members, {"group": members}, 10)
        for i, row in enumerate(rows):
            for token in members:
                self.assertEqual(row["members"][str(token)]["rank"],
                                 1 + int((scores[i] > scores[i, token]).sum()))

    def test_full_vocab_ranks_group_max_and_ties(self):
        result = summarize(torch.tensor([20., 8., 8., 10., 0.]), [1, 2, 4],
                           {"evaluation": [1, 2], "fiction": [4]}, 2)
        self.assertEqual(result["groups"]["evaluation"], {"winner_token_id": 1, "logit": 8., "rank": 3})
        self.assertEqual(result["members"]["4"]["rank"], 5)
        self.assertEqual(result["top_tokens"][0]["token_id"], 0)

    def test_replay_matches_cached_generation_and_cannot_see_future(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        hf = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                                         num_hidden_layers=2, num_attention_heads=2,
                                         num_key_value_heads=1, head_dim=8)).eval()
        class Tokenizer:
            bos_token_id = None
        model = jlens.from_hf(hf, Tokenizer(), force_bos=False)
        lens = jlens.JacobianLens({i: torch.eye(16) for i in range(2)}, n_prompts=1, d_model=16)
        ids = torch.tensor([[1, 5, 7, 3, 9]])
        full = {}
        handles = [layer.register_forward_hook(
            lambda m, inp, out, i=i: full.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach().clone()))
                   for i, layer in enumerate(model.layers)]
        with torch.no_grad():
            model.forward(ids)
        for handle in handles:
            handle.remove()
        cached = {i: [] for i in range(2)}
        handles = [layer.register_forward_hook(
            lambda m, inp, out, i=i: cached[i].append((out[0] if isinstance(out, tuple) else out).detach().clone()))
                   for i, layer in enumerate(model.layers)]
        with torch.no_grad():
            output = hf(input_ids=ids[:, :3], use_cache=True)
            for p in range(3, ids.shape[1]):
                output = hf(input_ids=ids[:, p:p+1], past_key_values=output.past_key_values, use_cache=True)
        for handle in handles:
            handle.remove()
        for layer in range(2):
            torch.testing.assert_close(full[layer], torch.cat(cached[layer], dim=1), atol=1e-6, rtol=1e-5)
        cells = []
        replay_readouts(model, lens, ids, 3, {"test": [1, 3]}, [0, 1], 2, 3, cells.append)
        self.assertEqual(len(cells), 10)
        self.assertEqual(sum(c["region"] == "continuation" for c in cells), 4)
        for cell in cells:
            logits = model.unembed(full[cell["layer"]][0, cell["position"]].float()).float()
            expected = summarize(logits, [1, 3], {"test": [1, 3]}, 3)
            self.assertAlmostEqual(cell["members"]["1"]["logit"], expected["members"]["1"]["logit"], places=5)
            self.assertEqual(cell["members"]["1"]["rank"], expected["members"]["1"]["rank"])
        self.assertTrue(all(not layer._forward_hooks for layer in model.layers))


if __name__ == "__main__":
    unittest.main()
