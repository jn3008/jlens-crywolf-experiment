import json
import tempfile
import unittest
from pathlib import Path

from crywolf.cli import concept_ids, read_prompts


class PilotTests(unittest.TestCase):
    def test_pilot_and_matched_task_content(self):
        rows = read_prompts(Path(__file__).resolve().parents[1] / "data/pilot.jsonl")
        self.assertEqual(len(rows), 10)
        for pair_id in {r["pair_id"] for r in rows} - {None}:
            pair = [r for r in rows if r["pair_id"] == pair_id]
            self.assertEqual(len(pair), 2)
            plain = next(r for r in pair if r["condition"] == "ordinary")
            cue = next(r for r in pair if r["condition"] == "evaluation_cue")
            self.assertTrue(cue["messages"][0]["content"].endswith(plain["messages"][0]["content"]))

    def test_reject_duplicate_ids_and_preassigned_recognition(self):
        row = dict(id="x", source="test", condition="ordinary", messages=[dict(role="user", content="hello")])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            for rows in [[row, row], [dict(row, recognition="clear_recognition")]]:
                path.write_text("\n".join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):
                    read_prompts(path)

    def test_concepts_use_whole_token_variants_and_exclude_specials(self):
        class Tokenizer:
            all_special_ids = [4]
            def __len__(self):
                return 5
            def decode(self, ids):
                return ["evaluation", " Evaluation", "eval", "uation", "evaluation"][ids[0]]

        self.assertEqual(concept_ids(Tokenizer(), ["evaluation"]), {"evaluation": [0, 1]})
        with self.assertRaises(ValueError):
            concept_ids(Tokenizer(), ["simulation"])


if __name__ == "__main__":
    unittest.main()
