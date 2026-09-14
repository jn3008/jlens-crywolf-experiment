import gzip
import json
import tempfile
import unittest
from pathlib import Path

from crywolf.inspect import load_example, write_directory_view


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / "examples/0000"
        self.directory.mkdir(parents=True)
        self.manifest = dict(schema_version=2, status="complete", resolved_layers=[0], group_token_ids={"eval": [1]}, config={"model": "test"})
        self.transcript = dict(id="test", status="complete", input_ids=[1], generated_ids=[2], prompt_length=1,
                               response="</script><script>alert(1)</script>", readout_count=2,
                               tokens=[dict(position=i, token_id=i+1, region="prompt" if i == 0 else "continuation", text="x") for i in range(2)])
        for name, data in [("manifest.json", self.manifest), ("vocabulary.json", {"1": "eval"})]:
            (self.root / name).write_text(json.dumps(data))
        (self.directory / "transcript.json").write_text(json.dumps(self.transcript))
        self.cells = [dict(layer=0, position=i, region="prompt" if i == 0 else "continuation",
                           members={"1": {"logit": 2.0, "rank": 1}},
                           groups={"eval": {"winner_token_id": 1, "logit": 2.0, "rank": 1}},
                           top_tokens=[{"token_id": 1, "logit": 2.0}]) for i in range(2)]
        self.save_cells()

    def save_cells(self):
        with gzip.open(self.directory / "readouts.jsonl.gz", "wt") as f:
            for cell in self.cells:
                f.write(json.dumps(cell) + "\n")

    def test_directory_view_keeps_data_out_of_html(self):
        target = self.root / "viewer"
        write_directory_view(self.root, target)
        viewer = (target / "viewer.html").read_text()
        self.assertNotIn(self.transcript["response"], viewer)
        self.assertIn('<details id="rawLogits">', viewer)
        self.assertTrue((target / "examples/0000.meta.json").exists())
        with self.assertRaises(FileExistsError):
            write_directory_view(self.root, target)

    def test_reject_missing_duplicate_and_bad_winner(self):
        original = list(self.cells)
        for corrupted in [original[:1], original + original[:1]]:
            self.cells = corrupted
            self.save_cells()
            with self.assertRaises(ValueError):
                load_example(self.root)
        self.cells = original
        self.cells[0]["groups"]["eval"]["logit"] = 99
        self.save_cells()
        with self.assertRaises(ValueError):
            load_example(self.root)

    def test_reject_partial_example(self):
        self.transcript["status"] = "generated"
        (self.directory / "transcript.json").write_text(json.dumps(self.transcript))
        with self.assertRaises(ValueError):
            load_example(self.root)

    def test_directory_view_marks_incomplete_examples(self):
        other = self.root / "examples/0001"
        other.mkdir()
        incomplete = dict(self.transcript, id="unfinished", status="generated")
        (other / "transcript.json").write_text(json.dumps(incomplete))
        target = self.root / "viewer"
        write_directory_view(self.root, target, initial_example="0001")
        html = (target / "viewer.html").read_text()
        data = json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
        self.assertEqual(data["initial_example"], "0000")
        self.assertEqual(data["examples"][1]["status"], "generated")
        self.assertNotIn("gzip_base64", html)
        meta = json.loads((target / "examples/0000.meta.json").read_text())
        self.assertEqual(meta["transcript"]["response"], self.transcript["response"])
        self.assertEqual(meta["readouts_url"], "0000.readouts.jsonl.gz")
        with self.assertRaises(FileExistsError):
            write_directory_view(self.root, target)


if __name__ == "__main__":
    unittest.main()
