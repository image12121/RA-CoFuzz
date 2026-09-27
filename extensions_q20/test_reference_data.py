"""Reference-data regression tests use neutral fixtures only."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reference_data as ref


class ReferenceData(unittest.TestCase):
    def test_all_bundled_references(self):
        for dataset in ref.SOURCES:
            ref.verify(dataset)

    def test_order_and_extra_columns_preserved(self):
        questions = [{"text": "B", "id": "002"}, {"text": "A", "id": "001"}]
        source = [{"q": "A", "t": "first"}, {"q": "B", "t": "second"}]
        rows, mapping = ref.align(questions, source, "q", "t")
        self.assertEqual([r["text"] for r in rows], ["B", "A"])
        self.assertEqual([r["id"] for r in rows], ["002", "001"])
        self.assertEqual([r["target"] for r in rows], ["second", "first"])
        self.assertEqual([r["source_row_zero_based"] for r in mapping], [1, 0])

    def test_missing_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference_match_not_unique"):
            ref.align([{"text": "A"}], [{"q": "B", "t": "ok"}], "q", "t")

    def test_duplicate_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference_match_not_unique"):
            ref.align([{"text": "A"}], [{"q": "A", "t": "ok"}] * 2, "q", "t")

    def test_blank_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty_reference"):
            ref.align([{"text": "A"}], [{"q": "A", "t": "  "}], "q", "t")

    def test_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference_schema_mismatch"):
            ref.align([{"text": "A"}], [{"q": "A"}], "q", "t")

    def test_no_fuzzy_or_whitespace_matching(self):
        with self.assertRaisesRegex(ValueError, "reference_match_not_unique"):
            ref.align([{"text": "A"}], [{"q": "A ", "t": "ok"}], "q", "t")

    def test_duplicate_question_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid_question_key"):
            ref.align([{"text": "A"}] * 2, [{"q": "A", "t": "ok"}], "q", "t")

    def test_csv_roundtrip(self):
        rows = [{"text": 'neutral, "quoted"\nsecond line', "target": "中文"}]
        self.assertEqual(ref.csv_rows(ref.encode_csv(rows, ["text", "target"])), rows)

    def test_changed_data_rejected(self):
        original, paired = ref.paths("advbench")
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "paired.csv"
            changed.write_bytes(paired.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "reference_provenance_mismatch"):
                ref.verify("advbench", original, changed)

    def test_conflict_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            datasets = {}
            for dataset in ref.SOURCES:
                rows = [{"text": f"neutral{i}"} for i in range(20)]
                (directory / f"{dataset}_q20_seed1234.csv").write_bytes(ref.encode_csv(rows, ["text"]))
                datasets[dataset] = {"url": dataset, "question_field": "q", "target_field": "t",
                                     "source_rows": 20}
            conflict = directory / "advbench_q20_seed1234_with_targets.csv"
            conflict.write_bytes(b"user owned data")
            source = ref.encode_csv([{"q": f"neutral{i}", "t": "ok"} for i in range(20)], ["q", "t"])
            with patch.object(ref, "QUESTIONS", directory), patch.object(ref, "ROOT", directory), \
                    patch.object(ref, "SOURCES", datasets), \
                    patch.object(ref, "MANIFEST", directory / "manifest.json"), \
                    patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(source)):
                with self.assertRaisesRegex(ValueError, "existing_reference_file_conflict"):
                    ref.restore()
            self.assertEqual(conflict.read_bytes(), b"user owned data")
            self.assertFalse((directory / "jailbreakbench_q20_seed1234_with_targets.csv").exists())


if __name__ == "__main__":
    unittest.main()
