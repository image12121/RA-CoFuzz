import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from support import (BudgetReached, EXT, LEGACY, PROTOCOL, Recorder,
                     normalize_tap, read_questions, verify_original)


class Interfaces(unittest.TestCase):
    def test_original_fingerprints(self):
        self.assertGreater(verify_original(), 20)

    def test_dataset_counts(self):
        for name in PROTOCOL["datasets"]:
            self.assertEqual(len(read_questions(name)[1]), 20)

    def test_reference_alignment(self):
        for dataset in PROTOCOL["datasets"]:
            self.assertEqual(len(read_questions(dataset, True)[1]), 20)

    def test_missing_references_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory, patch("support.LEGACY", Path(directory)):
            with self.assertRaisesRegex(ValueError, "target_column_file_required"):
                read_questions("advbench", True)

    def test_existing_methods_not_default_plan(self):
        result = subprocess.run([sys.executable, str(EXT / "run.py"), "plan"],
                                capture_output=True, text=True, check=True)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 9)
        self.assertEqual({r["method"] for r in rows}, {"tap", "renellm", "deepinception"})
        self.assertEqual({r["seed"] for r in rows}, {100, 200, 300})

    def test_original_entry_arguments_complete(self):
        tree = ast.parse((LEGACY / "gptfuzz.py").read_text(encoding="utf-8-sig"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        needed = {n.attr for n in ast.walk(main) if isinstance(n, ast.Attribute)
                  and isinstance(n.value, ast.Name) and n.value.id == "args"}
        wrapper = ast.parse((EXT / "run.py").read_text())
        assignment = next(n for n in ast.walk(wrapper) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "native_args" for t in n.targets))
        self.assertTrue(needed <= {k.arg for k in assignment.value.keywords})

    def test_tap_current_fields(self):
        source = "neutral {query} {subject} {target_str}"
        self.assertEqual(normalize_tap(source), source)

    def test_tap_alias(self):
        self.assertEqual(normalize_tap("{query} {reference_responses}"), "{query} {target_str}")

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError):
            normalize_tap("{query} {unexpected}")

    def test_malformed_braces_rejected(self):
        with self.assertRaises(ValueError):
            normalize_tap("{query} {")

    def test_escaped_literals_preserved(self):
        source = '{query} {{"neutral": 1}} {target_str}'
        self.assertEqual(normalize_tap(source), source)

    def test_per_question_budget_and_flush(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rows.jsonl"
            r = Recorder(path, budget=3, per_question=2)
            try:
                r.begin(0, "neutral0")
                r.generate(lambda _: "ok", "input")
                r.generate(lambda _: "ok", "input")
                with self.assertRaises(BudgetReached):
                    r.generate(lambda _: self.fail(), "input")
                r.begin(1, "neutral1")
                r.generate(lambda _: "ok", "input")
                with self.assertRaises(BudgetReached):
                    r.generate(lambda _: self.fail(), "input")
                self.assertEqual((r.calls, r.completed), (3, 3))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([x["question_index"] for x in rows], [0, 0, 1])
            finally:
                r.close()

    def test_failed_generation_not_recorded_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rows.jsonl"
            r = Recorder(path)
            def failed(_):
                raise RuntimeError("neutral")
            try:
                r.begin(0, "neutral")
                with self.assertRaises(RuntimeError):
                    r.generate(failed, "input")
                self.assertEqual((r.calls, r.completed), (1, 0))
                self.assertEqual(path.read_text(), "")
            finally:
                r.close()

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "existing.jsonl"
            path.write_text("existing")
            with self.assertRaises(FileExistsError):
                Recorder(path)
            self.assertEqual(path.read_text(), "existing")

    def test_invalid_target_format_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = Recorder(Path(d) / "rows.jsonl")
            try:
                r.begin(0, "neutral")
                with self.assertRaises(TypeError):
                    r.generate(lambda _: "ok", ["one", "two"])
                self.assertEqual(r.calls, 0)
            finally:
                r.close()


if __name__ == "__main__":
    unittest.main()
