import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def test_appendix_equivalent_docs_exist(self):
        required = (
            "docs/METHOD_AND_JUDGE_PROTOCOL.md",
            "docs/PAPER_REPOSITORY_MAP.md",
            "docs/ENVIRONMENT_LOCK.md",
            "docs/EXPERIMENT_PROTOCOL.md",
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_manifest_excludes_generated_cache_files(self):
        manifest = (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8")
        self.assertNotIn("__pycache__", manifest)
        self.assertNotIn(".ipynb_checkpoints", manifest)
        self.assertIsNone(re.search(r"\.py[co](?:\s|$)", manifest))

    def test_tree_contains_no_generated_cache_files(self):
        forbidden = []
        forbidden.extend(ROOT.rglob("__pycache__"))
        forbidden.extend(ROOT.rglob(".ipynb_checkpoints"))
        forbidden.extend(ROOT.rglob("*.pyc"))
        self.assertEqual([], forbidden)

    def test_private_artifacts_are_absent(self):
        forbidden_names = {".env", "logger.log"}
        found = [p for p in ROOT.rglob("*") if p.name in forbidden_names]
        found.extend(ROOT.rglob("*.jsonl"))
        found.extend(ROOT.rglob("*.pid"))
        self.assertEqual([], found)

    def test_active_tree_has_no_historical_backup_modules(self):
        active = ROOT / "GPTFuzz-master" / "gptfuzzer"
        forbidden = ("*before*.py", "*backup*.py")
        found = []
        for pattern in forbidden:
            found.extend(active.rglob(pattern))
        self.assertEqual([], found)

    def test_release_version(self):
        self.assertEqual("1.0.4", (ROOT / "VERSION").read_text().strip())

    def test_project_author_placeholders_are_absent(self):
        self.assertFalse((ROOT / "CITATION.cff").exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertNotIn("## Citation", readme)
        self.assertNotIn("RA-CoFuzz Authors", license_text)

    def test_exact_target_model_names_are_documented(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for model in (
            "Llama-3.2-3B-Instruct",
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Vicuna-7B-v1.5",
        ):
            self.assertIn(model, readme)

    def test_strongjudge_model_is_frozen(self):
        protocol = (ROOT / "docs/METHOD_AND_JUDGE_PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn("`deepseek-chat`", protocol)


if __name__ == "__main__":
    unittest.main()
