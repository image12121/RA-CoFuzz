import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ResumeCrossModel(unittest.TestCase):
    def test_exact_resume_order_avoids_completed_qwen_cells(self):
        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "project"
            extension = project / "extensions_q20"
            binary = Path(root) / "bin"
            extension.mkdir(parents=True)
            binary.mkdir()
            shutil.copy2(Path(__file__).with_name(
                "resume_cross_model_core_vicuna.sh"), extension)
            log = Path(root) / "calls.log"
            fake = binary / "python"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$RESUME_LOG\"\n")
            fake.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = str(binary) + os.pathsep + environment["PATH"]
            environment["RESUME_LOG"] = str(log)
            result = subprocess.run(
                ["bash", str(extension / "resume_cross_model_core_vicuna.sh")],
                cwd=project, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            calls = log.read_text().splitlines()
            self.assertEqual(calls[0], "extensions_q20/run.py audit")
            self.assertEqual(calls[1],
                "extensions_q20/run.py evaluate --method ra_cofuzz "
                "--dataset gptfuzzer --model vicuna_7b --seed 100")
            expected_cells = [
                ("strict_gptfuzzer", "100"), ("pair", "100"),
                ("ra_cofuzz", "200"), ("strict_gptfuzzer", "200"),
                ("pair", "200"), ("ra_cofuzz", "300"),
                ("strict_gptfuzzer", "300"), ("pair", "300"),
            ]
            generated = [line for line in calls if "run.py run " in line]
            evaluated = [line for line in calls if "run.py evaluate " in line][1:]
            self.assertEqual(len(generated), 8)
            self.assertEqual(len(evaluated), 8)
            for line, (method, seed) in zip(generated, expected_cells):
                self.assertIn(f"--method {method}", line)
                self.assertIn("--model vicuna_7b", line)
                self.assertIn(f"--seed {seed}", line)
                self.assertNotIn("qwen", line)
            self.assertEqual(calls[-1],
                             "extensions_q20/audit_cross_model.py --phase core")


if __name__ == "__main__":
    unittest.main()
