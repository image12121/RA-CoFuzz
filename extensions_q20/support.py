"""Small, dependency-free interface utilities. No original module is edited."""
import csv
import hashlib
import json
from pathlib import Path
from string import Formatter

ROOT = Path(__file__).resolve().parent.parent
LEGACY = ROOT / "GPTFuzz-master"
EXT = Path(__file__).resolve().parent
OUTPUT = ROOT / "extension_outputs"
PROTOCOL = json.loads((EXT / "protocol.json").read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_questions(dataset, needs_targets=False):
    path = LEGACY / "datasets/questions" / PROTOCOL["datasets"][dataset]
    paired = path.with_name(path.stem + "_with_targets.csv")
    if needs_targets:
        if not paired.is_file():
            raise ValueError("target_column_file_required")
        if dataset in {"advbench", "jailbreakbench"}:
            from reference_data import verify
            verify(dataset, path, paired)
        path = paired
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != PROTOCOL["questions"]:
        raise ValueError("question_count_mismatch")
    if any(not str(row.get("text", "")).strip() for row in rows):
        raise ValueError("empty_question")
    if len({row["text"].strip() for row in rows}) != len(rows):
        raise ValueError("duplicate_question")
    if needs_targets and any(not row.get("target", "").strip() for row in rows):
        raise ValueError("empty_reference")
    if needs_targets:
        with (LEGACY / "datasets/questions" / PROTOCOL["datasets"][dataset]).open(
            encoding="utf-8-sig", newline=""
        ) as f:
            original = list(csv.DictReader(f))
        if [r["text"] for r in original] != [r["text"] for r in rows]:
            raise ValueError("paired_question_order_mismatch")
    return path, rows


def normalize_tap(template):
    # Normalize a documented placeholder alias only; never alter template prose.
    template = template.replace("{reference_responses}", "{target_str}")
    fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
    if not fields <= {"query", "subject", "target_str"} or "query" not in fields:
        raise ValueError("unsupported_template_fields")
    template.format(query="neutral", subject="neutral", target_str="neutral")
    return template


class BudgetReached(BaseException):
    """An explicit outer-driver stop, not an API failure or a class label."""


class Recorder:
    def __init__(self, path, budget=200, per_question=10):
        self.file = Path(path).open("x", encoding="utf-8")
        self.budget, self.per_question = budget, per_question
        self.calls, self.completed, self.q_calls = 0, 0, 0
        self.index, self.question = None, None

    def begin(self, index, question):
        self.index, self.question, self.q_calls = index, question, 0

    def generate(self, generator, prompt):
        if self.index is None:
            raise ValueError("missing_question_context")
        if not isinstance(prompt, str):
            raise TypeError("unsupported_target_message_type")
        if self.calls >= self.budget or self.q_calls >= self.per_question:
            raise BudgetReached()
        self.calls += 1
        self.q_calls += 1
        response = generator(prompt)
        if not isinstance(response, str):
            raise TypeError("invalid_target_response_type")
        record = dict(question_index=self.index, question=self.question,
                      original_prompt=self.question, candidate_prompt=prompt,
                      response=response, target_call_index=self.calls)
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()
        self.completed += 1
        return response

    def close(self):
        self.file.close()


def verify_original():
    expected = json.loads((EXT / "original_hashes.json").read_text())
    for relative, expected_hash in expected.items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError("original_source_or_dataset_mismatch")
    return len(expected)


def generation_signature(method, dataset, model, seed, target):
    question_path, _ = read_questions(dataset, method in {"tap", "pair"})
    target = Path(target).resolve()
    if not (target / "config.json").is_file():
        raise ValueError("local_model_config_missing")
    return dict(method=method, dataset=dataset, model=model, seed=seed,
                question_sha256=digest(question_path),
                source_manifest=digest(EXT / "original_hashes.json"),
                extension_files={p.name: digest(p) for p in sorted(EXT.iterdir())
                                 if p.suffix in {".py", ".json"}},
                target=str(target), target_config_sha256=digest(target / "config.json"))
