"""Restore auxiliary target fields by exact joins; never generate answers.

No network access during normal runs/checks. --restore downloads the two pinned
public sources and creates missing files only; conflicting files are preserved.
Console output contains counts and fixed status codes, never dataset contents.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request

EXT = Path(__file__).resolve().parent
ROOT = EXT.parent
QUESTIONS = ROOT / "GPTFuzz-master/datasets/questions"
MANIFEST = EXT / "reference_provenance.json"
SOURCES = {
    "advbench": {
        "revision": "098262edf85f807224e70ecd87b9d83716bf6b73",
        "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv",
        "question_field": "goal", "target_field": "target",
        "source_rows": 520,
    },
    "jailbreakbench": {
        "revision": "886acc352a31533ffbcf4ef22c744658688086fc",
        "url": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/886acc352a31533ffbcf4ef22c744658688086fc/data/harmful-behaviors.csv",
        "question_field": "Goal", "target_field": "Target",
        "source_rows": 100,
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def align(questions, references, question_field, target_field):
    """Exact equality only: reject missing, ambiguous or empty references."""
    lookup = {}
    for index, row in enumerate(references):
        if question_field not in row or target_field not in row:
            raise ValueError("reference_schema_mismatch")
        lookup.setdefault(row[question_field], []).append((index, row[target_field]))
    output, mapping = [], []
    seen = set()
    for index, row in enumerate(questions):
        question = row.get("text")
        if not isinstance(question, str) or not question.strip() or question in seen:
            raise ValueError("invalid_question_key")
        seen.add(question)
        matches = lookup.get(question, [])
        if len(matches) != 1:
            raise ValueError("reference_match_not_unique")
        source_index, target = matches[0]
        if not isinstance(target, str) or not target.strip():
            raise ValueError("empty_reference")
        output.append({**row, "target": target})
        mapping.append({"q20_row_zero_based": index,
                        "source_row_zero_based": source_index,
                        "question_sha256": sha(question.encode()),
                        "target_sha256": sha(target.encode())})
    return output, mapping


def encode_csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def paths(dataset):
    original = QUESTIONS / f"{dataset}_q20_seed1234.csv"
    return original, original.with_name(original.stem + "_with_targets.csv")


def verify(dataset, original=None, paired=None):
    if dataset not in SOURCES:
        return
    if original is None or paired is None:
        original, paired = paths(dataset)
    try:
        manifest = json.loads(MANIFEST.read_text())
        entry = manifest["datasets"][dataset]
        if (manifest["schema_version"] != 1 or manifest["join_policy"] != "exact_text_no_normalization"
                or entry["source"] != SOURCES[dataset]
                or sha(original.read_bytes()) != entry["original_sha256"]
                or sha(paired.read_bytes()) != entry["paired_sha256"]):
            raise ValueError("reference_provenance_mismatch")
        rows = csv_rows(paired.read_bytes())
        base = csv_rows(original.read_bytes())
        if len(rows) != 20 or len(base) != 20 or len(entry["mapping"]) != 20:
            raise ValueError("reference_provenance_mismatch")
        for index, (row, source_row, record) in enumerate(zip(rows, base, entry["mapping"])):
            if (row["text"] != source_row["text"] or not row["target"].strip()
                    or record["q20_row_zero_based"] != index
                    or not 0 <= record["source_row_zero_based"] < SOURCES[dataset]["source_rows"]
                    or record["question_sha256"] != sha(row["text"].encode())
                    or record["target_sha256"] != sha(row["target"].encode())):
                raise ValueError("reference_provenance_mismatch")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("reference_provenance_mismatch") from exc


def restore():
    pending = []
    manifest = {"schema_version": 1, "join_policy": "exact_text_no_normalization",
                "target_policy": "public_source_auxiliary_prefix_not_full_answer_or_label",
                "datasets": {}}
    for dataset, source in SOURCES.items():
        original, paired = paths(dataset)
        original_bytes = original.read_bytes()
        questions = csv_rows(original_bytes)
        if len(questions) != 20:
            raise ValueError("question_count_mismatch")
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            source_bytes = response.read()
        references = csv_rows(source_bytes)
        if len(references) != source["source_rows"]:
            raise ValueError("reference_source_count_mismatch")
        rows, mapping = align(questions, references, source["question_field"], source["target_field"])
        output = encode_csv(rows, [*questions[0], "target"])
        manifest["datasets"][dataset] = {
            "source": source, "source_sha256": sha(source_bytes),
            "original_path": str(original.relative_to(ROOT)),
            "paired_path": str(paired.relative_to(ROOT)),
            "original_sha256": sha(original_bytes), "paired_sha256": sha(output),
            "exact_matches": len(rows), "mapping": mapping,
        }
        pending.append((paired, output))
    pending.append((MANIFEST, (json.dumps(manifest, indent=2) + "\n").encode()))
    # Check every destination before creating any file. Never overwrite user data.
    for path, data in pending:
        if path.exists() and path.read_bytes() != data:
            raise ValueError("existing_reference_file_conflict")
    for path, data in pending:
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="download pinned sources; never overwrite")
    args = parser.parse_args()
    if args.restore:
        restore()
    for dataset in SOURCES:
        verify(dataset)
        print(json.dumps({"status": "REFERENCE_DATA_PASS", "dataset": dataset, "exact_matches": 20}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "REFERENCE_DATA_FAILED", "error_type": type(exc).__name__}))
        sys.exit(1)
