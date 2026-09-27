#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${RACOFUZZ_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PAIR_DIR="$BASE/JailbreakingLLMs-official"
OFFICIAL_COMMIT="6379ef705a0fc745530f7d895963510c021b496a"
OFFICIAL_URL="https://github.com/patrickrchao/JailbreakingLLMs.git"
ARCHIVE_URL="https://codeload.github.com/patrickrchao/JailbreakingLLMs/tar.gz/$OFFICIAL_COMMIT"

if [ ! -d "$BASE/GPTFuzz-master" ]; then
  echo "Missing $BASE/GPTFuzz-master" >&2
  exit 1
fi

snapshot_is_valid() {
  [ -f "$PAIR_DIR/system_prompts.py" ] && \
    [ "$(cat "$PAIR_DIR/.ra_cofuzz_pair_commit" 2>/dev/null || true)" = "$OFFICIAL_COMMIT" ]
}

archive_existing_invalid_dir() {
  if [ -e "$PAIR_DIR" ]; then
    local backup="$PAIR_DIR.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "Moving incomplete PAIR directory to: $backup"
    mv "$PAIR_DIR" "$backup"
  fi
}

fetch_with_git() {
  local tmp="$BASE/.JailbreakingLLMs-git-$$"
  mkdir -p "$tmp"
  git -C "$tmp" init
  git -C "$tmp" remote add origin "$OFFICIAL_URL"

  local attempt
  for attempt in 1 2 3 4 5; do
    echo "Git fetch attempt $attempt/5 (pinned commit only)"
    if git -c http.version=HTTP/1.1 -C "$tmp" fetch --depth 1 origin "$OFFICIAL_COMMIT"; then
      git -C "$tmp" checkout --detach FETCH_HEAD
      printf '%s\n' "$OFFICIAL_COMMIT" > "$tmp/.ra_cofuzz_pair_commit"
      archive_existing_invalid_dir
      mv "$tmp" "$PAIR_DIR"
      return 0
    fi
    sleep $((attempt * 2))
  done

  rm -rf "$tmp"
  return 1
}

fetch_with_archive() {
  local archive="/tmp/JailbreakingLLMs-$OFFICIAL_COMMIT.tar.gz"
  local unpack="$BASE/.JailbreakingLLMs-archive-$$"
  mkdir -p "$unpack"

  echo "Git transport failed; trying the official GitHub source archive."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 8 --retry-delay 3 --retry-all-errors \
      -o "$archive" "$ARCHIVE_URL"
  else
    wget --tries=8 --waitretry=3 -O "$archive" "$ARCHIVE_URL"
  fi

  tar -xzf "$archive" -C "$unpack" --strip-components=1
  printf '%s\n' "$OFFICIAL_COMMIT" > "$unpack/.ra_cofuzz_pair_commit"
  archive_existing_invalid_dir
  mv "$unpack" "$PAIR_DIR"
}

if ! snapshot_is_valid; then
  if ! fetch_with_git; then
    fetch_with_archive
  fi
fi

if ! snapshot_is_valid; then
  echo "PAIR source acquisition failed: system_prompts.py is unavailable." >&2
  exit 1
fi

python -m pip install --upgrade "openai>=1.0" "fschat==0.2.36" pandas psutil

python - <<'PY'
import openai, pandas, torch, transformers
from fastchat.model import get_conversation_template
print("openai", openai.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("fastchat template", get_conversation_template("gpt-4").name)
PY

echo "PAIR official repository ready at: $PAIR_DIR"
echo "Pinned commit: $OFFICIAL_COMMIT"
