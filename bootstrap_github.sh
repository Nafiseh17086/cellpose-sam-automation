#!/usr/bin/env bash
set -euo pipefail
REPO_NAME="${1:?usage: $0 <repo-name> [--private|--public]}"
VISIBILITY="${2:---public}"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh not found. Install: brew install gh" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "error: not logged in. Run: gh auth login" >&2
  exit 1
fi

[ ! -d .git ] && git init -b main
git add .
git commit -m "initial commit: cellpose-SAM automation" || echo "(nothing to commit)"
gh repo create "$REPO_NAME" "$VISIBILITY" --source=. --remote=origin --push
echo "Done!"
gh repo view --web
