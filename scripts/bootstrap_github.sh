#!/usr/bin/env bash
# bootstrap_github.sh — one-shot: turn this folder into a fresh GitHub repo.
#
# Prereqs (one time):
#   brew install gh           # or apt / winget
#   gh auth login             # follow browser flow
#   git config --global user.name  "Your Name"
#   git config --global user.email "you@example.com"
#
# Usage:
#   ./scripts/bootstrap_github.sh <repo-name> [--private|--public]
#
# Example:
#   ./scripts/bootstrap_github.sh cellpose-sam-automation --public

set -euo pipefail

REPO_NAME="${1:?usage: $0 <repo-name> [--private|--public]}"
VISIBILITY="${2:---public}"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh (GitHub CLI) not found. Install: https://cli.github.com/" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: not logged in. Run: gh auth login" >&2
  exit 1
fi

# Always run from the repo root, not from scripts/
cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  git init -b main
fi
git add .
git commit -m "initial: cellpose-SAM automation following Pachitariu 2025" || echo "(nothing to commit)"
gh repo create "$REPO_NAME" "$VISIBILITY" --source=. --remote=origin --push

echo
echo "✅ Repo created and pushed."
gh repo view --web --json url -q .url
