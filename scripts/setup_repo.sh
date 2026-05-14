#!/usr/bin/env bash
# setup_repo.sh — one-shot bootstrap to publish a results folder to GitHub.
#
# Usage:
#   ./scripts/setup_repo.sh <output_dir> <repo_name> [public|private]
#
# Example:
#   ./scripts/setup_repo.sh ./out cellpose-results public
#
# Requires: git, gh (https://cli.github.com/), and `gh auth login` already done.

set -euo pipefail

DIR="${1:?usage: setup_repo.sh <output_dir> <repo_name> [public|private]}"
NAME="${2:?usage: setup_repo.sh <output_dir> <repo_name> [public|private]}"
VIS="${3:-private}"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI (gh) not installed. https://cli.github.com/" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "error: not logged in to gh. run: gh auth login" >&2
  exit 1
fi

mkdir -p "$DIR"
cd "$DIR"

# init git if needed
if [ ! -d .git ]; then
  git init -b main
fi

# minimal README so the repo isn't empty
if [ ! -f README.md ]; then
  cat > README.md <<EOF2
# $NAME

Automated outputs from \`cellpose-sam-automation\`.

- \`*_masks.tif\` — Cellpose-SAM instance masks (int32, opens in Fiji/napari)
- \`*_overlay.png\` — preview overlays
- \`*_cells.csv\` — per-cell measurements
- \`all_cells.csv\` — combined table across runs
EOF2
fi

git add -A
# only commit if there is something staged
if ! git diff --cached --quiet; then
  git commit -m "init: cellpose-SAM results repo"
fi

# create the GitHub repo and push
case "$VIS" in
  public)  gh repo create "$NAME" --public  --source=. --remote=origin --push ;;
  private) gh repo create "$NAME" --private --source=. --remote=origin --push ;;
  *) echo "error: visibility must be 'public' or 'private'" >&2; exit 1 ;;
esac

echo
echo "Repo created and pushed."
echo "Future runs will auto-commit when invoked with --git-push."
