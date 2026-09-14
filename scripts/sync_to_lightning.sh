#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-s_01m2fw60p95y4zweav7yree60q@ssh.lightning.ai}"
REMOTE_DIR="${REMOTE_DIR:-~/major-project}"

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"

rsync -avz --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude "data/" \
  --exclude "runs/" \
  ./ "$REMOTE:$REMOTE_DIR/"

echo "Synced to $REMOTE:$REMOTE_DIR"
