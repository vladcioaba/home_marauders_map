#!/usr/bin/env bash
# Sync this repo to a Raspberry Pi and run it there.
#
# Usage:
#   PI_HOST=pi@pi5.local ./scripts/deploy_to_pi.sh --setup      # first time
#   PI_HOST=pi@pi5.local ./scripts/deploy_to_pi.sh              # subsequent runs
#
# Environment:
#   PI_HOST   user@host of the Pi (required)
#   PI_DIR    remote project dir (default: ~/home_marauders_map)
#   BACKEND   ultralytics | hailo (default: ultralytics)
#   ARGS      extra args passed through to marauders (e.g. "--log tracks.db")
#
# --setup additionally creates a venv on the Pi and pip-installs the project.

set -euo pipefail

: "${PI_HOST:?set PI_HOST=user@host}"
PI_DIR="${PI_DIR:-~/home_marauders_map}"
BACKEND="${BACKEND:-ultralytics}"
ARGS="${ARGS:-}"
SETUP=0

for arg in "$@"; do
    case "$arg" in
        --setup) SETUP=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "→ syncing $ROOT → $PI_HOST:$PI_DIR"
rsync -azv --delete \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pt' \
    --exclude '*.onnx' \
    --exclude '*.hef' \
    --exclude '*.db*' \
    --exclude '.DS_Store' \
    "$ROOT"/ "$PI_HOST":"$PI_DIR"/

if (( SETUP )); then
    echo "→ creating venv + installing on $PI_HOST"
    ssh "$PI_HOST" "cd $PI_DIR && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ."
    echo "✔ setup complete. Re-run without --setup to start the app."
    exit 0
fi

echo "→ running marauders on $PI_HOST (backend=$BACKEND)"
# shellcheck disable=SC2029   (intentional: $BACKEND / $ARGS expand locally)
ssh -t "$PI_HOST" "cd $PI_DIR && .venv/bin/marauders --config config/house.yaml --backend $BACKEND $ARGS"
