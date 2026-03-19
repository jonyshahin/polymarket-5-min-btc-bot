#!/bin/bash
# run_bot.sh — Wrapper script for launchd auto-restart
# Activates venv, runs the bot, logs output

set -euo pipefail

BOT_DIR="$HOME/polymarketbot"
LOG_DIR="$BOT_DIR/logs"
VENV="$BOT_DIR/.venv"
STRATEGY="${1:-smc}"
MODE="${2:---dry-run}"

mkdir -p "$LOG_DIR"

# Log rotation: keep last 10 log files
cd "$LOG_DIR"
ls -t bot_*.log 2>/dev/null | tail -n +10 | xargs -r rm -f
cd "$BOT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bot_${TIMESTAMP}.log"

echo "[$(date)] Starting bot: strategy=$STRATEGY mode=$MODE" | tee -a "$LOG_FILE"

# Activate venv and run
source "$VENV/bin/activate"
exec python main.py --strategy "$STRATEGY" $MODE 2>&1 | tee -a "$LOG_FILE"
