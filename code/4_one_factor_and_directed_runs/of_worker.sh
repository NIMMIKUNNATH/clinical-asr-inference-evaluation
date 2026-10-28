#!/bin/bash
# one condition of the one-factor family, on the 40-recording Fareez development set
set -u
CONFIG=$1
PY=/home/nimmiubuntu/osce/venv/bin/python
LOG=/mnt/d/fareez_optuna/logs/onefactor
mkdir -p "$LOG"
eval "$(grep -m1 'HF_TOKEN=' ~/.bashrc)"
export HF_TOKEN
cd /mnt/d/fareez_optuna/code/final || exit 1
OUT="$LOG/$CONFIG.log"
echo "START $(date -Is) $CONFIG" >> "$OUT"
$PY final_run.py --config "$CONFIG" --set dev >> "$OUT" 2>&1
RC=$?
echo "END $(date -Is) exit=$RC $CONFIG" >> "$OUT"
