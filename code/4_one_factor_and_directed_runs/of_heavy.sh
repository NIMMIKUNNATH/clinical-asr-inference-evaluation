#!/bin/bash
# The three memory-hungry conditions, run one at a time after the main queue drains.
# Beam 8, the six-step temperature fallback and previous-text conditioning each allocate far
# more than the ~4.8 GB a light condition needs; three of them concurrently killed the context.
set -u
W=/mnt/d/fareez_optuna/code/final/of_worker.sh
L=/mnt/d/fareez_optuna/logs/onefactor

while ! grep -q "ONE-FACTOR FAMILY DONE" "$L/schedule.log" 2>/dev/null; do sleep 120; done

echo "heavy conditions, one at a time, started $(date -Is)" >> "$L/schedule.log"
for c in of_condprev_on of_temp_full of_beam_8; do
  bash "$W" "$c"
  n=$(ls "/mnt/d/fareez_optuna/final/$c" 2>/dev/null | grep -c '\.json$')
  echo "  $c finished with $n/40 recordings $(date -Is)" >> "$L/schedule.log"
done
echo "HEAVY CONDITIONS DONE $(date -Is)" >> "$L/schedule.log"
