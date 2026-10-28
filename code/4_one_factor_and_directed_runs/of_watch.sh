#!/bin/bash
# Wait for the card to free up, then run the full one-factor family at a slot count the
# available memory can actually support.  One large-v3 float16 instance needs about 6.0 GB;
# four of them on a 24 GB card is what produced cudaErrorUnknown, so the cap here is three.
set -u
W=/mnt/d/fareez_optuna/code/final/of_worker.sh
LOG=/mnt/d/fareez_optuna/logs/onefactor
mkdir -p "$LOG"
WATCH="$LOG/watcher.log"
PER_SLOT_MB=6200      # measured: 5.9 GB per instance, plus a little room
MARGIN_MB=1500        # leave the display and anything small its own space
MAX_SLOTS=3

cat > "$LOG/all_conditions.txt" <<'JOBS'
of_condprev_on
of_temp_full
of_beam_8
of_reference
of_vadsilence_2000
of_vadsilence_1000
of_vadthresh_050
of_vadthresh_020
of_reppen_125
of_reppen_110
of_logprob_100
of_logprob_030
of_nospeech_070
of_nospeech_030
of_beam_1
of_prompt_clinical
of_bestof_5
of_norepeat_3
of_patience_100
of_lenpen_080
of_compression_300
of_hallsil_off
of_vadminspeech_0
of_vadpad_400
JOBS

echo "watcher started $(date -Is); polling every 2 min for $((PER_SLOT_MB * 2 + MARGIN_MB)) MB free" > "$WATCH"

ok=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$free" ]; then free=0; fi
  slots=$(( (free - MARGIN_MB) / PER_SLOT_MB ))
  [ "$slots" -gt "$MAX_SLOTS" ] && slots=$MAX_SLOTS
  if [ "$slots" -ge 2 ]; then ok=$((ok + 1)); else ok=0; fi
  echo "$(date -Is)  free=${free}MB  slots=${slots}  stable=${ok}" >> "$WATCH"
  # two consecutive readings, so a momentary dip does not trigger a launch
  if [ "$ok" -ge 2 ]; then
    echo "$(date -Is)  LAUNCHING at ${slots} slots" >> "$WATCH"
    echo "one-factor family launched $(date -Is) at ${slots} slots" >> "$LOG/schedule.log"
    xargs -a "$LOG/all_conditions.txt" -P "$slots" -L 1 bash "$W"
    echo "ONE-FACTOR FAMILY DONE $(date -Is)" >> "$LOG/schedule.log"
    echo "$(date -Is)  family finished" >> "$WATCH"
    exit 0
  fi
  sleep 120
done
