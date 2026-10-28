#!/bin/bash
# Wait for the card, then run the two outstanding jobs:
#   1. of_vadfilter_off  - the last one-factor condition, 40 development recordings
#   2. joint_beam1_pad400 - the post hoc deployment candidate, 231 held-out recordings, 3 shards
# One large-v3 float16 instance needs about 6 GB; three is the ceiling on this card.
set -u
L=/mnt/d/fareez_optuna/logs/pending
mkdir -p "$L"
W="$L/watcher.log"
PY=/home/nimmiubuntu/osce/venv/bin/python
PER=6200; MARGIN=1500; MAX=3

echo "watcher started $(date -Is); needs $((PER * 2 + MARGIN)) MB free" > "$W"
ok=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$free" ] && free=0
  slots=$(( (free - MARGIN) / PER ))
  [ "$slots" -gt "$MAX" ] && slots=$MAX
  if [ "$slots" -ge 2 ]; then ok=$((ok + 1)); else ok=0; fi
  echo "$(date -Is)  free=${free}MB slots=${slots} stable=${ok}" >> "$W"
  [ "$ok" -ge 2 ] && break
  sleep 120
done

eval "$(grep -m1 'HF_TOKEN=' ~/.bashrc)"; export HF_TOKEN
cd /mnt/d/fareez_optuna/code/final || exit 1
echo "$(date -Is)  launching at ${slots} slots" >> "$W"

# the last one-factor condition first: it is small and the family is otherwise complete
$PY final_run.py --config of_vadfilter_off --set dev >> "$L/of_vadfilter_off.log" 2>&1
echo "$(date -Is)  of_vadfilter_off done ($(ls /mnt/d/fareez_optuna/final/of_vadfilter_off 2>/dev/null | grep -c json)/40)" >> "$W"

# then the held-out candidate, sharded across the available slots
for i in $(seq 0 $((slots - 1))); do
  $PY final_run.py --config joint_beam1_pad400 --shard "$i" --nshards "$slots" \
      >> "$L/joint_beam1_pad400_s$i.log" 2>&1 &
done
wait
echo "$(date -Is)  joint_beam1_pad400 done ($(ls /mnt/d/fareez_optuna/final/joint_beam1_pad400 2>/dev/null | grep -c json)/231)" >> "$W"
echo "ALL PENDING DONE $(date -Is)" >> "$W"
