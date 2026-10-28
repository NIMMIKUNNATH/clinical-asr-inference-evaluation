#!/bin/bash
# Third corpus: AfriSpeech-Dialog, held-out transfer only (nothing is tuned on it).
# Step 1 runs defaults alone, which also caches pyannote diarization once per conversation (pyannote and several
# Whisper instances together wedged the GPU earlier). Step 2 runs the four other fixed configurations in parallel,
# four model instances at most. Step 3 scores everything with the same scorer as PriMock57.
set -u
cd /mnt/d/fareez_optuna/code/final
eval "$(grep -m1 'HF_TOKEN=' ~/.bashrc)"; export HF_TOKEN
PY=~/osce/venv/bin/python
LOG=/mnt/d/fareez_optuna/logs; mkdir -p $LOG

echo "step 1: defaults + diarization cache $(date +%T)"
$PY run_primock.py --corpus afrispeech --config defaults > $LOG/afri_defaults.log 2>&1 || echo "defaults FAILED"

echo "step 2: four configurations in parallel $(date +%T)"
for c in tuned primock_tuned oldref_v2 trial32_v2; do
  $PY run_primock.py --corpus afrispeech --config $c > $LOG/afri_$c.log 2>&1 &
done
wait

echo "step 3: scoring $(date +%T)"
for c in defaults tuned primock_tuned oldref_v2 trial32_v2; do
  ~/eval_venv/bin/python score_primock.py --corpus afrispeech --config $c 2>/dev/null | grep -v "Assuming sort"
done
echo "all done $(date +%T)"
