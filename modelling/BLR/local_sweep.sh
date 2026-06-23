#!/bin/bash
# Local beta-binomial fit sweep on clpc95 (R 4.5.1 — the env proven to work).
# Fits every pulled split x {deprived_sev, deprived_mod}, 3 concurrent, idempotent.
set +e
set +o noclobber
cd /home/scratch/Dropbox/Seth/Research/MLGHrepos/multiview_remote_sensing
ROOT=modelling/BLR
maxjobs=3
mkdir -p "$ROOT/output_bb"
for d in "$ROOT"/data/sweep/spatial*; do
  clean=$(basename "$d")
  for tgt in deprived_sev deprived_mod; do
    out="$ROOT/output_bb/$clean"
    if [ -f "$out/Y_test_pred_${tgt}.csv" ]; then echo "SKIP $clean/$tgt (exists)"; continue; fi
    (
      echo "START $clean/$tgt $(date +%H:%M)"
      Rscript "$ROOT/blr_betabinom.R" "$d" "$out" "$tgt" 1000 0 normal 4 >| "$ROOT/log_${clean}_${tgt}.log" 2>&1
      res=$(grep -E "Overdispersion phi|Test prevalence MAE|coverage of k/n|Error|does not contain" "$ROOT/log_${clean}_${tgt}.log" | tr '\n' ' ')
      echo "DONE  $clean/$tgt :: $res"
    ) &
    while [ "$(jobs -r | wc -l)" -ge "$maxjobs" ]; do sleep 5; done
  done
done
wait
echo "=== ALL LOCAL FITS DONE $(date) ==="
