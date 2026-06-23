#!/usr/bin/env bash
#
# End-to-end DINOv3-SAT multi-view sweep: fine-tune -> extract features -> evaluate.
#
# Runs every (fold, view) on the available GPUs (NGPU in parallel), then fuses the
# cached features and reports single-view and multi-view MAE on the CPU.
#
# Usage:
#   CKPT=modelling/dino/ckpt/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
#   IMAGERY=/path/to/parent_imagery \
#   bash modelling/dino/run_dinov3_sweep.sh
#
# Skip stages with flags, e.g. re-evaluate without re-training:
#   DO_FINETUNE=0 DO_EXTRACT=0 bash modelling/dino/run_dinov3_sweep.sh
#
# Note: torch.hub.load clones facebookresearch/dinov3 on first call, so the machine
# needs internet (or pre-clone the repo and switch the scripts to source='local').

set -uo pipefail

# ---------------- config (override via env) ----------------
CKPT="${CKPT:-modelling/dino/ckpt/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth}"
IMAGERY="${IMAGERY:?Set IMAGERY=/path/to/parent_imagery_folder}"
MODEL="${MODEL:-dinov3_vitl16}"
SOURCE="${SOURCE:-S}"            # S=Sentinel, L=Landsat
MODE="${MODE:-spatial}"          # spatial | temporal | one_country
EPOCHS="${EPOCHS:-20}"
BATCH_FT="${BATCH_FT:-1}"        # fine-tune batch size (memory-bound at 992px)
BATCH_EXTRACT="${BATCH_EXTRACT:-4}"
NGPU="${NGPU:-2}"

DO_FINETUNE="${DO_FINETUNE:-1}"
DO_EXTRACT="${DO_EXTRACT:-1}"
DO_EVAL="${DO_EVAL:-1}"

FOLDS=( ${FOLDS:-1 2 3 4 5} )
# the four paper views: natural / false-colour / land-moisture / agriculture
VIEWS=( "4 3 2" "8 4 2" "12 1 3" "11 8 2" )
# -----------------------------------------------------------

mkdir -p modelling/dino/model modelling/dino/features modelling/dino/fused_features logs

run_one() {  # fold, "b1 b2 b3", gpu
  local fold="$1" bands="$2" gpu="$3"
  local tag="${bands// /-}"
  local log="logs/${MODEL}_${MODE}${SOURCE}_fold${fold}_bands${tag}.log"
  {
    echo "=== fold $fold | bands $bands | GPU $gpu | $(date) ==="
    if [ "$DO_FINETUNE" = "1" ]; then
      echo "--- fine-tune ---"
      CUDA_VISIBLE_DEVICES="$gpu" python modelling/dino/finetune_spatial_dinov3.py \
        --fold "$fold" --model_name "$MODEL" --pretrained_weights "$CKPT" \
        --imagery_path "$IMAGERY" --imagery_source "$SOURCE" \
        --batch_size "$BATCH_FT" --num_epochs "$EPOCHS" --grouped_bands $bands || { echo "FINETUNE FAILED"; exit 1; }
    fi
    if [ "$DO_EXTRACT" = "1" ]; then
      echo "--- extract ---"
      CUDA_VISIBLE_DEVICES="$gpu" python modelling/dino/extract_features_dinov3.py \
        --fold "$fold" --model_name "$MODEL" --pretrained_weights "$CKPT" \
        --imagery_path "$IMAGERY" --imagery_source "$SOURCE" --mode "$MODE" \
        --batch_size "$BATCH_EXTRACT" --grouped_bands $bands || { echo "EXTRACT FAILED"; exit 1; }
    fi
    echo "=== done fold $fold bands $bands $(date) ==="
  } > "$log" 2>&1
}

# -------- GPU stage: schedule (fold, view) jobs NGPU at a time --------
if [ "$DO_FINETUNE" = "1" ] || [ "$DO_EXTRACT" = "1" ]; then
  JOBS=()
  for fold in "${FOLDS[@]}"; do
    for bands in "${VIEWS[@]}"; do
      JOBS+=("${fold}|${bands}")
    done
  done

  echo "Scheduling ${#JOBS[@]} (fold,view) jobs across ${NGPU} GPU(s). Logs in logs/"
  i=0
  n=${#JOBS[@]}
  while [ "$i" -lt "$n" ]; do
    pids=()
    for ((g=0; g<NGPU && i<n; g++)); do
      IFS='|' read -r fold bands <<< "${JOBS[$i]}"
      echo "  launch [$((i+1))/$n] fold $fold bands '$bands' on GPU $g"
      run_one "$fold" "$bands" "$g" &
      pids+=("$!")
      i=$((i+1))
    done
    # wait for this batch; report any failures (logs hold details)
    for pid in "${pids[@]}"; do
      wait "$pid" || echo "  WARNING: a job (pid $pid) exited non-zero — check its log."
    done
  done
fi

# -------- CPU stage: fuse features and score --------
if [ "$DO_EVAL" = "1" ]; then
  FOLDS_ARG="${FOLDS[*]}"
  ALL_TAGS=()
  echo
  echo "================ single-view results ================"
  for bands in "${VIEWS[@]}"; do
    tag="${bands// /-}"
    ALL_TAGS+=("$tag")
    echo "--- view $tag ---"
    python modelling/dino/evaluate_dinov3.py --views "$tag" \
      --folds $FOLDS_ARG --mode "$MODE" --imagery_source "$SOURCE"
  done

  echo
  echo "================ multi-view (all views) ================"
  python modelling/dino/evaluate_dinov3.py --views "${ALL_TAGS[@]}" \
    --folds $FOLDS_ARG --mode "$MODE" --imagery_source "$SOURCE" --save_fused
fi
