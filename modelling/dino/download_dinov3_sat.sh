#!/usr/bin/env bash
#
# Download the DINOv3 SAT-493M checkpoint(s) on the GPU server.
#
# Usage:
#   bash download_dinov3_sat.sh [DEST_DIR]
#   DOWNLOAD_7B=1 bash download_dinov3_sat.sh /data/ckpts   # also grab the 27GB ViT-7B
#
# Notes:
#   - The URLs below are Meta presigned links tied to YOUR account. They are
#     time-limited (see EXPIRY check) and contain a private signature, so do NOT
#     commit this file or share it. *.pth is already gitignored; this .sh is not.
#   - If you get HTTP 403 / "expired", re-open the Meta DINOv3 download page/email
#     and paste fresh URLs over the two variables below.
#   - For fine-tuning on a 32GB GPU use the ViT-L. The ViT-7B is frozen-feature only.

set -euo pipefail

DEST_DIR="${1:-modelling/dino/ckpt}"
DOWNLOAD_7B="${DOWNLOAD_7B:-0}"

# --- presigned URLs (replace if expired) ------------------------------------
VITL_URL='https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiazl6anN1ZGYwb2piaHp6b2I3eDY0ZGp1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3ODIzNzkwODB9fX1dfQ__&Signature=Fr52eRUKWLk66LnT7rV-5Ku6SVCqby2yX6GNbfj%7EYtgsdACvtW3cmtBNCeVfzHbxf6C6hr-8i9SarPfxUUHBXSCDVuH6gSxbgABFUMjaZe-Wjrbj42doRMLajCL6MIynczWdgt6NKDrbROul65DgFFLfl2EmcULL9fipGjkhuKzochAsrrBJML0ot3QnD-WP2ROhWcOL0U2XOntRSiz0GxM6KDV68UZ6evyn4tyvv7I%7E34AUJY-FEDBh971Mj82ote1yNamk1HMLdxqLkFiPFTLddG7OH7l1PWlaK0HDakp8oy0ySn4q2MpUkyT13ERVsw4jCOTiRaW2NRw9n9skQA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1558031575678018'

VIT7B_URL='https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_sat493m-a6675841.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiazl6anN1ZGYwb2piaHp6b2I3eDY0ZGp1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3ODIzNzkwODB9fX1dfQ__&Signature=Fr52eRUKWLk66LnT7rV-5Ku6SVCqby2yX6GNbfj%7EYtgsdACvtW3cmtBNCeVfzHbxf6C6hr-8i9SarPfxUUHBXSCDVuH6gSxbgABFUMjaZe-Wjrbj42doRMLajCL6MIynczWdgt6NKDrbROul65DgFFLfl2EmcULL9fipGjkhuKzochAsrrBJML0ot3QnD-WP2ROhWcOL0U2XOntRSiz0GxM6KDV68UZ6evyn4tyvv7I%7E34AUJY-FEDBh971Mj82ote1yNamk1HMLdxqLkFiPFTLddG7OH7l1PWlaK0HDakp8oy0ySn4q2MpUkyT13ERVsw4jCOTiRaW2NRw9n9skQA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1558031575678018'
# ----------------------------------------------------------------------------

# warn if the signed links are past their expiry (epoch is embedded in the policy)
EXPIRY_EPOCH=1782379080
NOW_EPOCH="$(date +%s)"
if [ "$NOW_EPOCH" -ge "$EXPIRY_EPOCH" ]; then
  echo "WARNING: these presigned URLs expired on $(date -d @"$EXPIRY_EPOCH" 2>/dev/null || echo epoch:$EXPIRY_EPOCH)."
  echo "         Re-copy fresh URLs from the Meta DINOv3 download page before running."
fi

command -v curl >/dev/null 2>&1 || { echo "curl not found; install it or adapt to wget."; exit 1; }
mkdir -p "$DEST_DIR"

# download with resume + retries, then sanity-check the file
download() {
  local url="$1" out="$2"
  echo ">>> downloading $(basename "$out") -> $out"
  curl -fL -C - --retry 5 --retry-delay 5 -o "$out" "$url"

  # a torch .pth is a zip archive: first bytes must be 'PK\x03\x04'
  local magic
  magic="$(head -c 2 "$out" 2>/dev/null || true)"
  if [ "$magic" != "PK" ]; then
    echo "ERROR: $out does not look like a torch checkpoint (bad magic bytes)."
    echo "       The link may have expired or returned an error page. First bytes:"
    head -c 200 "$out"; echo
    exit 1
  fi
  echo "    OK: $(du -h "$out" | cut -f1)  $out"
}

VITL_OUT="$DEST_DIR/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
download "$VITL_URL" "$VITL_OUT"

if [ "$DOWNLOAD_7B" = "1" ]; then
  VIT7B_OUT="$DEST_DIR/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
  download "$VIT7B_URL" "$VIT7B_OUT"
fi

echo
echo "Done. Checkpoint(s) in: $DEST_DIR"
echo
echo "Use with the trainer, e.g.:"
echo "  python modelling/dino/finetune_spatial_dinov3.py --fold 1 --model_name dinov3_vitl16 \\"
echo "    --pretrained_weights $VITL_OUT \\"
echo "    --imagery_path \$IMAGERY_PATH --imagery_source S --batch_size 1 \\"
echo "    --num_epochs 20 --grouped_bands 4 3 2"
