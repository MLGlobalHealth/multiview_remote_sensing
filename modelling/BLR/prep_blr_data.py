"""Turn a SatMAE/DINO feature export into the CSVs that blr_binomial.R reads.

The eval step writes, per split, an (N, 1025) array (1024 embedding dims + the
prevalence target) plus a companion <name>_meta.csv carrying CENTROID_ID and the
binomial counts (deprived_{sev,mod}_{k,n}) in the SAME row order. This script
splits that into a feature matrix and a target table.

Usage:
    python -m modelling.BLR.prep_blr_data \
        --export_dir <dir with the .npy + _meta.csv> \
        --train_name finetuned_1 --test_name finetuned_1 \
        --out_dir modelling/BLR/data/multispectral
"""
import argparse
import os

import numpy as np
import pandas as pd


def load_split(export_dir, name, split):
    arr = np.load(os.path.join(export_dir, f"{split}_{name}.npy"))
    meta = pd.read_csv(os.path.join(export_dir, f"{split}_{name}_meta.csv"))
    assert len(arr) == len(meta), (
        f"{split} feature rows ({len(arr)}) != meta rows ({len(meta)}); "
        "the export and its _meta.csv are out of sync."
    )
    feats = arr[:, :1024]
    X = pd.DataFrame(feats, columns=[f"f{i}" for i in range(feats.shape[1])])
    return X, meta.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--train_name", required=True, help="e.g. finetuned_1")
    ap.add_argument("--test_name", required=True)
    ap.add_argument("--out_dir", default="modelling/BLR/data/multispectral")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split, name in [("train", args.train_name), ("test", args.test_name)]:
        X, meta = load_split(args.export_dir, name, split)
        X.to_csv(os.path.join(args.out_dir, f"X_{split}.csv"), index=False)
        meta.to_csv(os.path.join(args.out_dir, f"Y_{split}.csv"), index=False)
        nk = "deprived_sev_n"
        have = nk in meta.columns and meta[nk].notna().sum()
        print(f"{split}: {len(X)} rows, {X.shape[1]} feats; "
              f"clusters with valid deprived_sev_n: {have if have else 'NONE — re-run eval after re-processing'}")


if __name__ == "__main__":
    main()
