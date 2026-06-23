"""
GPU stage: extract and cache DINOv3 features for one (fold, view).

Run this once per view per fold. It loads the (optionally fine-tuned) DINOv3-SAT
backbone, runs every train/test image through it, and caches the feature matrix +
targets + metadata to disk. All downstream work (ridge, BLR, conformal, calibration,
active learning) then reads these caches and never touches a GPU again.

Cache layout (one dir per fold/view):
    {features_root}/{mode}_{source}_fold{fold}_bands{b1}-{b2}-{b3}/
        X_train.npy  y_train.npy  meta_train.csv
        X_test.npy   y_test.npy   meta_test.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DINOV3_EMB_SIZES = {
    'dinov3_vits16': 384, 'dinov3_vitb16': 768, 'dinov3_vitl16': 1024,
    'dinov3_vith16plus': 1280, 'dinov3_vit7b16': 4096,
}
DINOV3_PATCH_SIZE = 16
# metadata columns worth keeping for spatial UQ / active learning, if present
META_COLS = ['CENTROID_ID', 'deprived_sev', 'LATNUM', 'LONGNUM',
             'URBAN_RURA', 'YEAR', 'countrycode', 'country', 'DHSCC']


def feature_dir(root, mode, source, fold, bands):
    """Cache directory for a (fold, view) — must match the evaluator's reader."""
    band_tag = '-'.join(str(b) for b in bands)
    return os.path.join(root, f'{mode}_{source}_fold{fold}_bands{band_tag}')


def main(fold, model_name, pretrained_weights, imagery_path, imagery_source,
         grouped_bands, mode, checkpoint, use_pretrained, target, emb_size,
         img_size, batch_size, features_root, normalize, norm_mean, norm_std):

    if imagery_source == 'L':
        normalization = 30000.
        imagery_size = 336
    elif imagery_source == 'S':
        normalization = 3000.
        imagery_size = 992
    else:
        raise Exception("Unsupported imagery source")
    if img_size is not None:
        imagery_size = img_size
    if imagery_size % DINOV3_PATCH_SIZE != 0:
        raise ValueError(f"imagery_size {imagery_size} must be divisible by {DINOV3_PATCH_SIZE}")

    # locate the fine-tuned checkpoint (same naming finetune_spatial_dinov3.py writes)
    if checkpoint is None and not use_pretrained:
        gb_str = str(list(grouped_bands))  # e.g. "[4, 3, 2]"
        checkpoint = (f'modelling/dino/model/{model_name}_{fold}_{gb_str}'
                      f'all_cluster_best_{imagery_source}{target}_.pth')

    data_folder = r'survey_processing/processed_data/'
    if mode == 'temporal':
        train_df = pd.read_csv(f'{data_folder}before_2020.csv')
        test_df = pd.read_csv(f'{data_folder}after_2020.csv')
    else:  # 'spatial' or 'one_country' both read train/test fold files
        train_df = pd.read_csv(f'{data_folder}train_fold_{fold}.csv')
        test_df = pd.read_csv(f'{data_folder}test_fold_{fold}.csv')

    available_imagery = []
    for d in os.listdir(imagery_path):
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))

    def is_available(centroid_id):
        return any(centroid_id in c for c in available_imagery)

    def filter_contains(query):
        for item in available_imagery:
            if query in item:
                return item

    train_df = train_df[train_df['CENTROID_ID'].apply(is_available)].copy()
    test_df = test_df[test_df['CENTROID_ID'].apply(is_available)].copy()
    train_df['imagery_path'] = train_df['CENTROID_ID'].apply(filter_contains)
    test_df['imagery_path'] = test_df['CENTROID_ID'].apply(filter_contains)
    train_df = train_df[train_df['deprived_sev'].notna()].reset_index(drop=True)
    test_df = test_df[test_df['deprived_sev'].notna()].reset_index(drop=True)
    if test_df.empty:
        raise Exception("Empty test set")

    def load_and_preprocess_image(path):
        with rasterio.open(path) as src:
            b1 = src.read(grouped_bands[0])
            b2 = src.read(grouped_bands[1])
            b3 = src.read(grouped_bands[2])
            img = np.dstack((b1, b2, b3)) / normalization
        img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    transform_steps = [transforms.Resize((imagery_size, imagery_size)), transforms.ToTensor()]
    if normalize:
        mean = norm_mean if norm_mean is not None else [0.485, 0.456, 0.406]
        std = norm_std if norm_std is not None else [0.229, 0.224, 0.225]
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_steps)

    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item['imagery_path'])
            return transform(Image.fromarray(image)), np.float32(item['deprived_sev'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    base_model = torch.hub.load('facebookresearch/dinov3', model_name,
                                source='github', weights=pretrained_weights)

    # overlay fine-tuned backbone weights (we only need the encoder, not the head)
    if not use_pretrained:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Fine-tuned checkpoint not found: {checkpoint}\n"
                f"Run finetune_spatial_dinov3.py first, or pass --use_pretrained to use raw SAT weights."
            )
        sd = torch.load(checkpoint, map_location='cpu')['model_state_dict']
        base_sd = {k[len('base_model.'):]: v for k, v in sd.items() if k.startswith('base_model.')}
        missing, unexpected = base_model.load_state_dict(base_sd, strict=False)
        print(f"Loaded fine-tuned backbone from {checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("Using raw pretrained SAT backbone (no fine-tuned weights).")

    base_model.to(device).eval()

    def cls_feat(x):
        out = base_model(x)
        if isinstance(out, dict):
            out = out.get('x_norm_clstoken', out.get('cls_token'))
        return out

    def extract(df):
        loader = DataLoader(CustomDataset(df), batch_size=batch_size, shuffle=False, num_workers=4)
        feats, ys = [], []
        for images, targets in tqdm(loader):
            images = images.to(device)
            with torch.no_grad():
                out = cls_feat(images)
            feats.append(out.cpu().numpy())
            ys.append(targets.numpy())
        return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = extract(train_df)
    torch.cuda.empty_cache()
    X_test, y_test = extract(test_df)

    out_dir = feature_dir(features_root, mode, imagery_source, fold, grouped_bands)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    train_df[[c for c in META_COLS if c in train_df.columns]].to_csv(
        os.path.join(out_dir, 'meta_train.csv'), index=False)
    test_df[[c for c in META_COLS if c in test_df.columns]].to_csv(
        os.path.join(out_dir, 'meta_test.csv'), index=False)

    print(f"Saved features to {out_dir}  "
          f"(X_train {X_train.shape}, X_test {X_test.shape})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract & cache DINOv3 features for one fold/view.')
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='dinov3_vitl16')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                        help='Path to the DINOv3-SAT checkpoint .pth (the torch.hub backbone weights)')
    parser.add_argument('--imagery_path', type=str, required=True)
    parser.add_argument('--imagery_source', type=str, default='S', help='L (Landsat) or S (Sentinel)')
    parser.add_argument('--grouped_bands', type=int, nargs=3, required=True)
    parser.add_argument('--mode', type=str, default='spatial', choices=['spatial', 'temporal', 'one_country'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Explicit fine-tuned checkpoint path (overrides the auto-derived name)')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Skip fine-tuned weights; extract from the raw SAT backbone')
    parser.add_argument('--target', type=str, default='', help='Target used at fine-tune time (for checkpoint name)')
    parser.add_argument('--emb_size', type=int, default=None)
    parser.add_argument('--imagery_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--features_root', type=str, default='modelling/dino/features')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=None)
    parser.add_argument('--norm_std', type=float, nargs=3, default=None)
    args = parser.parse_args()

    emb_size = args.emb_size or DINOV3_EMB_SIZES.get(args.model_name)

    main(args.fold, args.model_name, args.pretrained_weights, args.imagery_path,
         args.imagery_source, args.grouped_bands, args.mode, args.checkpoint,
         args.use_pretrained, args.target, emb_size, args.imagery_size,
         args.batch_size, args.features_root, args.normalize, args.norm_mean, args.norm_std)
