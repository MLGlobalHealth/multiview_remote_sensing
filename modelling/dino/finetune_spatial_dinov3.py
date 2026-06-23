import argparse
import pandas as pd
from tqdm import tqdm
import os
import random
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image
import re
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn import L1Loss
import warnings
warnings.filterwarnings("ignore")


# DINOv3 ships several backbones via torch.hub. The satellite (SAT-493M) checkpoints
# are gated and must be downloaded locally, then passed in through --pretrained_weights.
# Embedding dims for the common variants (used to default emb_size):
DINOV3_EMB_SIZES = {
    'dinov3_vits16': 384,
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vith16plus': 1280,
    'dinov3_vit7b16': 4096,
}
DINOV3_PATCH_SIZE = 16  # all dinov3 ViTs use patch size 16 (dinov2 used 14)


def main(fold, model_name, target, imagery_path, imagery_source, emb_size, batch_size,
         num_epochs, pretrained_weights, img_size=None, grouped_bands=None,
         normalize=False, norm_mean=None, norm_std=None):

    if imagery_source == 'L':
        normalization = 30000.
        imagery_size = 336      # 336 / 16 = 21, already patch16-compatible
    elif imagery_source == 'S':
        normalization = 3000.
        imagery_size = 992      # 992 / 16 = 62  (dinov2 used 994 = 71 * 14)
    else:
        raise Exception("Unsupported imagery source")

    best_model = f'modelling/dino/model/{model_name}_{fold}_{str(grouped_bands)}all_cluster_best_{imagery_source}{target}_.pth'
    last_model = f'modelling/dino/model/{model_name}_{fold}_{str(grouped_bands)}all_cluster_last_{imagery_source}{target}_.pth'

    if grouped_bands is None:
        grouped_bands = [4, 3, 2]
    if img_size is not None:
        imagery_size = img_size

    if imagery_size % DINOV3_PATCH_SIZE != 0:
        raise ValueError(
            f"imagery_size ({imagery_size}) must be divisible by the dinov3 patch size "
            f"({DINOV3_PATCH_SIZE}). Try {imagery_size - (imagery_size % DINOV3_PATCH_SIZE)}."
        )

    data_folder = r'survey_processing/processed_data'

    train_df = pd.read_csv(f'{data_folder}/train_fold_{fold}.csv')
    test_df = pd.read_csv(f'{data_folder}/test_fold_{fold}.csv')

    available_imagery = []
    for d in os.listdir(imagery_path):
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))

    def is_available(centroid_id):
        for centroid in available_imagery:
            if centroid_id in centroid:
                return True
        return False
    train_df = train_df[train_df['CENTROID_ID'].apply(is_available)]
    test_df = test_df[test_df['CENTROID_ID'].apply(is_available)]

    def filter_contains(query):
        for item in available_imagery:
            if query in item:
                return item
    train_df['imagery_path'] = train_df['CENTROID_ID'].apply(filter_contains)
    test_df['imagery_path'] = test_df['CENTROID_ID'].apply(filter_contains)

    if target == '':
        predict_target = ['h10', 'h3', 'h31', 'h5', 'h7', 'h9', 'hc70', 'hv109', 'hv121',
                          'hv106', 'hv201', 'hv204', 'hv205', 'hv216', 'hv225', 'hv271', 'v312']
    else:
        predict_target = [target]

    filtered_predict_target = []
    for col in predict_target:
        filtered_predict_target.extend(
            [c for c in train_df.columns if c == col or re.match(f"^{col}_[^a-zA-Z]", c)]
        )
    train_df = train_df.dropna(subset=filtered_predict_target)
    predict_target = sorted(filtered_predict_target)

    def load_and_preprocess_image(path, grouped_bands=grouped_bands):
        with rasterio.open(path) as src:
            b1 = src.read(grouped_bands[0])
            b2 = src.read(grouped_bands[1])
            b3 = src.read(grouped_bands[2])

            img = np.dstack((b1, b2, b3))
            img = img / normalization

        img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed = 42
    set_seed(seed)
    train, validation = train_test_split(train_df, test_size=0.2, random_state=seed)

    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item['imagery_path'])
            image_tensor = self.transform(Image.fromarray(image))
            target = torch.tensor(item[predict_target], dtype=torch.float32)
            return image_tensor, target

    transform_steps = [
        transforms.Resize((imagery_size, imagery_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        # DINOv3-SAT was pretrained with its own channel statistics, not ImageNet's.
        # Pass the checkpoint card's exact constants via --norm_mean/--norm_std.
        mean = norm_mean if norm_mean is not None else [0.485, 0.456, 0.406]
        std = norm_std if norm_std is not None else [0.229, 0.224, 0.225]
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_steps)

    train_dataset = CustomDataset(train, transform)
    val_dataset = CustomDataset(validation, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load DINOv3 with the local satellite checkpoint. The gated SAT-493M weights must
    # already be downloaded; `weights` accepts a path to that .pth file.
    base_model = torch.hub.load(
        'facebookresearch/dinov3',
        model_name,
        source='github',
        weights=pretrained_weights,
    )

    def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, filename)

    torch.cuda.empty_cache()

    class ViTForRegression(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.regression_head = nn.Linear(emb_size, len(predict_target))

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            # dinov3 returns the pooled CLS embedding when called directly; guard in case
            # a build returns the forward_features dict instead.
            if isinstance(outputs, dict):
                outputs = outputs.get('x_norm_clstoken', outputs.get('cls_token'))
            return torch.sigmoid(self.regression_head(outputs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = ViTForRegression(base_model).to(device)
    if os.path.exists(last_model):
        last_state_dict = torch.load(last_model)
        best_error = torch.load(best_model)['loss']
        epochs_ran = last_state_dict['epoch']
        model.load_state_dict(last_state_dict['model_state_dict'])
        print('Found existing model')
    else:
        epochs_ran = 0
        best_error = np.inf

    model.to(device)

    base_model_params = {'params': model.base_model.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}
    head_params = {'params': model.regression_head.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}

    optimizer = torch.optim.Adam([base_model_params, head_params])
    loss_fn = L1Loss()

    for epoch in range(epochs_ran+1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print('Training...')
        for batch in tqdm(train_loader):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()

        model.eval()
        val_loss = []
        indiv_loss = []
        print('Validating...')
        for batch in val_loader:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                outputs = model(images)
            batch_loss = loss_fn(outputs, targets)
            val_loss.append(batch_loss.item())
            indiv_loss.append(torch.mean(torch.abs(outputs-targets), axis=0))

        mean_val_loss = np.mean(val_loss)
        mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

        if mean_val_loss < best_error:
            save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
            best_error = mean_val_loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}')
        save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune DINOv3 (satellite checkpoint) for socio-economic regression.')
    parser.add_argument('--fold', type=str, help='CV fold')
    parser.add_argument('--model_name', type=str, default='dinov3_vitl16',
                        help='DINOv3 hub entrypoint, e.g. dinov3_vitl16 / dinov3_vit7b16')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                        help='Path to the local (gated) DINOv3 satellite checkpoint .pth')
    parser.add_argument('--target', type=str, default='', help='Target variable')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--emb_size', type=int, default=None,
                        help='Backbone output dim. Defaults from model_name (vitl16 -> 1024).')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--imagery_size', type=int, help='Size of the imagery (must be divisible by 16)')
    parser.add_argument('--grouped_bands', type=int, nargs=3, help='Three integer grouped bands (e.g., 4 3 2)')
    parser.add_argument('--normalize', action='store_true',
                        help='Apply channel normalization (use the SAT checkpoint constants)')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=None, help='Per-channel mean for normalization')
    parser.add_argument('--norm_std', type=float, nargs=3, default=None, help='Per-channel std for normalization')
    args = parser.parse_args()

    emb_size = args.emb_size
    if emb_size is None:
        emb_size = DINOV3_EMB_SIZES.get(args.model_name)
        if emb_size is None:
            raise ValueError(
                f"Could not infer emb_size for model '{args.model_name}'. Pass --emb_size explicitly."
            )

    main(args.fold, args.model_name, args.target, args.imagery_path, args.imagery_source,
         emb_size, args.batch_size, args.num_epochs, args.pretrained_weights,
         args.imagery_size, args.grouped_bands, args.normalize, args.norm_mean, args.norm_std)
