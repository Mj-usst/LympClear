import os
import math
import glob
import random
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import jaccard_score, precision_score, recall_score

from CBAM_UNet import CBAMPlainConvUNet  # 确保与本脚本同目录

# ============ 工具函数 ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore_norm(v):
    m = np.mean(v)
    s = np.std(v) + 1e-8
    return (v - m) / s


def minmax_norm(v):
    vmin = np.percentile(v, 0.5)
    vmax = np.percentile(v, 99.5)
    v = np.clip(v, vmin, vmax)
    rng = (v.max() - v.min()) + 1e-8
    return (v - v.min()) / rng


def pad_or_crop_to(arr, target_dhw, pad_value=0):
    D, H, W = arr.shape[-3:]
    td, th, tw = target_dhw
    out = np.full((td, th, tw), pad_value, dtype=arr.dtype)
    sd0 = max(0, (D - td) // 2)
    sh0 = max(0, (H - th) // 2)
    sw0 = max(0, (W - tw) // 2)
    dd0 = max(0, (td - D) // 2)
    dh0 = max(0, (th - H) // 2)
    dw0 = max(0, (tw - W) // 2)
    sd1 = min(D, sd0 + td)
    sh1 = min(H, sh0 + th)
    sw1 = min(W, sw0 + tw)
    dd1 = dd0 + (sd1 - sd0)
    dh1 = dh0 + (sh1 - sh0)
    dw1 = dw0 + (sw1 - sw0)
    out[dd0:dd1, dh0:dh1, dw0:dw1] = arr[sd0:sd1, sh0:sh1, sw0:sw1]
    return out


def random_crop3d(img, lab, patch_size):
    D, H, W = img.shape[-3:]
    pd, ph, pw = patch_size
    if D <= pd or H <= ph or W <= pw:
        return pad_or_crop_to(img, patch_size), pad_or_crop_to(lab, patch_size)
    d0 = random.randint(0, D - pd)
    h0 = random.randint(0, H - ph)
    w0 = random.randint(0, W - pw)
    return (img[d0:d0+pd, h0:h0+ph, w0:w0+pw],
            lab[d0:d0+pd, h0:h0+ph, w0:w0+pw])


# ============ 数据集 ============
class Nifti3DDataset(Dataset):
    def __init__(self, images_dir, labels_dir, patch_size=(64, 128, 128),
                 norm='zscore', cache=False, foreground_crop_prob=0.5):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
        self.label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
        assert len(self.image_paths) == len(self.label_paths) and len(self.image_paths) > 0
        self.patch_size = patch_size
        self.norm = norm
        self.cache = cache
        self.fg_prob = foreground_crop_prob
        self._cache = {}

    def __len__(self):
        return len(self.image_paths)

    def _load_case(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]
        img = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        lab = nib.load(self.label_paths[idx]).get_fdata().astype(np.int64)
        if self.norm == 'zscore':
            img = zscore_norm(img)
        elif self.norm == 'minmax':
            img = minmax_norm(img)
        if self.cache:
            self._cache[idx] = (img, lab)
        return img, lab

    def __getitem__(self, idx):
        img, lab = self._load_case(idx)
        if random.random() < self.fg_prob and (lab > 0).any():
            fg_coords = np.argwhere(lab > 0)
            cz, cy, cx = fg_coords[np.random.choice(len(fg_coords))]
            pd, ph, pw = self.patch_size
            D, H, W = lab.shape
            d0 = np.clip(cz - pd // 2, 0, max(0, D - pd))
            h0 = np.clip(cy - ph // 2, 0, max(0, H - ph))
            w0 = np.clip(cx - pw // 2, 0, max(0, W - pw))
            img_patch = img[d0:d0+pd, h0:h0+ph, w0:w0+pw]
            lab_patch = lab[d0:d0+pd, h0:h0+ph, w0:w0+pw]
            if img_patch.shape != self.patch_size:
                img_patch = pad_or_crop_to(img_patch, self.patch_size)
                lab_patch = pad_or_crop_to(lab_patch, self.patch_size)
        else:
            img_patch, lab_patch = random_crop3d(img, lab, self.patch_size)
        img_t = torch.from_numpy(img_patch).unsqueeze(0)
        lab_t = torch.from_numpy(lab_patch).long()
        return img_t, lab_t


# ============ 指标 ============
def dice_per_class(pred, target, num_classes, eps=1e-6):
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        targ_c = (target == c).float()
        intersect = (pred_c * targ_c).sum()
        denom = pred_c.sum() + targ_c.sum()
        dice = (2 * intersect + eps) / (denom + eps)
        dices.append(dice.item())
    fg_mean = np.mean(dices[1:]) if num_classes > 1 else dices[0]
    return dices, float(fg_mean)


def flat_metrics_binary(pred, target):
    p = pred.detach().cpu().numpy().astype(int).ravel()
    t = target.detach().cpu().numpy().astype(int).ravel()
    return {
        "IoU": jaccard_score(t, p, average="binary"),
        "Precision": precision_score(t, p, zero_division=0),
        "Recall": recall_score(t, p, zero_division=0),
    }


# ============ 损失函数 ============
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        inputs = torch.softmax(inputs, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = torch.sum(inputs * targets_onehot, dims)
        cardinality = torch.sum(inputs + targets_onehot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice.mean()


# ============ 训练器 ============
class Trainer3D:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-4, pretrained_path=None, class_weights=None, dice_weight=0.5):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dice_loss = DiceLoss()
        w = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
        self.ce_loss = nn.CrossEntropyLoss(weight=w)
        self.dice_weight = dice_weight
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained weights from {pretrained_path}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_loss(self, outputs, labels):
        ce = self.ce_loss(outputs, labels)
        dice = self.dice_loss(outputs, labels)
        return self.dice_weight * dice + (1 - self.dice_weight) * ce, ce.item(), dice.item()

    def train_one_epoch(self):
        self.model.train()
        running_loss, running_ce, running_dice = 0.0, 0.0, 0.0
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            if isinstance(outputs, list):
                outputs = outputs[0]
            loss, ce_val, dice_val = self.compute_loss(outputs, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            running_ce += ce_val
            running_dice += dice_val
        n_batches = max(1, len(self.train_loader))
        return running_loss/n_batches, running_ce/n_batches, running_dice/n_batches

    @torch.no_grad()
    def validate(self, num_classes):
        """Patch-based validation (快速)"""
        self.model.eval()
        val_loss, fg_dices = 0.0, []
        per_class_accum = np.zeros(num_classes, dtype=np.float64)
        count_batches = 0
        ious, precs, recs = [], [], []

        for images, labels in tqdm(self.val_loader, desc="Patch Validation"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            if isinstance(outputs, list):
                outputs = outputs[0]
            loss, _, _ = self.compute_loss(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            dices, fg_mean = dice_per_class(preds, labels, num_classes)
            per_class_accum += np.array(dices, dtype=np.float64)
            fg_dices.append(fg_mean)

            if num_classes == 2:
                m = flat_metrics_binary(preds, labels)
                ious.append(m["IoU"])
                precs.append(m["Precision"])
                recs.append(m["Recall"])
            count_batches += 1

        results = {
            "val_loss": val_loss / max(1, count_batches),
            "dice_fg_mean": float(np.mean(fg_dices)) if fg_dices else 0.0,
            "dice_per_class": (per_class_accum / max(1, count_batches)).tolist()
        }
        if num_classes == 2 and ious:
            results.update({
                "IoU": float(np.mean(ious)),
                "Precision": float(np.mean(precs)),
                "Recall": float(np.mean(recs)),
            })
        return results

    @torch.no_grad()
    def inference_sliding_window(self, image, patch_size, overlap=0.5, num_classes=2):
        self.model.eval()
        D,H,W = image.shape
        pd,ph,pw = patch_size
        stride_d = max(1,int(pd*(1-overlap)))
        stride_h = max(1,int(ph*(1-overlap)))
        stride_w = max(1,int(pw*(1-overlap)))
        prob_map = torch.zeros((num_classes,D,H,W), device=self.device)
        count_map = torch.zeros((D,H,W), device=self.device)

        d_starts = list(range(0, max(1,D-pd+1), stride_d))
        if (D-pd)%stride_d != 0: d_starts.append(max(0,D-pd))
        h_starts = list(range(0, max(1,H-ph+1), stride_h))
        if (H-ph)%stride_h !=0: h_starts.append(max(0,H-ph))
        w_starts = list(range(0, max(1,W-pw+1), stride_w))
        if (W-pw)%stride_w !=0: w_starts.append(max(0,W-pw))

        for d in d_starts:
            for h in h_starts:
                for w in w_starts:
                    patch = pad_or_crop_to(image[d:d+pd,h:h+ph,w:w+pw], (pd,ph,pw))
                    patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                    logits = self.model(patch_t)
                    if isinstance(logits,list): logits=logits[0]
                    probs = torch.softmax(logits, dim=1)[0]
                    dd,dh,dw = min(pd,D-d), min(ph,H-h), min(pw,W-w)
                    prob_map[:,d:d+dd,h:h+dh,w:w+dw] += probs[:,:dd,:dh,:dw]
                    count_map[d:d+dd,h:h+dh,w:w+dw] += 1
        count_map[count_map==0]=1.0
        prob_map = prob_map / count_map.unsqueeze(0)
        pred = torch.argmax(prob_map, dim=0)
        return pred.cpu()

    @torch.no_grad()
    def validate_full_volume(self, dataset, indices, patch_size, overlap=0.5, num_classes=2):
        dices_per_case = []
        ious, precs, recs = [], [], []
        for idx in tqdm(indices, desc="Full-volume Validation"):
            img, lab = dataset._load_case(idx)
            pred = self.inference_sliding_window(img, patch_size, overlap, num_classes)
            lab_t = torch.from_numpy(lab).long()
            dices, fg = dice_per_class(pred, lab_t, num_classes)
            dices_per_case.append(dices)
            if num_classes==2:
                m = flat_metrics_binary(pred, lab_t)
                ious.append(m["IoU"])
                precs.append(m["Precision"])
                recs.append(m["Recall"])
        dices_per_case = np.array(dices_per_case)
        mean_dices = np.mean(dices_per_case, axis=0).tolist()
        results = {
            "dice_per_class": mean_dices,
            "dice_fg_mean": float(np.mean(mean_dices[1:])) if num_classes>1 else float(mean_dices[0])
        }
        if num_classes==2 and ious:
            results.update({
                "IoU": float(np.mean(ious)),
                "Precision": float(np.mean(precs)),
                "Recall": float(np.mean(recs)),
            })
        return results


# ============ 主入口 ============
def main():
    set_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_dir = "/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset027_upperlimb100/imagesTr"
    labels_dir = "/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset027_upperlimb100/labelsTr"
    save_path = "/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset027_upperlimb100/checkpoint_best.pth"
    pretrained_path = None

    num_classes = 2
    patch_size = (64,128,128)
    batch_size = 2
    epochs = 200
    patience = 20
    base_lr = 1e-4
    class_weights = None
    fullval_every = 20  # 每20轮做整卷验证

    full_dataset = Nifti3DDataset(images_dir, labels_dir, patch_size, norm='zscore')
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(2025))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1,batch_size//2), shuffle=False,
                            num_workers=4, pin_memory=True)

    model = CBAMPlainConvUNet(
        input_channels=1,
        n_stages=4,
        features_per_stage=[32,64,128,256],
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=[1,2,2,2],
        n_conv_per_stage=2,
        num_classes=num_classes,
        n_conv_per_stage_decoder=2,
        conv_bias=True,
        norm_op=nn.BatchNorm3d,
        norm_op_kwargs={"eps":1e-5,"affine":True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={"inplace":True},
        deep_supervision=False,
        nonlin_first=False
    )

    trainer = Trainer3D(model, train_loader, val_loader, device,
                        lr=base_lr, pretrained_path=pretrained_path,
                        class_weights=class_weights, dice_weight=0.5)

    csv_path = os.path.join(os.path.dirname(save_path), "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["Epoch","TrainLoss","CE_Loss","Dice_Loss","PatchValLoss","PatchValDice","FullValDice"]
    if num_classes==2:
        header += ["IoU","Precision","Recall"]
    csv_writer.writerow(header)

    best_fg_dice = -math.inf
    counter = 0
    val_indices = val_set.indices if isinstance(val_set, Subset) else list(range(n_val))

    for epoch in range(1, epochs+1):
        train_loss, ce_loss, dice_loss_val = trainer.train_one_epoch()
        patch_val_res = trainer.validate(num_classes)
        fullval_res = None
        if epoch % fullval_every == 0:
            fullval_res = trainer.validate_full_volume(full_dataset, val_indices, patch_size,
                                                       overlap=0.5, num_classes=num_classes)

        fg_dice = patch_val_res["dice_fg_mean"] if fullval_res is None else fullval_res["dice_fg_mean"]

        row = [epoch, train_loss, ce_loss, dice_loss_val,
               patch_val_res["val_loss"], patch_val_res["dice_fg_mean"],
               fg_dice]
        if num_classes==2:
            iou = patch_val_res.get("IoU",0.0) if fullval_res is None else fullval_res.get("IoU",0.0)
            prec = patch_val_res.get("Precision",0.0) if fullval_res is None else fullval_res.get("Precision",0.0)
            rec = patch_val_res.get("Recall",0.0) if fullval_res is None else fullval_res.get("Recall",0.0)
            row += [iou, prec, rec]
        csv_writer.writerow(row)
        csv_file.flush()

        if fg_dice > best_fg_dice:
            best_fg_dice = fg_dice
            torch.save(model.state_dict(), save_path)
            counter = 0
            print(f"✅ Epoch {epoch}: Best model saved, FG Dice={best_fg_dice:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"🔹 Early stopping at epoch {epoch}")
                break

    csv_file.close()


if __name__=="__main__":
    main()
