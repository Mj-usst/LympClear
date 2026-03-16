#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只评估 fold_0/validation 中存在预测结果的病例
计算 Dice / IoU / HD95(mm) / ASD(mm)，带进度条与运行中均值
优先使用 GPU (CuPy) 做 EDT 加速；无 CuPy 自动回退 CPU (SciPy)
新增：按病例并行（多进程），自动处理 CuPy 场景并发安全
"""

# —— 并行与数值库线程数限制（放最前，避免过度并发）——
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ===== 路径（按需修改）=====
GT_DIR   = "/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset113_Vein_all_5710/labelsTr"
PRED_DIR = "/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset113_Vein_all_5710/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"

# ===== 配置 =====
FOREGROUND_LABEL = 1        # 前景类别值（若不是1请改）
THRESH = 0.5                # 预测是概率图时的二值阈值
OUT_CSV = "seg_metrics_fold0.csv"
USE_ROI_CROP = True         # 是否裁剪到 (GT ∪ Pred) 的联合包围盒（强烈建议开启）
CROP_MARGIN = 8             # 裁剪外扩体素数

# ===== 并行参数 =====
PARALLEL = True
# 默认 worker 数：CPU 情况用物理核（或留 1 核做系统），GPU 情况自动降到 1
DEFAULT_WORKERS = max(cpu_count() - 1, 1)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", DEFAULT_WORKERS))
CHUNK_SIZE = 1   # imap_unordered 块大小

# ===== 可选：锁定显卡（配合 CUDA_VISIBLE_DEVICES 使用）=====
GPU_INDEX_WITHIN_VISIBLE = 0  # 在 CUDA_VISIBLE_DEVICES 生效后，这里的 0 指向那块“可见卡”的 0 号

# ===== 依赖：CuPy（GPU）优先，SciPy（CPU）回退 =====
HAS_CUPY = False
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cp_edt
    HAS_CUPY = True
    try:
        cp.cuda.Device(GPU_INDEX_WITHIN_VISIBLE).use()
    except Exception:
        pass
except Exception:
    HAS_CUPY = False

from scipy.ndimage import distance_transform_edt  # CPU 回退用
from scipy.ndimage import binary_erosion

# ===== 基础工具 =====
def dice_iou_binary(gt: np.ndarray, pr: np.ndarray):
    gt = gt.astype(bool); pr = pr.astype(bool)
    inter = np.logical_and(gt, pr).sum()
    gt_sum = gt.sum(); pr_sum = pr.sum()
    union = np.logical_or(gt, pr).sum()
    dice = (2.0 * inter) / (gt_sum + pr_sum + 1e-8)
    iou  = inter / (union + 1e-8)
    return float(dice), float(iou)

def crop_to_union_bbox(gt_bin: np.ndarray, pr_bin: np.ndarray, margin: int = 5):
    union = gt_bin | pr_bin
    if not union.any():
        return gt_bin, pr_bin
    # 注意：nibabel get_fdata() 默认数组轴顺序为 (X, Y, Z)
    x, y, z = np.where(union)
    x0, x1 = max(int(x.min())-margin, 0), int(x.max())+margin+1
    y0, y1 = max(int(y.min())-margin, 0), int(y.max())+margin+1
    z0, z1 = max(int(z.min())-margin, 0), int(z.max())+margin+1
    return gt_bin[x0:x1, y0:y1, z0:z1], pr_bin[x0:x1, y0:y1, z0:z1]

def surface_voxels(mask_bool: np.ndarray) -> np.ndarray:
    """表面体素（True/False）：一次腐蚀后取差。"""
    eroded = binary_erosion(mask_bool, iterations=1, border_value=0)
    return np.logical_and(mask_bool, np.logical_not(eroded))

# ===== 距离相关（GPU 优先 / CPU 回退）=====
def edt_distances_to_foreground(src_surface_bool: np.ndarray,
                                dst_mask_bool: np.ndarray,
                                spacing_xyz: tuple) -> np.ndarray:
    """
    计算从 src 表面每个点到 dst 前景(=1)最近点的欧氏距离（单位 mm）。
    使用 EDT(dst_foreground) 并在 src 表面采样。
    spacing_xyz 应与数组 (X,Y,Z) 轴顺序一致。
    返回 numpy 1D 距离数组。
    """
    spacing_xyz = tuple(map(float, spacing_xyz))
    if not src_surface_bool.any():
        return np.empty((0,), dtype=np.float32)

    if HAS_CUPY:
        # ---- GPU 路径 ----
        dst = cp.asarray(dst_mask_bool, dtype=cp.bool_)
        # EDT：距“最近的非零”(Foreground)的距离
        edt = cp_edt(dst, sampling=spacing_xyz)
        dists = edt[cp.asarray(src_surface_bool)]
        return cp.asnumpy(dists).astype(np.float32)
    else:
        # ---- CPU 路径 ----
        edt = distance_transform_edt(dst_mask_bool, sampling=spacing_xyz)
        return edt[src_surface_bool].astype(np.float32)

def compute_hd95_asd_mm(gt_bin: np.ndarray, pr_bin: np.ndarray, spacing_xyz: tuple):
    """
    用 EDT 快速计算 HD95 / ASD（mm）
    - 双向：GT→Pred 与 Pred→GT
    - HD95：两向距离合并取 95 百分位
    - ASD：两向平均的平均
    """
    # 表面
    gt_surf = surface_voxels(gt_bin.astype(bool))
    pr_surf = surface_voxels(pr_bin.astype(bool))

    if not gt_surf.any() and not pr_surf.any():
        return 0.0, 0.0
    if not gt_surf.any() or not pr_surf.any():
        return float("nan"), float("nan")

    d_gt_to_pr = edt_distances_to_foreground(gt_surf, pr_bin.astype(bool), spacing_xyz)
    d_pr_to_gt = edt_distances_to_foreground(pr_surf, gt_bin.astype(bool), spacing_xyz)

    both = np.concatenate([d_gt_to_pr, d_pr_to_gt]).astype(np.float32)
    finite = np.isfinite(both)
    hd95 = float(np.percentile(both[finite], 95)) if finite.any() else float("nan")

    def _avg(x):
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")
    asd = float(np.mean([_avg(d_gt_to_pr), _avg(d_pr_to_gt)]))
    return hd95, asd

# ===== 单病例评估（为并行设计）=====
def evaluate_one_case(pred_path: str):
    """
    输入：预测nii路径
    输出：dict: {case, dice, iou, hd95_mm, asd_mm, skipped(bool), warn(str or None)}
    """
    case = os.path.basename(pred_path)
    gt_path = os.path.join(GT_DIR, case)
    if not os.path.exists(gt_path):
        return dict(case=case, dice=np.nan, iou=np.nan, hd95_mm=np.nan, asd_mm=np.nan,
                    skipped=True, warn=f"GT not found for {case}")

    gt_img   = nib.load(gt_path)
    pred_img = nib.load(pred_path)

    if pred_img.shape != gt_img.shape:
        return dict(case=case, dice=np.nan, iou=np.nan, hd95_mm=np.nan, asd_mm=np.nan,
                    skipped=True, warn=f"Shape mismatch {case}: pred{pred_img.shape} vs gt{gt_img.shape}")

    gt_arr   = gt_img.get_fdata()
    pred_arr = pred_img.get_fdata()

    # 二值化
    gt_bin = (gt_arr == FOREGROUND_LABEL)
    if pred_arr.dtype.kind in ("f", "d"):
        pr_bin = (pred_arr > THRESH)
    else:
        pr_bin = (pred_arr == FOREGROUND_LABEL)

    # 空掩码稳健处理
    if gt_bin.sum() == 0 and pr_bin.sum() == 0:
        dice, iou, hd95, asd = 1.0, 1.0, 0.0, 0.0
    elif gt_bin.sum() == 0 or pr_bin.sum() == 0:
        dice, iou, hd95, asd = 0.0, 0.0, float("nan"), float("nan")
    else:
        # ROI 裁剪提速
        if USE_ROI_CROP:
            gt_bin, pr_bin = crop_to_union_bbox(gt_bin, pr_bin, margin=CROP_MARGIN)

        dice, iou = dice_iou_binary(gt_bin, pr_bin)

        # spacing：与数组轴顺序一致的 (sx, sy, sz)
        spacing_xyz = tuple(map(float, gt_img.header.get_zooms()[:3]))
        hd95, asd = compute_hd95_asd_mm(gt_bin, pr_bin, spacing_xyz)

    return dict(case=case, dice=float(dice), iou=float(iou),
                hd95_mm=float(hd95), asd_mm=float(asd),
                skipped=False, warn=None)

# ===== 主流程 =====
def main():
    pred_paths = sorted(glob.glob(os.path.join(PRED_DIR, "*.nii*")))
    if len(pred_paths) == 0:
        print(f"[ERROR] No predictions found in: {PRED_DIR}")
        return

    # CuPy 场景：默认强制单进程，避免多进程/多GPU上下文冲突
    workers = NUM_WORKERS
    if HAS_CUPY:
        if PARALLEL and workers > 1:
            print(f"[INFO] Detected CuPy. For safety, force NUM_WORKERS=1 (was {workers}). "
                  f"Set NUM_WORKERS>1 explicitly *only if* you know what you're doing.")
        workers = 1

    results = []
    warns = []

    if PARALLEL and workers > 1:
        print(f"[INFO] Parallel evaluation with {workers} workers (chunk_size={CHUNK_SIZE})")
        with Pool(processes=workers) as pool:
            for out in tqdm(pool.imap_unordered(evaluate_one_case, pred_paths, chunksize=CHUNK_SIZE),
                            total=len(pred_paths), desc="Evaluating (parallel)"):
                results.append(out)
                if out.get("warn"):
                    warns.append(out["warn"])
    else:
        # 串行（或GPU单进程）
        for p in tqdm(pred_paths, desc="Evaluating (serial/GPU-safe)"):
            out = evaluate_one_case(p)
            results.append(out)
            if out.get("warn"):
                warns.append(out["warn"])

    # 整理 DataFrame
    rows = [[r["case"], r["dice"], r["iou"], r["hd95_mm"], r["asd_mm"]] for r in results if not r["skipped"]]
    df = pd.DataFrame(rows, columns=["case", "dice", "iou", "hd95_mm", "asd_mm"])
    df.to_csv(OUT_CSV, index=False)

    # 运行中均值（并行下改为最终均值）
    if len(df):
        run_mean = {
            "dice": float(df["dice"].mean()),
            "iou":  float(df["iou"].mean()),
            "hd95": float(np.nanmean(df["hd95_mm"].values)),
            "asd":  float(np.nanmean(df["asd_mm"].values)),
        }
    else:
        run_mean = {"dice": np.nan, "iou": np.nan, "hd95": np.nan, "asd": np.nan}

    print("\n=== Summary (fold_0/validation only) ===")
    if len(df):
        print(df.describe(include="all"))
        print(f"\nRunning mean → Dice {run_mean['dice']:.3f} | IoU {run_mean['iou']:.3f} | "
              f"HD95 {run_mean['hd95']:.2f}mm | ASD {run_mean['asd']:.2f}mm")
        print(f"\nSaved: {OUT_CSV}  (n={len(df)})")
    else:
        print("No cases evaluated.")

    # 提示未评估或警告
    skipped = [r["case"] for r in results if r["skipped"]]
    if skipped:
        print(f"\n[WARN] {len(skipped)} cases skipped. Examples: {skipped[:5]}")
    if warns:
        print(f"\n[WARN] Details:")
        for w in warns[:10]:
            print("  -", w)
        if len(warns) > 10:
            print(f"  ... {len(warns)-10} more")

if __name__ == "__main__":
    main()
