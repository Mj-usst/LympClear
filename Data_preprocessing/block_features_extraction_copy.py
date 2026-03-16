import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import traceback

def calculate_block_features(mask_path, ct_image_path):
    try:
        # 读取mask文件和CT图像
        mask_img = sitk.ReadImage(mask_path)
        ct_img = sitk.ReadImage(ct_image_path)

        # 转换为numpy数组
        mask_array = sitk.GetArrayFromImage(mask_img)
        ct_array = sitk.GetArrayFromImage(ct_img)

        # 假设肿块值为2，如果掩码值为1，请改为 == 1
        tumor_label = 2

        # 肿块的CT值统计
        tumor_mask = (mask_array == tumor_label)
        masked_ct_values = ct_array[tumor_mask]
        tumor_stats = {
            'mean': np.nan if masked_ct_values.size == 0 else masked_ct_values.mean(),
            'min': np.nan if masked_ct_values.size == 0 else masked_ct_values.min(),
            'max': np.nan if masked_ct_values.size == 0 else masked_ct_values.max(),
            'std': np.nan if masked_ct_values.size == 0 else masked_ct_values.std(),
            'histogram': np.histogram(masked_ct_values, bins=50)[0] if masked_ct_values.size > 0 else []
        }

        # 外扩5mm区域的CT值统计
        spacing = mask_img.GetSpacing()
        dist_map = distance_transform_edt(~tumor_mask, sampling=spacing)
        outer_region_mask = (dist_map > 0) & (dist_map <= 5)
        outer_ct_values = ct_array[outer_region_mask]
        outer_stats = {
            'mean': np.nan if outer_ct_values.size == 0 else outer_ct_values.mean(),
            'min': np.nan if outer_ct_values.size == 0 else outer_ct_values.min(),
            'max': np.nan if outer_ct_values.size == 0 else outer_ct_values.max(),
            'std': np.nan if outer_ct_values.size == 0 else outer_ct_values.std(),
            'histogram': np.histogram(outer_ct_values, bins=50)[0] if outer_ct_values.size > 0 else []
        }

        return {
            'tumor_mean': tumor_stats['mean'],
            'tumor_min': tumor_stats['min'],
            'tumor_max': tumor_stats['max'],
            'tumor_std': tumor_stats['std'],
            'tumor_histogram': tumor_stats['histogram'].tolist(),
            'outer_mean': outer_stats['mean'],
            'outer_min': outer_stats['min'],
            'outer_max': outer_stats['max'],
            'outer_std': outer_stats['std'],
            'outer_histogram': outer_stats['histogram'].tolist(),
            'mask_path': mask_path
        }
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")
        traceback.print_exc()
        return None


def process_mask(mask_path, ct_folder):
    """
    单个mask处理，返回一个字典结果
    """
    base_name = os.path.basename(mask_path).split('.')[0]
    # 确保下面文件名对应你的CT文件，比如 XXX_0000.nii.gz
    ct_path = os.path.join(ct_folder, f"{base_name}_0000.nii.gz")
    try:
        return calculate_block_features(mask_path, ct_path)
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        traceback.print_exc()
        return None


def process_masks_concurrently(masks, ct_folder, max_workers=4):
    """
    并行处理所有的mask文件，返回结果列表
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_mask, masks, [ct_folder] * len(masks)),
                        total=len(masks), desc="Processing Masks"):
            if res is not None:
                results.append(res)
    return results


def save_results_to_excel(results, benign_excel, malignant_excel):
    """
    将处理结果写入良性和恶性Excel文件
    """
    benign_data = [r for r in results if "benign" in r['mask_path']]
    malignant_data = [r for r in results if "malignant" in r['mask_path']]

    if benign_data:
        df_benign = pd.DataFrame(benign_data)
        append_df_to_excel(df_benign, benign_excel)

    if malignant_data:
        df_malignant = pd.DataFrame(malignant_data)
        append_df_to_excel(df_malignant, malignant_excel)


def append_df_to_excel(df, excel_path, sheet_name="Results"):
    """
    追加写DataFrame到指定的Excel文件中
    """
    if not os.path.exists(excel_path):
        df.to_excel(excel_path, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            if sheet_name not in writer.book.sheetnames:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            else:
                startrow = writer.sheets[sheet_name].max_row
                df.to_excel(writer, index=False, header=False,
                            startrow=startrow, sheet_name=sheet_name)


if __name__ == "__main__":
    # 设置路径
    ct_folder = '/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset017_Breast_all/imagesTr/'
    benign_mask_folder = '/home/huawei/project/dataset/dataset/benign_label/'
    malignant_mask_folder = '/home/huawei/project/dataset/dataset/malignant_label/'

    benign_excel = "/home/huawei/project/nnUNet/Data_preprocessing/results_benign.xlsx"
    malignant_excel = "/home/huawei/project/nnUNet/Data_preprocessing/results_malignant.xlsx"

    # 获取所有mask文件
    benign_masks = [
        os.path.join(benign_mask_folder, f)
        for f in os.listdir(benign_mask_folder)
        if f.endswith('.nii.gz')
    ]
    malignant_masks = [
        os.path.join(malignant_mask_folder, f)
        for f in os.listdir(malignant_mask_folder)
        if f.endswith('.nii.gz')
    ]

    # 合并所有mask文件
    all_masks = benign_masks + malignant_masks

    # 设置并行处理的最大进程数
    max_workers = min(16, os.cpu_count())

    # 并行计算
    all_results = process_masks_concurrently(all_masks, ct_folder, max_workers=max_workers)

    # 保存结果到Excel
    save_results_to_excel(all_results, benign_excel, malignant_excel)

    print(f"Results saved to:\nBenign: {benign_excel}\nMalignant: {malignant_excel}")
