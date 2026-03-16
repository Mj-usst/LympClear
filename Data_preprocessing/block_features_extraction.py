import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, find_objects, distance_transform_edt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import traceback
# 如果你想用更准确的表面积，可以以后改用 skimage.measure.marching_cubes


def calculate_block_features(mask_path, ct_image_path):
    try:
        # 读取mask文件和CT图像
        mask_img = sitk.ReadImage(mask_path)
        ct_img = sitk.ReadImage(ct_image_path)

        # 转换为numpy数组
        mask_array = sitk.GetArrayFromImage(mask_img)
        ct_array = sitk.GetArrayFromImage(ct_img)

        # 这里假设肿块值 == 2，如果你的掩码实际为1，请改成 ==1
        tumor_label = 2

        # 计算肿块的密度均值
        masked_ct_values = ct_array[mask_array == tumor_label]
        mean_density = np.nan if masked_ct_values.size == 0 else masked_ct_values.mean()

        # 计算体积（使用像素间距）
        voxel_volume = np.prod(mask_img.GetSpacing())
        volume = np.sum(mask_array == tumor_label) * voxel_volume

        # 找前景，计算最小外接立方体
        labeled_array, num_features = label(mask_array == tumor_label)
        if num_features == 0:
            # 如果没有任何前景，返回空或默认值
            # 也可以直接 return None
            return {
                'mean_density': np.nan,
                'volume': 0,
                'min_cube_volume': 0,
                'length': 0,
                'width': 0,
                'height': 0,
                'surface_area': 0,
                'centroid_x': np.nan,
                'centroid_y': np.nan,
                'centroid_z': np.nan,
                'sphericity': 0,
                'mask_path': mask_path
            }

        # 只取第一个前景块，这里假定只有一个肿块，如果有多个需要自行处理
        obj_slices = find_objects(labeled_array)[0]
        min_box = np.array([
            obj_slices[0].start, obj_slices[0].stop,
            obj_slices[1].start, obj_slices[1].stop,
            obj_slices[2].start, obj_slices[2].stop
        ])
        length = (min_box[1] - min_box[0]) * mask_img.GetSpacing()[0]
        width  = (min_box[3] - min_box[2]) * mask_img.GetSpacing()[1]
        height = (min_box[5] - min_box[4]) * mask_img.GetSpacing()[2]
        min_cube_volume = length * width * height

        # 计算表面积（近似）
        surface_area = calculate_surface_area(mask_array == tumor_label)

        # 计算质心
        centroid = calculate_centroid(mask_array == tumor_label)

        # 计算球形度
        sphericity = calculate_sphericity(volume, surface_area)

        return {
            'mean_density': mean_density,
            'volume': volume,
            'min_cube_volume': min_cube_volume,
            'length': length,
            'width': width,
            'height': height,
            'surface_area': surface_area,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'centroid_z': centroid[2],
            'sphericity': sphericity,
            'mask_path': mask_path
        }
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")
        traceback.print_exc()
        return None


def calculate_surface_area(mask_bool_array):
    """
    计算肿块的表面积（近似）。
    注意：distance_transform_edt 得到的是浮点型，
    在3D中只有与背景距离刚好1个voxel的地方才计数为表面。
    """
    dist = distance_transform_edt(~mask_bool_array)  # mask_bool_array为True表示肿块，所以~表示背景
    # 只计算dist == 1的voxel数做近似
    surface_area = np.sum(dist == 1)
    return surface_area


def calculate_centroid(mask_bool_array):
    """
    计算肿块的质心（几何中心）
    """
    indices = np.array(np.nonzero(mask_bool_array))
    if indices.size == 0:
        return [np.nan, np.nan, np.nan]
    centroid = np.mean(indices, axis=1)
    return centroid


def calculate_sphericity(volume, surface_area):
    """
    计算球形度
    """
    if surface_area == 0:
        return 0
    return (36 * np.pi * (volume ** 2)) / (surface_area ** 3)


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
    # 注意：在 Windows 环境下，这个并行要放在 if __name__=="__main__": 保护下
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 返回的结果是按masks列表顺序排好的
        for res in tqdm(executor.map(process_mask, masks, [ct_folder]*len(masks)),
                        total=len(masks), desc="Processing Masks"):
            if res is not None:
                results.append(res)
    return results


def save_results_to_excel(results, benign_excel, malignant_excel):
    """
    将并行处理得到的所有结果一次性写入到良性/恶性 Excel 中
    （如果文件不存在则创建，存在就追加写）
    """
    # 拆分良性、恶性
    benign_data = [r for r in results if "benign" in r['mask_path']]
    malignant_data = [r for r in results if "malignant" in r['mask_path']]

    # 分别写
    if benign_data:
        df_benign = pd.DataFrame(benign_data)
        append_df_to_excel(df_benign, benign_excel)

    if malignant_data:
        df_malignant = pd.DataFrame(malignant_data)
        append_df_to_excel(df_malignant, malignant_excel)


def append_df_to_excel(df, excel_path, sheet_name="Results"):
    """
    追加写 DataFrame 到指定的 sheet 中，如果 Excel 不存在就创建。
    """
    if not os.path.exists(excel_path):
        # 文件不存在，直接写入（带header）
        df.to_excel(excel_path, index=False, sheet_name=sheet_name)
    else:
        # 文件存在，append模式
        # 注意：pandas>=1.2.0 才有 if_sheet_exists='overlay'/'replace'/'new'
        # 旧版可能叫 if_sheet_exists='replace'/'new'/'error'/'append'
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # 如果这个sheet不存在，会 KeyError，所以先确保它存在或捕获异常
            if sheet_name not in writer.book.sheetnames:
                # 不存在就创建
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            else:
                # 追加
                startrow = writer.sheets[sheet_name].max_row
                df.to_excel(writer, index=False, header=False,
                            startrow=startrow, sheet_name=sheet_name)


if __name__ == "__main__":
    # 设置路径
    ct_folder = '/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset007_Breast_all/imagesTr/'
    benign_mask_folder = '/home/huawei/project/dataset/dataset/benign_label/'
    malignant_mask_folder = '/home/huawei/project/dataset/dataset/malignant_label/'

    benign_excel = "/home/huawei/project/nnUNet/Data_preprocessing/results_benign.xlsx"
    malignant_excel = "/home/huawei/project/nnUNet/Data_preprocessing/results_malignant.xlsx"

    # 获取所有良性和恶性mask文件
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

    # 最后统一写 Excel
    save_results_to_excel(all_results, benign_excel, malignant_excel)

    print(f"Results saved to:\nBenign: {benign_excel}\nMalignant: {malignant_excel}")
