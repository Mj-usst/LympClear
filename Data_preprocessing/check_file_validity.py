import os
import nibabel as nib
import numpy as np

def calculate_block_features(mask_path, ct_path):
    # 这里你可以根据实际需要对mask和CT图像进行特征提取处理
    # 假设返回一个简单的特征（比如CT图像的平均值和mask的体积）
    
    try:
        mask_img = nib.load(mask_path)
        ct_img = nib.load(ct_path)
        
        mask_data = mask_img.get_fdata()
        ct_data = ct_img.get_fdata()

        if np.sum(mask_data) == 0:  # 检查mask是否为空
            print(f"Warning: Mask at {mask_path} is empty.")
            return None
        
        # 计算mask区域的CT图像值分布（简单的例子）
        masked_ct_values = ct_data[mask_data > 0]
        if len(masked_ct_values) == 0:  # 如果mask区域没有覆盖CT图像中的有效区域
            print(f"Warning: No valid CT data found for mask at {mask_path}.")
            return None
        
        return {
            'average_ct_value': np.mean(masked_ct_values), 
            'mask_volume': np.sum(mask_data)
        }

    except Exception as e:
        print(f"Error processing {mask_path} and {ct_path}: {e}")
        return None

def check_and_process_masks(mask_folder, ct_folder):
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.nii.gz')]
    results = []
    
    for mask_file in mask_files:
        base_name = os.path.basename(mask_file).split('.')[0]
        ct_path = os.path.join(ct_folder, f"{base_name}_0000.nii.gz")

        if os.path.exists(ct_path):
            print(f"Processing mask: {mask_file} with CT: {ct_path}")
            result = process_mask(mask_file, ct_folder)
            if result:
                results.append(result)
        else:
            print(f"CT file {ct_path} not found for mask {mask_file}. Skipping.")
    
    return results

def process_mask(mask, ct_folder):
    base_name = os.path.basename(mask).split('.')[0]  # 获取文件名（去掉扩展名）
    ct_path = os.path.join(ct_folder, f"{base_name}_0000.nii.gz")  # 假设CT图像文件名为{base_name}_0000.nii.gz
    return calculate_block_features(mask, ct_path)

# 示例调用
mask_folder = '/home/huawei/project/dataset/dataset/benign_label/'  # 请根据实际路径替换
ct_folder = '/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset007_Breast_all/imagesTr/'  # 请根据实际路径替换
results = check_and_process_masks(mask_folder, ct_folder)
