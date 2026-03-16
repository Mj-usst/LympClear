import os
import SimpleITK as sitk
import numpy as np

def merge_masks(gland_mask_dir, tumor_mask_dir, output_dir):
    """
    合并两个标签文件夹中的腺体和肿块mask为一个新的标签文件夹。
    如果标签图像的尺寸不一致，输出错误信息并跳过该对文件。
    
    :param gland_mask_dir: 腺体mask文件夹路径
    :param tumor_mask_dir: 肿块mask文件夹路径
    :param output_dir: 保存合并后mask的输出文件夹路径
    """
    # 获取腺体和肿块的标签文件
    gland_mask_files = [f for f in os.listdir(gland_mask_dir) if f.endswith('.nii.gz')]
    tumor_mask_files = [f for f in os.listdir(tumor_mask_dir) if f.endswith('.nii.gz')]

    # 确保腺体和肿块文件数量一致
    if len(gland_mask_files) != len(tumor_mask_files):
        print("腺体和肿块mask文件数量不匹配，请检查文件夹中的文件！")
        return

    # 遍历文件，进行合并
    for gland_file, tumor_file in zip(gland_mask_files, tumor_mask_files):
        # 读取腺体和肿块mask
        gland_path = os.path.join(gland_mask_dir, gland_file)
        tumor_path = os.path.join(tumor_mask_dir, tumor_file)
        
        gland_image = sitk.ReadImage(gland_path)
        tumor_image = sitk.ReadImage(tumor_path)
        
        # 检查两个图像的尺寸是否一致
        if gland_image.GetSize() != tumor_image.GetSize():
            print(f"错误: 图像尺寸不匹配: {gland_file} 和 {tumor_file}")
            print(f"腺体mask尺寸: {gland_image.GetSize()}, 肿块mask尺寸: {tumor_image.GetSize()}")
            continue  # 跳过该对文件
        
        # 将图像转换为numpy数组
        gland_array = sitk.GetArrayFromImage(gland_image)
        tumor_array = sitk.GetArrayFromImage(tumor_image)
        
        # 合并腺体和肿块的mask
        merged_array = np.zeros_like(gland_array)
        merged_array[gland_array == 1] = 1  # 腺体部分为1
        merged_array[tumor_array == 2] = 2  # 肿块部分为2
        
        # 将合并后的numpy数组转换回SimpleITK图像
        merged_image = sitk.GetImageFromArray(merged_array)
        
        # 保留原图像的元数据（空间信息等）
        merged_image.CopyInformation(gland_image)
        
        # 保存合并后的图像
        merged_path = os.path.join(output_dir, gland_file)  # 保持原文件名
        sitk.WriteImage(merged_image, merged_path)
        
        print(f'Processed and saved: {merged_path}')



# 使用
gland_mask_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset007_Breast007/nnUNetTrainer__nnUNetPlans__3d_fullres/set_nonzero_elements_to_one'  # 腺体mask文件夹路径
tumor_mask_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset007_Breast007/nnUNetTrainer__nnUNetPlans__3d_fullres/Mask with only tumor annotations'  # 肿块mask文件夹路径
output_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset007_Breast007/nnUNetTrainer__nnUNetPlans__3d_fullres/merge_tags'  # 合并后保存的文件夹路径

merge_masks(gland_mask_dir, tumor_mask_dir, output_dir)
