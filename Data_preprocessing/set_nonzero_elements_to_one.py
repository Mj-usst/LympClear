import os
import SimpleITK as sitk
import numpy as np

def set_nonzero_elements_to_one(labels_dir, output_dir):
    """
    遍历指定文件夹中的所有标签文件，将其中的非零元素设置为1，
    并将结果保存到指定的输出文件夹中，文件名与原文件名一致。

    :param labels_dir: 标签文件所在的目录路径
    :param output_dir: 保存处理后结果的目录路径
    """
    # 获取所有nii.gz标签文件的列表
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]

    # 遍历所有标签文件并检查标签值
    for label_file in label_files:
        # 读取标签图像
        label_path = os.path.join(labels_dir, label_file)
        label_image = sitk.ReadImage(label_path)

        # 将标签图像转换为numpy数组
        label_array = sitk.GetArrayFromImage(label_image)

        # 将非零元素设置为1
        label_array[label_array != 0] = 1

        # 将处理后的数组转换回SimpleITK图像
        updated_label_image = sitk.GetImageFromArray(label_array)

        # 保留原图像的元数据，例如原图的空间信息
        updated_label_image.CopyInformation(label_image)

        # 构建保存路径，并保存更新后的图像
        updated_label_path = os.path.join(output_dir, label_file)
        sitk.WriteImage(updated_label_image, updated_label_path)

        print(f'Processed and saved: {updated_label_path}')

# 示例使用
labels_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset007_Breast007/nnUNetTrainer__nnUNetPlans__3d_fullres/infer'
output_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset007_Breast007/nnUNetTrainer__nnUNetPlans__3d_fullres/set_nonzero_elements_to_one'  # 设置输出文件夹路径

set_nonzero_elements_to_one(labels_dir, output_dir)
