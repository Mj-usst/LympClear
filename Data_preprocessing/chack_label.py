



import os
import nibabel as nib
import numpy as np

# 设置标签文件所在的目录路径
labels_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset007_Breast007/nnUNetTrainer__nnUNetPlans__3d_fullres/Mask with only tumor annotations'

# 获取所有nii.gz标签文件的列表
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]

# 遍历所有标签文件并检查标签值
for label_file in label_files:
    label_path = os.path.join(labels_dir, label_file)
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()
    
    # 获取标签文件中包含的唯一标签值
    unique_labels = np.unique(label_data)
    print(f"标签文件 {label_file} 中包含的标签: {unique_labels}")
