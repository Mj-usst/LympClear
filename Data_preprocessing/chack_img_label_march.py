"""
它可以帮助你检查图像数据和标注数据之间的形状、空间分辨率和原点是否匹配。该代码将遍历数据集中所有的图像和标注文件，并输出不匹配的文件信息。"""

import os
import nibabel as nib
import numpy as np

# 设置图像和标签文件所在的目录
image_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset003_Breast_CT_malig_easy/imagesTr'
seg_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset003_Breast_CT_malig_easy/labelsTr'

# 获取所有图像和标签文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]

# 检查图像和标签文件是否一一对应
for image_file in image_files:
    # 获取图像对应的标签文件名（假设文件名相同，除去后缀部分）
    seg_file = image_file.replace('BREAST', 'BREAST').replace('_0000.nii.gz', '.nii.gz')
    
    if seg_file not in seg_files:
        print(f"标签文件 {seg_file} 不存在，跳过该图像文件 {image_file}。")
        continue

    # 加载图像和标签数据
    image_path = os.path.join(image_dir, image_file)
    seg_path = os.path.join(seg_dir, seg_file)
    
    img = nib.load(image_path)
    seg = nib.load(seg_path)
    
    # 获取图像和标签的形状、spacing和origin
    img_shape = img.shape
    seg_shape = seg.shape
    img_spacing = img.header.get_zooms()
    seg_spacing = seg.header.get_zooms()
    img_origin = img.affine[:3, 3]
    seg_origin = seg.affine[:3, 3]

    # 检查形状是否匹配
    if img_shape != seg_shape:
        print(f"形状不匹配: 图像 {image_file} 形状 {img_shape}，标签 {seg_file} 形状 {seg_shape}")

    # 检查空间分辨率 (spacing) 是否匹配
    if np.any(np.array(img_spacing) != np.array(seg_spacing)):
        print(f"空间分辨率不匹配: 图像 {image_file} spacing {img_spacing}，标签 {seg_file} spacing {seg_spacing}")

    # 检查原点 (origin) 是否匹配
    if not np.allclose(img_origin, seg_origin, atol=1e-5):
        print(f"原点不匹配: 图像 {image_file} origin {img_origin}，标签 {seg_file} origin {seg_origin}")
        
print("检查完成！")
