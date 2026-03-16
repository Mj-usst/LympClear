import os
import nibabel as nib

def check_image_label_match(img_dir, label_dir):
    """
    检查每个图像文件是否在标签文件夹中有对应的标签文件，并且检查它们的形状是否一致。
    
    :param img_dir: 图像文件夹路径
    :param label_dir: 标签文件夹路径
    """
    # 获取图像文件和标签文件的列表
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]
    
    # 提取图像文件的前缀（去除 "_0000.nii.gz"）
    img_prefixes = set(f.split('_')[0] + '_' + f.split('_')[1] for f in img_files)  # 只取前两部分作为前缀
    
    # 提取标签文件的前缀（去除 ".nii.gz"）
    label_prefixes = set(f.split('.')[0] for f in label_files)
    
    # 检查每个图像文件是否有对应的标签文件
    for img_file in img_files:
        # 获取图像文件的前缀，去掉 '_0000' 部分
        img_prefix = '_'.join(img_file.split('_')[:2])  # 例如 'BREAST_8004'
        
        corresponding_label = img_prefix + '.nii.gz'  # 构造标签文件的名称
        
        if corresponding_label not in label_files:
            print(f"Warning: 图像文件 {img_file} 没有对应的标签文件 {corresponding_label}！")
        else:
            # 加载图像文件和标签文件
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, corresponding_label)
            
            try:
                img_data = nib.load(img_path).get_fdata()
                label_data = nib.load(label_path).get_fdata()
                
                # 检查图像和标签的形状是否一致
                if img_data.shape == label_data.shape:
                    print(f"匹配成功: 图像文件 {img_file} 和 标签文件 {corresponding_label}，形状一致: {img_data.shape}")
                else:
                    print(f"Warning: 图像文件 {img_file} 和 标签文件 {corresponding_label} 的形状不一致！")
                    print(f"  图像形状: {img_data.shape}，标签形状: {label_data.shape}")
            except Exception as e:
                print(f"Error: 无法读取文件 {img_file} 或 {corresponding_label}，错误信息: {e}")
                continue  # 跳过损坏的文件

# 示例路径
img_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset007_Breast007/imagesTr'
label_dir = '/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset007_Breast007/labelsTr'

# 调用函数检查匹配
check_image_label_match(img_dir, label_dir)
