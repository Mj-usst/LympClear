import nrrd
import os
import numpy as np

# 设置标签文件所在的目录路径
labels_dir = '/home/huawei/project/dataset/vein_raw/tagging_count262'

# 获取所有 .nrrd 标签文件的列表
label_files = [f for f in os.listdir(labels_dir) ]

print(label_files)
# 遍历所有标签文件并检查标签值
for label_file in label_files:
    label_path = os.path.join(labels_dir, label_file)
    label_path = os.path.join(label_path, f"{label_file}.nrrd")

    
    # 使用 nrrd 读取图像数据和头信息
    label_data, header = nrrd.read(label_path)
    
    # 获取标签文件中包含的唯一标签值
    unique_labels = np.unique(label_data)
    print(f"标签文件 {label_file} 中包含的标签: {unique_labels}")
