


import os
import SimpleITK as sitk
import json

input_base_dir = r"/home/huawei/project/dataset/vein_raw"
output_base_dir = r"/home/huawei/project/dataset/vein_conv"

# 遍历患者文件夹
patient_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
# print("patient_dirs",patient_dirs)
print("list_length",len(patient_dirs))

for patient in patient_dirs:
    patient_dir = os.path.join(input_base_dir, patient)
    print("正在处理：",patient)
    # 查找 img 文件夹 修正患者文件夹下img文件夹与患者名不一致问题
    img_dir_name = None
    for item in os.listdir(patient_dir):
        item_path = os.path.join(patient_dir, item)
        if os.path.isdir(item_path):
            img_dir_name = item
            break
    
    if img_dir_name is None:
        print(f"患者 {patient} 没有找到 img 文件夹，跳过...")
        continue
    
    img_dir = os.path.join(patient_dir, img_dir_name)
    new_img_dir = os.path.join(patient_dir, patient)
    
    # 重命名 img 文件夹为患者名
    os.rename(img_dir, new_img_dir)