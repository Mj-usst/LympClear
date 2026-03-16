'''
面向标注好的数据，准换格式输入模型中训练，不是推理用的
# 1.数据文件转换：
# 将 .dcm 文件转换为 .nii 或 .nii.gz 文件。
# 将 .nrrd 文件转换为 .nii 或 .nii.gz 文件。

# 2.生成 dataset.json 文件： dataset.json 文件包含有关数据集的元信息，如类标签、训练和测试集的图像数量等。
 '''


import os
import SimpleITK as sitk
import json
import logging

input_base_dir = r"/home/huawei/project/dataset/vein_raw/tagging_count262"
output_base_dir = r"/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset112_Vein"
log_file = os.path.join(output_base_dir, "processing_log.txt")

# 配置日志记录
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建输出目录结构
os.makedirs(os.path.join(output_base_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "labelsTr"), exist_ok=True)

# 将多个DICOM文件合并为一个NIfTI文件
def convert_dcm_series_to_nifti(dcm_dir, output_file):
    try:
        reader = sitk.ImageSeriesReader()
        dcm_files = reader.GetGDCMSeriesFileNames(dcm_dir)
        reader.SetFileNames(dcm_files)
        image = reader.Execute()
        sitk.WriteImage(image, output_file)
    except Exception as e:
        logging.error(f"Failed to convert DICOM series {dcm_dir} to NIfTI: {e}")
        raise

# 将NRRD文件转换为NIfTI文件
def convert_nrrd_to_nifti(nrrd_file, output_file):
    try:
        image = sitk.ReadImage(nrrd_file)
        sitk.WriteImage(image, output_file)
    except Exception as e:
        logging.error(f"Failed to convert NRRD file {nrrd_file} to NIfTI: {e}")
        raise

# 检查图像和分割文件的方向、形状和间距是否一致
def check_image_compatibility(image_file, seg_file):
    try:
        img = sitk.ReadImage(image_file)
        seg = sitk.ReadImage(seg_file)

        img_direction = img.GetDirection()
        seg_direction = seg.GetDirection()
        img_shape = img.GetSize()
        seg_shape = seg.GetSize()
        img_spacing = img.GetSpacing()
        seg_spacing = seg.GetSpacing()

        if img_direction != seg_direction:
            raise ValueError(f"方向不匹配: 图像方向 {img_direction}, 分割方向 {seg_direction}")

        if img_shape != seg_shape:
            raise ValueError(f"形状不匹配: 图像形状 {img_shape}, 分割形状 {seg_shape}")

        if img_spacing != seg_spacing:
            raise ValueError(f"间距不匹配: 图像间距 {img_spacing}, 分割间距 {seg_spacing}")

    except Exception as e:
        logging.error(f"Failed to check compatibility for {image_file} and {seg_file}: {e}")
        raise

# 遍历患者文件夹
patient_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
print("患者文件夹数量:", len(patient_dirs))

# 初始化编号和编号对应关系
counter = 8001
patient_id_mapping = {}

for patient in patient_dirs:
    patient_dir = os.path.join(input_base_dir, patient)
    print("正在处理：", patient)

    # 查找图像和标签路径
    img_dir = os.path.join(patient_dir, patient)
    mask_file = os.path.join(patient_dir, f"{patient}.nrrd")
    
    # 生成编号，格式为4位数，不足补零
    patient_id = f"{counter:04d}"
    patient_id_mapping[patient] = patient_id
    
    # 输出文件路径
    output_image_file = os.path.join(output_base_dir, "imagesTr", f"vein_{patient_id}_0002.nii.gz")
    output_mask_file = os.path.join(output_base_dir, "labelsTr", f"vein_{patient_id}.nii.gz")
    
    # 打印输出路径
    print(f"输出图像文件路径: {output_image_file}")
    print(f"输出标签文件路径: {output_mask_file}")

    try:
        # 合并DICOM并转换为NIfTI
        convert_dcm_series_to_nifti(img_dir, output_image_file)
        
        # 转换NRRD为NIfTI
        convert_nrrd_to_nifti(mask_file, output_mask_file)
        
        # 检查图像和分割文件的兼容性
        check_image_compatibility(output_image_file, output_mask_file)
        
    except Exception as e:
        logging.error(f"Error processing patient {patient}: {e}")
        continue
    
    counter += 1

# 生成 dataset.json 文件
dataset_info = {
    "name": "vein",
    "description": "Lymphedema veins of lower limbs",
    "tensorImageSize": "3D",
    "channel_names": {
        "0": "mri"
    },
    "file_ending": ".nii.gz",
    "labels": {
        "background": "0",
        "vein": "1"
    },
    "numTraining": len(patient_dirs),
    "numTest": 0,
    "training": [],
    "test": []
}

for patient in patient_dirs:
    patient_id = patient_id_mapping[patient]
    
    output_image_file = os.path.join(output_base_dir, "imagesTr", f"vein_{patient_id}_0002.nii.gz")
    output_mask_file = os.path.join(output_base_dir, "labelsTr", f"vein_{patient_id}.nii.gz")
    
    dataset_info["training"].append({
        "image": f"./imagesTr/vein_{patient_id}_0002.nii.gz",
        "label": f"./labelsTr/vein_{patient_id}.nii.gz"
    })
    
    # 打印处理信息
    print(f"正在处理：{patient}")
    print(f"输出图像文件路径: {output_image_file}")
    print(f"输出标签文件路径: {output_mask_file}")

# 保存 dataset.json 文件
with open(os.path.join(output_base_dir, "dataset.json"), 'w') as f:
    json.dump(dataset_info, f, indent=4)

# 保存患者编号对应关系
with open(os.path.join(output_base_dir, "patient_id_mapping.json"), 'w') as f:
    json.dump(patient_id_mapping, f, indent=4)

print("数据转换和 dataset.json 文件创建完成。")
print("患者编号对应关系已保存到 patient_id_mapping.json")
print("日志记录已保存到", log_file)

