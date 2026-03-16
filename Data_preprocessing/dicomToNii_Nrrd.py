import os
import SimpleITK as sitk
import logging
import csv

"""
只处理DICOM (.dcm) 文件，将其转换为NIfTI (.nii.gz)，忽略NRRD文件，适用于已标注的数据。
dataset/
├── 患者id1
│   ├──患者id1
│       ├──001.dcm
│       ├──002.dcm
│       ├──003.dcm
│       ├── ...
│   ├──患者id1.nrrd (忽略)
├── 患者id2
├── 患者id3
...
"""

input_base_dir = r"/home/huawei/project/dataset/HuangQian"
output_base_dir = r"/home/huawei/project/nnUNet/DATASET/nnUNet_raw/Dataset112_Vein/imagesTs/HuangQian"
log_file = os.path.join(output_base_dir, "processing_log.txt")
id_mapping_file = os.path.join(output_base_dir, "id_mapping.csv")

# 配置日志记录
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 保存原始ID和修改后的ID对应关系
id_mapping = []

# 遍历患者文件夹
patient_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
print("患者文件夹数量:", len(patient_dirs))

def convert_dcm_series_to_nifti(dcm_dir, output_file):
    try:
        reader = sitk.ImageSeriesReader()
        dcm_files = reader.GetGDCMSeriesFileNames(dcm_dir)
        if not dcm_files:
            raise ValueError(f"No DICOM files found in directory: {dcm_dir}")
        reader.SetFileNames(dcm_files)
        image = reader.Execute()
        sitk.WriteImage(image, output_file)
    except Exception as e:
        logging.error(f"Failed to convert DICOM series {dcm_dir} to NIfTI: {e}")
        raise

# 创建输出目录结构
os.makedirs(output_base_dir, exist_ok=True)

# 初始化编号
counter = 1

for patient in patient_dirs:
    patient_dir = os.path.join(input_base_dir, patient)
    print("正在处理：", patient)

    # 获取该患者文件夹中的所有文件及子文件夹
    sub_dirs = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
    
    for sub_dir in sub_dirs:
        # 检查是否包含DICOM文件
        dcm_files = [f for f in os.listdir(sub_dir) if f.endswith('.dcm')]
        if not dcm_files:
            continue  # 跳过不包含DICOM文件的子文件夹

        # 生成编号，格式为3位数，不足补零
        patient_id = f"{counter:03d}"

        # 输出文件路径
        output_image_file = os.path.join(output_base_dir, f"vein_{patient_id}_0002.nii.gz")

        # 打印输出路径
        print(f"输出图像文件路径: {output_image_file}")

        try:
            # 合并DICOM并转换为NIfTI
            convert_dcm_series_to_nifti(sub_dir, output_image_file)
            
            # 记录原始ID和新的ID对应关系
            id_mapping.append([patient, patient_id])
        except Exception as e:
            logging.error(f"Error processing patient {patient} in subfolder {sub_dir}: {e}")
            continue

        counter += 1    

# 保存ID对应关系为CSV文件
with open(id_mapping_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Original Patient ID", "New Patient ID"])
    writer.writerows(id_mapping)

print("数据转换完成。")
print("日志记录已保存到", log_file)
print(f"ID对应关系已保存到 {id_mapping_file}")
