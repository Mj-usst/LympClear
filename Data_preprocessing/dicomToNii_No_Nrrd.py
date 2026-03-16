

"""
一般是用于新的数据
dataset/
├── 标注人员1
├── 标注人员2
│   ├── 患者id1
│   ├── 患者id2
│   ├── 患者id3
│       ├──001.dcm
│       ├──002.dcm
│       ├──003.dcm
│       ├──004.dcm
│       ├── ...

"""

import os
import SimpleITK as sitk
import logging
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

input_base_dir = r"/home/huawei/project/dataset/upload"
output_base_dir = r"/home/huawei/project/dataset/infer_nii"
log_file = os.path.join(output_base_dir, "processing_log.txt")
csv_file = os.path.join(output_base_dir, "id_mapping.csv")

# 配置日志记录
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建输出目录结构
os.makedirs(os.path.join(output_base_dir, "result"), exist_ok=True)
output_base_dir = os.path.join(output_base_dir, "result")

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

# 定义每个患者的处理任务
def process_patient(patient, counter, id_mapping):
    patient_dir = os.path.join(input_base_dir, patient)
    print("正在处理：", patient)
    
    # 生成编号，格式为3位数，不足补零
    patient_id = f"{counter:03d}"
    output_image_file = os.path.join(output_base_dir, f"vein_{patient_id}_0002.nii.gz")
    
    print(f"输出图像文件路径: {output_image_file}")
    
    try:
        convert_dcm_series_to_nifti(patient_dir, output_image_file)
        id_mapping.append([patient, output_image_file])  # 记录患者ID与文件名的对应关系
    except Exception as e:
        logging.error(f"Error processing patient {patient}: {e}")

# 初始化患者文件夹列表
patient_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
print("患者文件夹数量:", len(patient_dirs))

# 保存ID对应关系
id_mapping = []

# 使用线程池并行处理
with ThreadPoolExecutor(max_workers=4) as executor:  # 你可以根据你的CPU核心数调整线程数
    futures = {executor.submit(process_patient, patient, i + 1, id_mapping): patient for i, patient in enumerate(patient_dirs)}
    
    for future in as_completed(futures):
        patient = futures[future]
        try:
            future.result()
            print(f"{patient}处理完成")
        except Exception as e:
            print(f"{patient}处理时出错: {e}")

# 将ID映射关系保存为CSV文件
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["PatientID", "NIfTI File"])  # 写入标题行
    writer.writerows(id_mapping)  # 写入每一行患者ID和NIfTI文件路径的对应关系

print(f"ID映射关系已保存到 {csv_file}")
print("数据转换完成。")
print("日志记录已保存到", log_file)
