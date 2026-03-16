import SimpleITK as sitk

# 输入路径（nrrd 文件）
input_path = r"/home/huawei/project/dataset/nrrd/vein_001.nii.gz.nii.nrrd"

# 输出路径（nii.gz 文件）
output_path = r"/home/huawei/project/dataset/infer_result/vein_001.nii.gz"

# 读取 nrrd 文件
image = sitk.ReadImage(input_path)

# 保存为 nii.gz
sitk.WriteImage(image, output_path)

print("转换完成:", output_path)
