import json
import pandas as pd

# 读取 summary.json
json_path = "/home/huawei/project/nnUNet/DATASET/nnUNet_results/Dataset113_Vein_all_5710/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json"

with open(json_path, 'r') as f:
    data = json.load(f)

records = []
for case in data["metric_per_case"]:
    case_metrics = case["metrics"]["1"]  # 类别1的指标
    pred_file = case["prediction_file"].split("/")[-1]  # 只保留文件名
    ref_file = case["reference_file"].split("/")[-1]

    TP = case_metrics["TP"]
    TN = case_metrics["TN"]
    FP = case_metrics["FP"]
    FN = case_metrics["FN"]

    # 计算敏感性和特异性
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else None
    specificity = TN / (TN + FP) if (TN + FP) > 0 else None

    records.append({
        "Case": pred_file.replace(".nii.gz", ""),
        "Dice": case_metrics["Dice"],
        "IoU": case_metrics["IoU"],
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "n_pred": case_metrics["n_pred"],
        "n_ref": case_metrics["n_ref"]
    })

# 转为 DataFrame
df = pd.DataFrame(records)

# 保存为 CSV
csv_path = "fold4_validation_metrics.csv"
df.to_csv(csv_path, index=False)

print(f"保存完成: {csv_path}")
print(df.head())
