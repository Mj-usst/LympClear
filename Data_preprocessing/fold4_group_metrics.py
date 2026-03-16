import pandas as pd

# 读取 CSV 文件
csv_path = "/home/huawei/project/nnUNet/Data_preprocessing/fold4_validation_metrics.csv"
df = pd.read_csv(csv_path)

# 取第 100 行之后的 200 行
df = df.iloc[100:300].reset_index(drop=True)

# 检查数据行数
if len(df) < 200:
    raise ValueError(f"筛选后数据不足 200 行，当前行数: {len(df)}")

# 定义分组顺序和每组数量
groups = ["Stage 0/1", "Stage 2", "Stage 3", "Unilateral", "Bilateral"]
cases_per_group = 40

results = []

# 按顺序切片
for i, group in enumerate(groups):
    start_idx = i * cases_per_group
    end_idx = start_idx + cases_per_group
    subset = df.iloc[start_idx:end_idx]

    dice_mean = subset["Dice"].mean()
    dice_sd = subset["Dice"].std()

    iou_mean = subset["IoU"].mean()
    iou_sd = subset["IoU"].std()

    sens_mean = subset["Sensitivity"].mean()
    sens_sd = subset["Sensitivity"].std()

    spec_mean = subset["Specificity"].mean()
    spec_sd = subset["Specificity"].std()

    results.append({
        "Group": group,
        "n": cases_per_group,
        "Dice (mean ± SD)": f"{dice_mean:.4f} ± {dice_sd:.4f}",
        "IoU (mean ± SD)": f"{iou_mean:.4f} ± {iou_sd:.4f}",
        "Sensitivity (mean ± SD)": f"{sens_mean*100:.2f} ± {sens_sd*100:.2f}",  # 百分比
        "Specificity (mean ± SD)": f"{spec_mean*100:.2f} ± {spec_sd*100:.2f}"   # 百分比
    })

# 转成 DataFrame
results_df = pd.DataFrame(results)

# 保存结果
output_path = "/home/huawei/project/nnUNet/Data_preprocessing/fold4_group_metrics_subset.csv"
results_df.to_csv(output_path, index=False)

print("分组统计完成，保存到：", output_path)
print(results_df)
