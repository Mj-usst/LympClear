import pandas as pd
import numpy as np
from scipy import stats

# 读取 CSV
df = pd.read_csv("/home/huawei/project/nnUNet/Data_preprocessing/fold0_validation_metrics.csv")

# 需要计算的指标
metrics = ["Dice", "IoU", "Sensitivity", "Specificity"]

def mean_sd_ci(series, alpha=0.05):
    """返回均值、标准差、95%CI"""
    x = series.dropna().to_numpy()
    mean = np.mean(x)
    sd = np.std(x, ddof=1)
    n = len(x)
    se = sd / np.sqrt(n)
    tval = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_lo = mean - tval * se
    ci_hi = mean + tval * se
    return mean, sd, ci_lo, ci_hi

# 计算结果
results = {}
for m in metrics:
    mean, sd, lo, hi = mean_sd_ci(df[m])
    results[m] = {
        "mean": mean,
        "sd": sd,
        "95CI_low": lo,
        "95CI_high": hi
    }

# 打印结果
res_df = pd.DataFrame(results).T
print(res_df)

# 保存为 CSV
res_df.to_csv("fold0_variability_metrics.csv")
