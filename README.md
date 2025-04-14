# LympClear
Improving Diagnostic Precision for Lower Limb Lymphedema: Suppressing Vein Signal Interference Using Deep Learning in MR Lymphangiography

# LympClear: Deep Learning for Vein Suppression in MRL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.xxxx/zenodo.xxxxxx)  # 可选（若同步发布数据集）

> **Abstract**  
> 此处粘贴你的摘要（精简至 3-5 行）。

## 🚀 核心功能
- **静脉信号抑制**: Dice 系数 0.940，阅读时间减少 87%  
- **淋巴发育分级**: 与临床分期强相关（Kappa 0.91）  
- **动态可视化**: 对比剂时空流动展示（见 `notebooks/`）  

## Lymphatic Development Grading System

| Grade       | Description        | Visualization                      |
|-------------|--------------------|------------------------------------|
| **Grade I** | Aplasia            | ![Grade I](figures/grade_i.png)    |
| **Grade II**| Hypoplasia         | ![Grade II](figures/grade_ii.png)  |
| **Grade III**| Hyperplasia        | ![Grade III](figures/grade_iii.png)|
| **Grade IV**| Severe Hyperplasia | ![Grade IV](figures/grade_iv.png)  |

## Lymphatic Development Grading System

### Grade I: Aplasia
![Grade I](zeromip_image_comparison_vein_10716_0000.nii.png)  
*图示：淋巴管发育不良（Aplasia）*

### Grade II: Hypoplasia  
![Grade II](figures/grade_ii.png)  
*图示：淋巴管发育不全（Hypoplasia）*

### Grade III: Hyperplasia  
![Grade III](figures/grade_iii.png)  
*图示：淋巴管增生（Hyperplasia）*

### Grade IV: Severe Hyperplasia  
![Grade IV](figures/grade_iv.png)  
*图示：淋巴管严重增生（Severe Hyperplasia）*

## 📄 引用
```bibtex
@article{yourname2024lympclear,
  title={Improving Diagnostic Precision for Lower Limb Lymphedema...},
  author={Your Name, et al.},
  journal={Journal of Medical Imaging},
  year={2024}
}
