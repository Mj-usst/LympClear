# LympClear
Improving Diagnostic Precision for Lower Limb Lymphedema: Suppressing Vein Signal Interference Using Deep Learning in MR Lymphangiography

# LympClear: Deep Learning for Vein Suppression in MRL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.xxxx/zenodo.xxxxxx)  # 可选（若同步发布数据集）

# LympClear: Deep Learning for Venous Signal Suppression in MR Lymphangiography

![LympClear Workflow](figures/workflow.png)

## 🧠 Overview

**LympClear** is a deep learning–based framework developed to suppress venous signal interference in Magnetic Resonance Lymphangiography (MRL), significantly enhancing diagnostic accuracy for **lower extremity lymphedema (LE)**. This project introduces a novel signal suppression approach, dynamic image generation pipeline, and a lymphatic development grading system to improve anatomical clarity and aid clinical decision-making.

---

## 🔍 Background

MRL is commonly used for LE diagnosis, but high-intensity venous signals often obscure lymphatic structures, making interpretation difficult and time-consuming. LympClear addresses this issue using a model trained with a **brightness-matching technique**, based on the **nnUNet** architecture, to automatically remove venous signals and enhance lymphatic visibility.

---

## 🚀 Highlights

- 🧠 **Model**: Custom-trained nnUNet with brightness-matching strategy  
- 🧪 **Dataset**: 1022 patients, 6162 dynamic scans, multi-center, 2007–2024  
- 🎞 **Output**: High-quality static and dynamic MRL images  
- 📈 **Clinical Value**: Improved diagnosis, reduced reading time, enhanced consistency

---

## 📊 Results

| Metric                          | Before LympClear | After LympClear |
|---------------------------------|------------------|-----------------|
| Dice for vein segmentation      | N/A              | **0.940**       |
| Image quality (1–10)            | 6.7 ± 1.1        | **7.5 ± 0.9**   |
| Vein removal clarity            | 5.8 ± 1.2        | **8.8 ± 0.7**   |
| Lymphatic visibility            | 6.0 ± 1.3        | **8.3 ± 1.0**   |
| Radiologist reading time        | Baseline         | **↓ 87%**       |
| Reflux detection improvement    | -                | **↑ 18.7%**     |
| Missed diagnosis rate           | 15.4%            | **7.2%**        |
| Cohen’s Kappa (diagnostic κ)    | 0.65             | **0.91**        |

---

## 🧬 Method

<p align="center">
  <img src="figures/model_architecture.png" width="600"/>
</p>

1. **Input**: 3D dynamic MRL scans  
2. **Preprocessing**: Normalization, registration, manual vein annotation  
3. **Training**: nnUNet + brightness matching for venous signal suppression  
4. **Output**:  
   - Cleaned MRL images (vein-free)  
   - Dynamic visualization of contrast agent flow  
   - Lymphatic development grading  

---

## 📦 Repository Structure



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
![Grade II](zeromip_image_comparison_vein_10394_0000.nii.png)  
*图示：淋巴管发育不全（Hypoplasia）*

### Grade III: Hyperplasia  
![Grade III](zeromip_image_comparison_vein_10043_0000.nii.png)  
*图示：淋巴管增生（Hyperplasia）*

### Grade IV: Severe Hyperplasia  
![Grade IV](zeromip_image_comparison_vein_10897_0000.nii.png)  
*图示：淋巴管严重增生（Severe Hyperplasia）*


## 📄 引用
```bibtex
@article{yourname2024lympclear,
  title={Improving Diagnostic Precision for Lower Limb Lymphedema...},
  author={Your Name, et al.},
  journal={Journal of Medical Imaging},
  year={2024}
}
