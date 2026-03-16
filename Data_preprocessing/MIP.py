import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

def process_images(image_dir, label_dir, output_dir, vein_color=(0, 0, 0)):
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob(os.path.join(image_dir, "*.nii.gz"))

    for image_file in image_files:
        base_name = os.path.basename(image_file)
        parts = base_name.split('_')
        patient_id = parts[1]

        mask_file = os.path.join(label_dir, f"vein_{patient_id}.nii.gz")
        if not os.path.exists(mask_file):
            print(f"⚠️ 未找到 {mask_file}，跳过")
            continue

        print(f"✅ 正在处理: {patient_id}")
        img_nii = nib.load(image_file)
        mask_nii = nib.load(mask_file)
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        if img_data.shape != mask_data.shape:
            print(f"❌ 尺寸不匹配，跳过 {patient_id}")
            continue

        # 标准化 + 灰度图转换
        img_normalized = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        gray_values = (img_normalized * 255).astype(np.uint8)
        base_rgb = np.stack([gray_values] * 3, axis=-1)

        # 第一张图：原始图像
        original_img = base_rgb.copy()

        # 第二张图：提亮（全图 ×1.5）
        brightened = np.clip(base_rgb * 1.5, 0, 255).astype(np.uint8)

        # 第三张图：在提亮图上抑制静脉（mask > 0 区域设为黑色）
        suppressed = brightened.copy()
        suppressed[mask_data > 0] = vein_color  # 静脉设为黑色

        # MIP函数（矢状位）
        def generate_mip(data):
            mip = np.max(data, axis=2)
            mip = np.rot90(mip, k=1)
            mip = np.flipud(mip)
            return mip

        mip_original = generate_mip(original_img)
        mip_bright = generate_mip(brightened)
        mip_suppressed = generate_mip(suppressed)

        # 三合一可视化
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        titles = ['Original MIP', 'Brightened MIP', 'Vein Suppressed MIP']

        for ax, img, title in zip(axs, [mip_original, mip_bright, mip_suppressed], titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=18)
            ax.axis('off')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{patient_id}_3view_mip.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        print(f"✅ 已保存: {output_path}")

if __name__ == "__main__":
    image_dir = r'/home/huawei/project/dataset/infer_nii/result'
    label_dir = r'/home/huawei/project/dataset/infer_result'
    output_dir = r'/home/huawei/project/dataset/MIP'
    vein_color = (0, 0, 0)  # 静脉设为黑色

    process_images(image_dir, label_dir, output_dir, vein_color)
