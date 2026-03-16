import os
import glob
import numpy as np
import imageio.v2 as imageio  # 使用 v2 以避免弃用警告
from PIL import Image, ImageFilter
from PIL import Image
from PIL import ImageFile
import warnings

# 关闭 DecompressionBomb 警告
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# 可选：允许加载更大的图像（默认最大为 178956970 像素）
Image.MAX_IMAGE_PIXELS = None  # 或者设置为更大的值



# 设置图片所在的文件夹路径
image_folder = r'/home/huawei/project/dataset/MIP'  # ← 修改为你的文件夹路径

# 获取所有图片路径，支持png/jpg等，按文件名排序
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.*')))

# 读取所有图片并转换为 PIL.Image 对象，统一转换为 RGB 模式
images = [Image.open(img_path).convert('RGB') for img_path in image_paths]

# 应用轻微平滑滤镜美化图像（可根据需要修改或增加滤镜）
images = [img.filter(ImageFilter.SMOOTH) for img in images]

# 定义一个函数，在两张图片之间生成中间帧
def create_intermediate_frames(img1, img2, n_frames=5):
    intermediate_frames = []
    for i in range(1, n_frames + 1):
        alpha = i / (n_frames + 1)
        blended = Image.blend(img1, img2, alpha)
        intermediate_frames.append(blended)
    return intermediate_frames

# 构造包含原图及中间帧的 GIF 帧列表
gif_frames = []
n_intermediate = 3  # 每对连续图片之间补 3 帧
for i in range(len(images) - 1):
    gif_frames.append(images[i])
    # 生成并加入中间帧
    intermediates = create_intermediate_frames(images[i], images[i+1], n_frames=n_intermediate)
    gif_frames.extend(intermediates)
gif_frames.append(images[-1])

# 转换帧为 numpy 数组（imageio.mimsave 需要数组格式）
gif_frames_np = [np.array(frame) for frame in gif_frames]

# 保存为 GIF 动图，设置每帧显示时间（单位秒）
output_path = '/home/huawei/project/dataset/GIF/test.gif'
imageio.mimsave(output_path, gif_frames_np, duration=0.2)

print(f'GIF 动图已保存为 {output_path}')
